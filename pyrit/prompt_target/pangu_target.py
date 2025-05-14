import logging
import json
from typing import Optional

# Assuming these imports are available from the pyrit library
# If running this code standalone, you would need to install pyrit and its dependencies.
from pyrit.exceptions import EmptyResponseException, handle_bad_request_exception, pyrit_target_retry
from pyrit.models import PromptRequestResponse, construct_response_from_request
from pyrit.prompt_target import PromptTarget, limit_requests_per_minute
import httpx # Using httpx for async requests, consistent with CrucibleTarget

logger = logging.getLogger(__name__)

# PanguAgentTarget class, structured similarly to CrucibleTarget
class PanguAgentTarget(PromptTarget):

    def __init__(
        self,
        *,
        cookie_str: str,
        max_requests_per_minute: Optional[int] = None,
    ) -> None:
        """
        Initializes the PanguAgentTarget.

        Args:
            endpoint (str): The API endpoint for the Pangu Agent.
                            Defaults to a hardcoded URL with a specific conversation ID.
            max_requests_per_minute (Optional[int]): Maximum number of requests per minute to allow.
        """
        super().__init__(max_requests_per_minute=max_requests_per_minute)
        self._endpoint = "https://portal.huaweicloud.com/rest/apigw/cdi/rest/cdi/pangudoercoreservice/v1/conversations"
        self.build_cookie(cookie_str)
        self._header = {
            'accept': '*/*',
            'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8,ja;q=0.7,zh-TW;q=0.6',
            'cache-control': 'no-cache',
            'content-type': 'application/json',
            'origin': 'https://www.huaweicloud.com',
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': 'https://www.huaweicloud.com/',
            'sec-ch-ua': '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
            'wise-groupid': '3.d3d3Lmh1YXdlaWNsb3VkLmNvbS98L2h0bWwvYm9keS9kaXZbMl0vZGl2L2Rpdi9kaXZpL2Rpdi9kaXZbMV0vZGl2WzJdL2Rpdi9hL2Rpdi9kaXZvZGl2WzFdL2RpdnwKICAgICAgICAgICAgICAKICAgICAgICA=.e6fb51e1fe64495aabbfdfaff6d3716e.3m7w8PZj.1747038759880',
            }

    def build_cookie(self, cookie_str):
        self.cookies = {}
        for cookie_pair in cookie_str.split('; '):
            if '=' in cookie_pair:
                key, value = cookie_pair.split('=', 1)
                self.cookies[key] = value

    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Sends a normalized prompt asynchronously to the Pangu Agent.

        Args:
            prompt_request (PromptRequestResponse): The prompt request object containing the text prompt.

        Returns:
            PromptRequestResponse: The response from the Pangu Agent.
        """
        self._validate_request(prompt_request=prompt_request)
        request_piece = prompt_request.request_pieces[0]

        logger.info(f"Sending the following prompt to the Pangu Agent: {request_piece.converted_value}")

        try:
            response_text = await self._call_pangu_api_async(request_piece.converted_value)
            response_entry = construct_response_from_request(request=request_piece, response_text_pieces=[response_text])
        except httpx.HTTPStatusError as bre:
            # This mirrors CrucibleTarget's handling of HTTPStatusError
            if bre.response.status_code == 400:
                response_entry = handle_bad_request_exception(
                    response_text=bre.response.text, request=request_piece, is_content_filter=False # Adjust is_content_filter as per Pangu's behavior
                )
            else:
                logger.error(f"HTTP Status Error from Pangu Agent: {bre}")
                raise
        except httpx.RequestError as e:
            logger.error(f"An HTTPX request error occurred while communicating with Pangu Agent: {e}")
            raise # Re-raise the exception for PyRIT to handle
        except Exception as e:
            logger.error(f"An unexpected error occurred during Pangu Agent API call: {e}")
            raise # Re-raise the exception

        return response_entry

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        """
        Validates the prompt request to ensure it's suitable for the Pangu Agent.
        """
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("PanguAgentTarget only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("PanguAgentTarget only supports text prompt input.")

    @pyrit_target_retry
    async def _call_pangu_api_async(self, prompt: str) -> str:
        """
        Makes the asynchronous API call to the Pangu Agent, handling streaming responses.

        Args:
            prompt (str): The text prompt to send to the Pangu Agent.

        Returns:
            str: The accumulated response text from the Pangu Agent.

        Raises:
            httpx.HTTPStatusError: If the API request returns an HTTP error status.
            httpx.RequestError: If a network-related error occurs during the request.
            Exception: For any other unexpected errors during API interaction.
        """

        payload = {
            "context_info": {
                "device": {
                    "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
                    "type": "pc"
                },
                "referrer": {
                    "hidden_question": False
                }
            },
            "result_info": {},
            "version_id": "V1",
            "question": prompt,
            "is_refresh": 0
        }

        start_payload = {
            "context_info": {
                "device": {
                    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0",
                    "type":"pc"
                }
            }
        }

        final_response_text = ""

        try:
            conversation_id = None
            with httpx.Client() as client:
                response = client.post(f'{self._endpoint}/start', headers=self._header, cookies=self.cookies, json=start_payload)
                response.raise_for_status()
                conversation_id = response.json()["conversation_id"]

            async with httpx.AsyncClient() as client:
                # Make the POST request with stream=True to handle streaming data
                async with client.stream(
                    "POST",
                    f'{self._endpoint}/{conversation_id}/send-message',
                    headers=self._header,
                    cookies=self.cookies,
                    json=payload,
                    timeout=None # Set timeout to None for potentially long-running streams
                ) as response:
                    response.raise_for_status() # Raise an exception for 4xx/5xx responses

                    logger.info("Connected to SSE stream. Receiving data from Pangu Agent...")

                    async for line in response.aiter_lines():
                        if line:
                            decoded_line = line.strip()
                            # SSE data lines typically start with "data: "
                            if decoded_line.startswith('data:'):
                                data_str = decoded_line[len('data:'):].strip()
                                try:
                                    data_json = json.loads(data_str)
                                    # Based on the original Chrome trace, the content is directly under 'content' key.
                                    if 'content' in data_json and data_json['content'] is not None:
                                        final_response_text += data_json['content']
                                except json.JSONDecodeError:
                                    logger.warning(f"Could not decode JSON from data: '{data_str}'")
                                    # If it's not JSON, it might be a plain text chunk, or an empty line that was stripped.
                                    if data_str:
                                        final_response_text += data_str
                            elif decoded_line.startswith('event:'):
                                # Ignore event lines for final content accumulation
                                pass
                            elif decoded_line == '': # Empty line separating SSE events
                                pass
                            else:
                                # This case might occur if the server sends non-standard SSE or raw text that's not 'data:'
                                logger.debug(f"Non-data/event line received from Pangu Agent: {decoded_line}")

            if not final_response_text:
                raise EmptyResponseException("Pangu Agent returned an empty response.")

            logger.info(f"--- Final Accumulated Response from Pangu Agent ({len(final_response_text)} chars) ---")
            # logger.info(final_response_text) # Uncomment to see the full response in logs

            return final_response_text

        except httpx.HTTPStatusError as e:
            logger.error(f"Pangu Agent API request failed with status code: {e.response.status_code}. Response: {e.response.text}")
            raise e
        except httpx.RequestError as e:
            logger.error(f"Network or request error during Pangu Agent API call: {e}")
            raise e
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error for Pangu Agent API response: {e}")
            raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred during Pangu Agent API interaction: {e}")
            raise e
