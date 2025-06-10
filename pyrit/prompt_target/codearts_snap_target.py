import logging
import json
from typing import Optional

# Assuming these imports are available from the pyrit library
# If running this code standalone, you would need to install pyrit and its dependencies.
from pyrit.exceptions import EmptyResponseException, handle_bad_request_exception, pyrit_target_retry
from pyrit.models import PromptRequestResponse, construct_response_from_request
from pyrit.prompt_target import PromptTarget, limit_requests_per_minute
from pyrit.common import default_values
import httpx # Using httpx for async requests, consistent with CrucibleTarget

logger = logging.getLogger(__name__)

# PanguAgentTarget class, structured similarly to CrucibleTarget
class CodeArtsSnapTarget(PromptTarget):

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        domain: Optional[str] = None,
        refresh = True,
        max_requests_per_minute: Optional[int] = None,
    ) -> None:
        """
        Initializes the CodeArtsSnapTarget.

        Args:
            endpoint (str): The API endpoint for the Pangu Agent.
                            Defaults to a hardcoded URL with a specific conversation ID.
            max_requests_per_minute (Optional[int]): Maximum number of requests per minute to allow.
        """
        super().__init__(max_requests_per_minute=max_requests_per_minute)
        self._refresh = refresh
        self._conversation_id = None
        self._endpoint = "https://snap-access.cn-north-4.myhuaweicloud.com/v1/chat"
        self._token = default_values.get_required_value(
            env_var_name="HUAWEICLOUD_IAM_TOKEN", passed_value=token
        )
        self._domain = default_values.get_required_value(
            env_var_name="HUAWEICLOUD_IAM_DOMAIN", passed_value=domain
        )

        self._start_chat_header = {
            "X-Auth-Token": self._token,
            "Content-Type": "application/json",
            "Agent-Type": "ChatAgent",
        }

        self._chat_header = {
            "X-Auth-Token": self._token,
            "Content-Type": "application/json",
            "Agent-Type": "ChatAgent",
            "Accept": "text/event-stream",
            "X-Domain-Id": self._domain
        }

    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Sends a normalized prompt asynchronously to the Pangu Agent.

        Args:
            prompt_request (PromptRequestResponse): The prompt request object containing the text prompt.

        Returns:
            PromptRequestResponse: The response from the Pangu Agent.
        """
        request_piece = prompt_request.request_pieces[0]

        logger.info(f"Sending the following prompt to the Pangu Agent: {request_piece.converted_value}")

        try:
            response_text = await self._call_codearts_snap_api_async(request_piece.converted_value)
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
    async def _call_codearts_snap_api_async(self, prompt: str) -> str:
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
            "chat_id": "",
            "question": prompt,
            "client": "IDE",
            "task": "chat",
            "task_parameters": {
                "context_code": "",
                "ide": "CodeArts IDE"
            },
            "knowledge_ids": []
        }

        start_payload = {
            "client": "IDE"
        }

        final_response_text = ""

        try:
            if not self._conversation_id or self._refresh:
                with httpx.Client() as client:
                    response = client.post(f'{self._endpoint}/start-chat', headers=self._start_chat_header, json=start_payload)
                    response.raise_for_status()
                    self._conversation_id = response.json()["chat_id"]
                    payload["chat_id"] = self._conversation_id

            async with httpx.AsyncClient() as client:
                # Make the POST request with stream=True to handle streaming data
                async with client.stream(
                    "POST",
                    f'{self._endpoint}/chat',
                    headers=self._chat_header,
                    json=payload,
                    timeout=None # Set timeout to None for potentially long-running streams
                ) as response:
                    # Check status code before accessing streaming content
                    if response.status_code >= 400:
                        # Read the error response content first
                        error_content = await response.aread()
                        error_text = error_content.decode('utf-8')
                        logger.error(f"CodeArts Snap Agent API request failed with status code: {response.status_code}. Response: {error_text}")
                        response.raise_for_status() # This will now work properly

                    logger.info("Connected to SSE stream. Receiving data from  Agent...")
                    
                    last_text_content = ""  # 保存最后一个有效的text内容

                    async for line in response.aiter_lines():
                        if line:
                            decoded_line = line.strip()
                            # SSE data lines typically start with "data: "
                            if decoded_line.startswith('data:'):
                                data_str = decoded_line[len('data:'):].strip()
                                try:
                                    data_json = json.loads(data_str)
                                    # Based on the actual response trace, the content is under 'text' key
                                    if 'text' in data_json and data_json['text'] is not None:
                                        if data_json['text'] == '[DONE]':
                                            # 遇到[DONE]标记，结束处理
                                            break
                                        else:
                                            # 每个text都是完整的内容，保存最新的
                                            last_text_content = data_json['text']
                                except json.JSONDecodeError:
                                    logger.warning(f"Could not decode JSON from data: '{data_str}'")
                            elif decoded_line.startswith('event:'):
                                # Ignore event lines for final content accumulation
                                pass
                            elif decoded_line == '': # Empty line separating SSE events
                                pass
                            else:
                                # This case might occur if the server sends non-standard SSE or raw text that's not 'data:'
                                logger.debug(f"Non-data/event line received from CodeArts Snap Agent: {decoded_line}")
                    
                    # 使用最后一个有效的文本内容
                    final_response_text = last_text_content

            if not final_response_text:
                raise EmptyResponseException("CodeArts Snap Agent returned an empty response.")

            logger.info(f"--- Final Accumulated Response from CodeArts Snap Agent ({len(final_response_text)} chars) ---")
            # logger.info(final_response_text) # Uncomment to see the full response in logs

            return final_response_text

        except httpx.HTTPStatusError as e:
            # For streaming responses, the error text may have already been read and logged above
            try:
                error_text = e.response.text
            except Exception:
                error_text = "Error content not available (streaming response)"
            logger.error(f"CodeArts Snap Agent API request failed with status code: {e.response.status_code}. Response: {error_text}")
            raise e
        except httpx.RequestError as e:
            logger.error(f"Network or request error during CodeArts Snap Agent API call: {e}")
            raise e
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error for CodeArts Snap Agent API response: {e}")
            raise e
        except Exception as e:
            logger.error(f"An unexpected error occurred during CodeArts Snap Agent API interaction: {e}")
            raise e