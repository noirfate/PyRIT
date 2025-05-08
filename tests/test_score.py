import asyncio
from pyrit.common import initialize_pyrit
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.common import RDS, initialize_pyrit
from pyrit.memory import CentralMemory
from pyrit.score import (
    FloatScaleThresholdScorer,
    SelfAskScaleScorer,
)

CONNECTION_STRING = 'postgresql://xxx'
SCHEMA_NAME = 'pyrit'
DB_TYPE = 'postgresql'

memory_labels = {"xxx": "xxx"}
initialize_pyrit(memory_db_type=RDS, db_type=DB_TYPE, schema_name=SCHEMA_NAME, connection_string=CONNECTION_STRING)

score_target = OpenAIChatTarget()
objective_scorer = FloatScaleThresholdScorer(
    scorer=SelfAskScaleScorer(
        chat_target=score_target,
        scale_arguments_path=SelfAskScaleScorer.ScalePaths.TASK_ACHIEVED_SCALE.value,
        system_prompt_path=SelfAskScaleScorer.SystemPaths.RED_TEAMER_SYSTEM_PROMPT.value,
    ),
    threshold=0.7,
)

async def score_results():
    memory = CentralMemory.get_memory_instance()

    prompts = memory.get_prompt_request_pieces(labels=memory_labels)
    print(f"找到 {len(prompts)} 个提示")

    conversations = {}
    for piece in prompts:
        if piece.role == "system":
            continue
        conv_id = piece.conversation_id
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(piece)

    print(f"共有 {len(conversations)} 个对话")

    # 只适合一问一答类型，不适用于多轮会话
    for i, (conv_id, pieces) in enumerate(conversations.items()):
        print(f"为对话{i}: {conv_id} 创建评分...")
        pieces.sort(key=lambda x: x.sequence)
        scores = memory.get_scores_by_prompt_ids(prompt_request_response_ids=[str(pieces[1].id)])
        if not scores:
            await objective_scorer.score_async(request_response=pieces[1], task=pieces[0].converted_value)

asyncio.run(score_results())

# 查询成功的对话
#SELECT 
#    pme.id,
#    pme.role,
#    pme.conversation_id,
#    pme.converted_value,
#    se.task
#FROM pyrit."PromptMemoryEntries" AS pme
#JOIN pyrit."ScoreEntries" AS se
#    ON se.prompt_request_response_id = pme.id
#WHERE se.score_value = 'True';
