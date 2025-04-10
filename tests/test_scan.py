import asyncio
from pathlib import Path
import yaml

from pyrit.common import initialize_pyrit
from pyrit.cli.scanner_config import ScannerConfig
from pyrit.memory import CentralMemory
from pyrit.score import SelfAskRefusalScorer
from pyrit.prompt_target import OpenAIChatTarget
import sys

# 加载配置文件
script_dir = Path(__file__).parent
config_path = script_dir / "test_scan.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    raw_dict = yaml.safe_load(f)

# 解析配置
config = ScannerConfig(**raw_dict)

# 指定明确的数据库路径
db_path = script_dir / "results.duckdb"

# 初始化PyRIT，明确传递db_path参数
initialize_pyrit(memory_db_type=config.database.db_type, db_path=db_path)

def run_scan():
    from pyrit.cli.__main__ import run_scenarios_async
    asyncio.run(run_scenarios_async(config))

def load_results():
    memory = CentralMemory.get_memory_instance()

    # 使用你在配置文件中设置的内存标签
    memory_labels = {
        "operator": "tester",
        "operation": "op_test"
    }

    # 从数据库获取已存储的提示
    prompts = memory.get_prompt_request_pieces()
    print(f"找到 {len(prompts)} 个提示")

    user_prompts = []
    model_responses = []

    for piece in prompts:
        if piece.role == "user":
            user_prompts.append(piece)
        elif piece.role == "assistant":
            model_responses.append(piece)

    print(f"用户提示: {len(user_prompts)} 个")
    print(f"模型回复: {len(model_responses)} 个")
    
    # 检查是否有未评分的回复，如果有则进行评分
    async def score_unevaluated_responses():
        scorer = SelfAskRefusalScorer(chat_target=OpenAIChatTarget())
        scored_count = 0
        
        for response in model_responses:
            # 检查这个回复是否已经有评分
            existing_scores = memory.get_scores_by_prompt_ids(prompt_request_response_ids=[str(response.id)])
            if not existing_scores:
                print(f"为ID {response.id} 的回复创建评分...")
                # 获取对应的用户提示作为task
                user_prompt = None
                for p in user_prompts:
                    if p.conversation_id == response.conversation_id and p.sequence < response.sequence:
                        user_prompt = p
                
                if user_prompt:
                    try:
                        # 执行评分
                        scores = await scorer.score_async(request_response=response, task=user_prompt.converted_value)
                        print(f"评分创建成功: {scores[0].get_value()}")
                        scored_count += 1
                    except Exception as e:
                        print(f"评分失败: {str(e)}")
        
        print(f"总共为 {scored_count} 个回复创建了新的评分")
    
    # 执行评分
    if len(model_responses) > 0:
        print("\n===== 检查和创建评分 =====")
        asyncio.run(score_unevaluated_responses())
    
    # 获取并打印评分结果
    print("\n===== 评分结果 =====")
    
    #all_scores = memory.get_scores()
    #print(f"数据库中总共有 {len(all_scores)} 个评分")
    
    scores = memory.get_scores_by_memory_labels(memory_labels=memory_labels)
    print(f"使用标签 {memory_labels} 找到 {len(scores)} 个评分")
    
    # 打印详细的评分信息
    for score in scores:
        prompt_id = score.prompt_request_response_id
        prompt_pieces = memory.get_prompt_request_pieces(prompt_ids=[str(prompt_id)])
        
        if prompt_pieces:
            prompt = prompt_pieces[0]
            print(f"\n评分ID: {score.id}")
            print(f"提示角色: {prompt.role}")
            print(f"提示内容: {prompt.converted_value[:100]}..." if len(prompt.converted_value) > 100 else prompt.converted_value)
            print(f"评分值: {score.get_value()}")
            print(f"评分理由: {score.score_rationale}")
            print("-" * 50)

if __name__ == "__main__":
    # 提供选项运行扫描或仅查看结果
    if len(sys.argv) > 1 and sys.argv[1] == "--scan":
        print("执行扫描...")
        run_scan()
        print("扫描完成，现在加载结果...")
    
    # 加载并显示结果
    load_results()
    
    # 关闭连接
    CentralMemory.get_memory_instance().dispose_engine()
