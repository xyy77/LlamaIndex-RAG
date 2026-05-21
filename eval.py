import os

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_dataset import LabelledRagDataset
from utils import get_chat_engine, ALL_MODELS, get_llm
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
import pandas as pd

dataset_path = "datasets/rag_dataset.json"

load_dotenv()

eval_model = "Qwen Max"
model_name = "Llama 3.1 8B"
enable_semantic_splitter = False
chunk_size = 1024
chunk_overlap = 150
chat_engine = get_chat_engine(model_name,
                              docs=SimpleDirectoryReader('Files').load_data(),
                              m_size=ALL_MODELS[model_name]['context_window'] * 0.5,
                              enable_semantic_splitter=enable_semantic_splitter,
                              enable_reranker=False,
                              verbose=False)

eval_llm = get_llm(eval_model)

faith_evaluator = FaithfulnessEvaluator(llm=eval_llm)
relev_evaluator = RelevancyEvaluator(llm=eval_llm)

rag_dataset = LabelledRagDataset.from_json(dataset_path)
results_list = []

print("开始评估...")
for i, example in enumerate(rag_dataset.examples):

    query = example.query
    ref_answer = example.reference_answer
    print(f"[{i + 1}/{len(rag_dataset.examples)}] 正在处理问题: {example.query[:30]}...")

    try:
        response_obj = chat_engine.chat(query)
        generated_answer = response_obj.response

        faith_result = faith_evaluator.evaluate_response(response=response_obj)

        relev_result = relev_evaluator.evaluate_response(query=query, response=response_obj)

        results_list.append({
            "Question": query,
            "Reference": ref_answer,
            "Response": generated_answer,
            "Faithfulness": faith_result.passing,
            "Relevancy": relev_result.passing,
        })

    except Exception as e:
        print(f"Error processing example {i + 1}: {str(e)}")
        results_list.append({
            "Question": query,
            "Response": f"ERROR: {str(e)}",
            "Faithfulness": None,
            "Relevancy": None,
        })


df = pd.DataFrame(results_list)

os.makedirs("eval", exist_ok=True)
output_file = f"eval/{model_name.lower().replace(' ', '_')}_result{1 if enable_semantic_splitter else 0}.csv"
print(f"评估完成！结果已保存至: {output_file}\n")

# 计算指标
faith_rate = df["Faithfulness"].mean()
relev_rate = df["Relevancy"].mean()
df.to_csv(output_file, index=False, encoding='utf-8')

print("\n" + "="*40)
print(f"      RAG 评估报告 - {model_name}")
print("="*40)
print(f"配置信息:")
print(f" - 模型名称:   {model_name}")
if not enable_semantic_splitter:
    print(f" - 分块大小:   {chunk_size}")
    print(f" - 分块重叠:   {chunk_overlap}")
else:
    print(f" - 分块策略:   语义分块")
print("-" * 40)
print(f"核心指标:")
print(f" - 忠实度 (Faithfulness): {faith_rate:.2%}")
print(f" - 相关性 (Relevancy):    {relev_rate:.2%}")
print("-" * 40)
print("="*40 + "\n")
