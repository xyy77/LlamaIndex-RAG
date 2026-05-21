import pandas as pd

enable_semantic_splitter = True
chunk_size = 1024
chunk_overlap = 150
model_name = "Mistral Small 3"

df = pd.read_csv("eval/mistral_small_3_result1.csv", encoding="utf-8")

faith_rate = df["Faithfulness"].mean()
relev_rate = df["Relevancy"].mean()

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