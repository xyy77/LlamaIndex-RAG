---
title: LlamaIndex RAG Chat
sdk: streamlit
app_file: app.py
pinned: false
---

# RAG Chat - PDF 智能问答助手

[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face%20Demo-blue)](https://huggingface.co/spaces/tina-su/LlamaIndex-RAG)

基于 **LlamaIndex** + **Streamlit** 实现的 RAG（检索增强生成）应用。用户上传 PDF 文件后，可以用自然语言提问，系统根据文档内容给出准确、可靠的回答。

支持 **阿里云通义千问（DashScope）** 和 **OpenRouter** 多种大语言模型，并内置 RAG 质量评测功能。

## 项目目标与价值

- 支持单份 PDF 的智能问答
- 同时使用向量检索（适合找细节、引用、事实）和摘要检索（适合总结、整体理解）
- 通过路由机制（Router Retriever）让 LLM 自动判断使用哪种检索方式
- 使用重排序（Reranker）提升检索结果的精确性
- 支持流式输出 + 模型推理过程展示，提升用户体验
- 内置评估管线，便于量化 RAG 效果
- 适用场景：论文阅读、合同审查、技术手册查询、内部知识库问答等

## 技术栈一览

| 类别           | 技术 / 库                            | 主要用途                          |
|----------------|--------------------------------------|-----------------------------------|
| 前端界面       | Streamlit                           | 快速构建交互式聊天界面            |
| RAG 核心框架   | LlamaIndex                          | 文档加载、分块、索引、检索、对话引擎 |
| 大模型         | 通义千问 (qwen-max)、OpenRouter (Mistral Small 3, Llama 3.1 8B, Qwen3.6 Plus) | 文本生成、路由选择 |
| 向量嵌入       | DashScope text-embedding-v2         | 生成文本向量                      |
| PDF/TXT 解析   | SimpleDirectoryReader               | 加载 PDF/TXT 文档                 |
| 索引类型       | VectorStoreIndex + SummaryIndex     | 向量检索 + 文档摘要检索           |
| 检索路由       | RouterRetriever + LLMSingleSelector | 智能选择检索工具                  |
| 重排序         | DashScopeRerank (gte-rerank)        | 对检索结果重排，提升 Top-N 精确性  |
| 对话记忆       | ChatMemoryBuffer (可配置 token 上限) | 支持多轮上下文对话               |
| 评估指标       | FaithfulnessEvaluator + RelevancyEvaluator | 忠实度与相关性评测           |
| 环境变量管理   | python-dotenv                       | 安全管理 API Key                  |

## 核心实现逻辑

### 1. 文档处理流程
- 用户上传 PDF/TXT → 保存到临时目录
- 使用 SimpleDirectoryReader 解析文档
- SentenceSplitter（块大小 1024，交叠 150）或 SemanticSplitter 进行分块

### 2. 双索引设计
- **VectorStoreIndex**：用于精确定位细节、引用、具体事实
- **SummaryIndex**：用于回答概括性、整体理解类问题

### 3. 智能路由检索
```python
router_retriever = RouterRetriever(
    selector=LLMSingleSelector.from_defaults(),
    retriever_tools=[summary_tool, vector_tool],
    verbose=True
)
```
LLM 根据问题自动判断：
- 用 summary_tool（问主题、总结、大方向）
- 用 vector_tool（问具体段落、数据、引用）

### 4. 重排序（Reranker）
检索结果经过 DashScope gte-rerank 模型重排，从 Top-K（默认 15）中筛选出最相关的 Top-N（默认 5）节点，提升回答质量。

### 5. 对话引擎
- 使用 ContextChatEngine（保留完整上下文）
- 自定义 system prompt，强制模型基于上传的 PDF 内容回答，不编造内容
- 支持 streaming 流式输出（逐 token 显示）
- 内置 ChatMemoryBuffer 支持多轮对话

### 6. 思考过程可视化
- 自动识别模型输出中的 `` 标签
- 用 `st.expander` 折叠展示推理过程，主回答保持简洁

## 模块说明

| 文件                  | 功能                                     |
|-----------------------|------------------------------------------|
| `app.py`              | Streamlit 前端应用入口（HF Spaces 默认入口） |
| `utils.py`            | 核心工具函数：LLM 配置、索引构建、路由检索引擎 |
| `eval.py`             | 对数据集进行逐项评估，输出忠实度与相关性指标 |
| `generate_report.py`  | 从 CSV 结果生成格式化评估报告摘要 |

### 分块策略

项目支持两种文本分块方式：
- **SentenceSplitter**（默认）：按固定大小和交叠切分，适合大多数场景
- **SemanticSplitterNodeParser**：基于嵌入相似度动态切分，在语义边界处断开分块，适合结构复杂的文档

在 `utils.py` 的 `get_chat_engine()` 中通过 `enable_semantic_splitter` 参数切换。

### 重排序开关

重排序（Reranker）在聊天引擎中默认启用，可通过 `enable_reranker=False` 关闭。评估脚本 `eval.py` 中默认关闭以对比效果。

### 路由 Prompt 优化

为提升路由选择的稳定性，自定义了 `LLMSingleSelector` 的 prompt 模板，要求 LLM 输出标准 JSON 格式，确保双引号包裹键值对，避免解析失败。

## 如何运行

### 环境要求

- Python 3.8+
- 可联网访问 dashscope.aliyuncs.com 和 openrouter.ai

### 安装依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 已包含在项目中。

### 配置 API Key

复制 `.env.example` 为 `.env` 并填入真实 API Key：

```bash
cp .env.example .env
# 编辑 .env，替换为你的真实 Key
```

> 两个 Key 均需配置，DashScope 用于嵌入模型和 Qwen 模型，OpenRouter 用于免费模型。

### 启动应用

```bash
streamlit run app.py
```

浏览器会自动打开：http://localhost:8501

## 使用步骤

1. 侧边栏选择模型（Qwen Max / Mistral Small 3 / Llama 3.1 8B / Qwen3.6 Plus）
2. 上传一份 PDF 或 TXT 文件
3. 等待「PDF 加载完成」和「索引构建完成」
4. 在下方输入框提问
5. 可点击「清空对话」重置所有状态

## 评估

将数据集 JSON 和 PDF 文件准备好后，运行：

```bash
python eval.py
```

脚本会：
1. 加载评估数据集
2. 使用 ChatEngine 逐一生成回答
3. 分别评估 **忠实度（Faithfulness）** 和 **相关性（Relevancy）**
4. 输出汇总结果到 `eval/` 目录，文件名格式为 `{model_name}_result{0或1}.csv`，数字表示是否启用语义分块
5. 终端打印评估报告摘要

评估指标说明：
- **忠实度（Faithfulness）**：回答是否基于检索到的上下文，是否存在幻觉/编造
- **相关性（Relevancy）**：回答是否与问题相关

## 部署到 Hugging Face Spaces

1. 在 [huggingface.co/spaces](https://huggingface.co/spaces) 创建新 Space，SDK 选择 **Streamlit**
2. 将本项目文件（除 `.env` 外）推送到 Space 仓库
3. 在 Space Settings → Secrets 中添加：
   - `DASHSCOPE_API_KEY` = 你的 DashScope API Key
   - `OPENROUTER_API_KEY` = 你的 OpenRouter API Key
4. Space 会自动安装 `requirements.txt` 并启动 `app.py`

本地部署只需 `cp .env.example .env` 填入真实的 Key 后运行 `streamlit run app.py`。

## 当前限制 & 未来改进方向

- 仅支持单份文件（可扩展为多文件/文件夹）
- 未显示引用来源/页码（可添加 Citation 后处理）
- 未处理表格、图片内容（可引入 LlamaParse 或 Unstructured）
- TXT 文件支持已添加，但分块策略对非结构化文本的效果可能不如 PDF
- 可进一步加入 hybrid search 提升检索召回率
- 语义分块对长文档构建索引较慢，建议先用 SentenceSplitter 快速验证
