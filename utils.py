import os

from dotenv import load_dotenv
from llama_index.core import Settings, SummaryIndex, VectorStoreIndex
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.selectors.llm_selectors import DEFAULT_SINGLE_SELECT_PROMPT_TMPL
from llama_index.core.tools import RetrieverTool
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openrouter import OpenRouter
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
import dashscope
load_dotenv()

# 确保 DashScope SDK 能正确读取 API Key
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

ALL_MODELS = {
    # 阿里云 DashScope 模型
    "Qwen Max": {
        "provider": "dashscope",
        "model_id": "qwen-max",
        "context_window": 1000000
    },
    # OpenRouter 模型
    "Mistral Small 3": {
        "provider": "openrouter",
        "model_id": "mistralai/mistral-small-24b-instruct-2501",
        "context_window": 32768
    },
    "Llama 3.1 8B": {
        "provider": "openrouter",
        "model_id": "meta-llama/llama-3.1-8b-instruct",
        "context_window": 131072
    },
    "Qwen3.6 Plus": {
        "provider": "openrouter",
        "model_id": "qwen/qwen3.6-plus",
        "context_window": 1000000
    }
}


def get_llm(selected_label):
    config = ALL_MODELS.get(selected_label)
    if not config:
        raise ValueError(f"未定义的模型选择: {selected_label}")

    provider = config["provider"]
    model_id = config["model_id"]
    ctx_window = config["context_window"]

    if provider == "openrouter":
        return OpenRouter(
            model=model_id,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0,
            max_tokens=2048,
            context_window=ctx_window,
        )

    elif provider == "dashscope":
        return OpenAILike(
            model=model_id,
            api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            is_chat_model=True,
            temperature=0,
            max_tokens=2048,
            context_window=ctx_window,
        )
    return None


def get_chat_engine(selected_model,
                    docs=None,
                    top_k=15,
                    top_n=5,
                    m_size=40960,
                    enable_reranker=True,
                    enable_semantic_splitter=False,
                    chk_size=1024,
                    chk_overlap=150,
                    verbose=True):

    llm = get_llm(selected_model)

    embed_model = DashScopeEmbedding(
        model_name="text-embedding-v2",
        api_key=os.getenv("DASHSCOPE_API_KEY")
    )
    Settings.llm = llm
    Settings.embed_model = embed_model

    if enable_semantic_splitter:
        splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model)
    else:
        splitter = SentenceSplitter(chunk_size=chk_size, chunk_overlap=chk_overlap)

    nodes = splitter.get_nodes_from_documents(docs)

    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes)

    summary_retriever = summary_index.as_retriever()
    vector_retriever = vector_index.as_retriever(similarity_top_k=top_k)

    summary_tool = RetrieverTool.from_defaults(
        retriever=summary_retriever,
        description="Useful for summarization questions and getting a high-level overview."
    )

    vector_tool = RetrieverTool.from_defaults(
        retriever=vector_retriever,
        description="Useful for retrieving specific context, details, quotes, or facts from the document."
    )

    # 在默认prompt后加入格式要求，确保输出稳定可解析
    new_tmpl = DEFAULT_SINGLE_SELECT_PROMPT_TMPL + "\nCRITICAL: You must use double quotes for all JSON keys and string values. Do not use single quotes or unquoted keys."

    router_retriever = RouterRetriever(
        selector=LLMSingleSelector.from_defaults(prompt_template_str=new_tmpl),
        retriever_tools=[summary_tool, vector_tool],
        verbose=True
    )

    memory = ChatMemoryBuffer.from_defaults(
        token_limit=m_size,
    )

    # 添加阿里云的rerank重排模型来增加精确性
    post_processors = []
    if enable_reranker:
        reranker = DashScopeRerank(model="gte-rerank", top_n=top_n, api_key=os.getenv("DASHSCOPE_API_KEY"))
        post_processors.append(reranker)

    chat_engine = ContextChatEngine.from_defaults(
        retriever=router_retriever,
        memory=memory,
        system_prompt=(
            "你是一个专业的PDF问答助手，要根据提供的文件内容进行回答，"
            "不要编造内容，优先基于提供的PDF内容。"
        ),
        node_postprocessors=post_processors,
        verbose=verbose
    )

    return chat_engine
