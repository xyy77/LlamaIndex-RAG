import os
import re

from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openrouter import OpenRouter
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from llama_index.retrievers.bm25 import BM25Retriever
import jieba
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


# ── 动态检索：查询复杂度分类 ──
_EXACT_ARTICLE_RE = re.compile(r'第[零〇一二三四五六七八九十百千万0-9]+条')
_COMPLEX_SIGNALS = [
    '什么情况下', '如何', '哪些', '条件', '区别', '比较',
    '分别', '同时', '以及', '或者', '还是', '怎么办',
]


def get_adaptive_top_k(query: str) -> int:
    """根据查询复杂度返回合适的 top_k。"""
    if _EXACT_ARTICLE_RE.search(query):
        return 5
    if len(query) < 10:
        return 8
    if len(query) > 50 or any(sig in query for sig in _COMPLEX_SIGNALS):
        return 25
    return 15


def route_query(query: str, available_laws: list) -> str | None:
    """判断查询是否应限定到某部法律（关键词匹配）。"""
    for law in available_laws:
        if law in query:
            return law
    return None


def filter_nodes_by_law(nodes, law_name: str | None):
    """如果指定了 law_name，只保留匹配的节点。"""
    if law_name is None:
        return nodes
    return [n for n in nodes if n.metadata.get("law_name") == law_name]


class AdaptiveRetriever(BaseRetriever):
    """包装融合检索器，检索时根据 query 动态调整 top_k，并内置 rerank。"""

    def __init__(self, fusion_retriever, reranker=None, top_n=5, available_laws=None):
        super().__init__()
        self._fusion = fusion_retriever
        self._reranker = reranker
        self._top_n = top_n
        self._available_laws = available_laws or []

    def _retrieve(self, query, **kwargs):
        query_str = query.query_str if hasattr(query, 'query_str') else str(query)
        top_k = get_adaptive_top_k(query_str)
        target_law = route_query(query_str, self._available_laws)
        for r in self._fusion._retrievers:
            r._similarity_top_k = top_k
        self._fusion._similarity_top_k = top_k
        nodes = self._fusion._retrieve(query)
        nodes = filter_nodes_by_law(nodes, target_law)
        if self._reranker:
            nodes = self._reranker.postprocess_nodes(nodes, query)
        return nodes[:self._top_n]


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
                    custom_splitter=None,
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

    if custom_splitter:
        splitter = custom_splitter
    elif enable_semantic_splitter:
        splitter = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model)
    else:
        splitter = SentenceSplitter(chunk_size=chk_size, chunk_overlap=chk_overlap)

    nodes = splitter.get_nodes_from_documents(docs)

    # 向量索引 + 语义检索器
    vector_index = VectorStoreIndex(nodes)
    vector_retriever = vector_index.as_retriever(similarity_top_k=top_k)

    # BM25 关键词检索器（精确术语匹配，使用 jieba 中文分词）
    def chinese_tokenizer(text: str):
        return list(jieba.cut(text))

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
        tokenizer=chinese_tokenizer,
    )

    # RRF 融合：结合语义和关键词两路检索结果
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=top_k,
        mode=FUSION_MODES.RECIPROCAL_RANK,
        verbose=verbose,
    )

    memory = ChatMemoryBuffer.from_defaults(
        token_limit=m_size,
    )

    # 阿里云 rerank 重排模型
    reranker = None
    if enable_reranker:
        reranker = DashScopeRerank(model="gte-rerank", top_n=top_n, api_key=os.getenv("DASHSCOPE_API_KEY"))

    # 收集所有文档中的法律名称，用于查询路由
    available_laws = list({n.metadata.get("law_name", "") for n in nodes if n.metadata.get("law_name")})

    # AdaptiveRetriever：查询时动态调整 top_k + 路由 + 内置 rerank
    adaptive_retriever = AdaptiveRetriever(
        fusion_retriever=fusion_retriever,
        reranker=reranker,
        top_n=top_n,
        available_laws=available_laws,
    )

    chat_engine = ContextChatEngine.from_defaults(
        retriever=adaptive_retriever,
        memory=memory,
        system_prompt=(
            "你是一个专业的法律条文助手，专注于回答基于《民法典》《劳动合同法》等中国法律的问题。"
            "你必须严格依据提供的法律条款内容回答，不得编造任何法律条文。"
            "如果检索到的条款不能完全回答用户问题，请明确说明：根据现有条款，无法完全回答，建议咨询专业律师。"
            "回答时请标明所引用的具体条款（例如：依据《民法典》第XX条）。"
            "最后必须加上免责声明：以上内容由AI生成，不构成正式法律意见，如有疑问请咨询执业律师。"
        ),
        verbose=verbose
    )

    return chat_engine
