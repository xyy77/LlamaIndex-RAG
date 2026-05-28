import streamlit as st
import os
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
import tempfile
import shutil
import re

load_dotenv()

import jieba

from utils import get_chat_engine, ALL_MODELS
from legal_chunker import LegalArticleSplitter

LEGAL_DISCLAIMER = (
    ":warning: **免责声明**：本工具生成的回答基于公开法律条文，"
    "由 AI 模型输出，仅供参考，不构成正式法律意见。"
    "涉及具体法律事务，请咨询执业律师。"
)

LEGAL_SUGGESTIONS = {
    "试用期": "试用期的时长和工资标准是如何规定的？",
    "经济补偿": "用人单位需要支付经济补偿金的情形有哪些？",
    "经济补偿金": "经济补偿金的计算标准是什么？",
    "赔偿金": "违法解除劳动合同的赔偿金如何计算？",
    "违约金": "劳动者需要支付违约金的情形有哪些？",
    "劳动合同": "劳动合同的必备条款包括哪些？",
    "解除劳动合同": "用人单位可以单方解除劳动合同的条件是什么？",
    "终止劳动合同": "劳动合同终止后用人单位有哪些义务？",
    "工伤": "哪些情形可以认定为工伤？",
    "加班": "加班费的计算标准是什么？",
    "年休假": "带薪年休假的天数如何确定？",
    "社会保险": "用人单位未缴纳社会保险的法律后果是什么？",
    "竞业限制": "竞业限制的期限和补偿标准是什么？",
    "无效": "合同无效的法律后果是什么？",
    "违约责任": "违约责任的承担方式有哪些？",
    "诉讼时效": "民事权利的诉讼时效是多久？",
    "婚姻": "哪些情形下婚姻无效？",
    "继承": "法定继承的顺序是什么？",
    "消费者": "消费者享有哪些基本权利？",
    "欺诈": "经营者存在欺诈行为时消费者如何维权？",
    "董事": "公司董事的忠实义务包括哪些？",
    "股东": "股东的权利包括哪些？",
    "商标": "商标侵权的认定标准是什么？",
    "著作权": "著作权的保护期限是多久？",
    "专利权": "专利权的保护范围如何确定？",
}

ARTICLE_PATTERN = re.compile(r'第[零〇一二三四五六七八九十百千万0-9]+条')

# ═══════════════════════════════════════════════════════════════
# 温暖亲和风 — 深灰蓝 + 琥珀（适合普通公众）
# ═══════════════════════════════════════════════════════════════
THEME_CSS = """
<style>
/* ── 全局 ── */
.stApp {
    background: #f5f3ef;
}
header[data-testid="stHeader"] {
    background: #2c3e50;
}
section[data-testid="stSidebar"] {
    background: #2c3e50;
}
section[data-testid="stSidebar"] * {
    color: #d5d9de !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stFileUploader label {
    color: #e09a5e !important;
}
section[data-testid="stSidebar"] .stButton button {
    color: #d5d9de !important;
    border-color: #d5d9de44 !important;
}

/* ── 聊天气泡 ── */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    border-radius: 10px;
    padding: 12px 18px;
}
div[data-testid="stChatMessage"]:has(.stChatMessage[data-testid="stChatMessageIconUser"])
    [data-testid="stMarkdownContainer"] {
    background: #2c3e50;
    color: #f0ede8;
}
div[data-testid="stChatMessage"]:has(.stChatMessage[data-testid="stChatMessageIconAssistant"])
    [data-testid="stMarkdownContainer"] {
    background: #ffffff;
    border: 1px solid #ece8e2;
}

/* ── 欢迎卡片 ── */
.welcome-card {
    text-align: center;
    padding: 60px 24px;
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #ece8e2;
}
.welcome-card h1 {
    color: #2c3e50;
    font-size: 1.6rem;
}
.welcome-card p {
    color: #7a756e;
}

/* ── 条款引用卡片 ── */
.article-cite {
    background: #f5f3ef;
    border-left: 3px solid #c97d3e;
    padding: 8px 14px;
    margin: 6px 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.88em;
}
.article-cite .label {
    color: #c97d3e;
    font-weight: 600;
    font-size: 0.82em;
}
.article-cite .text {
    color: #7a756e;
    margin-top: 4px;
}

/* ── 免责声明 ── */
.legal-disclaimer {
    margin-top: 14px;
    padding: 8px 14px;
    background: #fdf6ee;
    border-radius: 6px;
    font-size: 0.78em;
    color: #8c6a3d;
}

/* ── Sidebar 文档卡片 ── */
.doc-card {
    background: #3d5568;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 8px 0;
}
.doc-card .name {
    color: #e09a5e;
    font-weight: 600;
    font-size: 0.9em;
}
.doc-card .meta {
    color: #d5d9de;
    font-size: 0.78em;
    margin-top: 2px;
}

/* ── Header ── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0;
}
.app-header .brand {
    font-size: 1.4rem;
    font-weight: 700;
    color: #2c3e50;
}
.app-header .subtitle {
    font-size: 0.85rem;
    color: #7a756e;
    margin-left: 0.5rem;
}

/* ── 输入框 ── */
.stChatInput textarea {
    border: 1px solid #ece8e2 !important;
    border-radius: 8px !important;
}
.stChatInput textarea:focus {
    border-color: #c97d3e !important;
    box-shadow: 0 0 0 1px #c97d3e22 !important;
}

/* ── 展开器 ── */
.streamlit-expanderHeader {
    border: 1px solid #ece8e2;
    border-radius: 8px;
    background: #ffffff;
}
</style>
"""


def format_reasoning_response(thinking_content):
    """清理 <think> 标签（如果模型输出带有）。"""
    return (
        thinking_content.replace("<think>\n\n</think>", "")
        .replace("<think>", "")
        .replace("</think>", "")
        .strip()
    )


def display_assistant_message(content):
    """显示 assistant 回复，自动展开 thinking 部分（如果有）。"""
    pattern = r"<think>(.*?)</think>"
    think_match = re.search(pattern, content, re.DOTALL)

    if think_match:
        think_block = think_match.group(0)
        main_response = content.replace(think_block, "").strip()
        think_clean = format_reasoning_response(think_block)

        with st.expander("模型推理过程"):
            st.markdown(think_clean)
        st.markdown(main_response)
    else:
        st.markdown(content)


def generate_suggestions(documents, max_suggestions=5):
    """从文档中提取法律关键词，匹配建议问题。"""
    all_text = " ".join(doc.text for doc in documents)
    words = set(jieba.cut(all_text))
    matched = []
    for keyword in LEGAL_SUGGESTIONS:
        if keyword in words or keyword in all_text:
            matched.append((keyword, LEGAL_SUGGESTIONS[keyword]))
    seen = set()
    unique = []
    for kw, q in matched:
        if q not in seen:
            seen.add(q)
            unique.append((kw, q))
    return unique[:max_suggestions]


def main():
    st.set_page_config(page_title="法智问答", page_icon="⚖", layout="wide")

    # ── session_state 初始化 ──
    defaults = {
        "messages": [],
        "docs_loaded": False,
        "temp_dir": None,
        "file_fingerprint": None,
        "chat_engine": None,
        "pending_question": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ── CSS 注入 ──
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        # 模型选择
        model_options = list(ALL_MODELS.keys())
        selected_model = st.selectbox("选择生成模型", model_options, index=0)

        st.divider()

        # 文件上传（支持多文档联合检索）
        st.subheader("上传法律文档")
        uploaded_files = st.file_uploader(
            "选择文件",
            type=["pdf", "txt", "docx", "doc", "md", "markdown", "csv", "json", "html", "htm"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        # 从文件名推断法律名称
        def infer_law_name(filename: str) -> str:
            return os.path.splitext(filename)[0]

        # 文档加载与索引构建
        if uploaded_files:
            current_fp = tuple(f.name + str(f.size) for f in uploaded_files)
            if current_fp != st.session_state.get("file_fingerprint"):
                st.session_state.file_fingerprint = current_fp
                st.session_state.chat_engine = None

                try:
                    if not os.getenv("DASHSCOPE_API_KEY"):
                        st.error("请在 .env 文件中设置 DASHSCOPE_API_KEY")
                        st.stop()
                    if not os.getenv("OPENROUTER_API_KEY"):
                        st.error("请在 .env 文件中设置 OPENROUTER_API_KEY")
                        st.stop()

                    if st.session_state.temp_dir:
                        shutil.rmtree(st.session_state.temp_dir)

                    st.session_state.temp_dir = tempfile.mkdtemp()
                    all_documents = []

                    with st.spinner("正在解析文档..."):
                        for uf in uploaded_files:
                            file_path = os.path.join(st.session_state.temp_dir, uf.name)
                            with open(file_path, "wb") as f:
                                f.write(uf.getbuffer())
                            docs = SimpleDirectoryReader(
                                input_files=[file_path]
                            ).load_data()
                            law_name = infer_law_name(uf.name)
                            for d in docs:
                                d.metadata["law_name"] = law_name
                            all_documents.extend(docs)

                    st.session_state.docs_loaded = True
                    st.session_state.documents = all_documents

                except Exception as e:
                    st.error(f"加载失败：{str(e)}")

            # 显示已加载文档卡片
            if st.session_state.docs_loaded:
                docs = st.session_state.documents
                is_legal = any(ARTICLE_PATTERN.search(d.text) for d in docs)
                strategy = "条款级切分" if is_legal else "段落级切分"
                article_count = len(ARTICLE_PATTERN.findall(
                    "\n".join(d.text for d in docs)
                ))
                law_names = {d.metadata.get("law_name", "") for d in docs}
                law_label = "、".join(law_names) if law_names else uploaded_files[0].name
                st.markdown(f"""
                <div class="doc-card">
                    <div class="name">{law_label}</div>
                    <div class="meta">检测到 {article_count} 个条款 · {strategy} · {len(uploaded_files)} 个文件</div>
                </div>
                """, unsafe_allow_html=True)

            # 构建索引
            if st.session_state.docs_loaded and st.session_state.chat_engine is None:
                with st.spinner("正在构建法律索引..."):
                    docs = st.session_state.documents
                    is_legal = any(ARTICLE_PATTERN.search(d.text) for d in docs)
                    splitter = LegalArticleSplitter() if is_legal else None

                    st.session_state.chat_engine = get_chat_engine(
                        selected_model,
                        docs=docs,
                        m_size=ALL_MODELS[selected_model]["context_window"] * 0.5,
                        enable_semantic_splitter=False,
                        custom_splitter=splitter,
                    )
                st.success("索引构建完成")

        st.divider()
        st.caption(LEGAL_DISCLAIMER)

    # ── Header ──
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(
            f'<div class="app-header">'
            f'<span class="brand">⚖ 法智问答</span>'
            f'<span class="subtitle">法律条文智能助手</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.session_state.docs_loaded = False
            if st.session_state.temp_dir:
                shutil.rmtree(st.session_state.temp_dir)
                st.session_state.temp_dir = None
            st.session_state.file_fingerprint = None
            st.session_state.chat_engine = None
            st.rerun()

    # ── 主区域：欢迎页 or 聊天界面 ──
    if not st.session_state.docs_loaded:
        # 欢迎引导页
        st.markdown("""
        <div class="welcome-card">
            <h1>欢迎使用法智问答</h1>
            <p>上传法律文档，即可对条文内容进行智能问答</p>
            <p style="font-size:0.85em; color:#7a756e; margin-top:20px;">
                支持文档：民法典 · 劳动合同法 · 刑法 · 公司法 · 消费者权益保护法 等<br>
                支持格式：PDF · Word · Markdown · TXT · CSV · JSON · HTML
            </p>
            <p style="font-size:0.78em; color:#7a756e; margin-top:8px;">
                上传后自动检测法律条文，按条款级精准切分
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # 聊天消息历史
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    display_assistant_message(message["content"])
                else:
                    st.markdown(message["content"])

    # ── 聊天输入 ──
    prompt = st.chat_input("请输入您的法律问题...")

    # 桥接：建议按钮点击 → 下一轮自动填入
    if prompt is None and st.session_state.pending_question:
        prompt = st.session_state.pending_question
        st.session_state.pending_question = None

    # ── 智能示例建议（仅当文档已加载且无活跃提问时展示）──
    if st.session_state.docs_loaded and prompt is None:
        suggestions = generate_suggestions(st.session_state.documents)
        if suggestions:
            st.caption("试试这些问题：")
            cols = st.columns(min(len(suggestions), 5))
            for i, (kw, question) in enumerate(suggestions):
                with cols[i]:
                    if st.button(
                        question, key=f"sugg_{i}",
                        use_container_width=True,
                    ):
                        st.session_state.pending_question = question
                        st.rerun()

    if prompt:
        if not st.session_state.docs_loaded:
            st.error("请先在左侧上传法律文档")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("正在思考..."):
                try:
                    message_placeholder = st.empty()
                    full_response = ""

                    stream_response = st.session_state.chat_engine.stream_chat(prompt)

                    for token in stream_response.response_gen:
                        full_response += token
                        message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)

                    # 参考条款原文（HTML 卡片替代 expander）
                    if hasattr(stream_response, "source_nodes") and stream_response.source_nodes:
                        with st.expander("参考条款原文"):
                            for node in stream_response.source_nodes:
                                article = node.metadata.get("article", "")
                                law = node.metadata.get("law_name", "")
                                label = f"{law} {article}" if law else (article or "来源")
                                try:
                                    text = node.get_content()
                                except Exception:
                                    text = str(node)
                                st.markdown(f"""
                                <div class="article-cite">
                                    <div class="label">{label}</div>
                                    <div class="text">{text[:500]}</div>
                                </div>
                                """, unsafe_allow_html=True)

                    # 免责声明
                    st.markdown(
                        f'<div class="legal-disclaimer">{LEGAL_DISCLAIMER}</div>',
                        unsafe_allow_html=True,
                    )

                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"生成失败：{str(e)}")


if __name__ == "__main__":
    main()
