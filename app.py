import streamlit as st
import os
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
import tempfile
import shutil
import re

load_dotenv()

from utils import get_chat_engine, ALL_MODELS


def format_reasoning_response(thinking_content):
    """清理 <think> 标签（如果模型输出带有）"""
    return (
        thinking_content.replace("<think>\n\n</think>", "")
        .replace("<think>", "")
        .replace("</think>", "")
        .strip()
    )


def display_assistant_message(content):
    """显示 assistant 回复，自动展开 thinking 部分（如果有）"""
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


def main():
    st.set_page_config(page_title="RAG Chat", layout="wide")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "docs_loaded" not in st.session_state:
        st.session_state.docs_loaded = False
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = None
    if "current_pdf" not in st.session_state:
        st.session_state.current_pdf = None
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None

    st.title("RAG Chat with LlamaIndex")

    if st.button(" 清空对话"):
        st.session_state.messages = []
        st.session_state.docs_loaded = False
        if st.session_state.temp_dir:
            shutil.rmtree(st.session_state.temp_dir)
            st.session_state.temp_dir = None
        st.session_state.current_pdf = None
        st.session_state.chat_engine = None
        st.rerun()

    with st.sidebar:
        model_options = ALL_MODELS.keys()
        selected_model = st.selectbox(
            "选择生成模型",
            model_options,
            index=0
        )

        st.divider()

        st.subheader("上传文件")
        uploaded_file = st.file_uploader(
            "选择文件",
            type=["pdf", "txt"],
            accept_multiple_files=False
        )

        if uploaded_file is not None:
            if uploaded_file != st.session_state.current_pdf:
                st.session_state.current_pdf = uploaded_file
                st.session_state.chat_engine = None  # 重置引擎，触发重建索引

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
                    file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)

                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    with st.spinner("正在加载文档..."):
                        documents = SimpleDirectoryReader(
                            st.session_state.temp_dir,
                        ).load_data()

                        st.session_state.docs_loaded = True
                        st.session_state.documents = documents
                        st.success("文档加载完成")

                except Exception as e:
                    st.error(f"加载失败：{str(e)}")

            if st.session_state.docs_loaded and st.session_state.chat_engine is None:
                with st.spinner("正在构建索引..."):
                    st.session_state.chat_engine = get_chat_engine(
                        selected_model,
                        docs=st.session_state.documents,
                        m_size=ALL_MODELS[selected_model]['context_window'] * 0.5,
                        enable_semantic_splitter=False
                    )
                st.success("索引构建完成")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                display_assistant_message(message["content"])
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("请问关于这份文档的任何问题..."):
        if not st.session_state.docs_loaded:
            st.error("请先上传文档")
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

                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"生成失败：{str(e)}")


if __name__ == "__main__":
    main()
