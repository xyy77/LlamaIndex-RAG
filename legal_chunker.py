import re
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import NodeParser


def split_by_article(text: str, law_name: str) -> List[Document]:
    """按法律条款（第X条）切分文本，每个条款作为一个独立 Document。"""
    pattern = re.compile(
        r'(第[零〇一二三四五六七八九十百千万0-9]+条[\s\S]*?)'
        r'(?=第[零〇一二三四五六七八九十百千万0-9]+条|\Z)'
    )
    matches = pattern.findall(text)

    docs = []
    for match in matches:
        article_num = re.search(r'第([零〇一二三四五六七八九十百千万0-9]+)条', match)
        article_label = article_num.group(0) if article_num else "未知条款"
        docs.append(Document(
            text=match.strip(),
            metadata={
                "law_name": law_name,
                "article": article_label,
            }
        ))
    return docs


class LegalArticleSplitter(NodeParser):
    """LlamaIndex 兼容的条款级分块器，用于替代 SentenceSplitter。"""

    law_name: str = ""

    def _parse_nodes(self, documents: List[Document], **kwargs):
        all_docs = []
        for doc in documents:
            # 优先从 metadata 取，其次从文件名推断
            law = (doc.metadata.get("law_name")
                   or doc.metadata.get("file_name")
                   or self.law_name
                   or "未知法律")
            all_docs.extend(split_by_article(doc.text, law))
        return all_docs
