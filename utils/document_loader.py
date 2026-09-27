from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from config.settings import KNOWLEDGE_BASE_DIR

_LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".md": UnstructuredMarkdownLoader,
}


def load_documents(directory: str | Path = KNOWLEDGE_BASE_DIR) -> list:
    """遍历目录，加载所有支持格式的文件为 LangChain Document 列表。"""
    directory = Path(directory)
    documents = []
    for file_path in sorted(directory.rglob("*")):
        loader_factory = _LOADER_MAP.get(file_path.suffix.lower())
        if loader_factory is None:
            continue
        loader = loader_factory(str(file_path))
        documents.extend(loader.load())
        print(f"已加载：{file_path.name}")
    return documents
