from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_OVERLAP, CHUNK_SIZE

def split_documents(documents) -> list:
    """文本切分"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    return splitter.split_documents(documents)
