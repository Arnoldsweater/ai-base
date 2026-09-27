import os
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from config.settings import VECTOR_STORE_DIR, VECTOR_STORE_NAME


def build_and_save(documents, embeddings) -> FAISS:
    """从文档构建 FAISS 索引并持久化到磁盘。"""
    db = FAISS.from_documents(
        documents, embeddings, distance_strategy=DistanceStrategy.COSINE
    )
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    db.save_local(str(VECTOR_STORE_DIR), index_name=VECTOR_STORE_NAME)
    return db


def load_or_none(embeddings) -> FAISS | None:
    """若本地索引存在则加载，否则返回 None。"""
    index_file = os.path.join(VECTOR_STORE_DIR, f"{VECTOR_STORE_NAME}.faiss")
    if not os.path.exists(index_file):
        return None
    return FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        index_name=VECTOR_STORE_NAME,
        distance_strategy=DistanceStrategy.COSINE,
        allow_dangerous_deserialization=True,
    )


def add_and_save(db: FAISS, documents, embeddings) -> FAISS:
    """增量添加文档并保存（保持长期存储）。"""
    db.add_documents(documents)
    db.save_local(str(VECTOR_STORE_DIR), index_name=VECTOR_STORE_NAME)
    return db
