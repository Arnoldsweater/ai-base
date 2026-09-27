"""RAG 检索 + 生成链（含可选本地重排）。"""
from typing import List

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from config.settings import (
    FETCH_K,
    FINAL_K,
    SEARCH_TYPE,
    TOP_K,
    USE_RERANKER,
)
from model.local_models import LocalReranker

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个严谨的企业制度问答助手。请遵守以下规则：\n"
            "1. 只依据下面提供的『上下文』作答，不要使用你原有的通用知识。\n"
            "2. 若上下文与问题无关、或信息不足以回答，只回复：知识库中未找到相关信息。\n"
            "3. 回答要简洁直接，不要复述问题，不要出现“根据上下文”“作为一个AI”之类的话。",
        ),
        ("human", "上下文：\n{context}\n\n问题：{input}"),
    ]
)


class RerankRetriever(BaseRetriever):
    """在基础检索结果上做本地重排的自定义检索器。"""

    base_retriever: BaseRetriever = Field(...)
    reranker: object = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        if self.reranker is not None:
            docs = self.reranker.rerank(query, docs, top_k=FINAL_K)
        return docs


def build_qa_chain(llm, db) -> callable:
    """构建检索增强问答链。"""
    base_retriever = db.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={"k": TOP_K, "fetch_k": FETCH_K},
    )
    retriever = RerankRetriever(
        base_retriever=base_retriever,
        reranker=LocalReranker() if USE_RERANKER else None,
    )
    combine_chain = create_stuff_documents_chain(llm, _PROMPT)
    return create_retrieval_chain(retriever, combine_chain)
