import os
import sys
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from model.local_models import load_embeddings, load_llm
from service.rag_chain import build_qa_chain
from service.vector_store import build_and_save, load_or_none
from utils.document_loader import load_documents
from utils.text_splitter import split_documents


_state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    embeddings = load_embeddings()
    db = load_or_none(embeddings)

    if db is None:
        print("未检测到本地索引，正在构建 FAISS 索引……")
        documents = split_documents(load_documents())
        db = build_and_save(documents, embeddings)
        print(f"索引构建完成，共 {len(documents)} 个片段。")
    else:
        print("已加载本地 FAISS 索引。")

    llm = load_llm()
    _state["qa_chain"] = build_qa_chain(llm, db)
    print("RAG 已就绪。")
    yield
    _state.clear()


app = FastAPI(title="杭州金房科技有限公司员工手册问答服务系统", version="1.0", lifespan=lifespan)

class AskIn(BaseModel):
    question: str


class SourceOut(BaseModel):
    source: Optional[str] = None
    page: Optional[int] = None
    snippet: Optional[str] = None


class AskOut(BaseModel):
    answer: str
    sources: List[SourceOut] = []


# ------------------------------------------------------------------
# 4) 接口
# ------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "RAG 问答服务运行中", "docs": "/docs"}


@app.post("/ask", response_model=AskOut)
def ask(payload: AskIn):
    """问答接口。同步函数会自动跑在线程池里，不阻塞事件循环。"""
    result = _state["qa_chain"].invoke({"input": payload.question})

    sources = []
    for doc in result.get("context", [])[:5]:
        meta = doc.metadata or {}
        sources.append(
            SourceOut(
                source=meta.get("source"),
                page=meta.get("page"),
                snippet=doc.page_content[:100],
            )
        )
    return AskOut(answer=result["answer"], sources=sources)

def main() -> None:
    embeddings = load_embeddings()
    db = load_or_none(embeddings)

    if db is None:
        print("未检测到本地索引，正在构建 FAISS 索引……")
        documents = split_documents(load_documents())
        db = build_and_save(documents, embeddings)
        print(f"索引构建完成，共 {len(documents)} 个片段。")
    else:
        print("已加载本地 FAISS 索引。")

    llm = load_llm()
    qa_chain = build_qa_chain(llm, db)

    print("RAG 已就绪，输入问题开始对话（输入 exit 退出）：")
    while True:
        query = input("你：").strip()
        if query.lower() in {"exit", "quit", "q"}:
            break
        if not query:
            continue
        result = qa_chain.invoke({"input": query})
        print("助手：", result["answer"])


if __name__ == "__main__":
    if "--cli" in sys.argv:
        main()
    else:
        # 直接以 Web 服务方式启动：python main.py
        try:
            import uvicorn
        except ModuleNotFoundError:
            print("当前解释器缺少 uvicorn，无法启动 Web 服务。")
            print("请改用已安装依赖的解释器，例如：")
            print(r"    D:\anaconda\python.exe main.py")
            print("或一键启动（会自动挑对解释器）：")
            print("    python run_all.py")
            raise SystemExit(1)

        host = os.environ.get("RAG_HOST", "127.0.0.1")
        port = int(os.environ.get("RAG_PORT", "8000"))
        print(f"启动 RAG 服务：http://{host}:{port}   （接口文档 /docs）")
        uvicorn.run(app, host=host, port=port)
