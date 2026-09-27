import os
from pathlib import Path
import torch


BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
VECTOR_STORE_DIR = BASE_DIR / "vectory_store" / "faiss_index"
VECTOR_STORE_NAME = "company_regulations"  # 持久化文件前缀


MODEL_DIR = BASE_DIR / "model"
EMBEDDING_MODEL_PATH = str(MODEL_DIR / "embedding" / "bge-small-zh-v1.5")
RERANKER_MODEL_PATH = str(MODEL_DIR / "reranker" / "bge-reranker-base")


def _find_model_dir(parent: Path, *must_contain: str) -> str:
    """按子串匹配目录名（规避目录名中的特殊连字符等隐藏字符问题）。"""
    for name in sorted(os.listdir(parent)):
        if all(s in name for s in must_contain) and (parent / name).is_dir():
            return str(parent / name)
    raise FileNotFoundError(f"在 {parent} 下未找到同时包含 {must_contain} 的模型目录")


# Qwen 目录名含非断行连字符（U+2011），无法用普通 '-' 精确书写，故按子串匹配
LLM_MODEL_PATH = _find_model_dir(MODEL_DIR / "llm" / "Qwen", "Qwen2.5", "0.5B", "Instruct")


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

TOP_K = 6                  # 初检召回数
FINAL_K = 3                # 重排后保留数
USE_RERANKER = True        # 是否启用本地重排模型
SEARCH_TYPE = "mmr"        # similarity / mmr
FETCH_K = 20               # mmr 候选池大小


MAX_NEW_TOKENS = 512
TEMPERATURE = 0.1
DO_SAMPLE = False          # False=贪婪解码，问答更稳定、可复现

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
