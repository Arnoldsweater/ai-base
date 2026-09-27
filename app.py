import json
import uuid
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
ASK_URL = f"{API_BASE}/ask"

_NO_PROXY = {"http": None, "https": None}
CHAT_DIR = Path(__file__).resolve().parent / "chat_history"
CHAT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="企业制度RAG", page_icon="💬", layout="wide")

def _conv_path(cid: str) -> Path:
    return CHAT_DIR / f"{cid}.json"


def load_conversations() -> dict:
    """从磁盘加载全部历史对话，返回 {id: 对话字典}。"""
    items: dict = {}
    for file in CHAT_DIR.glob("*.json"):
        try:
            items[file.stem] = json.loads(file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return items


def save_conversation(conv: dict) -> None:
    _conv_path(conv["id"]).write_text(
        json.dumps(conv, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def delete_conversation(cid: str) -> None:
    path = _conv_path(cid)
    if path.exists():
        path.unlink()


def create_conversation() -> dict:
    cid = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    return {
        "id": cid,
        "title": "新对话",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [],
    }


def check_backend() -> bool:
    """探测后端是否在线。"""
    try:
        return requests.get(
            f"{API_BASE}/", timeout=2, proxies=_NO_PROXY
        ).status_code == 200
    except requests.RequestException:
        return False



# 会话状态初始化

if "conversations" not in st.session_state:
    st.session_state.conversations = load_conversations()

if (
    st.session_state.get("current_id") not in st.session_state.conversations
):
    if st.session_state.conversations:
        st.session_state.current_id = max(
            st.session_state.conversations,
            key=lambda k: st.session_state.conversations[k]["created_at"],
        )
    else:
        conv = create_conversation()
        st.session_state.conversations[conv["id"]] = conv
        save_conversation(conv)
        st.session_state.current_id = conv["id"]


def render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander("查看引用来源"):
        for s in sources:
            src = s.get("source") or "未知来源"
            page = s.get("page")
            page_txt = f"（第 {page} 页）" if page is not None else ""
            st.caption(f"{src}{page_txt}：{s.get('snippet', '')}")



with st.sidebar:
    st.subheader("💬 对话")

    if st.button("＋ 新建对话", use_container_width=True, type="primary"):
        conv = create_conversation()
        st.session_state.conversations[conv["id"]] = conv
        st.session_state.current_id = conv["id"]
        save_conversation(conv)
        st.rerun()

    st.divider()

    ordered = sorted(
        st.session_state.conversations.values(),
        key=lambda c: c["created_at"],
        reverse=True,
    )
    for conv in ordered:
        cid = conv["id"]
        active = cid == st.session_state.current_id
        col_name, col_del = st.columns([0.8, 0.2])
        label = ("▶ " if active else "") + conv["title"]
        if col_name.button(
            label, key=f"sel_{cid}", use_container_width=True
        ):
            st.session_state.current_id = cid
            st.rerun()
        if col_del.button("🗑", key=f"del_{cid}", use_container_width=True):
            delete_conversation(cid)
            st.session_state.conversations.pop(cid, None)
            if st.session_state.current_id == cid:
                remaining = st.session_state.conversations
                if remaining:
                    st.session_state.current_id = max(
                        remaining, key=lambda k: remaining[k]["created_at"]
                    )
                else:
                    new_conv = create_conversation()
                    st.session_state.conversations[new_conv["id"]] = new_conv
                    save_conversation(new_conv)
                    st.session_state.current_id = new_conv["id"]
            st.rerun()

    st.divider()
    if check_backend():
        st.caption(f"✅ 后端已连接 · {API_BASE}")
    else:
        st.caption(f"❌ 后端未连接 · {API_BASE}")



conv = st.session_state.conversations[st.session_state.current_id]
st.title("员工手册问答助手")

if not check_backend():
    st.warning(
        "未检测到后端服务。请在项目目录下任选一种方式启动：\n\n"
        "- `python run_all.py` —— 一键启动前后端（推荐，会自动挑对 Python 环境）\n"
        "- `python main.py` —— 只启动后端\n"
        "- `uvicorn main:app --reload` —— 需要热重载时用\n\n"
        "注意：请用**装了依赖的解释器**（如 `D:\\anaconda\\python.exe`），"
        "否则会报 `No module named uvicorn`。"
    )

for msg in conv["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources", []))

if prompt := st.chat_input("请输入你的问题"):
    conv["messages"].append({"role": "user", "content": prompt})
    if conv["title"] in ("新对话", ""):
        conv["title"] = prompt[:12]
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("检索中…"):
            try:
                resp = requests.post(
                    ASK_URL,
                    json={"question": prompt},
                    timeout=300,
                    proxies=_NO_PROXY,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                sources = data.get("sources", [])
            except requests.RequestException as exc:
                answer = (
                    f"调用后端失败：{exc}\n\n"
                    "请确认后端已启动，或改用 `python run_all.py` 一键启动前后端。"
                )
                sources = []
        st.markdown(answer)
        render_sources(sources)

    conv["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    save_conversation(conv)
