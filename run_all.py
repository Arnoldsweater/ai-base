from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) or "."
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
FRONTEND_PORT = 8501
HEALTH_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/"

# 前后端跑起来所必需的第三方模块（缺一不可）
REQUIRED_MODULES = (
    "uvicorn",
    "fastapi",
    "streamlit",
    "requests",
    "langchain",
    "faiss",
    "torch",
    "transformers",
    "sentence_transformers",
)

# 健康检查用：显式绕开系统代理，避免 http_proxy 把 127.0.0.1 也代理掉
_URL_OPENER = build_opener(ProxyHandler({}))


# ----------------------------------------------------------------------
# 解释器探测
# ----------------------------------------------------------------------
def _candidates() -> list[list[str]]:
    """返回候选解释器命令列表（每项是 argv 前缀，如 ["py","-3.12"]）。"""
    cands: list[list[str]] = []

    def add(cmd: list[str] | None) -> None:
        if cmd and cmd not in cands:
            cands.append(cmd)

    env_py = os.environ.get("RAG_PYTHON")
    if env_py:
        add([env_py])

    # 1) 当前解释器：只要依赖齐全就直接用，零开销
    add([sys.executable])

    # 2) 常见 Conda / 系统安装位置
    if os.name == "nt":
        for p in (
            r"D:\anaconda\python.exe",
            r"C:\ProgramData\anaconda3\python.exe",
            os.path.expanduser(r"~\anaconda3\python.exe"),
            os.path.expanduser(r"~\miniconda3\python.exe"),
        ):
            if os.path.isfile(p):
                add([p])
        py_launcher = shutil.which("py")
        if py_launcher:
            for ver in ("-3.12", "-3.13", "-3.11", "-3.10"):
                add([py_launcher, ver])
    else:
        for p in ("/usr/bin/python3", "/usr/local/bin/python3", "/opt/conda/bin/python"):
            if os.path.isfile(p):
                add([p])

    # 3) PATH 上的 python
    for name in ("python", "python3"):
        found = shutil.which(name)
        if found:
            add([found])

    return cands


def _missing_modules(cmd: list[str]) -> list[str] | None:
    """探测某解释器缺少哪些模块。解释器本身跑不起来时返回 None。"""
    code = (
        "import importlib.util as u, sys;"
        "print(','.join(n for n in sys.argv[1:] if u.find_spec(n) is None))"
    )
    try:
        proc = subprocess.run(
            [*cmd, "-c", code, *REQUIRED_MODULES],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return [n for n in proc.stdout.strip().split(",") if n]


def _version_of(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(
            [*cmd, "-c", "import sys; print(sys.version.split()[0])"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return proc.stdout.strip() or "?"
    except Exception:
        return "?"


def pick_python() -> tuple[list[str] | None, list[tuple[list[str], list[str] | None]]]:
    """挑选依赖齐全的解释器，返回 (命令, 探测记录)。"""
    tried: list[tuple[list[str], list[str] | None]] = []
    for cmd in _candidates():
        missing = _missing_modules(cmd)
        if missing is not None and not missing:
            return cmd, tried
        tried.append((cmd, missing))
    return None, tried


def report_failure(tried: list[tuple[list[str], list[str] | None]]) -> None:
    print("=" * 68)
    print("没有找到依赖齐全的 Python 解释器，无法启动。探测结果：")
    print("=" * 68)
    for cmd, missing in tried:
        shown = " ".join(cmd)
        if missing is None:
            print(f"  [x] {shown}\n      无法运行")
        else:
            print(f"  [-] {shown}\n      缺少: {', '.join(missing)}")
    print("-" * 68)
    print("解决办法（任选其一）：")
    print("  1) 用已经装好依赖的解释器直接运行本脚本，例如：")
    print(r'       D:\anaconda\python.exe run_all.py')
    print("  2) 通过环境变量指定解释器：")
    print(r'       $env:RAG_PYTHON="D:\anaconda\python.exe"; python run_all.py')
    print("  3) 给当前解释器补依赖：")
    print(r'       python -m pip install -r requirements.txt')
    print("=" * 68)


# ----------------------------------------------------------------------
# 端口与服务
# ----------------------------------------------------------------------
def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.6)
        return sock.connect_ex((host, port)) == 0


def wait_backend(timeout: int = 900, interval: int = 2) -> bool:
    """轮询后端健康检查（首次加载本地模型较慢，默认最多等 15 分钟）。"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with _URL_OPENER.open(HEALTH_URL, timeout=3) as resp:
                if getattr(resp, "status", 200) == 200:
                    return True
        except (URLError, OSError):
            pass
        time.sleep(interval)
    return False


def terminate(proc: subprocess.Popen | None) -> None:
    """结束进程；Windows 下连子进程一起杀。"""
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
        )
    else:
        proc.terminate()


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> int:
    args = set(sys.argv[1:])

    python_cmd, tried = pick_python()
    if python_cmd is None:
        report_failure(tried)
        return 1

    py_ver = _version_of(python_cmd)
    print(f"[环境] 使用解释器: {' '.join(python_cmd)}  (Python {py_ver})")
    if os.path.normcase(python_cmd[0]) != os.path.normcase(sys.executable):
        print(f"[环境] 本文件由 {sys.executable} 运行，已自动切换到上面的解释器。")

    if "--doctor" in args:
        print("[体检] 依赖齐全，环境正常。未启动任何服务。")
        return 0

    start_backend = "--frontend-only" not in args
    start_frontend = "--backend-only" not in args

    # 子进程环境：强制本地直连，避免系统代理拦掉 127.0.0.1
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    no_proxy = ",".join(filter(None, [env.get("NO_PROXY", ""), env.get("no_proxy", ""),
                                      "127.0.0.1", "localhost"]))
    env["NO_PROXY"] = no_proxy
    env["no_proxy"] = no_proxy

    backend: subprocess.Popen | None = None
    frontend: subprocess.Popen | None = None

    try:
        if start_backend:
            if is_port_open(BACKEND_HOST, BACKEND_PORT):
                print(f"[1/2] 端口 {BACKEND_PORT} 已在监听，复用现有后端。")
            else:
                print(f"[1/2] 启动后端 uvicorn -> http://{BACKEND_HOST}:{BACKEND_PORT}"
                      "（加载本地模型，请稍候）")
                backend = subprocess.Popen(
                    [*python_cmd, "-m", "uvicorn", "main:app",
                     "--host", BACKEND_HOST, "--port", str(BACKEND_PORT),
                     "--app-dir", BASE_DIR],
                    cwd=BASE_DIR, env=env,
                )
                if backend.poll() is not None:
                    print("[1/2] 后端启动失败，请查看上方报错信息。")
                    return 1
                if wait_backend():
                    print("[1/2] 后端已就绪。")
                else:
                    print("[1/2] 等待后端超时（首次建索引可能很久）。若前端报连接失败，"
                          "请等后端打印『RAG 已就绪』后重试。")
        else:
            print("[1/2] 已跳过后端（--frontend-only）。")

        if start_frontend:
            if is_port_open("127.0.0.1", FRONTEND_PORT):
                print(f"[2/2] 端口 {FRONTEND_PORT} 已在监听，复用现有前端。")
            else:
                print(f"[2/2] 启动前端 streamlit -> http://127.0.0.1:{FRONTEND_PORT}")
                frontend = subprocess.Popen(
                    [*python_cmd, "-m", "streamlit", "run",
                     os.path.join(BASE_DIR, "app.py"),
                     "--server.address", "127.0.0.1",
                     "--server.port", str(FRONTEND_PORT),
                     "--server.headless", "true"],
                    cwd=BASE_DIR, env=env,
                )

            if "--no-browser" not in args:
                time.sleep(3)
                webbrowser.open(f"http://127.0.0.1:{FRONTEND_PORT}")
        else:
            print("[2/2] 已跳过前端（--backend-only）。后端地址 "
                  f"http://{BACKEND_HOST}:{BACKEND_PORT}/docs")

        print("\n服务运行中，按 Ctrl+C 退出。")
        if frontend is not None:
            frontend.wait()
        elif backend is not None:
            backend.wait()
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到退出信号，正在关闭……")
    finally:
        terminate(frontend)
        terminate(backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
