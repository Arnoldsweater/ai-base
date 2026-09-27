"""
FastAPI 入门示例
================
运行步骤：
  1) pip install fastapi uvicorn
  2) 在终端执行：uvicorn main:app --reload
  3) 浏览器打开 http://127.0.0.1:8000/docs   （自动生成的 Swagger 文档）

这个文件覆盖了 FastAPI 最常用的 5 个能力：
  - 基础 GET 路由
  - 路径参数（path parameter）
  - 查询参数（query parameter）
  - Pydantic 请求体（request body + 自动校验）
  - 响应模型（response model）
"""

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

# 1) 创建应用实例：整份程序就靠这个 app 对象
app = FastAPI(title="FastAPI 入门示例", version="1.0")


# ------------------------------------------------------------------
# 2) 请求体模型：用 Pydantic 定义「客户端要传什么数据」
#    FastAPI 会自动校验类型、自动生成 JSON Schema、自动生成文档
# ------------------------------------------------------------------
class ItemIn(BaseModel):
    name: str
    price: float
    # Optional 表示可选字段，不传时默认为 None
    description: Optional[str] = None
    tax: Optional[float] = None


# ------------------------------------------------------------------
# 3) 响应模型：定义「接口返回什么结构」，对外屏蔽内部多余字段
# ------------------------------------------------------------------
class ItemOut(BaseModel):
    id: int
    name: str
    price: float
    description: Optional[str] = None


# ------------------------------------------------------------------
# 4) 路由：把 URL + HTTP 方法 映射到你的 Python 函数
# ------------------------------------------------------------------

# 4.1 最基础的 GET 接口
@app.get("/")
def read_root():
    return {"message": "Hello FastAPI", "docs": "/docs"}


# 4.2 路径参数：{item_id} 直接写进路径，FastAPI 自动解析并做类型转换
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    # q 是查询参数（?q=xxx），可选
    return {"item_id": item_id, "q": q}


# 4.3 查询参数：函数里「不是路径参数、也不是模型字段」的普通参数就是查询参数
@app.get("/search/")
def search(keyword: str, limit: int = 10, full: bool = False):
    return {"keyword": keyword, "limit": limit, "full": full}


# 4.4 POST + 请求体：客户端在 body 里传 JSON，FastAPI 用 ItemIn 校验
@app.post("/items/", response_model=ItemOut)
def create_item(item: ItemIn):
    # 真实项目中这里会写数据库；示例里直接返回（模拟 id=1）
    return ItemOut(
        id=1,
        name=item.name,
        price=item.price,
        description=item.description,
    )
