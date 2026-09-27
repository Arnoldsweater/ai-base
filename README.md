# RAG问答系统
基于本地大模型实现知识库问答，检索内部文档，回答相关问题。

## ✨项目功能
- 加载本地PDF文档构建知识库
- 文档切片、向量化存入向量库
- RAG检索增强生成，基于知识库回答问题
- 支持对话历史记录，日志输出

## 📂项目目录说明
```
├── knowledge_base/     # 存放企业制度 PDF 文档
├── chat_history/       # 对话历史保存目录
├── logs/               # 运行日志
├── vectory_store/      # 向量数据库（本地生成，不上传 git）
├── main.py             # 主程序入口
├── app.py              # Web 服务入口
├── rag_chain.py        # RAG 检索链逻辑
├── .env                # 环境配置文件（密钥，请勿上传）
```


## 🛠环境依赖
Python >=3.10

安装依赖：
```
bash
pip install -r requirements.txt
```
## 启动服务
```
python run_all.py
```
