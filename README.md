# 电商智能客服问答系统 (RAG)

基于 FastAPI 的电商智能客服问答系统，使用 RAG（检索增强生成）架构实现语义相似度检索。

## 技术栈

- **FastAPI**: Web 框架
- **PostgreSQL**: 数据库
- **Sentence-Transformers**: 文本向量化（BAAI/bge-small-zh-v1.5）
- **FAISS**: 向量相似度检索
- **PyTorch**: 深度学习框架

## 项目结构

```
fastApiRAG/
├── main.py              # FastAPI 应用入口
├── config.py            # 配置管理
├── database.py          # 数据库连接
├── embedding.py         # 向量化模型
├── vector_store.py      # FAISS 向量存储
├── rag_service.py       # RAG 核心服务
├── requirements.txt     # 依赖包
├── init_db.sql         # 数据库初始化脚本
└── .env.example        # 环境变量示例
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置数据库

创建 PostgreSQL 数据库并执行初始化脚本：

```bash
psql -U postgres -f init_db.sql
```

或手动执行 SQL：

```sql
CREATE DATABASE ecommerce;
\c ecommerce;

CREATE TABLE faq (
    id SERIAL PRIMARY KEY,
    question VARCHAR(500) NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env` 并修改配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ecommerce
DB_USER=postgres
DB_PASSWORD=your_password
```

### 4. 运行服务

```bash
uvicorn main:app --reload
```

服务将在 http://127.0.0.1:8000 启动

## API 接口

### POST /ask

提交问题，获取最匹配的答案。

**请求示例：**

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "怎么查询订单？"}'
```

**响应示例：**

```json
{
  "answer": "您可以在"我的订单"页面查看订单状态，包括待付款、待发货、已发货、已完成等状态。",
  "confidence": 0.92,
  "matched_question": "如何查询订单状态？",
  "faq_id": 1
}
```

### GET /

系统状态检查

**响应：**

```json
{
  "message": "E-commerce Intelligent Customer Service RAG System"
}
```

## 配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| DB_HOST | 数据库主机 | localhost |
| DB_PORT | 数据库端口 | 5432 |
| DB_NAME | 数据库名称 | ecommerce |
| DB_USER | 数据库用户 | postgres |
| DB_PASSWORD | 数据库密码 | postgres |
| EMBEDDING_MODEL | 向量化模型 | BAAI/bge-small-zh-v1.5 |
| EMBEDDING_DEVICE | 计算设备 | cpu |
| TOP_K | 检索返回数量 | 3 |
| CONFIDENCE_THRESHOLD | 置信度阈值 | 0.5 |

## 工作原理

1. **初始化阶段**：
   - 从 PostgreSQL 数据库加载 FAQ 数据
   - 使用 BAAI/bge-small-zh-v1.5 模型对问题进行向量化
   - 使用 FAISS 构建向量索引

2. **查询阶段**：
   - 用户提交问题
   - 将问题向量化
   - 使用 FAISS 进行相似度检索
   - 返回最匹配的答案和置信度

## 注意事项

- 首次运行时会自动下载 BAAI/bge-small-zh-v1.5 模型（约 400MB）
- 确保数据库连接配置正确
- 建议使用 GPU 加速向量化（修改 EMBEDDING_DEVICE=cuda）

## 许可证

MIT License
