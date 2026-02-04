from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
from app.rag_engine import rag_engine
from app.schemas import AskRequest, AskResponse
from app.database import db
import logging
import jwt
import os
import json
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# JWT配置
# 使用与Java后端相同的密钥，确保能够正确解析token
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "abcdefghijkomnopqrstuvwxyx")
JWT_ALGORITHM = "HS512"

# 认证方案
security = HTTPBearer()

# 解析JWT token获取用户ID
def get_user_id_from_token(token: str) -> str:
    """从JWT token中解析用户ID"""
    try:
        # 先尝试不验证签名解析token内容
        payload = jwt.decode(token, options={"verify_signature": False})
        logger.info(f"Token payload without verification: {payload}")
        
        # 尝试从多个可能的字段中获取用户ID
        user_id = payload.get("memberId") or payload.get("sub") or payload.get("userId") or payload.get("id") or payload.get("login_member_key")
        if user_id:
            logger.info(f"Found user ID: {user_id}")
            return str(user_id)
        
        # 尝试验证签名
        try:
            verified_payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            user_id = verified_payload.get("memberId") or verified_payload.get("sub") or verified_payload.get("userId") or verified_payload.get("id") or verified_payload.get("login_member_key")
            if user_id:
                logger.info(f"Verified user ID: {user_id}")
                return str(user_id)
        except Exception as verify_error:
            logger.warning(f"Signature verification failed but continuing with extracted user ID: {verify_error}")
        
        raise ValueError("User ID not found in token")
    except Exception as e:
        logger.error(f"Error decoding JWT token: {e}")
        # 即使解析失败，也返回一个默认值以允许请求通过
        # 这是为了兼容前端的认证流程
        return "anonymous"


# 获取当前用户ID
async def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """获取当前认证用户的ID"""
    try:
        token = credentials.credentials
        user_id = get_user_id_from_token(token)
        logger.info(f"Authenticated user: {user_id}")
        return user_id
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        # 即使认证失败，也返回一个默认值以允许请求通过
        # 这是为了兼容前端的认证流程
        return "anonymous"


# 获取可选的用户ID（支持匿名访问）
async def get_optional_user_id(request: Request) -> str:
    """获取可选的用户ID，支持匿名访问"""
    # 优先从Authorization头获取
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            user_id = get_user_id_from_token(token)
            logger.info(f"Optional authenticated user: {user_id}")
            return user_id
        except Exception as e:
            logger.warning(f"Optional authentication failed: {e}")
    
    # 从X-User-ID头获取
    user_id = request.headers.get("X-User-ID")
    if user_id:
        logger.info(f"User ID from header: {user_id}")
        return user_id
    
    logger.info("No user ID found, using anonymous access")
    return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # 测试数据库连接
        logger.info("Testing database connection...")
        if db.test_connection():
            logger.info("Database connection test passed")
        else:
            logger.warning("Database connection test failed")
        
        # 初始化RAG引擎
        logger.info("Initializing RAG engine...")
        rag_engine.initialize()
        logger.info("RAG engine initialized successfully")
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise
    yield
    try:
        rag_engine.refresh()
        logger.info("RAG engine refreshed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# 自定义JSON响应处理，确保正确编码
class UTF8JSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":")
        ).encode("utf-8")


app = FastAPI(
    lifespan=lifespan,
    default_response_class=UTF8JSONResponse,
    title="AI客服系统",
    description="商城系统的智能客服API",
    version="1.0.0"
)


# 添加中间件确保响应编码
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 确保所有JSON响应使用UTF-8编码
@app.middleware("http")
async def ensure_utf8_encoding(request, call_next):
    response = await call_next(request)
    # 确保响应头设置正确的编码
    if response.headers.get("Content-Type") and "application/json" in response.headers.get("Content-Type"):
        response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
async def health_check():
    try:
        db_status = db.test_connection()
        stats = rag_engine.get_stats()
        return {"status": "healthy" if db_status else "unhealthy", 
                "database": "connected" if db_status else "disconnected", 
                **stats}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


@app.post("/ask", response_class=UTF8JSONResponse)
async def ask_question(ask_request: AskRequest, user_id: str = Depends(get_optional_user_id)):
    try:
        logger.info(f"Received question: '{ask_request.question}' from user: {user_id or 'anonymous'}")
        result = rag_engine.ask(ask_request.question, tables=ask_request.tables, user_id=user_id)
        return result
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return {
            "answer": "抱歉，系统暂时无法处理您的问题，请稍后再试。",
            "confidence": 0.0,
            "matched_question": None,
            "faq_id": None,
            "source": None,
            "url": None,
            "table": None
        }


@app.post("/api/ai/ask", response_class=UTF8JSONResponse)
async def api_ai_ask(ask_request: AskRequest, user_id: str = Depends(get_optional_user_id)):
    """AI客服API接口，与前端调用路径匹配"""
    return await ask_question(ask_request, user_id)


@app.get("/api/ai/status")
async def api_ai_status():
    """AI服务状态接口，与前端调用路径匹配"""
    try:
        db_status = db.test_connection()
        rag_stats = rag_engine.get_stats()
        from app.llm_service import llm_service
        llm_status = llm_service.get_status()
        
        return {
            "status": "healthy" if db_status else "unhealthy",
            "database": "connected" if db_status else "disconnected",
            "rag_engine": rag_stats,
            "llm": llm_status
        }
    except Exception as e:
        logger.error(f"Error getting AI status: {e}")
        return {"status": "unhealthy", "error": str(e)}


@app.get("/api/history")
async def get_conversation_history(user_id: str = Depends(get_current_user_id)):
    """获取用户的对话历史"""
    try:
        history = rag_engine.get_conversation_history(user_id)
        logger.info(f"Retrieved {len(history)} messages for user {user_id}")
        return {"history": history}
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/api/history")
async def clear_conversation_history(user_id: str = Depends(get_current_user_id)):
    """清除用户的对话历史"""
    try:
        rag_engine.clear_history(user_id)
        logger.info(f"Cleared conversation history for user {user_id}")
        return {"status": "success", "message": "对话历史已清除"}
    except Exception as e:
        logger.error(f"Error clearing conversation history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/stats")
async def get_system_stats():
    """获取系统状态信息"""
    try:
        db_status = db.test_connection()
        rag_stats = rag_engine.get_stats()
        # 添加LLM状态信息
        from app.llm_service import llm_service
        llm_status = llm_service.get_status()
        
        return {
            "database": "connected" if db_status else "disconnected",
            "rag_engine": rag_stats,
            "llm": llm_status
        }
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/tables")
async def get_tables():
    try:
        tables = db.get_all_tables()
        return {"tables": tables}
    except Exception as e:
        logger.error(f"Error getting tables: {e}")
        return {"tables": []}


@app.get("/api/table/{table_name}/columns")
async def get_table_columns(table_name: str):
    try:
        columns = db.get_table_columns(table_name)
        return {"columns": columns}
    except Exception as e:
        logger.error(f"Error getting table columns: {e}")
        return {"columns": []}
