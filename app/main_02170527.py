from fastapi import FastAPI
from app.db.init_db import init_db
from app.api.v1 import routes_chat, routes_health, routes_context
from app.api.v1.routes_sessions import router as sessions_router
from app.api.v1.routes_telegram import router as telegram_router
from dotenv import load_dotenv
load_dotenv()  # 这会从项目根目录的 .env 文件读取环境变量
from app.api.v1.routes_anchor_mcp import router as anchor_mcp_router
from app.api.v1.routes_openai_proxy import router as openai_proxy_router
from app.api.v1.routes_gateway_ctx import router as gateway_ctx_router

app = FastAPI(title="Listopia Gateway", version="0.1.0")

# 初始化数据库（建表）
init_db()

# 注册路由
app.include_router(routes_health.router, prefix="/api/v1")
app.include_router(routes_chat.router, prefix="/api/v1")
app.include_router(sessions_router, prefix="/api/v1")  
app.include_router(routes_context.router, prefix="/api/v1")
app.include_router(telegram_router, prefix="/api/v1")



# MCP：两个入口（Rikkahub 只用 gateway_ctx 即可）
app.include_router(anchor_mcp_router, prefix="/api/v1/mcp", tags=["mcp"])
app.include_router(gateway_ctx_router, prefix="/api/v1/mcp", tags=["mcp"])

# OpenAI proxy（保持原样）
app.include_router(openai_proxy_router)

from fastapi import APIRouter
from fastapi.responses import JSONResponse

debug_router = APIRouter()

@debug_router.get("/debug/routes")
def debug_routes():
    # 只列出路径，方便你确认有没有 /api/v1/mcp/gateway_ctx
    return JSONResponse([getattr(r, "path", str(r)) for r in app.router.routes])

app.include_router(debug_router, prefix="/api/v1")
