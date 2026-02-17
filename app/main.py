from fastapi import FastAPI, APIRouter
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from app.db.init_db import init_db
from app.api.v1 import routes_chat, routes_health, routes_context
from app.api.v1.routes_sessions import router as sessions_router
from app.api.v1.routes_telegram import router as telegram_router

from app.api.v1.routes_anchor_mcp import router as anchor_mcp_router
from app.api.v1.routes_gateway_ctx import router as gateway_ctx_router
from app.api.v1.routes_openai_proxy import router as openai_proxy_router

load_dotenv()

app = FastAPI(title="Listopia Gateway", version="0.1.0")

# 初始化数据库（建表）
init_db()

# 注册路由
app.include_router(routes_health.router, prefix="/api/v1")
app.include_router(routes_chat.router, prefix="/api/v1")
app.include_router(sessions_router, prefix="/api/v1")
app.include_router(routes_context.router, prefix="/api/v1")
app.include_router(telegram_router, prefix="/api/v1")

# MCP：只在这里加一次 /api/v1/mcp
app.include_router(anchor_mcp_router, prefix="/api/v1/mcp", tags=["mcp"])
app.include_router(gateway_ctx_router, prefix="/api/v1/mcp", tags=["mcp"])


# OpenAI proxy（保持原样）
app.include_router(openai_proxy_router)

# Debug: 列出所有路由路径
debug_router = APIRouter()

@debug_router.get("/debug/routes")
def debug_routes():
    return JSONResponse([getattr(r, "path", str(r)) for r in app.router.routes])

app.include_router(debug_router, prefix="/api/v1")



@app.get("/ping")
def ping():
    return {"ok": True}
