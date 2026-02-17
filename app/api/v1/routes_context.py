# app/api/v1/routes_context.py

import json
from fastapi import APIRouter, Query, Request
from sqlalchemy.orm import Session as OrmSession

from app.db.session import SessionLocal
from app.db.models import Message, SummaryS4, SummaryS60

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _safe_json_loads(s: str):
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        # 如果历史脏数据不是 JSON，就原样返回
        return s


# ----------------------------
# MCP: gateway_ctx
# 说明：
# - 这里不要写 /api/v1 前缀（外面通常已经加了 /api/v1）
# - GET 用于“我还活着”
# - POST 用于 tools/list（平台扫描工具时会用到）
# ----------------------------

    # 平台要工具列表
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": data.get("id", 1),
            "result": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Return the input text as output.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    }
                ]
            },
        }

    # 其他方法先明确报错（别给 404）
    return {
        "jsonrpc": "2.0",
        "id": data.get("id", 1),
        "error": {"code": -32601, "message": f"Unsupported method: {method}"},
    }


@router.get("/sessions/{session_id}/context")
def get_context(
    session_id: str,
    recent: int = Query(16, ge=1, le=200),
):
    """
    统一出口：返回 ContextPack（最新 s4 / s60 + 最近消息）
    """
    db: OrmSession = next(get_db())

    # 最新 S4 / S60
    s4_row = (
        db.query(SummaryS4)
        .filter(SummaryS4.session_id == session_id)
        .order_by(SummaryS4.to_turn.desc())
        .first()
    )
    s60_row = (
        db.query(SummaryS60)
        .filter(SummaryS60.session_id == session_id)
        .order_by(SummaryS60.to_turn.desc())
        .first()
    )

    # 最近 N 条消息（按 turn_id）
    msgs = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.turn_id.desc())
        .limit(recent)
        .all()
    )
    msgs = list(reversed(msgs))

    latest = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.turn_id.desc())
        .first()
    )

    out = {
        "session_id": session_id,
        "s4": None,
        "s60": None,
        "recent": [
            {
                "role": m.role,
                "turn_id": m.turn_id,
                "user_turn": m.user_turn,
                "content": m.content,
            }
            for m in msgs
        ],
        "meta": {
            "latest_turn_id": latest.turn_id if latest else None,
            "latest_user_turn": latest.user_turn if latest else None,
        },
    }

    if s4_row:
        out["s4"] = {
            "range": [s4_row.from_turn, s4_row.to_turn],
            "summary": _safe_json_loads(s4_row.summary_json),
            "created_at": s4_row.created_at.isoformat() if s4_row.created_at else None,
            "model": s4_row.model,
        }

    if s60_row:
        out["s60"] = {
            "range": [s60_row.from_turn, s60_row.to_turn],
            "summary": _safe_json_loads(s60_row.summary_json),
            "created_at": s60_row.created_at.isoformat() if s60_row.created_at else None,
            "model": s60_row.model,
        }

    return out

