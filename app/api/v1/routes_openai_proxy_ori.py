# app/api/v1/routes_openai_proxy.py
from __future__ import annotations

import os
import json
import uuid
from typing import Any, Dict, List, Optional, AsyncGenerator

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from sqlalchemy.orm import Session as OrmSession

from app.db.session import SessionLocal
from app.db.models import SummaryS4, SummaryS60
from app.services.chat_service import append_user_and_assistant

import asyncio
router = APIRouter()


# -----------------------------
# DB helper
# -----------------------------
def get_db() -> OrmSession:
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise


# -----------------------------
# Utils
# -----------------------------
def _safe_json_loads(s: str):
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return s


def _pick_session_id(payload: Dict[str, Any], req: Request) -> str:
    """
    尽量稳定地从请求中拿到 session_id：
    - header: X-Session-Id
    - payload.user / payload.metadata.session_id / payload.metadata.conversation_id
    - 否则生成一个临时的
    """
    h = req.headers.get("x-session-id") or req.headers.get("X-Session-Id")
    if h:
        return f"rk:{h}"

    if isinstance(payload.get("user"), str) and payload["user"].strip():
        return f"rk:{payload['user'].strip()}"

    meta = payload.get("metadata") or {}
    if isinstance(meta, dict):
        for k in ("session_id", "conversation_id", "chat_id"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return f"rk:{v.strip()}"

    return f"rk:tmp:{uuid.uuid4().hex[:12]}"


def _last_user_text(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
    return ""


def _compact_summary_block(s4: Optional[Dict[str, Any]], s60: Optional[Dict[str, Any]]) -> str:
    parts = []
    if s4 and s4.get("summary"):
        parts.append("S4: " + json.dumps(s4["summary"], ensure_ascii=False))
    if s60 and s60.get("summary"):
        parts.append("S60: " + json.dumps(s60["summary"], ensure_ascii=False))

    if not parts:
        return ""

    return (
        "【Internal Memory（只用于你在心里对齐上下文与语气，不要在回复中提到“摘要/记忆/系统”）】\n"
        + "\n".join(parts)
        + "\n【End】"
    )


def _fetch_latest_summaries(db: OrmSession, session_id: str) -> Dict[str, Any]:
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

    out = {"s4": None, "s60": None}

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


def _inject_system(messages: List[Dict[str, Any]], system_block: str) -> List[Dict[str, Any]]:
    if not system_block:
        return messages
    injected = [{"role": "system", "content": system_block}]
    injected.extend(messages or [])
    return injected


def _build_upstream_url(upstream_base: str) -> str:
    """
    兼容：
    - https://openrouter.ai/api/v1           -> /chat/completions
    - https://api.openai.com                -> /v1/chat/completions
    - https://api.openai.com/v1             -> /chat/completions
    - 用户直接给完整 .../chat/completions
    """
    base = (upstream_base or "").strip().rstrip("/")
    if not base:
        base = "https://api.openai.com"

    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _build_upstream_headers() -> Dict[str, str]:
    upstream_key = os.getenv("UPSTREAM_API_KEY", "").strip()
    if not upstream_key:
        raise RuntimeError("UPSTREAM_API_KEY is empty")

    headers = {
        "Authorization": f"Bearer {upstream_key}",
        "Content-Type": "application/json",
    }

    # OpenRouter optional attribution headers
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    title = os.getenv("OPENROUTER_X_TITLE", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    return headers


async def _proxy_stream_and_store(
    upstream_url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    *,
    session_id: str,
    user_text: str,
    model_name: str,
) -> AsyncGenerator[bytes, None]:
    """
    单次请求：边转发 SSE 给客户端，边解析 delta 拼全文，流结束后写库/触发 summarizer。
    关键点：httpx.AsyncClient 的生命周期在 generator 内部，避免“return 后 client 被关闭”。
    """
    full_parts: List[str] = []
    done = False

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", upstream_url, headers=headers, json=body) as r:
            if r.status_code >= 400:
                raw = await r.aread()
                try:
                    j = json.loads(raw.decode("utf-8", errors="ignore") or "{}")
                    msg = j.get("error", {}).get("message") or j.get("message") or raw.decode("utf-8", errors="ignore")
                except Exception:
                    msg = raw.decode("utf-8", errors="ignore")
                err = {"error": {"message": msg, "type": "upstream_error", "status": r.status_code}}
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
                return

            async for line in r.aiter_lines():
                print(f"[RAW] {line}") 
                if line is None:
                    continue
                if line == "":
                    yield b"\n"
                    continue

                out_line = (line + "\n").encode("utf-8")
                yield out_line

                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    done = True
                    break

                try:
                    j = json.loads(data)
                    delta = (j.get("choices") or [{}])[0].get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        full_parts.append(piece)
                except Exception:
                    continue

    full_text = "".join(full_parts).strip()
    # ---------- 添加日志：打印流式收集的完整回复 ----------
    print("=== 流式收集的完整回复 ===")
    print(full_text if full_text else "(空)")
    print("=== 流式结束 ===")

    if full_text:
        db2 = get_db()
        try:
            append_user_and_assistant(
                db2,
                session_id=session_id,
                user_text=user_text,
                assistant_text=full_text,
                model_name=model_name,
                s4_every_user_turns=int(os.getenv("S4_EVERY_USER_TURNS", "4")),
                s60_every_user_turns=int(os.getenv("S60_EVERY_USER_TURNS", "30")),
                s4_window_user_turns=int(os.getenv("S4_WINDOW_USER_TURNS", "4")),
                s60_window_user_turns=int(os.getenv("S60_WINDOW_USER_TURNS", "30")),
            )
        finally:
            db2.close()

    if not done:
        yield b"\ndata: [DONE]\n\n"


# -----------------------------
# Main route: OpenAI compatible
# -----------------------------
@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload: Dict[str, Any] = await request.json()

    session_id = _pick_session_id(payload, request)

    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        return JSONResponse({"error": {"message": "messages must be a list"}}, status_code=400)

    user_text = _last_user_text(messages)

    db = get_db()
    try:
        sums = _fetch_latest_summaries(db, session_id=session_id)
    finally:
        db.close()

    system_block = _compact_summary_block(sums.get("s4"), sums.get("s60"))
    messages2 = _inject_system(messages, system_block)

    upstream_base = os.getenv("UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1")
    try:
        headers = _build_upstream_headers()
    except RuntimeError as e:
        return JSONResponse({"error": {"message": str(e)}}, status_code=500)

    upstream_url = _build_upstream_url(upstream_base)

    body = dict(payload)
    body["messages"] = messages2

    # 提取 stream 参数（必须在打印日志之前，因为日志要用到 stream）
    stream = bool(body.get("stream", False))
    model_name = str(body.get("model") or "unknown")

    # ---------- 打印最终请求体（用于对比） ----------
    print("=== 最终请求体 (流式=" + str(stream) + ") ===")
    log_body = body.copy()
    print(json.dumps(log_body, ensure_ascii=False, indent=2))
    print("=== 请求头（不含 Authorization）===")
    safe_headers = {k: v for k, v in headers.items() if k.lower() != 'authorization'}
    print(json.dumps(safe_headers, indent=2))

    if stream:
        return StreamingResponse(
            _proxy_stream_and_store(
                upstream_url,
                headers,
                body,
                session_id=session_id,
                user_text=user_text,
                model_name=model_name,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    
    # non-stream
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(upstream_url, headers=headers, json=body)
        if r.status_code >= 400:
            ct = r.headers.get("content-type", "")
            if ct.startswith("application/json"):
                return JSONResponse(r.json(), status_code=r.status_code)
            return JSONResponse({"error": {"message": r.text}}, status_code=r.status_code)

        data = r.json()

    assistant_text = ""
    try:
        assistant_text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    except Exception:
        assistant_text = ""

    # ---------- 打印非流式响应内容 ----------
    print("=== 非流式上游响应 (assistant_text) ===")
    print(assistant_text if assistant_text else "(空)")
    print("=== 非流式结束 ===")

    if assistant_text:
        db2 = get_db()
        try:
            append_user_and_assistant(
                db2,
                session_id=session_id,
                user_text=user_text,
                assistant_text=assistant_text,
                model_name=model_name,
                s4_every_user_turns=int(os.getenv("S4_EVERY_USER_TURNS", "4")),
                s60_every_user_turns=int(os.getenv("S60_EVERY_USER_TURNS", "30")),
                s4_window_user_turns=int(os.getenv("S4_WINDOW_USER_TURNS", "4")),
                s60_window_user_turns=int(os.getenv("S60_WINDOW_USER_TURNS", "30")),
            )
        finally:
            db2.close()

    return JSONResponse(data)