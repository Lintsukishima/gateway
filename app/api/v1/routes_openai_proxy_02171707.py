from __future__ import annotations

import os
import json
import uuid
import re
from typing import Any, Dict, List, Optional, AsyncGenerator, Tuple

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from sqlalchemy.orm import Session as OrmSession

from app.db.session import SessionLocal
from app.db.models import SummaryS4, SummaryS60
from app.services.chat_service import append_user_and_assistant

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
        parts.append("S4 (recent): " + json.dumps(s4["summary"], ensure_ascii=False))
    if s60 and s60.get("summary"):
        parts.append("S60 (long): " + json.dumps(s60["summary"], ensure_ascii=False))

    if not parts:
        return ""

    return (
        "【Internal Memory摘要（仅用于你在心里对齐语气与上下文，不要在回复中提到“摘要/记忆/系统”）】\n"
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

def _inject_system(messages: List[Dict[str, Any]], system_blocks: List[str]) -> List[Dict[str, Any]]:
    blocks = [b for b in (system_blocks or []) if b and b.strip()]
    if not blocks:
        return messages
    injected = [{"role": "system", "content": "\n\n".join(blocks)}]
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

def _parse_stream_flag(body: Dict[str, Any]) -> bool:
    sv = body.get("stream", False)
    if sv is True:
        return True
    if sv is False or sv is None:
        return False
    return str(sv).lower() == "true"

# -----------------------------
# IMPORTANT: sanitize ONLY broken tool traces (avoid upstream 400)
# 保留“完整的工具轮次”：assistant(tool_calls) + tool(tool_call_id)
# -----------------------------
def _sanitize_messages_for_upstream(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    人话规则：
    - 如果历史里出现 assistant.tool_calls，那么后面必须跟着对应的 role=tool(tool_call_id) 才能发给上游。
    - 不完整（缺 tool 结果）的那一轮：我们就把那条 assistant 里的 tool_calls 字段删掉（content 还在就保留），避免上游 400。
    - 孤儿 tool 消息（找不到对应的 tool_call_id）丢弃。
    """
    if not isinstance(messages, list):
        return []

    cleaned: List[Dict[str, Any]] = []
    pending: set[str] = set()

    def strip_last_assistant_tool_fields():
        for i in range(len(cleaned) - 1, -1, -1):
            if cleaned[i].get("role") == "assistant":
                m2 = dict(cleaned[i])
                m2.pop("tool_calls", None)
                m2.pop("function_call", None)
                cleaned[i] = m2
                return

    for m in messages:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip()

        # tool message：只有匹配 pending 才保留
        if role == "tool":
            tcid = (m.get("tool_call_id") or "").strip()
            if tcid and tcid in pending:
                cleaned.append(m)
                pending.discard(tcid)
            else:
                # 孤儿 tool -> 丢
                continue
            continue

        # 如果 pending 没清完，但遇到非 tool，说明缺 tool 结果 -> 删掉上一条 assistant 的 tool_calls
        if pending and role != "tool":
            strip_last_assistant_tool_fields()
            pending.clear()

        # assistant 带 tool_calls：记下 ids（但先放行这一条，后面等 tool 消息）
        if role == "assistant" and m.get("tool_calls") is not None:
            tc = m.get("tool_calls") or []
            ids: List[str] = []
            if isinstance(tc, list):
                for t in tc:
                    if isinstance(t, dict):
                        _id = (t.get("id") or "").strip()
                        if _id:
                            ids.append(_id)
            pending = set(ids)
            cleaned.append(m)
            continue

        # 老式 function_call 但没 content 的“纯标记”直接丢
        if role == "assistant" and m.get("function_call") is not None:
            content = m.get("content")
            if content is None or (isinstance(content, str) and not content.strip()):
                continue
            cleaned.append(m)
            continue

        cleaned.append(m)

    # 末尾还有 pending，说明最后一轮 tool 结果缺失 -> 同样删掉最后 assistant 的 tool_calls
    if pending:
        strip_last_assistant_tool_fields()

    return cleaned

# -----------------------------
# Anchor keyword extraction (from your 1543_sec)
# -----------------------------
_STOPWORDS = {
    "我","你","他","她","它","我们","你们","他们","她们",
    "的","了","啊","呀","呢","吧","吗","喵","哥哥","小猫咪","小命",
    "就是","但是","然后","所以","因为","如果","能不能","怎么",
    "这个","那个","现在","今天","明天","刚才","感觉","有点",
    "接着","拿起","提前","给","当是","好啦","嗯","唉呀","唔",
}
_EMO_PAT = re.compile(r"[😂🤣😭🥺😙😗😸😺😿😽💦💖💕❤️✨🎭🖤]+")
_TECH_PAT = re.compile(r"(uvicorn|python|notion|dify|mcp|rag|api|http|db|sql|error|bug|traceback|token|stream|openrouter|rikkahub|telegram)", re.I)

def _is_smalltalk_emotion(text: str) -> bool:
    if not text:
        return True
    t = text.strip()
    if not t:
        return True
    if _TECH_PAT.search(t):
        return False
    if len(t) <= 18 and any(x in t for x in ["哥哥", "猫咪", "小猫咪", "小命", "宝宝", "在吗", "早安", "晚安", "嘿嘿", "喵"]):
        return True
    emo_hits = len(_EMO_PAT.findall(t))
    if emo_hits >= 2:
        return True
    if t.count("~") >= 2 or t.count("…") >= 2:
        return True
    if t.count("喵") >= 2 or t.count("嘿嘿") >= 2:
        return True
    if any(x in t for x in ["想你", "抱抱", "亲亲", "贴贴", "陪我", "我回来啦", "我来啦", "我走啦", "加油", "辛苦啦"]):
        return True
    return False

_TOPIC_PRI = [
    ("除夕", ["除夕","年","过年"]),
    ("鞭炮", ["鞭炮","噼里啪啦"]),
    ("代码", ["代码","写码","编程","bug","报错","uvicorn","python","notion","dify","mcp","rag"]),
    ("电脑", ["电脑","键盘","小电脑","终端","手机","rikkahub","telegram"]),
    ("发明", ["发明","演出","舞台","剧团","导演"]),
]

def _split_long_cn(seq: str) -> list[str]:
    seps = ["，","。","！","？","…","～","—","(",")","（","）"," ", "\n",
            "又","接着","拿起","就当","当是","今天","提前","给","好啦","于是","然后","所以","但是","因为","不过"]
    s = seq
    for sp in seps:
        s = s.replace(sp, "|")
    parts = [p for p in s.split("|") if p]
    return parts

def _extract_keywords(text: str, k: int = 2) -> str:
    if not text:
        return "猫咪,哥哥"
    if _is_smalltalk_emotion(text):
        return "撒娇,哥哥"

    for key, vocab in _TOPIC_PRI:
        for w in vocab:
            if w and w in text:
                return f"{key},猫咪"

    cn_seqs = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    cand: list[str] = []
    for seq in cn_seqs:
        parts = _split_long_cn(seq) if len(seq) > 6 else [seq]
        for p in parts:
            p = p.strip()
            if not p or p in _STOPWORDS:
                continue
            if len(p) > 6:
                p1 = p[:4]
                p2 = p[-4:]
                for pp in (p1, p2):
                    if 2 <= len(pp) <= 6 and pp not in _STOPWORDS:
                        cand.append(pp)
            else:
                cand.append(p)

    if not cand:
        return "猫咪,哥哥"

    seen = set()
    uniq = []
    for t in cand:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    def score(t: str) -> Tuple[int,int]:
        bonus = 0
        if any(x in t for x in ["代码","电脑","键盘","报错","终端","除夕","鞭炮","发明","演出"]):
            bonus += 5
        return (bonus, len(t))

    uniq.sort(key=lambda x: (score(x)[0], score(x)[1]), reverse=True)

    picked = uniq[:k]
    if "猫咪" not in picked and k >= 2:
        picked = picked[:k-1] + ["猫咪"]
    return ",".join(picked)

# -----------------------------
# Anchor workflow call (Dify)
# -----------------------------
def _build_anchor_system_block(snippet: str) -> str:
    snippet = (snippet or "").strip()
    if not snippet:
        return ""
    return (
        "【Persona Anchor（仅用于你在心里模仿语气与节奏，不要在回复中提到“锚点/检索/工具/系统”）】\n"
        "规则：不要复述/引用下面的任何原句内容，把它当作你自己的记忆和语气示范；只需学习用词、节奏、口癖、和动作描写，把它自然融入你的回应。\n"
        f"{snippet}\n"
        "【End】"
    )

def _first(s: str, n: int = 300) -> str:
    s = s or ""
    s = s.replace("\r", "").replace("\n", "\\n")
    return s[:n]

async def _call_dify_workflow_anchor(keyword: str, user_id: str) -> str:
    run_url = os.getenv("DIFY_WORKFLOW_RUN_URL", "").strip()
    api_key = os.getenv("DIFY_WORKFLOW_API_KEY", "").strip()
    timeout = float(os.getenv("ANCHOR_TIMEOUT_SECS", "20"))
    max_chars = int(os.getenv("ANCHOR_SNIP_MAX_CHARS", "360"))

    if not run_url or not api_key:
        return ""

    payload = {
        "inputs": {"keyword": keyword},
        "response_mode": "blocking",
        "user": user_id,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(run_url, headers=headers, json=payload)

        ct = (r.headers.get("content-type") or "").lower()
        text = r.text or ""
        if r.status_code >= 400:
            return ""
        if "application/json" not in ct:
            return ""

        data = r.json()
        outputs = None
        if isinstance(data, dict):
            outputs = (data.get("data") or {}).get("outputs")
            if outputs is None:
                outputs = data.get("outputs")
        if not isinstance(outputs, dict):
            return ""

        cand = outputs.get("chat_text") or outputs.get("result") or ""
        if not isinstance(cand, str):
            cand = json.dumps(cand, ensure_ascii=False)
        cand = cand.strip()
        if not cand:
            return ""

        marker = "[ChatHistory]"
        if marker in cand:
            cand = cand.split(marker, 1)[1].strip()

        cand = re.sub(r"\s+\n", "\n", cand)
        cand = re.sub(r"[ \t]{2,}", " ", cand).strip()

        if len(cand) > max_chars:
            cand = cand[:max_chars].rstrip() + "…"
        return cand

    except Exception:
        return ""

# -----------------------------
# Streaming proxy: single stream + collect + store (stable)
# -----------------------------
async def _proxy_stream_and_store(
    upstream_url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    *,
    session_id: str,
    user_text: str,
    model_name: str,
) -> AsyncGenerator[bytes, None]:
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
                if line is None:
                    continue
                if line == "":
                    yield b"\n"
                    continue

                yield (line + "\n").encode("utf-8")

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

    # ✅ 关键：只清理“坏的工具痕迹”，完整轮次保留
    messages = _sanitize_messages_for_upstream(messages)

    user_text = _last_user_text(messages)

    db = get_db()
    try:
        sums = _fetch_latest_summaries(db, session_id=session_id)
    finally:
        db.close()

    s_block = _compact_summary_block(sums.get("s4"), sums.get("s60"))

    # Anchor inject
    anchor_block = ""

    # 强制每轮都做一次“gateway/anchor 预取”（你要的“必跑”）
    # 同时：把 .env 的 ANCHOR_INJECT_ENABLED 当作“总开关”
    if os.getenv("ANCHOR_INJECT_ENABLED", "1") == "1" and FORCE_GATEWAY_EVERY_TURN:
        kw = _extract_keywords(user_text, k=2)
        snip = await _call_dify_workflow_anchor(keyword=kw, user_id=session_id)
        anchor_block = _build_anchor_system_block(snip)

    system_blocks = []
    if s_block:
        system_blocks.append(s_block)
    if anchor_block:
        system_blocks.append(anchor_block)

    messages2 = _inject_system(messages, system_blocks)

    upstream_base = os.getenv("UPSTREAM_BASE_URL", "https://openrouter.ai/api/v1")
    try:
        headers = _build_upstream_headers()
    except RuntimeError as e:
        return JSONResponse({"error": {"message": str(e)}}, status_code=500)

    upstream_url = _build_upstream_url(upstream_base)

    body = dict(payload)
    body["messages"] = messages2

    # ✅ 关键：不再删 tools / tool_choice / functions
    # 让上游模型能“看到工具”，从而发起 tool_calls，RikkaHub 再去执行工具

    stream = _parse_stream_flag(body)
    model_name = str(body.get("model") or "unknown")

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
                "X-Upstream-URL": upstream_url,   # 新增行
            },
        )

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

    resp = JSONResponse(data) 
    resp.headers["x-upstream-url"] = upstream_url
    resp.headers["x-session-id"] = session_id
    return resp
