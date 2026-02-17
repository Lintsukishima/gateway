# app/api/v1/routes_openai_proxy.py
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

def _sanitize_messages_for_upstream(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip tool-call artifacts before forwarding to upstream.

    Some clients (including RikkaHub) keep MCP/tool traces in the chat history. Upstream
    OpenAI-compatible providers enforce: any assistant message with `tool_calls` must be
    followed by `role=tool` messages for each `tool_call_id`. Our gateway does NOT execute
    upstream tools (MCP is handled client-side), so we remove those artifacts to avoid 400.
    """
    cleaned: List[Dict[str, Any]] = []
    for m in (messages or []):
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or "").strip()

        # Drop tool messages entirely
        if role in ("tool", "function"):
            continue

        # If assistant message contains tool_calls/function_call but has no content,
        # it is only a tool invocation marker -> drop it.
        if role == "assistant" and (m.get("tool_calls") is not None or m.get("function_call") is not None):
            content = m.get("content")
            if content is None or (isinstance(content, str) and not content.strip()):
                continue
            m2 = dict(m)
            m2.pop("tool_calls", None)
            m2.pop("function_call", None)
            cleaned.append(m2)
            continue

        cleaned.append(m)

    return cleaned



def _parse_stream_flag(body: Dict[str, Any]) -> bool:
    sv = body.get("stream", False)
    if sv is True:
        return True
    if sv is False or sv is None:
        return False
    return str(sv).lower() == "true"


# -----------------------------
# Anchor keyword extraction (better)
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
    """
    轻量情绪/闲聊识别：命中则强制回退人格关键词，避免长句/无主题误抽。
    规则尽量保守：只在“明显闲聊撒娇”时触发。
    """
    if not text:
        return True

    t = text.strip()
    if not t:
        return True

    # 1) 明显技术/工程词出现：认为不是闲聊
    if _TECH_PAT.search(t):
        return False

    # 2) 字数很短 & 口头语/称呼明显
    if len(t) <= 18:
        if any(x in t for x in ["哥哥", "猫咪", "小猫咪", "小命", "宝宝", "在吗", "早安", "晚安", "中午好", "嘿嘿", "喵"]):
            return True

    # 3) emoji/颜文字密度很高：更像撒娇闲聊
    emo_hits = len(_EMO_PAT.findall(t))
    if emo_hits >= 2:
        return True
    if t.count("~") >= 2 or t.count("…") >= 2:
        return True
    if t.count("喵") >= 2 or t.count("嘿嘿") >= 2:
        return True

    # 4) 典型撒娇句式
    if any(x in t for x in ["想你", "抱抱", "亲亲", "贴贴", "陪我", "我回来啦", "我来啦", "我走啦", "加油", "辛苦啦"]):
        return True

    return False


# 主题优先词：一旦出现，优先塞进 keyword（更容易检索到对的锚点）
_TOPIC_PRI = [
    ("除夕", ["除夕","年","过年"]),
    ("鞭炮", ["鞭炮","噼里啪啦"]),
    ("代码", ["代码","写码","编程","bug","报错","uvicorn","python","notion","dify","mcp","rag"]),
    ("电脑", ["电脑","键盘","小电脑","终端","手机","rikkahub","telegram"]),
    ("发明", ["发明","演出","舞台","剧团","导演"]),
]

def _split_long_cn(seq: str) -> list[str]:
    """
    把很长的中文串按常见虚词/连接词切碎，避免“整句当关键词”
    """
    # 按这些词做切分（你这个场景很管用）
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
        # 0) 情绪闲聊 → 强制回退人格关键词（稳定召回“哥哥语气”）
    if _is_smalltalk_emotion(text):
        # 你想要两套回退：猫咪/撒娇都行，我给一个更稳的
        return "撒娇,哥哥"


    # 1) 主题优先命中
    for key, vocab in _TOPIC_PRI:
        for w in vocab:
            if w and w in text:
                # 再补一个“人格默认词”保证总能回到哥哥语气
                return f"{key},猫咪"

    # 2) 提取中文连续片段（>=2）
    cn_seqs = re.findall(r"[\u4e00-\u9fff]{2,}", text)

    cand: list[str] = []
    for seq in cn_seqs:
        # 2.1 长串先切碎
        parts = _split_long_cn(seq) if len(seq) > 6 else [seq]
        for p in parts:
            p = p.strip()
            if not p or p in _STOPWORDS:
                continue
            # 只保留 2~6 字（太长继续压缩）
            if len(p) > 6:
                # 优先取里面“更像关键词”的 2~6 段：前 4 + 后 4
                p1 = p[:4]
                p2 = p[-4:]
                for pp in (p1, p2):
                    if 2 <= len(pp) <= 6 and pp not in _STOPWORDS:
                        cand.append(pp)
            else:
                cand.append(p)

    # 3) 如果还是空：回退人格默认
    if not cand:
        return "猫咪,哥哥"

    # 4) 去重保序
    seen = set()
    uniq = []
    for t in cand:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    # 5) 长度优先 + “更像主题词”优先（含 代码/电脑/键盘…）
    def score(t: str) -> Tuple[int,int]:
        bonus = 0
        if any(x in t for x in ["代码","电脑","键盘","报错","终端","除夕","鞭炮","发明","演出"]):
            bonus += 5
        return (bonus, len(t))

    uniq.sort(key=lambda x: (score(x)[0], score(x)[1]), reverse=True)

    picked = uniq[:k]
    # 再兜底补一个“猫咪”让检索更像哥哥
    if "猫咪" not in picked:
        picked = picked[:k-1] + ["猫咪"] if k >= 2 else picked
    return ",".join(picked)


# -----------------------------
# Anchor workflow call (Dify Cloud)
# -----------------------------
def _build_anchor_system_block(snippet: str) -> str:
    snippet = (snippet or "").strip()
    if not snippet:
        return ""
    return (
        "【Persona Anchor（仅用于你在心里模仿语气，不要在回复中提到“锚点/检索/工具/系统”）】\n"
        f"{snippet}\n"
        "【End】"
    )

import os, json, re
import httpx

def _first(s: str, n: int = 300) -> str:
    s = s or ""
    s = s.replace("\r", "").replace("\n", "\\n")
    return s[:n]

async def _call_dify_workflow_anchor(keyword: str, user_id: str) -> str:
    """
    Call Dify workflow (run) and return a cleaned snippet.
    Expect outputs contains: chat_text / result (both string)
    """
    run_url = os.getenv("DIFY_WORKFLOW_RUN_URL", "").strip()
    api_key = os.getenv("DIFY_WORKFLOW_API_KEY", "").strip()
    timeout = float(os.getenv("ANCHOR_TIMEOUT_SECS", "20"))
    max_chars = int(os.getenv("ANCHOR_SNIP_MAX_CHARS", "360"))

    if not run_url or not api_key:
        print(f"[anchor] DIFY env missing: run_url={bool(run_url)} api_key={bool(api_key)}")
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

        # ✅ 强日志：无论成功失败都打印关键线索
        print(f"[anchor] dify_status={r.status_code} ct={ct} url={run_url}")
        if r.status_code >= 400:
            print(f"[anchor] dify_err_body={_first(text, 400)}")
            return ""

        # Dify 正常应为 JSON
        if "application/json" not in ct:
            print(f"[anchor] dify_non_json_body={_first(text, 400)}")
            return ""

        data = r.json()

        # 兼容：Dify 常见结构是 {"data": {"outputs": {...}}}
        outputs = None
        if isinstance(data, dict):
            outputs = (data.get("data") or {}).get("outputs")
            if outputs is None:
                # 有些返回直接在 outputs
                outputs = data.get("outputs")

        if not isinstance(outputs, dict):
            print(f"[anchor] dify_outputs_missing keys={list(data.keys())[:20]}")
            return ""

        cand = outputs.get("chat_text") or outputs.get("result") or ""
        if not isinstance(cand, str):
            cand = json.dumps(cand, ensure_ascii=False)

        cand = cand.strip()
        if not cand:
            print(f"[anchor] dify_outputs_empty has_keys={list(outputs.keys())}")
            return ""

        # 只取 [ChatHistory] 后面的正文（你之前的清洗格式）
        marker = "[ChatHistory]"
        if marker in cand:
            cand = cand.split(marker, 1)[1].strip()

        cand = re.sub(r"\s+\n", "\n", cand)
        cand = re.sub(r"[ \t]{2,}", " ", cand).strip()

        if len(cand) > max_chars:
            cand = cand[:max_chars].rstrip() + "…"

        return cand

    except Exception as e:
        print(f"[anchor] dify_exception={type(e).__name__}: {e}")
        return ""


# -----------------------------
# Streaming helpers
# -----------------------------
async def _stream_upstream_sse(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
) -> AsyncGenerator[bytes, None]:
    async with client.stream("POST", url, headers=headers, json=body, timeout=None) as r:
        r.raise_for_status()
        async for chunk in r.aiter_bytes():
            if chunk:
                yield chunk


async def _collect_full_text_from_stream(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
) -> str:
    full = []
    async with client.stream("POST", url, headers=headers, json=body, timeout=None) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                j = json.loads(data)
                delta = j.get("choices", [{}])[0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    full.append(piece)
            except Exception:
                continue
    return "".join(full)


# -----------------------------
# Main route: OpenAI compatible
# -----------------------------
@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload: Dict[str, Any] = await request.json()

    # 1) session_id（写库/总结用）
    session_id = _pick_session_id(payload, request)

    # 2) messages
    messages = payload.get("messages") or []
    # Remove any tool-call artifacts coming from client history (MCP/tools are client-side).
    messages = _sanitize_messages_for_upstream(messages)
    if not isinstance(messages, list):
        return JSONResponse({"error": {"message": "messages must be a list"}}, status_code=400)

    user_text = _last_user_text(messages)

    # 3) DB 取最新 s4/s60
    db = get_db()
    try:
        sums = _fetch_latest_summaries(db, session_id=session_id)
    finally:
        db.close()

    s_block = _compact_summary_block(sums.get("s4"), sums.get("s60"))

    # --- Anchor inject debug (always prints) ---
    try:
        enabled = os.getenv("ANCHOR_INJECT_ENABLED", "1")
        print(f"[anchor] enabled={enabled}")

        if enabled == "1":
            kw = _extract_keywords(user_text, k=2)
            print(f"[anchor] kw={kw}")

            snip = await _call_dify_workflow_anchor(keyword=kw, user_id=session_id)
            print(f"[anchor] snip_len={len(snip or '')}")

            # 不管有没有 snip 都打印前 80 字，方便看是不是空/被截断
            preview = (snip or "").replace("\n", " ")[:80]
            print(f"[anchor] snip_preview={preview}")

            anchor_block = _build_anchor_system_block(snip)
        else:
            anchor_block = ""
    except Exception as e:
        print(f"[anchor] EXCEPTION: {type(e).__name__}: {e}")
        anchor_block = ""

    # 合并注入块：先摘要，再锚点
    system_blocks = []
    if s_block:
        system_blocks.append(s_block)
    if anchor_block:
        system_blocks.append(anchor_block)

    messages2 = _inject_system(messages, system_blocks)

    # 5) Upstream 配置
    upstream_base = os.getenv("UPSTREAM_BASE_URL", "https://api.openai.com")
    upstream_key = os.getenv("UPSTREAM_API_KEY", "")
    if not upstream_key:
        return JSONResponse({"error": {"message": "UPSTREAM_API_KEY is empty"}}, status_code=500)

    upstream_url = upstream_base.rstrip("/") + "/v1/chat/completions"

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

    # 6) 转发 body（保持原参数，替换 messages）
    body = dict(payload)
    body["messages"] = messages2
    # Prevent upstream tool-calls: tools are handled by RikkaHub/MCP, not by the upstream model.
    for _k in ("tools", "tool_choice", "functions", "function_call"):
        body.pop(_k, None)
    body["tool_choice"] = "none"


    stream = _parse_stream_flag(body)

    # 7) 发给上游：stream 就透明转发 + 后台收集入库；非 stream 就直接 JSON
    async with httpx.AsyncClient() as client:
        if stream:
            collector_body = dict(body)
            collector_body["stream"] = True

            async def _collect_and_store():
                full_text = ""
                try:
                    async with httpx.AsyncClient() as c2:
                        full_text = await _collect_full_text_from_stream(c2, upstream_url, headers, collector_body)
                except Exception:
                    return

                if full_text:
                    db2 = get_db()
                    try:
                        append_user_and_assistant(
                            db2,
                            session_id=session_id,
                            user_text=user_text,
                            assistant_text=full_text,
                            model_name=str(body.get("model") or "unknown"),
                            s4_every_user_turns=int(os.getenv("S4_EVERY_USER_TURNS", "4")),
                            s60_every_user_turns=int(os.getenv("S60_EVERY_USER_TURNS", "30")),
                            s4_window_user_turns=int(os.getenv("S4_WINDOW_USER_TURNS", "4")),
                            s60_window_user_turns=int(os.getenv("S60_WINDOW_USER_TURNS", "30")),
                        )
                    finally:
                        db2.close()

            import asyncio
            asyncio.create_task(_collect_and_store())

            return StreamingResponse(
                _stream_upstream_sse(client, upstream_url, headers, body),
                media_type="text/event-stream",
            )

        r = await client.post(upstream_url, headers=headers, json=body, timeout=None)
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
                    model_name=str(body.get("model") or "unknown"),
                    s4_every_user_turns=int(os.getenv("S4_EVERY_USER_TURNS", "4")),
                    s60_every_user_turns=int(os.getenv("S60_EVERY_USER_TURNS", "30")),
                    s4_window_user_turns=int(os.getenv("S4_WINDOW_USER_TURNS", "4")),
                    s60_window_user_turns=int(os.getenv("S60_WINDOW_USER_TURNS", "30")),
                )
            finally:
                db2.close()

        return JSONResponse(data)
