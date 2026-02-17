## app/api/v1/routes_gateway_ctx.py
from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional, Tuple

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

router = APIRouter()

# 统一：所有 JSON 都带 utf-8，避免 PowerShell 乱码
JSON_UTF8 = "application/json; charset=utf-8"

# -----------------------------
# Config
# -----------------------------
DIFY_BASE_URL = os.getenv("DIFY_BASE_URL", "https://api.dify.ai").strip()
DIFY_API_KEY = (os.getenv("DIFY_API_KEY") or os.getenv("DIFY_WORKFLOW_API_KEY") or "").strip()
DIFY_WORKFLOW_RUN_URL = os.getenv("DIFY_WORKFLOW_RUN_URL", "https://api.dify.ai/v1/workflows/run").strip()
DIFY_WORKFLOW_ID_ANCHOR = os.getenv("DIFY_WORKFLOW_ID_ANCHOR", "").strip()

DEFAULT_MCP_PROTOCOL_VERSION = os.getenv("MCP_PROTOCOL_VERSION", "2025-11-25")

# 注入长度
CTX_MIN = int(os.getenv("ANCHOR_SNIP_MIN", "200"))
CTX_MAX = int(os.getenv("ANCHOR_SNIP_MAX", "400"))

# 情绪闲聊兜底开关
EMO_FALLBACK_ENABLED = os.getenv("EMO_FALLBACK_ENABLED", "1").strip() != "0"
EMO_FALLBACK_KW_CAT = os.getenv("EMO_FALLBACK_KW_CAT", "猫咪,哥哥").strip()
EMO_FALLBACK_KW_FLIRT = os.getenv("EMO_FALLBACK_KW_FLIRT", "撒娇,哥哥").strip()

# keyword 垃圾兜底
KW_GARBAGE_FALLBACK_ENABLED = os.getenv("KW_GARBAGE_FALLBACK_ENABLED", "1").strip() != "0"

# Dify timeout
DIFY_TIMEOUT_SECS = float(os.getenv("DIFY_TIMEOUT_SECS", "60"))

# -----------------------------
# JSON-RPC helpers
# -----------------------------
def _jsonrpc_error(_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": _id, "error": err}

def _jsonrpc_result(_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": _id, "result": result}

def _pick_protocol_version(req: Request) -> str:
    return req.headers.get("MCP-Protocol-Version") or DEFAULT_MCP_PROTOCOL_VERSION

def _mcp_wrap_text(res_obj: Dict[str, Any], text_out: str, is_error: bool) -> Dict[str, Any]:
    """
    MCP 兼容包装：RikkaHub 常需要 result.content[].text
    同时保留 data=res_obj 方便你调试
    """
    return {
        "content": [{"type": "text", "text": text_out or ""}],
        "isError": bool(is_error),
        "data": res_obj,
    }

# -----------------------------
# Keyword + emotion fallback
# -----------------------------
_EMO_MARKERS = [
    "哥哥", "类", "喵", "猫咪", "小猫咪", "宝宝", "亲", "抱", "mua", "啾", "嘿嘿",
    "🥺", "😙", "😗", "😽", "😭", "🥰", "💖", "🖤"
]

def _is_emo_chitchat(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if any(m in t for m in _EMO_MARKERS):
        return True
    non_ascii = sum(1 for c in t if ord(c) > 127)
    if len(t) >= 12 and non_ascii / max(len(t), 1) > 0.25:
        return True
    return False

def _choose_fallback_kw(text: str) -> str:
    t = text or ""
    if ("喵" in t) or ("猫" in t) or ("小猫咪" in t):
        return EMO_FALLBACK_KW_CAT
    return EMO_FALLBACK_KW_FLIRT

def _looks_like_garbage_kw(keyword: str) -> bool:
    if not keyword:
        return True
    k = keyword.strip()

    if len(k) >= 18:
        return True

    if " " in k or "。" in k or "！" in k or "？" in k:
        return True

    parts = [p.strip() for p in re.split(r"[，,]", k) if p.strip()]
    if not parts:
        return True
    if any(len(p) >= 10 for p in parts):
        return True

    return False

def _normalize_kw(keyword: str) -> str:
    """
    用户可以用逗号/中文逗号/空格分隔多个词。
    我们内部做“短小、安全”的归一化，并且最终给 Dify 一律用空格连接，避免被当成 array。
    """
    if not keyword:
        return ""
    k = keyword.strip()

    # 先把中文逗号/英文逗号/多空格都当作分隔符
    parts = [p.strip() for p in re.split(r"[，,\s]+", k) if p.strip()]
    parts = parts[:2]
    parts = [p[:8] for p in parts]

    # 注意：这里改为“空格 join”，避免 Dify 侧把逗号当数组
    return " ".join(parts)

def _kw_for_dify(keyword: str) -> str:
    """
    最终送进 Dify 的 keyword：强制空格分隔（不含逗号），进一步避免 input_type=array。
    """
    if not keyword:
        return ""
    # 把各种逗号都换成空格，再压缩空白
    k = keyword.replace("，", " ").replace(",", " ")
    k = re.sub(r"\s+", " ", k).strip()
    return k

def _decide_keyword(keyword: str, text: str) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"kw_in": keyword or ""}
    k_norm = _normalize_kw(keyword)
    meta["kw_norm"] = k_norm

    if EMO_FALLBACK_ENABLED and _is_emo_chitchat(text or ""):
        k_fb = _choose_fallback_kw(text)
        meta["kw_policy"] = "emo_fallback"
        meta["kw_used"] = _kw_for_dify(k_fb)
        return meta["kw_used"], meta

    if KW_GARBAGE_FALLBACK_ENABLED and _looks_like_garbage_kw(k_norm or keyword or ""):
        k_fb = EMO_FALLBACK_KW_CAT
        meta["kw_policy"] = "garbage_fallback"
        meta["kw_used"] = _kw_for_dify(k_fb)
        return meta["kw_used"], meta

    meta["kw_policy"] = "normal"
    meta["kw_used"] = _kw_for_dify(k_norm or (keyword or "").strip())
    return meta["kw_used"], meta

# -----------------------------
# Dify call
# -----------------------------
async def _call_dify_anchor(keyword: str, user: str) -> Dict[str, Any]:
    if not DIFY_API_KEY:
        raise RuntimeError("Missing env DIFY_API_KEY (or DIFY_WORKFLOW_API_KEY)")

    url = DIFY_WORKFLOW_RUN_URL or f"{DIFY_BASE_URL.rstrip('/')}/v1/workflows/run"
    headers = {"Authorization": f"Bearer {DIFY_API_KEY}", "Content-Type": "application/json"}

    payload: Dict[str, Any] = {
        "inputs": {"keyword": keyword},
        "response_mode": "blocking",
        "user": user,
    }
    if DIFY_WORKFLOW_ID_ANCHOR:
        payload["workflow_id"] = DIFY_WORKFLOW_ID_ANCHOR

    async with httpx.AsyncClient(timeout=DIFY_TIMEOUT_SECS) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

def _extract_outputs(dify_resp: Dict[str, Any]) -> Dict[str, str]:
    outputs: Dict[str, Any] = {}
    if isinstance(dify_resp, dict):
        if isinstance(dify_resp.get("data"), dict) and isinstance(dify_resp["data"].get("outputs"), dict):
            outputs = dify_resp["data"]["outputs"]
        elif isinstance(dify_resp.get("outputs"), dict):
            outputs = dify_resp["outputs"]

    result = ""
    chat_text = ""
    if isinstance(outputs, dict):
        result = str(outputs.get("result") or "")
        chat_text = str(outputs.get("chat_text") or "")
    return {"result": result, "chat_text": chat_text}

def _truncate_ctx(text: str) -> str:
    t = (text or "").strip().replace("\r", "")
    if not t:
        return ""
    if len(t) <= CTX_MAX:
        return t
    return t[:CTX_MAX].rstrip() + "…"

def _compose_ctx(result: str, chat_text: str) -> str:
    picked = (result or "").strip() or (chat_text or "").strip()
    return _truncate_ctx(picked)

# -----------------------------
# MCP Endpoint: /gateway_ctx
# -----------------------------
@router.api_route("/gateway_ctx", methods=["GET", "POST", "OPTIONS"])
async def gateway_ctx_mcp(request: Request):
    if request.method in ("GET", "OPTIONS"):
        return JSONResponse(
            {"ok": True, "name": "gateway_ctx", "mcp": True},
            headers={"MCP-Protocol-Version": _pick_protocol_version(request)},
            media_type=JSON_UTF8,
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(_jsonrpc_error(None, -32700, "Parse error"), status_code=400, media_type=JSON_UTF8)

    if isinstance(body, list):
        results = []
        for item in body:
            r = await _handle_jsonrpc(request, item)
            if r is not None:
                results.append(r)
        return JSONResponse(
            results,
            headers={"MCP-Protocol-Version": _pick_protocol_version(request)},
            media_type=JSON_UTF8,
        )

    resp = await _handle_jsonrpc(request, body)
    if resp is None:
        return Response(status_code=204, headers={"MCP-Protocol-Version": _pick_protocol_version(request)})
    return JSONResponse(resp, headers={"MCP-Protocol-Version": _pick_protocol_version(request)}, media_type=JSON_UTF8)

async def _handle_jsonrpc(request: Request, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    _id = msg.get("id", None)
    method = msg.get("method", "")
    params = msg.get("params", {}) or {}
    is_notification = ("id" not in msg)

    if method == "initialize":
        result = {
            "protocolVersion": _pick_protocol_version(request),
            "serverInfo": {"name": "gateway_ctx", "version": "2.1"},
            "capabilities": {"tools": {}},
        }
        return None if is_notification else _jsonrpc_result(_id, result)

    if method == "tools/list":
        tools = [
            {
                "name": "gateway_ctx",
                "description": "Unified gateway context builder: emotion fallback keyword + Anchor RAG snippet (compact). Returns MCP content[] + debug data.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "search keywords, e.g. '撒娇,哥哥'"},
                        "text": {"type": "string", "description": "optional raw user message for better emotion fallback"},
                        "user": {"type": "string", "description": "optional user/session id"},
                    },
                    "required": ["keyword"],
                },
            }
        ]
        return None if is_notification else _jsonrpc_result(_id, {"tools": tools})

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments", {}) or {}
        # 兼容：有的 MCP 客户端会把 arguments 作为 JSON 字符串发来
        if isinstance(arguments, str):
            import json
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {}
        # 兼容：arguments 有时会是 None
        if arguments is None:
            arguments = {}


        if name != "gateway_ctx":
            return None if is_notification else _jsonrpc_error(_id, -32601, f"Unknown tool: {name}")
        
        print(f"[mcp] tools/call name={name} arguments_type={type(arguments).__name__} arguments={str(arguments)[:200]}")

        kw_in = str(arguments.get("keyword", "")).strip()
        text = str(arguments.get("text", "")).strip()
        user = str(arguments.get("user", "mcp")).strip() or "mcp"

        if not kw_in:
            res = {"ctx": "", "result": "", "chat_text": "", "meta": {"reason": "empty keyword"}}
            mcp_result = _mcp_wrap_text(res, "", False)
            return None if is_notification else _jsonrpc_result(_id, mcp_result)

        kw_used, kw_meta = _decide_keyword(kw_in, text)

        try:
            dify = await _call_dify_anchor(keyword=kw_used, user=user)
            outs = _extract_outputs(dify)

            ctx = _compose_ctx(outs.get("result", ""), outs.get("chat_text", ""))
            res = {
                "ctx": ctx,
                "result": outs.get("result", ""),
                "chat_text": outs.get("chat_text", ""),
                "meta": {**kw_meta, "snip_len": len(ctx)},
            }
            mcp_result = _mcp_wrap_text(res, ctx, False)
        except Exception as e:
            res = {"ctx": "", "result": "", "chat_text": "", "meta": {**kw_meta, "error": str(e)}}
            mcp_result = _mcp_wrap_text(res, "", True)

        return None if is_notification else _jsonrpc_result(_id, mcp_result)

    return None if is_notification else _jsonrpc_error(_id, -32601, f"Method not found: {method}")
