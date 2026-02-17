from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

router = APIRouter()

JSON_UTF8 = "application/json; charset=utf-8"

DIFY_BASE_URL = os.getenv("DIFY_BASE_URL", "https://api.dify.ai").strip()
DIFY_API_KEY = (os.getenv("DIFY_API_KEY") or os.getenv("DIFY_WORKFLOW_API_KEY") or "").strip()
DIFY_WORKFLOW_RUN_URL = os.getenv("DIFY_WORKFLOW_RUN_URL", "https://api.dify.ai/v1/workflows/run").strip()
DIFY_WORKFLOW_ID_ANCHOR = os.getenv("DIFY_WORKFLOW_ID_ANCHOR", "").strip()

# ✅ 默认降级：别用 2025-11-25，RikkaHub 很可能不支持
DEFAULT_MCP_PROTOCOL_VERSION = os.getenv("MCP_PROTOCOL_VERSION", "2025-06-18").strip()

# ✅ 兼容集合（按需增删）
SUPPORTED_VERSIONS = {
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
    "2024-10-07",
}

CTX_MAX = int(os.getenv("ANCHOR_SNIP_MAX", "400"))
DIFY_TIMEOUT_SECS = float(os.getenv("DIFY_TIMEOUT_SECS", "60"))

_EMO_MARKERS = ["哥哥", "类", "喵", "猫咪", "小猫咪", "宝宝", "亲", "抱", "mua", "啾", "嘿嘿", "🥺", "😙", "😗", "😽", "😭", "🥰", "💖", "🖤"]

def _jsonrpc_error(_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": _id, "error": err}

def _jsonrpc_result(_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": _id, "result": result}

def _negotiate_protocol_version(request: Request, params: Dict[str, Any]) -> str:
    # 1) initialize 里客户端会传 params.protocolVersion
    pv = str((params or {}).get("protocolVersion") or "").strip()
    if pv and pv in SUPPORTED_VERSIONS:
        return pv

    # 2) HTTP header 里也可能带 MCP-Protocol-Version
    hv = (request.headers.get("MCP-Protocol-Version") or "").strip()
    if hv and hv in SUPPORTED_VERSIONS:
        return hv

    # 3) 最后才用服务端默认
    return DEFAULT_MCP_PROTOCOL_VERSION if DEFAULT_MCP_PROTOCOL_VERSION in SUPPORTED_VERSIONS else "2025-06-18"

def _mcp_wrap_text(res_obj: Dict[str, Any], text_out: str, is_error: bool) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": text_out or ""}], "isError": bool(is_error), "data": res_obj}

def _is_emo_chitchat(text: str) -> bool:
    t = (text or "").strip()
    return any(m in t for m in _EMO_MARKERS)

def _truncate_ctx(text: str) -> str:
    t = (text or "").strip().replace("\r", "")
    if not t:
        return ""
    if len(t) <= CTX_MAX:
        return t
    return t[:CTX_MAX].rstrip() + "…"

async def _call_dify_anchor(keyword: str, user: str = "mcp") -> Dict[str, Any]:
    if not DIFY_API_KEY:
        raise RuntimeError("Missing env DIFY_API_KEY (or DIFY_WORKFLOW_API_KEY)")

    url = DIFY_WORKFLOW_RUN_URL or f"{DIFY_BASE_URL.rstrip('/')}/v1/workflows/run"
    headers = {"Authorization": f"Bearer {DIFY_API_KEY}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"inputs": {"keyword": keyword}, "response_mode": "blocking", "user": user}
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

    return {"result": str(outputs.get("result") or ""), "chat_text": str(outputs.get("chat_text") or "")}

async def _handle_jsonrpc(request: Request, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    _id = msg.get("id", None)
    method = msg.get("method", "")
    params = (msg.get("params", {}) or {}) if isinstance(msg, dict) else {}
    is_notification = isinstance(msg, dict) and ("id" not in msg)

    # ✅ 版本谈判：每个请求都算一次，initialize 最关键
    pv = _negotiate_protocol_version(request, params)
    request.state.mcp_pv = pv  # 给外层响应 header 用

    if method == "initialize":
        result = {
            "protocolVersion": pv,
            "serverInfo": {"name": "gateway_ctx", "version": "2.1"},
            "capabilities": {"tools": {}},
        }
        return None if is_notification else _jsonrpc_result(_id, result)

    if method == "tools/list":
        tools = [{
            "name": "gateway_ctx",
            "description": "Unified gateway context builder: keyword + Anchor RAG snippet (compact). Returns MCP content[].text + debug data.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "search keywords"},
                    "text": {"type": "string", "description": "optional raw user message"},
                    "user": {"type": "string", "description": "optional user/session id"},
                },
                "required": ["keyword"],
            },
        }]
        return None if is_notification else _jsonrpc_result(_id, {"tools": tools})

    if method != "tools/call":
        return None if is_notification else _jsonrpc_error(_id, -32601, f"Method not found: {method}")

    name = params.get("name")
    arguments = params.get("arguments", {}) or {}
    if name != "gateway_ctx":
        return None if is_notification else _jsonrpc_error(_id, -32601, f"Unknown tool: {name}")

    keyword = str(arguments.get("keyword") or "").strip()
    text = str(arguments.get("text") or "").strip()
    user = str(arguments.get("user") or "mcp").strip()

    if not keyword:
        # 兜底：至少别空
        keyword = "猫咪,哥哥" if _is_emo_chitchat(text) else "撒娇,哥哥"

    try:
        dify = await _call_dify_anchor(keyword=keyword, user=user)
        outs = _extract_outputs(dify)
        picked = (outs.get("result") or "").strip() or (outs.get("chat_text") or "").strip()
        ctx = _truncate_ctx(picked)
        res_obj = {"keyword": keyword, "ctx": ctx, "raw": outs}
        return None if is_notification else _jsonrpc_result(_id, _mcp_wrap_text(res_obj, ctx, is_error=False))
    except Exception as e:
        res_obj = {"keyword": keyword, "error": str(e)}
        return None if is_notification else _jsonrpc_result(_id, _mcp_wrap_text(res_obj, str(e), is_error=True))

@router.api_route("/gateway_ctx", methods=["GET", "POST", "OPTIONS"])
async def gateway_ctx_mcp(request: Request):
    # 默认 header 先给个保守版本
    default_pv = DEFAULT_MCP_PROTOCOL_VERSION if DEFAULT_MCP_PROTOCOL_VERSION in SUPPORTED_VERSIONS else "2025-06-18"

    if request.method in ("GET", "OPTIONS"):
        return JSONResponse(
            {"ok": True, "name": "gateway_ctx", "mcp": True},
            headers={"MCP-Protocol-Version": default_pv},
            media_type=JSON_UTF8,
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse(_jsonrpc_error(None, -32700, "Parse error"), status_code=400, media_type=JSON_UTF8)

    if isinstance(body, list):
        results = []
        for item in body:
            r = await _handle_jsonrpc(request, item if isinstance(item, dict) else {})
            if r is not None:
                results.append(r)
        pv = getattr(request.state, "mcp_pv", default_pv)
        return JSONResponse(results, headers={"MCP-Protocol-Version": pv}, media_type=JSON_UTF8)

    resp = await _handle_jsonrpc(request, body if isinstance(body, dict) else {})
    pv = getattr(request.state, "mcp_pv", default_pv)
    if resp is None:
        return Response(status_code=204, headers={"MCP-Protocol-Version": pv})
    return JSONResponse(resp, headers={"MCP-Protocol-Version": pv}, media_type=JSON_UTF8)
