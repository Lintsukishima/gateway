# app/api/v1/routes_anchor_mcp.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

router = APIRouter()

JSON_UTF8 = "application/json; charset=utf-8"

# -----------------------------
# Config
# -----------------------------
DIFY_BASE_URL = os.getenv("DIFY_BASE_URL", "https://api.dify.ai").strip()
DIFY_API_KEY = (os.getenv("DIFY_API_KEY") or os.getenv("DIFY_WORKFLOW_API_KEY") or "").strip()
DIFY_WORKFLOW_ID_ANCHOR = os.getenv("DIFY_WORKFLOW_ID_ANCHOR", "").strip()
DIFY_WORKFLOW_RUN_URL = os.getenv("DIFY_WORKFLOW_RUN_URL", "https://api.dify.ai/v1/workflows/run").strip()

DEFAULT_MCP_PROTOCOL_VERSION = os.getenv("MCP_PROTOCOL_VERSION", "2025-11-25")

ANCHOR_SNIP_MIN = int(os.getenv("ANCHOR_SNIP_MIN", "200"))
ANCHOR_SNIP_MAX = int(os.getenv("ANCHOR_SNIP_MAX", "400"))
DIFY_TIMEOUT_SECS = float(os.getenv("DIFY_TIMEOUT_SECS", "60"))

# -----------------------------
# Helpers
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
    return {
        "content": [{"type": "text", "text": text_out or ""}],
        "isError": bool(is_error),
        "data": res_obj,
    }

def _truncate_to_range(text: str, min_chars: int, max_chars: int) -> str:
    t = (text or "").strip().replace("\r", "")
    if not t:
        return ""
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "…"

def _compose_anchor_block(notion_result: str, kb_chat_text: str) -> str:
    primary = (notion_result or "").strip()
    secondary = (kb_chat_text or "").strip()
    picked = primary if primary else secondary
    return _truncate_to_range(picked, ANCHOR_SNIP_MIN, ANCHOR_SNIP_MAX)

async def _call_dify_workflow(keyword: str, user: str = "mcp") -> Dict[str, Any]:
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

    result = ""
    chat_text = ""
    if isinstance(outputs, dict):
        result = str(outputs.get("result") or "")
        chat_text = str(outputs.get("chat_text") or "")
    return {"result": result, "chat_text": chat_text}

# -----------------------------
# MCP Endpoint: /anchor_rag
# -----------------------------
@router.api_route("/anchor_rag", methods=["GET", "POST", "OPTIONS"])
async def anchor_rag_mcp(request: Request):
    if request.method in ("GET", "OPTIONS"):
        return JSONResponse(
            {"ok": True, "name": "anchor_rag", "mcp": True},
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
        return JSONResponse(results, headers={"MCP-Protocol-Version": _pick_protocol_version(request)}, media_type=JSON_UTF8)

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
            "serverInfo": {"name": "anchor_rag", "version": "2.1"},
            "capabilities": {"tools": {}},
        }
        return None if is_notification else _jsonrpc_result(_id, result)

    if method == "tools/list":
        tools = [
            {
                "name": "anchor_rag",
                "description": "Search persona anchors via Dify workflow and return a compact style snippet for prompt injection.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"keyword": {"type": "string"}, "user": {"type": "string"}},
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


        if name != "anchor_rag":
            return None if is_notification else _jsonrpc_error(_id, -32601, f"Unknown tool: {name}")
        
        print(f"[mcp] tools/call name={name} arguments_type={type(arguments).__name__} arguments={str(arguments)[:200]}")

        keyword = str(arguments.get("keyword", "")).strip()
        user = str(arguments.get("user", "mcp")).strip() or "mcp"

        if not keyword:
            res = {"text": "", "meta": {"reason": "empty keyword"}}
            mcp_result = _mcp_wrap_text(res, "", False)
            return None if is_notification else _jsonrpc_result(_id, mcp_result)

        try:
            dify = await _call_dify_workflow(keyword=keyword, user=user)
            outs = _extract_outputs(dify)
            snip = _compose_anchor_block(outs.get("result", ""), outs.get("chat_text", ""))
            res = {"text": snip, "meta": {"keyword": keyword, "snip_len": len(snip)}}
            mcp_result = _mcp_wrap_text(res, snip, False)
        except Exception as e:
            res = {"text": "", "meta": {"keyword": keyword, "error": str(e)}}
            mcp_result = _mcp_wrap_text(res, "", True)

        return None if is_notification else _jsonrpc_result(_id, mcp_result)

    return None if is_notification else _jsonrpc_error(_id, -32601, f"Method not found: {method}")
