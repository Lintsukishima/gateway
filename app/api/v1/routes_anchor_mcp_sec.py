# app/api/v1/routes_anchor_mcp.py
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

router = APIRouter()

# -----------------------------
# Config
# -----------------------------
DIFY_BASE_URL = os.getenv("DIFY_BASE_URL", "https://api.dify.ai")
DIFY_API_KEY = os.getenv("DIFY_API_KEY", "")
DIFY_WORKFLOW_ID_ANCHOR = os.getenv("DIFY_WORKFLOW_ID_ANCHOR", "")  # 你 Anchor_RAG 工作流的 ID（推荐填）
# 也允许你直接填完整 URL（两者二选一，URL 优先）
DIFY_WORKFLOW_RUN_URL = os.getenv("DIFY_WORKFLOW_RUN_URL", "").strip()

# MCP 协议版本：尽量兼容（很多客户端会发 MCP-Protocol-Version）
DEFAULT_MCP_PROTOCOL_VERSION = os.getenv("MCP_PROTOCOL_VERSION", "2025-11-25")

# 轻注入长度限制（你想要后续方便调）
ANCHOR_SNIP_MIN = int(os.getenv("ANCHOR_SNIP_MIN", "200"))
ANCHOR_SNIP_MAX = int(os.getenv("ANCHOR_SNIP_MAX", "400"))

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

def _truncate_to_range(text: str, min_chars: int, max_chars: int) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if len(t) <= max_chars:
        return t
    # 简单截断（你后面想更“自然”可以改成按句号/换行截断）
    return t[:max_chars].rstrip() + "…"

def _compose_anchor_block(notion_result: str, kb_chat_text: str) -> str:
    """
    只注入 1 段，200~400 字为目标。
    优先使用更“像人格锚点原句”的片段（通常 Notion result 会更像原文）
    """
    # 候选来源：Notion result > chat_text
    primary = (notion_result or "").strip()
    secondary = (kb_chat_text or "").strip()

    # primary 里往往有 “Notion hits...” 等前缀，你之前清洗节点已经处理得比较干净
    # 这里再做一次“取 content 部分”的轻处理：直接拿整段前若干字
    picked = primary if primary else secondary
    picked = picked.replace("\r", "").strip()

    # 目标：200~400
    snip = _truncate_to_range(picked, ANCHOR_SNIP_MIN, ANCHOR_SNIP_MAX)
    return snip

async def _call_dify_workflow(keyword: str, user: str = "mcp") -> Dict[str, Any]:
    """
    调用 Dify workflow run
    期望返回字段里能拿到 outputs: { result: str, chat_text: str }
    """
    if not DIFY_API_KEY:
        raise RuntimeError("Missing env DIFY_API_KEY")

    if DIFY_WORKFLOW_RUN_URL:
        url = DIFY_WORKFLOW_RUN_URL
    else:
        if not DIFY_WORKFLOW_ID_ANCHOR:
            raise RuntimeError("Missing env DIFY_WORKFLOW_ID_ANCHOR (or DIFY_WORKFLOW_RUN_URL)")
        url = f"{DIFY_BASE_URL.rstrip('/')}/v1/workflows/run"

    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json",
    }

    # Dify workflow run 标准 body
    # workflow_id 方案：走 /v1/workflows/run 时需要 workflow_id
    payload: Dict[str, Any] = {
        "inputs": {"keyword": keyword},
        "response_mode": "blocking",
        "user": user,
    }
    if not DIFY_WORKFLOW_RUN_URL:
        payload["workflow_id"] = DIFY_WORKFLOW_ID_ANCHOR

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    return data

def _extract_outputs(dify_resp: Dict[str, Any]) -> Dict[str, str]:
    """
    兼容不同 Dify 返回形态：outputs/result/chat_text 位置可能不同
    """
    # 常见：{"data": {"outputs": {...}}}
    outputs = {}
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
# Routes (MCP endpoint)
# -----------------------------
@router.api_route("/gateway_ctx", methods=["GET", "POST", "OPTIONS"])
async def gateway_ctx(request: Request):
    # 让客户端探测不报 405
    if request.method in ("GET", "OPTIONS"):
        return JSONResponse(
            {
                "ok": True,
                "name": "gateway_ctx",
                "mcp": True,
                "hint": "Use POST JSON-RPC: initialize, tools/list, tools/call",
            },
            headers={"MCP-Protocol-Version": _pick_protocol_version(request)},
        )

    # POST: JSON-RPC
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(_jsonrpc_error(None, -32700, "Parse error"), status_code=400)

    # 支持 batch（有些客户端会批量发）
    if isinstance(body, list):
        results = []
        for item in body:
            results.append(await _handle_jsonrpc(request, item))
        # JSON-RPC batch：如果全是通知（id=None），可返回空
        results = [x for x in results if x is not None]
        return JSONResponse(results, headers={"MCP-Protocol-Version": _pick_protocol_version(request)})

    resp = await _handle_jsonrpc(request, body)
    if resp is None:
        # notification：不返回
        return Response(status_code=204, headers={"MCP-Protocol-Version": _pick_protocol_version(request)})
    return JSONResponse(resp, headers={"MCP-Protocol-Version": _pick_protocol_version(request)})

async def _handle_jsonrpc(request: Request, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    _id = msg.get("id", None)
    method = msg.get("method", "")
    params = msg.get("params", {}) or {}

    # notification: id 缺失时不返回
    is_notification = ("id" not in msg)

    # --- initialize ---
    if method == "initialize":
        result = {
            "protocolVersion": _pick_protocol_version(request),
            "serverInfo": {"name": "gateway_ctx", "version": "2.0"},
            "capabilities": {
                "tools": {},
                # 你后面要扩展 roots / resources / prompts 都可以在这儿加
            },
        }
        return None if is_notification else _jsonrpc_result(_id, result)

    # --- tools/list ---
    if method == "tools/list":
        tools = [
            {
                "name": "anchor_rag",
                "description": "Search persona anchors via Dify workflow and return a compact style snippet for prompt injection.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "keywords for searching anchors, e.g. '撒娇,哥哥'"},
                        "user": {"type": "string", "description": "optional user/session id"},
                    },
                    "required": ["keyword"],
                },
            }
        ]
        return None if is_notification else _jsonrpc_result(_id, {"tools": tools})

    # --- tools/call ---
    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments", {}) or {}

        if name == "anchor_rag":
            keyword = str(arguments.get("keyword", "")).strip()
            user = str(arguments.get("user", "mcp")).strip() or "mcp"

            if not keyword:
                res = {"text": "", "meta": {"reason": "empty keyword"}}
                return None if is_notification else _jsonrpc_result(_id, res)

            try:
                dify = await _call_dify_workflow(keyword=keyword, user=user)
                outs = _extract_outputs(dify)
                snip = _compose_anchor_block(outs.get("result", ""), outs.get("chat_text", ""))
                res = {
                    "text": snip,
                    "meta": {
                        "keyword": keyword,
                        "snip_len": len(snip),
                    },
                }
            except Exception as e:
                res = {
                    "text": "",
                    "meta": {"keyword": keyword, "error": str(e)},
                }

            return None if is_notification else _jsonrpc_result(_id, res)

        return None if is_notification else _jsonrpc_error(_id, -32601, f"Unknown tool: {name}")

    # unknown method
    return None if is_notification else _jsonrpc_error(_id, -32601, f"Method not found: {method}")
