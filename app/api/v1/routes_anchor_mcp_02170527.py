# app/api/v1/routes_anchor_mcp.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

router = APIRouter()

JSON_UTF8 = "application/json; charset=utf-8"

DIFY_API_KEY = (os.getenv("DIFY_API_KEY") or os.getenv("DIFY_WORKFLOW_API_KEY") or "").strip()
DIFY_WORKFLOW_RUN_URL = os.getenv("DIFY_WORKFLOW_RUN_URL", "https://api.dify.ai/v1/workflows/run").strip()
DIFY_WORKFLOW_ID_ANCHOR = os.getenv("DIFY_WORKFLOW_ID_ANCHOR", "").strip()
DIFY_TIMEOUT_SECS = float(os.getenv("DIFY_TIMEOUT_SECS", "60"))

DEFAULT_MCP_PROTOCOL_VERSION = os.getenv("MCP_PROTOCOL_VERSION", "2025-11-25")


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


async def _call_dify_workflow(keyword: str, user: str) -> Dict[str, Any]:
    if not DIFY_API_KEY:
        raise RuntimeError("Missing DIFY_API_KEY / DIFY_WORKFLOW_API_KEY")

    headers = {"Authorization": f"Bearer {DIFY_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "inputs": {"keyword": keyword},
        "response_mode": "blocking",
        "user": user,
    }
    if DIFY_WORKFLOW_ID_ANCHOR:
        payload["workflow_id"] = DIFY_WORKFLOW_ID_ANCHOR

    async with httpx.AsyncClient(timeout=DIFY_TIMEOUT_SECS) as client:
        r = await client.post(DIFY_WORKFLOW_RUN_URL, headers=headers, json=payload)
        ct = (r.headers.get("content-type") or "").lower()
        print(f"[anchor] dify_status={r.status_code} ct={ct} url={DIFY_WORKFLOW_RUN_URL}", flush=True)
        r.raise_for_status()
        return r.json()


def _extract_outputs(dify: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(dify, dict):
        return {}
    if isinstance(dify.get("data"), dict) and isinstance(dify["data"].get("outputs"), dict):
        return dify["data"]["outputs"]
    if isinstance(dify.get("result"), dict) and isinstance(dify["result"].get("outputs"), dict):
        return dify["result"]["outputs"]
    if isinstance(dify.get("outputs"), dict):
        return dify["outputs"]
    return {}


def _compose_anchor_block(result_text: str, chat_text: str) -> str:
    rt = (result_text or "").strip()
    ct = (chat_text or "").strip()
    return rt if rt else ct


@router.post("/api/v1/mcp/anchor_rag")
async def mcp_anchor_rag(request: Request):
    # 先读 raw bytes，定位“中文是否在客户端阶段被替换成 ?”
    raw = await request.body()
    raw_preview = raw[:400]
    raw_preview_text = raw_preview.decode("utf-8", errors="replace")
    print(
        f"[mcp] raw_bytes_len={len(raw)} content_type={request.headers.get('content-type')} "
        f"raw_preview_utf8={raw_preview_text!r}",
        flush=True,
    )

    try:
        body = json.loads(raw) if raw else {}
    except Exception as e:
        return JSONResponse(
            _jsonrpc_error(None, -32700, "Parse error", data={"err": str(e)}),
            status_code=400,
            media_type=JSON_UTF8,
        )

    # batch
    if isinstance(body, list):
        results = []
        for item in body:
            r = await _handle_jsonrpc(item)
            if r is not None:
                results.append(r)
        return JSONResponse(
            results,
            headers={"MCP-Protocol-Version": _pick_protocol_version(request)},
            media_type=JSON_UTF8,
        )

    resp = await _handle_jsonrpc(body)
    if resp is None:
        return Response(status_code=204, headers={"MCP-Protocol-Version": _pick_protocol_version(request)})
    return JSONResponse(resp, headers={"MCP-Protocol-Version": _pick_protocol_version(request)}, media_type=JSON_UTF8)


async def _handle_jsonrpc(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    _id = msg.get("id", None)
    method = msg.get("method", "")
    params = msg.get("params", {}) or {}
    is_notification = ("id" not in msg)

    if method == "initialize":
        result = {
            "protocolVersion": DEFAULT_MCP_PROTOCOL_VERSION,
            "serverInfo": {"name": "anchor_rag", "version": "2.2"},
            "capabilities": {"tools": {}},
        }
        return None if is_notification else _jsonrpc_result(_id, result)

    if method == "tools/list":
        tools = [
            {
                "name": "anchor_rag",
                "description": "Anchor RAG tool (direct). Returns MCP content[] + debug data.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string"},
                        "user": {"type": "string"},
                    },
                    "required": ["keyword"],
                },
            }
        ]
        return None if is_notification else _jsonrpc_result(_id, {"tools": tools})

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments", {}) or {}

        # compat: arguments may be JSON string
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {}
        if arguments is None:
            arguments = {}

        print(
            f"[mcp] tools/call name={name} arguments_type={type(arguments).__name__} "
            f"keyword_repr={repr((arguments or {}).get('keyword'))}",
            flush=True,
        )

        if name != "anchor_rag":
            return None if is_notification else _jsonrpc_error(_id, -32601, f"Unknown tool: {name}")

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
            print(f"[anchor] snip_len={len(snip)} snip_preview={snip[:200]}", flush=True)
            res = {"text": snip, "meta": {"keyword": keyword, "snip_len": len(snip)}}
            mcp_result = _mcp_wrap_text(res, snip, False)
        except Exception as e:
            res = {"text": "", "meta": {"keyword": keyword, "error": str(e)}}
            mcp_result = _mcp_wrap_text(res, "", True)

        return None if is_notification else _jsonrpc_result(_id, mcp_result)

    return None if is_notification else _jsonrpc_error(_id, -32601, f"Method not found: {method}")
