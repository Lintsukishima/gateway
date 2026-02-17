# app/api/v1/routes_gateway_ctx.py
from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Dict, Optional, Tuple

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

router = APIRouter()

# 统一：所有 JSON 都带 utf-8（PowerShell / 某些客户端更稳）
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

# 情绪闲聊兜底开关（你如果不想 fallback，可把环境变量设 0）
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
    "🥺", "😙", "😗", "😽", "😭", "🥰", "💖", "🖤",
]


def _is_emo_chitchat(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    if len(t) <= 60:
        for m in _EMO_MARKERS:
            if m in t:
                return True
    return False


def _looks_like_garbled_qmarks(s: str) -> bool:
    """
    判断是否“已经被客户端替换成 ?”的典型情况：
    - 含大量 ? 且几乎没有中文
    - 或者完全由 ?、逗号、空格构成
    """
    if not s:
        return False
    # 去掉常见分隔符后看是否只剩 ?
    core = re.sub(r"[,\s]+", "", s)
    if core and set(core) <= {"?"}:
        return True
    q = s.count("?")
    if q >= max(3, int(len(s) * 0.3)):
        # 如果本应是中文关键词，但一个中文都没有，也很可疑
        if not re.search(r"[\u4e00-\u9fff]", s):
            return True
    return False


def _maybe_b64_decode(v: str) -> str:
    """
    支持 keyword_b64 / text_b64：ASCII 永不乱码。
    """
    if not v:
        return ""
    try:
        raw = base64.b64decode(v, validate=True)
        # 尽量 utf-8，失败再 gbk
        try:
            return raw.decode("utf-8")
        except Exception:
            return raw.decode("gbk", errors="replace")
    except Exception:
        return ""


def _decide_keyword(kw_in: str, text: str) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"kw_in": kw_in, "fallback": False, "reason": ""}

    if not kw_in:
        meta["fallback"] = True
        meta["reason"] = "empty keyword"
        return EMO_FALLBACK_KW_CAT, meta

    # 如果 keyword 变成 ??，说明客户端已经丢信息了
    if _looks_like_garbled_qmarks(kw_in) or _looks_like_garbled_qmarks(text):
        meta["fallback"] = True
        meta["reason"] = "client_garbled_to_question_marks"
        # 这里不强行用“猫咪兜底”，优先尝试从 text 判定情绪分类再给较合理 kw
        if EMO_FALLBACK_ENABLED and _is_emo_chitchat(text):
            return EMO_FALLBACK_KW_CAT, meta
        # 你如果真的不想 fallback，可把 EMO_FALLBACK_ENABLED=0，然后它会继续用 kw_in（哪怕是 ??）
        return (kw_in if not EMO_FALLBACK_ENABLED else EMO_FALLBACK_KW_CAT), meta

    # keyword 垃圾（长句）-> 兜底
    if KW_GARBAGE_FALLBACK_ENABLED and len(kw_in) > 30 and ("," not in kw_in and "，" not in kw_in):
        if EMO_FALLBACK_ENABLED and _is_emo_chitchat(text):
            meta["fallback"] = True
            meta["reason"] = "kw_garbage_long_sentence"
            return EMO_FALLBACK_KW_CAT, meta

    # 情绪闲聊：可选兜底
    if EMO_FALLBACK_ENABLED and _is_emo_chitchat(text):
        meta["fallback"] = True
        meta["reason"] = "emo_chitchat_detected"
        return EMO_FALLBACK_KW_CAT, meta

    return kw_in, meta


# -----------------------------
# Dify call + output parse
# -----------------------------
async def _call_dify_anchor(keyword: str, user: str) -> Dict[str, Any]:
    if not DIFY_API_KEY:
        raise RuntimeError("Missing DIFY_API_KEY / DIFY_WORKFLOW_API_KEY")
    if not DIFY_WORKFLOW_ID_ANCHOR:
        # 有些人用 workflow_id 放在 inputs 里跑；你这里如果不是必须，可自行改
        # 我先保持你之前行为：允许为空但仍可跑（Dify 侧如果需要会报错）
        pass

    headers = {"Authorization": f"Bearer {DIFY_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "inputs": {"keyword": keyword},
        "response_mode": "blocking",
        "user": user,
    }
    # 如果你 Dify workflow 需要 workflow_id，可在这里补：
    if DIFY_WORKFLOW_ID_ANCHOR:
        payload["workflow_id"] = DIFY_WORKFLOW_ID_ANCHOR

    async with httpx.AsyncClient(timeout=DIFY_TIMEOUT_SECS) as client:
        r = await client.post(DIFY_WORKFLOW_RUN_URL, headers=headers, json=payload)
        ct = (r.headers.get("content-type") or "").lower()
        print(
            f"[anchor] enabled=1\n[anchor] kw={keyword}\n[anchor] dify_status={r.status_code} ct={ct} url={DIFY_WORKFLOW_RUN_URL}",
            flush=True,
        )
        r.raise_for_status()
        return r.json()


def _extract_outputs(dify: Dict[str, Any]) -> Dict[str, Any]:
    """
    兼容 Dify workflow run 常见结构：
    - outputs 在 data.outputs
    - 或者 result.outputs / outputs
    """
    if not isinstance(dify, dict):
        return {}
    if isinstance(dify.get("data"), dict) and isinstance(dify["data"].get("outputs"), dict):
        return dify["data"]["outputs"]
    if isinstance(dify.get("result"), dict) and isinstance(dify["result"].get("outputs"), dict):
        return dify["result"]["outputs"]
    if isinstance(dify.get("outputs"), dict):
        return dify["outputs"]
    return {}


def _compact_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def _trim_to_range(s: str, min_len: int, max_len: int) -> str:
    s = _compact_text(s)
    if len(s) <= max_len:
        return s
    # 简单裁剪：优先在句号/换行附近截断
    cut = s[:max_len]
    for sep in ["。\n", "。\r\n", "。", "\n\n", "\n"]:
        idx = cut.rfind(sep)
        if idx >= min_len:
            return cut[: idx + len(sep)].strip()
    return cut.strip()


def _compose_ctx(result_text: str, chat_text: str) -> str:
    """
    你之前的“Dify KB hits: ... + top blocks”输出风格，这里保留。
    优先用 result，再补 chat_text。
    """
    rt = _compact_text(result_text)
    ct = _compact_text(chat_text)
    combined = rt if rt else ct
    return _trim_to_range(combined, CTX_MIN, CTX_MAX)


# -----------------------------
# Routes
# -----------------------------
@router.post("/api/v1/mcp/gateway_ctx")
async def mcp_gateway_ctx(request: Request):
    """
    MCP JSON-RPC endpoint
    """
    # 先读原始 bytes，方便你抓“到底是谁把中文变成 ?”
    raw = await request.body()

    # 打印前 400 bytes 的可视化（不污染终端太多）
    raw_preview = raw[:400]
    # 同时做一个“可读字符串”预览（replace 只是用于日志，不影响解析）
    raw_preview_text = raw_preview.decode("utf-8", errors="replace")

    print(
        f"[mcp] raw_bytes_len={len(raw)} content_type={request.headers.get('content-type')} "
        f"raw_preview_utf8={raw_preview_text!r}",
        flush=True,
    )

    try:
        # json.loads 支持 bytes（按 utf-8-sig 解码）
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
            "serverInfo": {"name": "gateway_ctx", "version": "2.2"},
            "capabilities": {"tools": {}},
        }
        return None if is_notification else _jsonrpc_result(_id, result)

    if method == "tools/list":
        tools = [
            {
                "name": "gateway_ctx",
                "description": "Unified gateway context builder: Anchor RAG snippet (compact). Returns MCP content[] + debug data.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "search keywords, e.g. '撒娇,哥哥'"},
                        "text": {"type": "string", "description": "optional raw user message for better matching"},
                        "user": {"type": "string", "description": "optional user/session id"},
                        # 关键：ASCII 通道，避免客户端把中文变成 ?
                        "keyword_b64": {"type": "string", "description": "base64(keyword utf-8), safer than keyword when client encoding is broken"},
                        "text_b64": {"type": "string", "description": "base64(text utf-8), safer than text when client encoding is broken"},
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

        # debug: print what client actually sent
        print(
            f"[mcp] tools/call name={name} arguments_type={type(arguments).__name__} "
            f"keyword_repr={repr((arguments or {}).get('keyword'))} text_repr={repr((arguments or {}).get('text'))}",
            flush=True,
        )

        if name != "gateway_ctx":
            return None if is_notification else _jsonrpc_error(_id, -32601, f"Unknown tool: {name}")

        # 先取普通字段
        kw_in = str(arguments.get("keyword", "")).strip()
        text = str(arguments.get("text", "")).strip()
        user = str(arguments.get("user", "mcp")).strip() or "mcp"

        # 如果客户端把中文替换成 ?，这里允许用 b64 传真实值（不影响你本来的 keyword 方案）
        kw_b64 = str(arguments.get("keyword_b64", "")).strip()
        tx_b64 = str(arguments.get("text_b64", "")).strip()
        if kw_b64:
            decoded = _maybe_b64_decode(kw_b64).strip()
            if decoded:
                kw_in = decoded
        if tx_b64:
            decoded = _maybe_b64_decode(tx_b64).strip()
            if decoded:
                text = decoded

        if not kw_in:
            res = {"ctx": "", "result": "", "chat_text": "", "meta": {"reason": "empty keyword"}}
            mcp_result = _mcp_wrap_text(res, "", False)
            return None if is_notification else _jsonrpc_result(_id, mcp_result)

        kw_used, kw_meta = _decide_keyword(kw_in, text)

        try:
            dify = await _call_dify_anchor(keyword=kw_used, user=user)
            outs = _extract_outputs(dify)

            # 给你一个非常明确的“这次注入的 ctx”输出
            ctx = _compose_ctx(outs.get("result", ""), outs.get("chat_text", ""))

            # 日志：你想看的 print
            print(
                f"[anchor] snip_len={len(ctx)}\n[anchor] snip_preview={ctx[:200]}",
                flush=True,
            )

            res = {
                "ctx": ctx,
                "result": outs.get("result", ""),
                "chat_text": outs.get("chat_text", ""),
                "meta": {**kw_meta, "kw_used": kw_used, "snip_len": len(ctx)},
            }
            mcp_result = _mcp_wrap_text(res, ctx, False)
        except Exception as e:
            res = {"ctx": "", "result": "", "chat_text": "", "meta": {**kw_meta, "kw_used": kw_used, "error": str(e)}}
            mcp_result = _mcp_wrap_text(res, "", True)

        return None if is_notification else _jsonrpc_result(_id, mcp_result)

    return None if is_notification else _jsonrpc_error(_id, -32601, f"Method not found: {method}")
