from __future__ import annotations

import os
import time
import re
from typing import Any, Dict, Optional, Tuple

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

router = APIRouter()
JSON_UTF8 = "application/json; charset=utf-8"

DIFY_BASE_URL = os.getenv("DIFY_BASE_URL", "https://api.dify.ai").strip()
DIFY_API_KEY = (os.getenv("DIFY_API_KEY") or os.getenv("DIFY_WORKFLOW_API_KEY") or "").strip()
DIFY_WORKFLOW_RUN_URL = os.getenv("DIFY_WORKFLOW_RUN_URL", "https://api.dify.ai/v1/workflows/run").strip()
DIFY_WORKFLOW_ID_ANCHOR = os.getenv("DIFY_WORKFLOW_ID_ANCHOR", "").strip()

# ✅ 默认别用 2025-11-25（你之前就被这个坑过）
DEFAULT_MCP_PROTOCOL_VERSION = os.getenv("MCP_PROTOCOL_VERSION", "2025-06-18").strip()

SUPPORTED_VERSIONS = {
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
    "2024-10-07",
}

# 原句截断长度（你可以在 .env 调）
CTX_MAX = int(os.getenv("ANCHOR_SNIP_MAX", "400"))
GATEWAY_CTX_DEBUG = os.getenv("GATEWAY_CTX_DEBUG", "0").strip().lower() in ("1", "true", "yes")
DIFY_TIMEOUT_SECS = float(os.getenv("DIFY_TIMEOUT_SECS", "30"))

# 关键词乱码修复开关：当 keyword 里大部分都是 '?' 时，优先用 text 重新推导中文关键词（而不是直接走撒娇/猫咪兜底）
GARBLED_KW_REPAIR_ENABLED = os.getenv("GARBLED_KW_REPAIR_ENABLED", "1").strip().lower() in ("1", "true", "yes")

# 用于判断 '?' 乱码：只要非空且 '?' 占比高，就视为乱码 keyword
_QMARK = "?"
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def _looks_garbled_keyword(keyword: str) -> bool:
    kw = (keyword or "").strip()
    if not kw:
        return False
    q = kw.count(_QMARK)
    # 关键：客户端把中文变成 '?' 时，往往会出现 '??' 或 '??,???'
    if q == 0:
        return False
    # 忽略分隔符后的长度
    total = sum(1 for ch in kw if ch not in " ,，;；|/\t\r\n")
    if total <= 0:
        return True
    return (q / total) >= 0.4


# 从 text 推导“中文关键词检索”用的 keyword（仅在 keyword 缺失/乱码时使用）
_STOP_TOKENS = set([
    "哥哥", "哥", "类", "神代", "喵", "猫咪", "小猫咪", "宝宝", "亲", "抱", "mua", "啾", "嘿嘿",
])


def _derive_kw_from_text(text: str, k: int = 2) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # 1) 先抓中文连续片段
    seqs = _CJK_RE.findall(t)
    cands: list[str] = []
    for s in seqs:
        s = s.strip()
        if not s:
            continue
        # 去掉纯情绪/称呼词
        if s in _STOP_TOKENS:
            continue
        # 过滤太短/太长
        if len(s) < 2:
            continue
        # 常见口语词也别当关键词
        if s in ("就是", "然后", "那个", "这个", "怎么", "为什么", "可以", "不要", "不是"):
            continue
        if s not in cands:
            cands.append(s)
        if len(cands) >= k:
            break
    if not cands:
        return ""
    return ",".join(cands)


# 轻量缓存（同 keyword 短时间重复调用就直接复用）
CACHE_TTL_SECS = float(os.getenv("GATEWAY_CTX_CACHE_TTL", "20"))
MAX_CACHE_SIZE = int(os.getenv("GATEWAY_CTX_CACHE_MAX", "256"))
_cache: Dict[str, Tuple[float, str, Dict[str, Any]]] = {}

_EMO_MARKERS = [
    "哥哥", "类", "喵", "猫咪", "小猫咪", "宝宝", "亲", "抱", "mua", "啾", "嘿嘿",
    "🥺", "😙", "😗", "😽", "😭", "🥰", "💖", "🖤",
]


def _jsonrpc_error(_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": _id, "error": err}


def _jsonrpc_result(_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": _id, "result": result}


def _negotiate_protocol_version(request: Request, params: Dict[str, Any]) -> str:
    pv = str((params or {}).get("protocolVersion") or "").strip()
    if pv and pv in SUPPORTED_VERSIONS:
        return pv

    hv = (request.headers.get("MCP-Protocol-Version") or "").strip()
    if hv and hv in SUPPORTED_VERSIONS:
        return hv

    return DEFAULT_MCP_PROTOCOL_VERSION if DEFAULT_MCP_PROTOCOL_VERSION in SUPPORTED_VERSIONS else "2025-06-18"


def _mcp_wrap_text(res_obj: Dict[str, Any], text_out: str, is_error: bool) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": text_out or ""}], "isError": bool(is_error), "data": res_obj}


def _is_emo_chitchat(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(m in t for m in _EMO_MARKERS)


def _truncate_ctx(text: str) -> str:
    t = (text or "").strip().replace("\r", "")
    if not t:
        return ""
    if len(t) <= CTX_MAX:
        return t
    return t[:CTX_MAX].rstrip() + "…"


def _normalize_kw(keyword: str) -> str:
    """Normalize keyword string to stabilize caching."""
    kw = (keyword or "").strip()
    if not kw:
        return ""
    # unify separators
    kw = kw.replace("，", ",").replace(";", ",").replace("；", ",")
    parts = [p.strip() for p in kw.split(",") if p.strip()]
    # de-dup while preserving order
    seen = set()
    uniq = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return ",".join(uniq)


async def _call_dify_anchor(keyword: str, user: str = "mcp") -> Dict[str, Any]:
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


async def _handle_jsonrpc(request: Request, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    _id = msg.get("id", None)
    method = msg.get("method", "")
    params = (msg.get("params", {}) or {}) if isinstance(msg, dict) else {}
    is_notification = isinstance(msg, dict) and ("id" not in msg)

    pv = _negotiate_protocol_version(request, params)
    request.state.mcp_pv = pv

    if method == "initialize":
        result = {
            "protocolVersion": pv,
            "serverInfo": {"name": "gateway_ctx", "version": "2.3"},
            "capabilities": {"tools": {}},
        }
        return None if is_notification else _jsonrpc_result(_id, result)

    if method == "tools/list":
        tools = [{
            "name": "gateway_ctx",
            "description": "Unified gateway context builder: keyword + Anchor RAG snippet. Returns MCP content[].text + debug data.",
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

    # 1) 先确定 primary keyword（优先使用上游抽取结果；仅在缺失/乱码时，才用 text 推导中文关键词）
    primary_keyword_raw = keyword

    # 1.1 缺失 / 乱码 -> 用 text 推导中文关键词（尽量保持“中文关键词检索”，不要直接掉到撒娇/猫咪兜底）
    if (not keyword) or (GARBLED_KW_REPAIR_ENABLED and _looks_garbled_keyword(keyword)):
        derived = _derive_kw_from_text(text)
        if derived:
            if GARBLED_KW_REPAIR_ENABLED and _looks_garbled_keyword(primary_keyword_raw):
                print(f"[gateway_ctx] repair_garbled_kw from={primary_keyword_raw!r} to={derived!r}")
            keyword = derived
        else:
            keyword = ""

    # 1.2 可选：垃圾 keyword -> 也尝试用 text 推导
    try:
        if keyword and "_is_garbage_kw" in globals() and _is_garbage_kw(keyword):
            derived = _derive_kw_from_text(text)
            keyword = derived or ""
    except Exception:
        pass

    # 1.3 如果最终仍然没有 keyword（例如 text 也抽不到），才用情绪兜底 keyword
    if not keyword:
        keyword = "哥哥,小猫咪" if _is_emo_chitchat(text) else "哥哥,撒娇"

    # 2) 再生成 cache_key（必须在 keyword 最终确定之后）
    keyword = _normalize_kw(keyword)
    primary_keyword = keyword
    cache_key = f"{user}||{primary_keyword}"
    t0 = time.perf_counter()
    if GATEWAY_CTX_DEBUG:
        print(f"[gateway_ctx] pid={os.getpid()} cache_size={len(_cache)} kw={keyword!r}")
        print(f"[gateway_ctx] user={user!r} cache_key={cache_key!r} ttl={CACHE_TTL_SECS}")

    # cache hit?
    now = time.time()
    hit = _cache.get(cache_key)
    if hit and (now - hit[0] <= CACHE_TTL_SECS):
        ctx, res_obj = hit[1], hit[2]
        dt = (time.perf_counter() - t0) * 1000
        print(f"[gateway_ctx] cache_hit kw={keyword!r} ms={dt:.1f} len={len(ctx)}")
        return None if is_notification else _jsonrpc_result(_id, _mcp_wrap_text(res_obj, ctx, is_error=False))

    # cache miss -> call dify
    try:
        t1 = time.perf_counter()
        dify = await _call_dify_anchor(keyword=keyword, user=user)
        ms_dify = (time.perf_counter() - t1) * 1000

        outs = _extract_outputs(dify)
        picked = (outs.get("result") or "").strip() or (outs.get("chat_text") or "").strip()
        ctx = _truncate_ctx(picked)

        used_keyword = primary_keyword
        ms_dify_primary = ms_dify
        ms_dify_used = ms_dify

        # 3.1 如果 primary keyword 没命中（ctx 为空），再按“撒娇程度”路由到亲密兜底 keyword，并重试一次
        if not ctx:
            fallback_keyword = _normalize_kw("哥哥,小猫咪" if _is_emo_chitchat(text) else "哥哥,撒娇")
            # 避免 primary 本来就是兜底 keyword 时重复调用
            if fallback_keyword and fallback_keyword != primary_keyword:
                if GATEWAY_CTX_DEBUG:
                    print(f"[gateway_ctx] primary_miss kw={primary_keyword!r} -> fallback={fallback_keyword!r}")
                t2 = time.perf_counter()
                dify2 = await _call_dify_anchor(keyword=fallback_keyword, user=user)
                ms_dify2 = (time.perf_counter() - t2) * 1000
                outs2 = _extract_outputs(dify2)
                picked2 = (outs2.get("result") or "").strip() or (outs2.get("chat_text") or "").strip()
                ctx2 = _truncate_ctx(picked2)
                if ctx2:
                    used_keyword = fallback_keyword
                    ctx = ctx2
                    outs = outs2
                    ms_dify_used = ms_dify2

        res_obj = {
            "keyword": used_keyword,
            "keyword_primary": primary_keyword,
            "ctx": ctx,
            "raw": outs,
            "ms_dify_primary": round(ms_dify_primary, 1),
            "ms_dify_used": round(ms_dify_used, 1),
        }

        # ✅ 写入缓存时用最新 now（更符合 TTL 语义）
        _cache[cache_key] = (time.time(), ctx, res_obj)
        # simple eviction (oldest-first) to cap memory
        if len(_cache) > MAX_CACHE_SIZE:
            oldest_key = min(_cache.items(), key=lambda kv: kv[1][0])[0]
            _cache.pop(oldest_key, None)

        ms_all = (time.perf_counter() - t0) * 1000
        print(f"[gateway_ctx] miss kw={primary_keyword!r} used={res_obj.get('keyword')!r} ms_all={ms_all:.1f} ms_dify={ms_dify:.1f} len={len(ctx)}")
        return None if is_notification else _jsonrpc_result(_id, _mcp_wrap_text(res_obj, ctx, is_error=False))

    except Exception as e:
        ms_all = (time.perf_counter() - t0) * 1000
        print(f"[gateway_ctx] ERROR kw={keyword!r} ms_all={ms_all:.1f} err={e}")
        res_obj = {"keyword": keyword, "error": str(e)}
        return None if is_notification else _jsonrpc_result(_id, _mcp_wrap_text(res_obj, str(e), is_error=True))


@router.api_route("/gateway_ctx", methods=["GET", "POST", "OPTIONS"])
async def gateway_ctx_mcp(request: Request):
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
        return JSONResponse(_jsonrpc_error(None, -32700, "Parse error"), headers={"MCP-Protocol-Version": default_pv}, media_type=JSON_UTF8)

    # batch?
    if isinstance(body, list):
        results = []
        for msg in body:
            if isinstance(msg, dict):
                r = await _handle_jsonrpc(request, msg)
                if r is not None:
                    results.append(r)
        pv = getattr(request.state, "mcp_pv", default_pv)
        return JSONResponse(results, headers={"MCP-Protocol-Version": pv}, media_type=JSON_UTF8)

    resp = await _handle_jsonrpc(request, body if isinstance(body, dict) else {})
    pv = getattr(request.state, "mcp_pv", default_pv)
    if resp is None:
        return Response(status_code=204, headers={"MCP-Protocol-Version": pv})
    return JSONResponse(resp, headers={"MCP-Protocol-Version": pv}, media_type=JSON_UTF8)
