from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from app.services.reranker import build_reranked_context_text, rerank_evidences

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
RERANK_TOPK = int(os.getenv("RERANK_TOPK", "5"))

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


def _extract_outputs(dify_resp: Dict[str, Any]) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {}
    data_block: Dict[str, Any] = {}
    if isinstance(dify_resp, dict):
        if isinstance(dify_resp.get("data"), dict) and isinstance(dify_resp["data"].get("outputs"), dict):
            data_block = dify_resp["data"]
            outputs = dify_resp["data"]["outputs"]
        elif isinstance(dify_resp.get("outputs"), dict):
            outputs = dify_resp["outputs"]

    extracted: Dict[str, Any] = {}
    evidences: List[Dict[str, Any]] = []
    if isinstance(outputs, dict):
        for k, v in outputs.items():
            if isinstance(v, str):
                extracted[k] = v
            elif isinstance(v, list) and k in {"evidences", "evidence", "documents"}:
                evidences = [item for item in v if isinstance(item, dict)]
            elif v is not None:
                extracted[k] = str(v)

    if not evidences and isinstance(data_block.get("retriever_resources"), list):
        evidences = [item for item in data_block.get("retriever_resources") if isinstance(item, dict)]

    extracted.setdefault("result", "")
    extracted.setdefault("chat_text", "")
    extracted["evidences"] = evidences
    return extracted


def _infer_source_by_key(output_key: str) -> str:
    key = (output_key or "").lower()
    if "s60" in key:
        return "summary_s60"
    if "s4" in key:
        return "summary_s4"
    if key in {"result", "kb", "knowledge", "dify_kb"}:
        return "dify_kb"
    return "anchor_fallback"


def _build_evidence(keyword: str, outs: Dict[str, Any]) -> List[Dict[str, Any]]:
    now = int(time.time())
    hit_keywords = [p.strip() for p in str(keyword or "").split(",") if p.strip()]

    ordered_keys = ["result", "chat_text", "summary_s4", "summary_s60"]
    candidates: List[Tuple[str, str]] = []
    for key in ordered_keys:
        candidates.append((key, (outs.get(key) or "").strip()))
    for key, value in outs.items():
        if key not in ordered_keys and isinstance(value, str):
            candidates.append((key, (value or "").strip()))

    evidence: List[Dict[str, Any]] = []
    for idx, (key, text) in enumerate(candidates, start=1):
        if not text:
            continue
        evidence.append({
            "id": f"ev_{idx}",
            "text": text,
            "source": _infer_source_by_key(key),
            "type": "context",
            "score_vec": None,
            "score_key": None,
            "score_final": None,
            "hit_keywords": hit_keywords,
            "created_at": now,
            "meta": {"output_key": key},
        })

    if evidence:
        return evidence

    return [{
        "id": "ev_1",
        "text": "",
        "source": "anchor_fallback",
        "type": "context",
        "score_vec": None,
        "score_key": None,
        "score_final": None,
        "hit_keywords": hit_keywords,
        "created_at": now,
        "meta": {"reason": "empty_outputs"},
    }]


def _render_ctx_from_evidence(evidence: List[Dict[str, Any]]) -> str:
    text_out = "\n\n".join(str(item.get("text") or "").strip() for item in evidence if str(item.get("text") or "").strip())
    return _truncate_ctx(text_out)


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

    # 1) 先修 keyword（兜底 / 垃圾 kw）
    if not keyword:
        keyword = "猫咪,哥哥" if _is_emo_chitchat(text) else "撒娇,哥哥"

    # 如果你有 _is_garbage_kw，就在这里也加一层（可选）
    try:
        if "_is_garbage_kw" in globals() and _is_garbage_kw(keyword):
            keyword = "猫咪,哥哥" if _is_emo_chitchat(text) else "撒娇,哥哥"
    except Exception:
        pass

    # 2) 再生成 cache_key（必须在 keyword 最终确定之后）
    keyword = _normalize_kw(keyword)
    cache_key = f"{user}||{keyword}"

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
        upstream_evidences = outs.get("evidences") or []
        evidence = upstream_evidences if upstream_evidences else _build_evidence(keyword=keyword, outs=outs)

        # 即使 Dify 已启用混合检索与重排，网关仍保留轻量二次重排（融合 S4/S60 与多源统一规则）。
        reranked_evidence = rerank_evidences(evidence, top_k=RERANK_TOPK)
        reranked_ctx = build_reranked_context_text(reranked_evidence)
        ctx = _truncate_ctx(reranked_ctx) if reranked_ctx else _render_ctx_from_evidence(reranked_evidence)

        res_obj = {
            "keyword": keyword,
            "ctx": ctx,
            "evidence": reranked_evidence,
            "raw": outs,
            "ms_dify": round(ms_dify, 1),
        }

        # ✅ 写入缓存时用最新 now（更符合 TTL 语义）
        _cache[cache_key] = (time.time(), ctx, res_obj)
        # simple eviction (oldest-first) to cap memory
        if len(_cache) > MAX_CACHE_SIZE:
            oldest_key = min(_cache.items(), key=lambda kv: kv[1][0])[0]
            _cache.pop(oldest_key, None)

        ms_all = (time.perf_counter() - t0) * 1000
        print(f"[gateway_ctx] miss kw={keyword!r} ms_all={ms_all:.1f} ms_dify={ms_dify:.1f} len={len(ctx)}")
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
