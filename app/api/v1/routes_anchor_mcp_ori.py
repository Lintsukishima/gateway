# app/api/v1/routes_anchor_mcp.py
from __future__ import annotations

import os
from fastapi import APIRouter, Request

from app.services.anchor_rag_clean import (
    query_anchor_snippets_clean,
    format_anchor_block,
)

router = APIRouter()


@router.post("/mcp/anchor_rag")
async def mcp_anchor_rag(req: Request):
    """
    MCP Tool endpoint for Rikkahub.
    Returns a compact anchor_block for style imitation.
    """
    payload = await req.json()

    # 兼容各种字段名
    q = (
        payload.get("keyword")
        or payload.get("q")
        or payload.get("query")
        or payload.get("text")
        or ""
    )
    q = str(q).strip()

    # 可选参数
    k = int(payload.get("k") or os.getenv("ANCHOR_RAG_K", "3"))
    max_chars = int(payload.get("max_chars") or os.getenv("ANCHOR_RAG_MAX_CHARS", "180"))
    allow_context = payload.get("allow_context")
    layer = payload.get("layer")  # "Style" / "Core" / None
    role = payload.get("role") or "assistant"
    score_min = payload.get("score_min")
    try:
        score_min = float(score_min) if score_min is not None else None
    except Exception:
        score_min = None

    snippets = []
    anchor_block = ""

    if q:
        snippets = query_anchor_snippets_clean(
            q,
            k=k,
            max_chars=max_chars,
            allow_context=allow_context,
            score_min=score_min,
            layer=layer,
            role=role,
        )
        anchor_block = format_anchor_block(snippets)

    # MCP 常见：返回一个 text 字段即可
    # 同时也把 snippets 数组带上，方便你调试（不想要也可以删）
    return {
        "text": anchor_block,
        "snippets": snippets,
        "meta": {"k": k, "max_chars": max_chars, "layer": layer, "allow_context": allow_context},
    }
