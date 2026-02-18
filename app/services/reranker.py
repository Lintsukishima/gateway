from __future__ import annotations

import os
from typing import Any, Dict, List

# 第一版固定混合权重
SCORE_VEC_WEIGHT = 0.7
SCORE_KEY_WEIGHT = 0.3

# 可通过 .env 调整的默认常量
DEFAULT_W_FACT = float(os.getenv("RERANK_W_FACT", "0.2"))
DEFAULT_D_TIME = float(os.getenv("RERANK_DEFAULT_D_TIME", "1.0"))
DEFAULT_B_HITS = float(os.getenv("RERANK_DEFAULT_B_HITS", "1.0"))
DEFAULT_TOPK = int(os.getenv("RERANK_TOPK", "5"))


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def score_final(
    score_vec: Any,
    score_key: Any,
    d_time: Any = DEFAULT_D_TIME,
    b_hits: Any = DEFAULT_B_HITS,
    w_fact: float = DEFAULT_W_FACT,
) -> float:
    """score_mix = score_vec*0.7 + score_key*0.3
    score_final = score_mix * (1 + W_fact * D_time * B_hits)
    """
    vec = _to_float(score_vec, 0.0)
    key = _to_float(score_key, 0.0)
    d_time_value = _to_float(d_time, DEFAULT_D_TIME)
    b_hits_value = _to_float(b_hits, DEFAULT_B_HITS)

    score_mix = vec * SCORE_VEC_WEIGHT + key * SCORE_KEY_WEIGHT
    return score_mix * (1 + w_fact * d_time_value * b_hits_value)


def _pick_evidence_text(evidence: Dict[str, Any]) -> str:
    for key in ("content", "text", "snippet", "chunk", "document", "segment"):
        value = evidence.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    # Dify retriever_resources 常见结构：segment.content
    segment = evidence.get("segment")
    if isinstance(segment, dict):
        value = segment.get("content") or segment.get("text")
        if isinstance(value, str) and value.strip():
            return value.strip()

    metadata = evidence.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("content") or metadata.get("text")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _normalize_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(evidence)

    # 多种字段兜底：vector/hybrid 检索产物字段不一致
    score_vec = item.get("score_vec", item.get("vector_score", item.get("score", 0.0)))
    score_key = item.get("score_key", item.get("keyword_score", item.get("bm25_score", 0.0)))
    d_time = item.get("D_time", item.get("d_time", item.get("time_decay", DEFAULT_D_TIME)))
    b_hits = item.get("B_hits", item.get("b_hits", item.get("hit_boost", DEFAULT_B_HITS)))

    item["score_vec"] = _to_float(score_vec, 0.0)
    item["score_key"] = _to_float(score_key, 0.0)
    item["D_time"] = _to_float(d_time, DEFAULT_D_TIME)
    item["B_hits"] = _to_float(b_hits, DEFAULT_B_HITS)
    item["score_mix"] = item["score_vec"] * SCORE_VEC_WEIGHT + item["score_key"] * SCORE_KEY_WEIGHT
    item["score_final"] = score_final(
        score_vec=item["score_vec"],
        score_key=item["score_key"],
        d_time=item["D_time"],
        b_hits=item["B_hits"],
    )

    return item


def rerank_evidences(evidences: List[Dict[str, Any]], top_k: int = DEFAULT_TOPK) -> List[Dict[str, Any]]:
    ranked = [_normalize_evidence(ev) for ev in (evidences or []) if isinstance(ev, dict)]
    ranked.sort(key=lambda x: x.get("score_final", 0.0), reverse=True)
    return ranked[:top_k] if top_k > 0 else ranked


def build_reranked_context_text(evidences: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for evidence in evidences or []:
        text = _pick_evidence_text(evidence)
        if text:
            lines.append(text)
    return "\n\n".join(lines).strip()
