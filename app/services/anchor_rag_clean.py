# app/services/anchor_rag_clean.py
from __future__ import annotations

import os
import re
import requests
from typing import Any, Dict, List, Optional

NOTION_API = "https://api.notion.com/v1"


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()


def _notion_headers() -> Dict[str, str]:
    token = _env("NOTION_TOKEN")
    version = _env("NOTION_VERSION", "2025-09-03")
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": version,
        "Content-Type": "application/json",
    }


def _extract_keywords(text: str, *, max_kw: int = 6) -> List[str]:
    """
    极简关键词提取：尽量挑“能命中 Anchor Text 的词”
    - 英文/数字：按 token
    - 中文/日文：取连续片段的前 2-4 字
    """
    t = (text or "").strip()
    if not t:
        return []

    t = re.sub(r"[^\w\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+", " ", t)

    latin = [w.lower() for w in re.findall(r"[A-Za-z0-9_]{2,}", t)]
    cjk_chunks = re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]{2,}", t)

    cjk_tokens: List[str] = []
    for ch in cjk_chunks:
        ch = ch.strip()
        for L in (4, 3, 2):
            if len(ch) >= L:
                cjk_tokens.append(ch[:L])
                break

    seen = set()
    out: List[str] = []
    for w in latin + cjk_tokens:
        if w and w not in seen:
            out.append(w)
            seen.add(w)
        if len(out) >= max_kw:
            break
    return out


def _get_rich_text(props: Dict[str, Any], name: str) -> str:
    """
    Notion properties[name].rich_text -> plain_text 拼接
    """
    p = props.get(name) or {}
    rt = p.get("rich_text") or []
    return "".join([x.get("plain_text", "") for x in rt]).strip()


def query_anchor_snippets_clean(
    query_text: str,
    *,
    k: int = 3,
    max_chars: int = 180,
    allow_context: Optional[str] = None,
    score_min: Optional[float] = None,
    layer: Optional[str] = None,  # "Style" / "Core"
    role: str = "assistant",
) -> List[str]:
    """
    返回 k 条短 snippet（只保留可模仿语气的原句片段）
    """
    token = _env("NOTION_TOKEN")
    ds_id = _env("NOTION_ANCHOR_DATA_SOURCE_ID") or _env("NOTION_ANCHOR_DB_ID")
    if not token or not ds_id:
        return []

    try:
        k = int(_env("ANCHOR_RAG_K", str(k)))
    except Exception:
        pass
    try:
        max_chars = int(_env("ANCHOR_RAG_MAX_CHARS", str(max_chars)))
    except Exception:
        pass

    keywords = _extract_keywords(query_text, max_kw=6)

    # ------- 先尝试带 filter（更相关） -------
    ors: List[Dict[str, Any]] = []
    for kw in keywords:
        # 只围绕“Anchor Text / Signals / Category”做 contains
        ors.append({"property": "Anchor Text", "rich_text": {"contains": kw}})
        ors.append({"property": "Signals", "multi_select": {"contains": kw}})
        ors.append({"property": "Category", "multi_select": {"contains": kw}})

    ands: List[Dict[str, Any]] = []
    if allow_context:
        ands.append({"property": "Allow Context", "multi_select": {"contains": allow_context}})
    if layer:
        ands.append({"property": "Layer", "select": {"equals": layer}})
    if role:
        ands.append({"property": "Role", "select": {"equals": role}})
    if score_min is not None:
        ands.append({"property": "Score", "number": {"greater_than_or_equal_to": float(score_min)}})

    flt: Optional[Dict[str, Any]] = None
    if ors and ands:
        flt = {"and": ands + [{"or": ors}]}
    elif ors:
        flt = {"or": ors}
    elif ands:
        flt = {"and": ands}
    else:
        flt = None

    body: Dict[str, Any] = {
        "page_size": min(20, max(10, k * 6)),
        "sorts": [{"property": "Score", "direction": "descending"}],
    }
    if flt:
        body["filter"] = flt

    url = f"{NOTION_API}/data_sources/{ds_id}/query"
    r = requests.post(url, headers=_notion_headers(), json=body, timeout=25)

    # ------- 如果 filter 把自己筛死 or 请求失败：回退“直接取 top” -------
    if not r.ok:
        # 回退（不带 filter）
        body2: Dict[str, Any] = {
            "page_size": min(20, max(10, k * 6)),
            "sorts": [{"property": "Score", "direction": "descending"}],
        }
        # 保留强过滤（Layer/Role/Score），不保留关键词 or
        ands2: List[Dict[str, Any]] = []
        if allow_context:
            ands2.append({"property": "Allow Context", "multi_select": {"contains": allow_context}})
        if layer:
            ands2.append({"property": "Layer", "select": {"equals": layer}})
        if role:
            ands2.append({"property": "Role", "select": {"equals": role}})
        if score_min is not None:
            ands2.append({"property": "Score", "number": {"greater_than_or_equal_to": float(score_min)}})
        if ands2:
            body2["filter"] = {"and": ands2}

        r = requests.post(url, headers=_notion_headers(), json=body2, timeout=25)
        r.raise_for_status()

    r.raise_for_status()
    data = r.json()
    results = data.get("results") or []

    snippets: List[str] = []
    for page in results:
        props = page.get("properties") or {}
        text = _get_rich_text(props, "Anchor Text")
        if not text:
            continue
        text = text.replace("\n", " ").strip()
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        snippets.append(text)
        if len(snippets) >= k:
            break

    return snippets


def format_anchor_block(snippets: List[str]) -> str:
    if not snippets:
        return ""
    lines = ["[ANCHOR_SNIPPETS | use as memory + style exemplars]"]
    for i, s in enumerate(snippets, 1):
        lines.append(f'{i}) “{s}”')
    lines += [
        "Rules:",
        "- 不要提到检索/数据库/Notion/MCP/记录/工具。",
        "- 你必须在本次回复中自然体现至少 1 个锚点风格特征（例如“哦呀/呼呼/小猫咪/括号动作/舞台导演感”），但不要生硬堆砌。",
        "- 可以模仿用词、节奏、称呼与动作描写；不要长段逐字复读。",
        "- 若话题偏严肃：保留口癖与节奏即可，可适当降低亲密强度。",
    ]
    return "\n".join(lines)
