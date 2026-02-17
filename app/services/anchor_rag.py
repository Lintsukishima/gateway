# app/services/anchor_rag.py
from __future__ import annotations

import os
import re
import requests
from typing import List, Dict, Any, Optional

NOTION_API = "https://api.notion.com/v1"


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _extract_keywords(text: str, *, max_kw: int = 6) -> List[str]:
    """
    轻量关键词提取（不做中文分词，避免引入依赖）：
    - 英文/数字：按单词
    - CJK：按连续片段，并取 2-4 长度的子串作为候选（很粗糙但够用）
    """
    t = (text or "").strip()
    if not t:
        return []

    # 过滤常见噪音符号
    t = re.sub(r"[^\w\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+", " ", t)

    # 英文/数字 token
    latin = [w.lower() for w in re.findall(r"[A-Za-z0-9_]{2,}", t)]

    # CJK 连续片段
    cjk_chunks = re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]{2,}", t)

    cjk_tokens: List[str] = []
    for ch in cjk_chunks:
        ch = ch.strip()
        # 取 2-4 的子串，优先靠前（足够做 contains）
        # 例： "Telegram Bot" 这种不会在这里；中文会产生些片段
        for L in (4, 3, 2):
            if len(ch) >= L:
                cjk_tokens.append(ch[:L])
                break

    # 合并去重，保留顺序
    seen = set()
    out: List[str] = []
    for w in latin + cjk_tokens:
        if w and w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= max_kw:
            break
    return out


def _notion_headers() -> Dict[str, str]:
    token = _env("NOTION_TOKEN")
    version = _env("NOTION_VERSION", "2022-06-28")
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": version,
        "Content-Type": "application/json",
    }


def _build_filter(keywords: List[str], allow_context: Optional[str]) -> Dict[str, Any]:
    """
    Notion Database Query filter:
    - Anchor Text contains keyword（rich_text contains）
    - Signals/Category contains keyword（multi_select contains）
    - 可选：Allow Context contains allow_context（multi_select contains）
    """
    ors: List[Dict[str, Any]] = []

    for kw in keywords:
        ors.append({"property": "Anchor Text", "rich_text": {"contains": kw}})
        # 你的字段是多选标签：Signals / Category / Allow Context
        ors.append({"property": "Signals", "multi_select": {"contains": kw}})
        ors.append({"property": "Category", "multi_select": {"contains": kw}})

    base = {"or": ors} if ors else {}

    if allow_context:
        ctx_filter = {"property": "Allow Context", "multi_select": {"contains": allow_context}}
        if base:
            return {"and": [ctx_filter, base]}
        return ctx_filter

    return base or {}  # empty allowed


def query_anchor_snippets(
    user_text: str,
    *,
    allow_context: Optional[str] = None,
    k: int = 3,
    max_chars: int = 220,
) -> List[str]:
    """
    返回 k 条“哥哥原句片段”，用于 style exemplars。
    """
    token = _env("NOTION_TOKEN")
    db_id = _env("NOTION_ANCHOR_DATA_SOURCE_ID") or _env("NOTION_ANCHOR_DB_ID")
    if not token or not db_id:
        return []

    try:
        k = int(_env("ANCHOR_RAG_K", str(k)))
    except Exception:
        pass
    try:
        max_chars = int(_env("ANCHOR_RAG_MAX_CHARS", str(max_chars)))
    except Exception:
        pass

    keywords = _extract_keywords(user_text, max_kw=6)

    body: Dict[str, Any] = {
        "page_size": min(20, max(10, k * 5)),  # 多取一点，后面再挑
        "sorts": [
            {"property": "Score", "direction": "descending"},
            # 如果你的 Source Time ISO 是 date 类型，也可以加排序；不是也没关系
            # {"property": "Source Time ISO", "direction": "descending"},
        ],
    }

    flt = _build_filter(keywords, allow_context)
    if flt:
        body["filter"] = flt

    url = f"{NOTION_API}/data_sources/{db_id}/query"
    r = requests.post(url, headers=_notion_headers(), json=body, timeout=25)
    r.raise_for_status()
    data = r.json()

    results = data.get("results") or []
    snippets: List[str] = []

    for page in results:
        props = page.get("properties") or {}
        # Anchor Text 是 rich_text
        rich = props.get("Anchor Text", {}).get("rich_text") or []
        text = "".join([x.get("plain_text", "") for x in rich]).strip()
        if not text:
            continue
        # 只取一小段，保证“像原句”
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
        "- 不要提到检索/数据库/Notion/记录/工具。",
        "- 你必须在本次回复中自然出现至少一个锚点中的口癖/节奏特征（例如“哦呀”“呼呼”等），但不要生硬堆砌。",
        "- 用锚点的说话节奏与称呼方式来回复；内容要贴合用户问题。",
    ]
    return "\n".join(lines)
