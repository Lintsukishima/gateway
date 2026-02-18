from __future__ import annotations

from typing import Any, Dict, List


def build_fact_constraint_block(evidence: List[Dict[str, Any]], grounding_mode: str) -> str:
    if not evidence:
        return ""

    lines: List[str] = []
    for ev in evidence:
        text = str(ev.get("text") or "").strip()
        if not text:
            continue
        ev_type = str(ev.get("type") or "anchor")
        weight = float(ev.get("W_fact") or 1.0)
        lines.append(f"- ({ev_type}, W_fact={weight:.2f}) {text}")

    if not lines:
        return ""

    return (
        "【事实约束（Telegram / RikkaHub 通用）】\n"
        "规则：优先依据以下 evidence 回答事实与进展；summary_s4/summary_s60 的事实权重高于普通 anchor。\n"
        "当 evidence 不足时可以自然回答，但不要伪造具体事实。\n"
        f"grounding_mode={grounding_mode}\n"
        + "\n".join(lines)
        + "\n【End】"
    )
