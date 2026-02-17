import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from sqlalchemy.orm import Session

from app.db.models import Message, SummaryS4, SummaryS60

# ===== 在文件顶部 imports 下面（或任意位置）新增 =====

HELP_SEEKING_HINTS = [
    "借钱", "借我", "转账", "打钱", "资助", "赞助", "给我钱", "求助", "救济",
    "能不能给", "能否给", "帮我出", "帮我付", "你出钱", "帮我转", "给点钱"
]

def _has_help_seeking(transcript: str) -> bool:
    t = transcript or ""
    return any(h in t for h in HELP_SEEKING_HINTS)

def _sanitize_summary(transcript: str, obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    轻量纠偏：当对话里没有明确求助/借钱语句时，禁止 summary 里出现“寻求经济帮助/对方愿意提供帮助”等推断。
    只做保守清理，不做复杂重写，避免引入新幻觉。
    """
    if not isinstance(obj, dict):
        return _default_summary_schema()

    want_help = _has_help_seeking(transcript)

    # 统一字段类型（避免 LLM 给错类型导致 downstream 崩）
    obj.setdefault("goal", "")
    obj.setdefault("state", "")
    obj.setdefault("open_loops", [])
    obj.setdefault("constraints", [])
    obj.setdefault("tone_notes", [])

    if not isinstance(obj["open_loops"], list):
        obj["open_loops"] = [str(obj["open_loops"])]
    if not isinstance(obj["constraints"], list):
        obj["constraints"] = [str(obj["constraints"])]
    if not isinstance(obj["tone_notes"], list):
        obj["tone_notes"] = [str(obj["tone_notes"])]

    # 如果没明确求助，就把“经济帮助/愿意提供帮助”这类推断移除/弱化
    if not want_help:
        bad_goal_phrases = ["寻求经济上的帮助", "寻求经济帮助", "请求经济帮助", "求助对方", "让对方出钱"]
        for p in bad_goal_phrases:
            if p in (obj.get("goal") or ""):
                obj["goal"] = (obj["goal"] or "").replace(p, "").strip("，。;； ")

        bad_state_phrases = ["愿意提供帮助", "表示愿意提供帮助", "同意提供帮助", "已提供帮助", "答应提供帮助"]
        for p in bad_state_phrases:
            if p in (obj.get("state") or ""):
                obj["state"] = (obj["state"] or "").replace(p, "").strip("，。;； ")

        # open_loops 里也清理“需要解决经济困难具体方案”这种强推断措辞
        cleaned_loops = []
        for x in obj["open_loops"]:
            s = str(x)
            s = s.replace("需要解决经济困难的具体方案", "需要明确下一步安排/计划").strip("，。;； ")
            cleaned_loops.append(s)
        obj["open_loops"] = cleaned_loops

    # goal/state 为空时给一个保底（仍然只基于显式内容）
    if not obj["goal"]:
        obj["goal"] = "概括本段对话的显式主题（若无明确目标则写‘闲聊/状态更新’）"
    if not obj["state"]:
        obj["state"] = "概括当前显式进展（若无进展则写‘无明显推进’）"

    return obj


# ========= 基础 =========

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _safe_json_loads(s: Optional[str]) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return {"_raw": s}


# ========= LLM（OpenAI-Compatible）=========


def call_llm_json(
    *,
    system: str,
    user: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.2,
    timeout_s: int = 45,
) -> Dict[str, Any]:
    """调用 OpenAI-compatible 的 /chat/completions，要求返回 JSON。

    - 兼容 OpenRouter、LiteLLM、各种中转。
    - 失败会抛异常，外层会记录 failed。
    """

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # 强制 JSON（多数兼容实现都支持；不支持也没关系，我们后面会兜底解析）
        "response_format": {"type": "json_object"},
    }

    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    if not content:
        raise RuntimeError(f"LLM empty content: {data}")

    # 有些实现会把 JSON 包在 ```json``` 里
    if content.startswith("```"):
        content = content.strip("`")
        content = content.replace("json\n", "", 1).strip()

    try:
        return json.loads(content)
    except Exception as e:
        raise RuntimeError(f"LLM returned non-JSON: {content[:200]}...") from e


def _default_summary_schema() -> Dict[str, Any]:
    return {
        "goal": "当前在推进什么",
        "state": "进度到哪/现在什么状态",
        "open_loops": ["未完成事项/待决定点"],
        "constraints": ["硬约束：工具/时间/边界/要求"],
        "tone_notes": ["语气与互动基调提示（很短）"],
    }


def _render_transcript(msgs: List[Message]) -> str:
    lines = []
    for m in msgs:
        role = m.role
        # Telegram 会带很多表情，保留即可
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def _summarize_with_optional_llm(transcript: str, *, level: str) -> Dict[str, Any]:
    """有配置就用 LLM，没配置就返回占位 schema（保证系统仍然跑通）。"""

    base_url = os.getenv("SUMMARIZER_BASE_URL", "").strip()
    api_key = os.getenv("SUMMARIZER_API_KEY", "").strip()
    model = os.getenv("SUMMARIZER_MODEL", "").strip() or "gpt-4o-mini"

    if not base_url or not api_key:
        return _default_summary_schema()

    system = (
        "你是会话总结器。你必须严格基于对话中的『显式文本』提取信息，禁止推断、禁止脑补、禁止编造。"
        "只输出 JSON 对象，不要输出任何多余文字。"
        "字段必须包含：goal, state, open_loops(list), constraints(list), tone_notes(list)。\n\n"
        "硬规则：\n"
        "1) goal：只能写对话里出现过的明确目标/意图；如果用户只是表达情绪或陈述事实，不要写成‘寻求帮助/求助/想让对方做X’，除非用户明确提出请求。\n"
        "2) state：只能写明确发生过的进展；不要写‘对方愿意提供帮助/已确认…’，除非对话里有清晰承诺或确认。\n"
        "3) open_loops/constraints：只列出对话中明确未解决的问题/明确限制；不要新增‘具体方案’这类你自己设定的任务。\n"
        "4) tone_notes：只写非常短的语气标签，如‘关心/轻松/焦急’。\n"
        "5) 如果信息不足，宁可写‘无明显推进/未提及’，不要编造。"
    )

    user = (
        f"请对下面对话做{level}总结，输出 JSON。\n"
        "注意：不要使用‘寻求经济帮助/请求资助/对方愿意帮忙’等推断性措辞，除非原文明确提出。"
        "\n\n--- 对话 ---\n"
        f"{transcript}\n"
        "--- 结束 ---"
    )

    obj = call_llm_json(
        system=system,
        user=user,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.2,
    )

    return _sanitize_summary(transcript, obj)


# ========= 对外：S4 / S60 =========


def run_s4(
    db: Session,
    *,
    session_id: str,
    to_user_turn: int,
    window_user_turn: int = 4,
    model_name: str = "summarizer_mvp",
) -> Dict[str, Any]:
    """短期总结：按 user_turn 窗口。"""

    start_ut = max(1, to_user_turn - window_user_turn + 1)

    # 取窗口内所有消息（user + assistant），按 turn_id 还原
    msgs = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .filter(Message.user_turn >= start_ut)
        .filter(Message.user_turn <= to_user_turn)
        .order_by(Message.turn_id.asc())
        .all()
    )

    if not msgs:
        return {"skipped": True, "reason": "no messages"}

    from_turn = min(m.turn_id for m in msgs)
    to_turn = max(m.turn_id for m in msgs)

    # 幂等：同一个 to_turn 只写一次
    existed = (
        db.query(SummaryS4)
        .filter(SummaryS4.session_id == session_id)
        .filter(SummaryS4.to_turn == to_turn)
        .first()
    )
    if existed:
        return {"skipped": True, "reason": "exists", "to_turn": to_turn}

    transcript = _render_transcript(msgs)
    summary_obj = _summarize_with_optional_llm(transcript, level="短期")

    row = SummaryS4(
        session_id=session_id,
        from_turn=from_turn,
        to_turn=to_turn,
        summary_json=_safe_json_dumps(summary_obj),
        model=model_name,
        created_at=_now(),
        meta_json=_safe_json_dumps(
            {
                "to_user_turn": to_user_turn,
                "window_user_turn": window_user_turn,
            }
        ),
    )
    db.add(row)
    db.commit()

    return {
        "range": [from_turn, to_turn],
        "summary": summary_obj,
        "created_at": row.created_at.isoformat(),
        "model": model_name,
    }


def run_s60(
    db: Session,
    *,
    session_id: str,
    to_user_turn: int,
    window_user_turn: int = 30,
    model_name: str = "summarizer_mvp",
) -> Dict[str, Any]:
    """长期总结：你现在要的是 30 轮 user 消息。"""

    start_ut = max(1, to_user_turn - window_user_turn + 1)

    msgs = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .filter(Message.user_turn >= start_ut)
        .filter(Message.user_turn <= to_user_turn)
        .order_by(Message.turn_id.asc())
        .all()
    )

    if not msgs:
        return {"skipped": True, "reason": "no messages"}

    from_turn = min(m.turn_id for m in msgs)
    to_turn = max(m.turn_id for m in msgs)

    existed = (
        db.query(SummaryS60)
        .filter(SummaryS60.session_id == session_id)
        .filter(SummaryS60.to_turn == to_turn)
        .first()
    )
    if existed:
        return {"skipped": True, "reason": "exists", "to_turn": to_turn}

    transcript = _render_transcript(msgs)
    summary_obj = _summarize_with_optional_llm(transcript, level="长期")

    row = SummaryS60(
        session_id=session_id,
        from_turn=from_turn,
        to_turn=to_turn,
        summary_json=_safe_json_dumps(summary_obj),
        model=model_name,
        created_at=_now(),
        meta_json=_safe_json_dumps(
            {
                "to_user_turn": to_user_turn,
                "window_user_turn": window_user_turn,
            }
        ),
    )
    db.add(row)
    db.commit()

    return {
        "range": [from_turn, to_turn],
        "summary": summary_obj,
        "created_at": row.created_at.isoformat(),
        "model": model_name,
    }
