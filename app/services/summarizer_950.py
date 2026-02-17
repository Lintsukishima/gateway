import json
from datetime import datetime
from sqlalchemy.orm import Session as OrmSession
from app.db.models import Message, SummaryS4, SummaryS60


def _now():
    return datetime.utcnow()

S4_SCHEMA = {
  "goal": "当前在推进什么",
  "state": "进度到哪/现在什么状态",
  "open_loops": ["未完成事项/待决定点"],
  "constraints": ["硬约束：工具/时间/边界/要求"],
  "tone_notes": ["语气与互动基调提示（很短）"]
}

S60_SCHEMA = {
  "relationship_canon": ["关系与互动的稳定事实（短条）"],
  "projects": [{"name": "项目名", "status": "一句话进度", "open_loops": ["..."]}],
  "preferences": ["稳定偏好（短条）"],
  "boundaries": ["边界/禁忌/必须遵守（短条）"],
  "important_events": [{"when": "turn_61-120", "what": "发生了什么（短）"}]
}

def build_transcript(messages: list[Message]) -> str:
    # 尽量稳定格式，后续也方便 debug
    lines = []
    for m in messages:
        role = m.role.upper()
        lines.append(f"{role}#{m.turn_id}: {m.content}")
    return "\n".join(lines)

def call_llm_json(system_prompt: str, user_prompt: str) -> dict:
    """
    你在这里接你自己的模型。
    约定：返回 dict；如果失败就抛异常或返回 {}。
    """
    # MVP：先返回空壳，保证流程跑通
    # 你接入 LLM 后，把这里替换掉即可
    return {}

def _safe_merge(schema: dict, model_out: dict) -> dict:
    # 防止模型漏字段 / 填错类型导致崩
    out = dict(schema)
    if isinstance(model_out, dict):
        for k in out.keys():
            if k in model_out:
                out[k] = model_out[k]
    return out

def run_s4(db: OrmSession, session_id: str, to_turn: int, window: int = 8, model_name: str = "unknown"):
    from_turn = max(1, to_turn - window + 1)
    msgs = (db.query(Message)
              .filter(Message.session_id == session_id, Message.turn_id >= from_turn, Message.turn_id <= to_turn)
              .order_by(Message.turn_id.asc())
              .all())
    transcript = build_transcript(msgs)

    system_prompt = (
        "You are a strict conversation summarizer.\n"
        "Rules:\n"
        "1) Only summarize what is explicitly in the transcript. No inference.\n"
        "2) Output MUST be valid JSON only.\n"
        "3) Follow the given schema keys exactly.\n"
        "4) Keep it short and action-oriented.\n"
    )
    user_prompt = (
        "Summarize the following transcript into S4 schema:\n"
        f"S4_SCHEMA={json.dumps(S4_SCHEMA, ensure_ascii=False)}\n\n"
        f"TRANSCRIPT:\n{transcript}\n"
    )

    model_out = call_llm_json(system_prompt, user_prompt)
    summary = _safe_merge(S4_SCHEMA, model_out)

    row = SummaryS4(
        session_id=session_id,
        from_turn=from_turn,
        to_turn=to_turn,
        summary_json=json.dumps(summary, ensure_ascii=False),
        model=model_name,
        created_at=_now(),
        meta_json=json.dumps({"window": window}, ensure_ascii=False),
    )
    db.add(row)
    db.commit()
    return summary

def run_s60(db: OrmSession, session_id: str, to_turn: int, window: int = 60, model_name: str = "unknown"):
    from_turn = max(1, to_turn - window + 1)
    msgs = (db.query(Message)
              .filter(Message.session_id == session_id, Message.turn_id >= from_turn, Message.turn_id <= to_turn)
              .order_by(Message.turn_id.asc())
              .all())
    transcript = build_transcript(msgs)

    system_prompt = (
        "You are a strict long-horizon conversation summarizer.\n"
        "Rules:\n"
        "1) Only record stable facts/progress explicitly stated. No invention.\n"
        "2) Output MUST be valid JSON only.\n"
        "3) Follow the given schema keys exactly.\n"
        "4) Prefer bullet-like short strings.\n"
    )
    user_prompt = (
        "Summarize the following transcript into S60 schema:\n"
        f"S60_SCHEMA={json.dumps(S60_SCHEMA, ensure_ascii=False)}\n\n"
        f"TRANSCRIPT:\n{transcript}\n"
    )

    model_out = call_llm_json(system_prompt, user_prompt)
    summary = _safe_merge(S60_SCHEMA, model_out)

    row = SummaryS60(
        session_id=session_id,
        from_turn=from_turn,
        to_turn=to_turn,
        summary_json=json.dumps(summary, ensure_ascii=False),
        model=model_name,
        created_at=_now(),
        meta_json=json.dumps({"window": window}, ensure_ascii=False),
    )
    db.add(row)
    db.commit()
    return summary
