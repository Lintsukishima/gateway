from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from sqlalchemy.orm import Session as OrmSession

# 你项目里的模型路径可能不同：如果这里报错，把 traceback 发我
from app.db.models import Session, Message
from app.services.summarizer import run_s4, run_s60


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _get_or_create_session(db: OrmSession, session_id: str) -> Session:
    s = db.query(Session).filter(Session.id == session_id).first()
    if s:
        return s
    s = Session(id=session_id)
    # 如果你 Session 表有 created_at/updated_at 等字段，这里可补
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _next_turn_id(db: OrmSession, session_id: str) -> int:
    last = (
        db.query(Message.turn_id)
        .filter(Message.session_id == session_id)
        .order_by(Message.turn_id.desc())
        .first()
    )
    return (last[0] if last else 0) + 1


def _next_user_turn(db: OrmSession, session_id: str) -> int:
    # user_turn 在你的表里：对 user 消息递增；assistant 用同一个值
    last = (
        db.query(Message.user_turn)
        .filter(Message.session_id == session_id)
        .order_by(Message.turn_id.desc())
        .first()
    )
    last_ut = last[0] if last and last[0] is not None else 0
    return last_ut + 1


@dataclass
class ChatOnceResult:
    session_id: str
    user_turn: int
    user_turn_triggered_s4: bool
    user_turn_triggered_s60: bool
    user_message_turn_id: int
    assistant_message_turn_id: int


def chat_once(
    db: OrmSession,
    session_id: str,
    user_text: str,
    assistant_text: str,
    *,
    model_name: str = "unknown",
    s4_every_user_turns: int = 4,
    s60_every_user_turns: int = 30,
    s4_window_user_turns: int = 4,
    s60_window_user_turns: int = 30,
) -> ChatOnceResult:
    """
    写入一轮 user + assistant 消息，并按 user_turn 触发滚动总结：
      - S4：每 4 条用户消息触发一次（window=4 个 user_turn）
      - S60：每 30 条用户消息触发一次（window=30 个 user_turn）

    assistant 的 user_turn 与当轮 user 相同，不递增。
    """

    session = _get_or_create_session(db, session_id)

    # 计算 turn/user_turn
    user_turn = _next_user_turn(db, session_id)
    user_turn_id = _next_turn_id(db, session_id)
    assistant_turn_id = user_turn_id + 1

    # 写 user message
    m_user = Message(
        session_id=session_id,
        turn_id=user_turn_id,
        user_turn=user_turn,
        role="user",
        content=user_text,
        created_at=_now_utc(),
        meta_json="{}",
    )
    db.add(m_user)

    # 写 assistant message（user_turn 不变）
    m_asst = Message(
        session_id=session_id,
        turn_id=assistant_turn_id,
        user_turn=user_turn,
        role="assistant",
        content=assistant_text,
        created_at=_now_utc(),
        meta_json="{}",
    )
    db.add(m_asst)

    # 更新 session 的 last_turn_id / last_user_turn（如果 Session 有这些字段）
    if hasattr(session, "last_turn_id"):
        session.last_turn_id = assistant_turn_id
    if hasattr(session, "last_user_turn"):
        session.last_user_turn = user_turn

    db.commit()

    # 触发 summarizer（注意：这里传的是 to_user_turn 语义）
    triggered_s4 = (user_turn % s4_every_user_turns == 0)
    triggered_s60 = (user_turn % s60_every_user_turns == 0)

    if triggered_s4:
        run_s4(
            db,
            session_id=session_id,
            to_user_turn=user_turn,
            window_user_turn=s4_window_user_turns,
            model_name=model_name,
        )

    if triggered_s60:
        run_s60(
            db,
            session_id=session_id,
            to_user_turn=user_turn,
            window_user_turn=s60_window_user_turns,
            model_name=model_name,
        )

    return ChatOnceResult(
        session_id=session_id,
        user_turn=user_turn,
        user_turn_triggered_s4=triggered_s4,
        user_turn_triggered_s60=triggered_s60,
        user_message_turn_id=user_turn_id,
        assistant_message_turn_id=assistant_turn_id,
    )


# 兼容：如果别的地方 import chat（可选）
chat = chat_once


def _normalize_legacy_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    兼容旧参数命名（单数）→ 新参数命名（复数）
    你现在 routes_telegram 传的是 s4_every_user_turn / s60_every_user_turn
    """
    if not kwargs:
        return {}

    out = dict(kwargs)

    # 旧 -> 新（频率）
    if "s4_every_user_turn" in out and "s4_every_user_turns" not in out:
        out["s4_every_user_turns"] = out.pop("s4_every_user_turn")
    if "s60_every_user_turn" in out and "s60_every_user_turns" not in out:
        out["s60_every_user_turns"] = out.pop("s60_every_user_turn")

    # 旧 -> 新（窗口）
    if "s4_window_user_turn" in out and "s4_window_user_turns" not in out:
        out["s4_window_user_turns"] = out.pop("s4_window_user_turn")
    if "s60_window_user_turn" in out and "s60_window_user_turns" not in out:
        out["s60_window_user_turns"] = out.pop("s60_window_user_turn")

    # 保险：如果有人传了奇怪的 None
    for k in [
        "s4_every_user_turns",
        "s60_every_user_turns",
        "s4_window_user_turns",
        "s60_window_user_turns",
    ]:
        if k in out and out[k] is None:
            out.pop(k)

    return out


# 兼容：旧代码可能用这个名字
def append_user_and_assistant(
    db: OrmSession,
    session_id: str,
    user_text: str,
    assistant_text: str,
    **kwargs,
) -> ChatOnceResult:
    """
    Backward-compatible wrapper.
    旧路由/任务如果还在 import `append_user_and_assistant`，直接转调 `chat_once`。
    """
    kwargs = _normalize_legacy_kwargs(kwargs)
    return chat_once(
        db,
        session_id=session_id,
        user_text=user_text,
        assistant_text=assistant_text,
        **kwargs,
    )
