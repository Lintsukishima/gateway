from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from app.db.models import Message
from app.services.summarizer import run_s4, run_s60


@dataclass
class AppendResult:
    session_id: str
    user_turn: int
    user_turn_id: int
    assistant_turn_id: Optional[int]


def _get_last_message(db: Session, session_id: str) -> Optional[Message]:
    return (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.turn_id.desc())
        .first()
    )


def _next_turn_ids(db: Session, session_id: str) -> Tuple[int, int, int]:
    """返回 (next_user_turn_id, next_assistant_turn_id, next_user_turn_index)."""
    last = _get_last_message(db, session_id)
    if not last:
        # 约定：第一条 user turn_id=1，assistant=2
        return 1, 2, 1

    next_user_turn_id = last.turn_id + 1
    next_assistant_turn_id = last.turn_id + 2
    last_ut = last.user_turn or 0
    next_user_turn_index = last_ut + 1
    return next_user_turn_id, next_assistant_turn_id, next_user_turn_index


def append_user_and_assistant(
    db: Session,
    *,
    session_id: str,
    user_text: str,
    assistant_text: str,
    # 触发策略：短期 4 轮 user，总结一次；长期 30 轮 user，总结一次
    s4_every_user_turn: int = 4,
    s60_every_user_turn: int = 30,
    model_name: str = "summarizer_mvp",
) -> AppendResult:
    """写入一轮对话（user + assistant），并按 user_turn 触发 S4/S60。

    你现在 Telegram / Rikkahub 都可以复用这一套写入逻辑。
    """

    user_tid, asst_tid, ut = _next_turn_ids(db, session_id)

    user_msg = Message(
        session_id=session_id,
        turn_id=user_tid,
        user_turn=ut,
        role="user",
        content=user_text,
    )
    asst_msg = Message(
        session_id=session_id,
        turn_id=asst_tid,
        user_turn=ut,  # 关键：assistant 不递增
        role="assistant",
        content=assistant_text,
    )

    db.add(user_msg)
    db.add(asst_msg)
    db.commit()

    # 触发总结（幂等由 summarizer 内部保证）
    if ut % s4_every_user_turn == 0:
        run_s4(db, session_id=session_id, to_user_turn=ut, window_user_turn=s4_every_user_turn, model_name=model_name)

    if ut % s60_every_user_turn == 0:
        run_s60(db, session_id=session_id, to_user_turn=ut, window_user_turn=s60_every_user_turn, model_name=model_name)

    return AppendResult(
        session_id=session_id,
        user_turn=ut,
        user_turn_id=user_tid,
        assistant_turn_id=asst_tid,
    )
