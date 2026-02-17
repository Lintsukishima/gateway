import json
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session as OrmSession

from app.db.models import Message, SummaryS4, SummaryS60


def _safe_json_loads(s: Optional[str]) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        # 如果 summary_json 存了脏内容，至少别让接口炸
        return {"_raw": s}


def build_context_pack(
    db: OrmSession,
    session_id: str,
    recent: int = 16,
    include_meta: bool = True,
) -> Dict[str, Any]:
    """统一上下文包（给 Telegram / Rikkahub 都能用）。

    - 最新 S4
    - 最新 S60
    - 最近 N 条原文消息

    注意：这里“返回给前端”用；真正喂给 LLM 的 prompt 文本建议在 decider/telegram handler 里再组装。
    """

    s4_row = (
        db.query(SummaryS4)
        .filter(SummaryS4.session_id == session_id)
        .order_by(SummaryS4.to_turn.desc())
        .first()
    )
    s60_row = (
        db.query(SummaryS60)
        .filter(SummaryS60.session_id == session_id)
        .order_by(SummaryS60.to_turn.desc())
        .first()
    )

    msgs: List[Message] = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.turn_id.desc())
        .limit(recent)
        .all()
    )
    msgs = list(reversed(msgs))

    latest = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.turn_id.desc())
        .first()
    )

    pack: Dict[str, Any] = {
        "session_id": session_id,
        "s4": None,
        "s60": None,
        "recent": [
            {
                "role": m.role,
                "turn_id": m.turn_id,
                "user_turn": m.user_turn,
                "content": m.content,
            }
            for m in msgs
        ],
    }

    if s4_row:
        pack["s4"] = {
            "range": [s4_row.from_turn, s4_row.to_turn],
            "summary": _safe_json_loads(s4_row.summary_json),
            "created_at": s4_row.created_at.isoformat() if s4_row.created_at else None,
            "model": s4_row.model,
        }

    if s60_row:
        pack["s60"] = {
            "range": [s60_row.from_turn, s60_row.to_turn],
            "summary": _safe_json_loads(s60_row.summary_json),
            "created_at": s60_row.created_at.isoformat() if s60_row.created_at else None,
            "model": s60_row.model,
        }

    if include_meta:
        pack["meta"] = {
            "latest_turn_id": latest.turn_id if latest else None,
            "latest_user_turn": latest.user_turn if latest else None,
        }

    return pack
