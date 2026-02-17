import json
from sqlalchemy.orm import Session as OrmSession
from app.db.models import Message, SummaryS4, SummaryS60

def build_prompt(db: OrmSession, session_id: str, user_text: str, recent_limit: int = 16) -> dict:
    # 取最近摘要
    s4 = (db.query(SummaryS4).filter(SummaryS4.session_id == session_id)
          .order_by(SummaryS4.to_turn.desc()).first())
    s60 = (db.query(SummaryS60).filter(SummaryS60.session_id == session_id)
           .order_by(SummaryS60.to_turn.desc()).first())

    # 取最近消息
    msgs = (db.query(Message).filter(Message.session_id == session_id)
            .order_by(Message.turn_id.desc()).limit(recent_limit).all()[::-1])

    # 这里先不塞 canon/echo，后面加
    context = {
        "s60": json.loads(s60.summary_json) if s60 else None,
        "s4": json.loads(s4.summary_json) if s4 else None,
        "recent": [{"role": m.role, "turn": m.turn_id, "content": m.content} for m in msgs],
        "user_text": user_text
    }
    return context
