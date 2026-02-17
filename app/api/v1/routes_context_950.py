import json
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as OrmSession
from app.db.session import SessionLocal
from app.db.models import Message, SummaryS4, SummaryS60

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/sessions/{session_id}/context")
def get_context(session_id: str, recent: int = 16, db: OrmSession = Depends(get_db)):
    # 只取 “窗口=16”的最新短期摘要
    s4 = db.query(SummaryS4).filter(
        SummaryS4.session_id == session_id
    ).order_by(SummaryS4.to_turn.desc()).first() 

    s60 = db.query(SummaryS60).filter(
        SummaryS60.session_id == session_id
    ).order_by(SummaryS60.to_turn.desc()).first()  

     # 最近 N 条原文
    msgs = (db.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.turn_id.desc())
            .limit(recent)
            .all())[::-1]

    latest = (db.query(Message)
              .filter(Message.session_id == session_id)
              .order_by(Message.turn_id.desc())
              .first())

    return {
        "session_id": session_id,
        "s4": {
            "range": [s4.from_turn, s4.to_turn],
            "summary": json.loads(s4.summary_json),
            "created_at": s4.created_at.isoformat(),
            "model": s4.model,
        } if s4 else None,
        "s60": {
            "range": [s60.from_turn, s60.to_turn],
            "summary": json.loads(s60.summary_json),
            "created_at": s60.created_at.isoformat(),
            "model": s60.model,
        } if s60 else None,
        "recent": [
            {"role": m.role, "turn_id": m.turn_id, "user_turn": m.user_turn, "content": m.content}
            for m in msgs
        ],
        "meta": {
            "latest_turn_id": latest.turn_id if latest else None,
            "latest_user_turn": latest.user_turn if latest else None,
        }
    }
