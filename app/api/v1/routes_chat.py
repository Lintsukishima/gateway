from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as OrmSession
from app.db.session import SessionLocal
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import chat_once

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, db: OrmSession = Depends(get_db)):
    session_id, turn_id, reply = chat_once(
        db=db,
        session_id=req.session_id,
        user_id=req.user_id,
        user_text=req.message,
        meta=req.meta,
    )
    return ChatResponse(session_id=session_id, turn_id=turn_id, reply=reply)
