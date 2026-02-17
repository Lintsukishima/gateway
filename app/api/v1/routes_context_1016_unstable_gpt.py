from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session as OrmSession

from app.db.deps import get_db
from app.services.context_builder import build_context_pack

router = APIRouter()


@router.get("/sessions/{session_id}/context")
def get_context(session_id: str, recent: int = 16, db: OrmSession = Depends(get_db)):
    # 统一出口：返回结构化 ContextPack（给你调试/给上层平台拉取用）
    return build_context_pack(db=db, session_id=session_id, recent=recent)
