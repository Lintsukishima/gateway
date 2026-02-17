from pydantic import BaseModel
from typing import Optional, Dict, Any

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    user_id: Optional[str] = "default_user"
    message: str
    meta: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    session_id: str
    turn_id: int
    reply: str
