import json
import uuid
from datetime import datetime
from sqlalchemy.orm import Session as OrmSession
from app.db.models import Session, Message
from app.services.summarizer import run_s4, run_s60
from sqlalchemy import func  # 加上这一行


def _now():
    return datetime.utcnow()

def ensure_session(db: OrmSession, session_id: str | None, user_id: str) -> Session:
    """确保会话存在，没有则创建"""
    if not session_id:
        session_id = str(uuid.uuid4())
    s = db.query(Session).filter(Session.id == session_id).first()
    if not s:
        s = Session(
            id=session_id,
            user_id=user_id,
            created_at=_now(),
            updated_at=_now(),
            last_turn_id=0
        )
        db.add(s)
        db.commit()
        db.refresh(s)
    return s


def append_message(db, session, role, content, user_turn: int, meta=None):
    session.last_turn_id += 1
    session.updated_at = _now()

    msg = Message(
        session_id=session.id,
        turn_id=session.last_turn_id,
        role=role,
        content=content,
        created_at=_now(),
        meta_json=json.dumps(meta or {}, ensure_ascii=False),
        lang="mix",
        user_turn = user_turn,  # 可后续优化
    )
    db.add(msg)
    db.add(session)
    db.commit()
    db.refresh(msg)
    return msg

def get_recent_messages(db: OrmSession, session_id: str, limit: int = 20) -> list[Message]:
    """获取最近的 N 条消息（按时间升序）"""
    return (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.turn_id.desc())
        .limit(limit)
        .all()[::-1]
    )

def naive_echo_reply(user_text: str) -> str:
    """简单的回声回复（MVP用，以后替换为真正的 LLM 调用）"""
    return f"你刚才说：{user_text}\n（这是临时回声，后续会接入真正的智能回复）"

def chat_once(db: OrmSession, session_id: str | None, user_id: str, user_text: str, meta: dict | None = None):
    """处理一次对话"""
    session = ensure_session(db, session_id, user_id)

    # ---- 新增：计算本轮的用户轮次 ----
    max_user_turn = db.query(func.max(Message.user_turn)).filter(Message.session_id == session.id).scalar() or 0
    new_user_turn = max_user_turn + 1
    # ---------------------------------

    # 1. 存储用户消息
    append_message(db, session, "user", user_text, user_turn=new_user_turn, meta=meta)
    


    # 2. （这里未来会触发 S4/S60 总结，并检索记忆，拼装 prompt，调用 LLM）
    #    现在先用简单回声
    reply = naive_echo_reply(user_text)

    # 3. 存储助手回复
    
    append_message(db, session, "assistant", reply, user_turn=new_user_turn, meta={"generated_at": _now().isoformat()})


     # 4. 触发 S4/S60 总结（每 4 轮--相当于用户说4句/ 30 轮）
    if session.last_turn_id % 8 == 0:
        run_s4(db, session_id=session.id, to_turn=session.last_turn_id, window=16, model_name="summarizer_mvp")
    
    if session.last_turn_id % 30 == 0:
        run_s60(db, session_id=session.id, to_turn=session.last_turn_id, window=60, model_name="summarizer_mvp")


    return session.id, session.last_turn_id, reply
