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

def append_message(db: OrmSession, session: Session, role: str, content: str, meta: dict | None = None) -> Message:
    """追加一条消息，自动增加 turn_id"""
    # NOTE:
    # - turn_id：每写入一条 Message 就 +1（user/assistant 都算）
    # - user_turn：只在 role=user 的时候 +1；assistant 跟随同一轮 user_turn
    session.last_turn_id += 1
    session.updated_at = _now()

    # 计算本条消息的 user_turn
    if role == "user":
        last_ut = (
            db.query(func.max(Message.user_turn))
            .filter(Message.session_id == session.id)
            .scalar()
        ) or 0
        msg_user_turn = last_ut + 1
    else:
        # assistant/system：默认跟随当前最大 user_turn
        msg_user_turn = (
            db.query(func.max(Message.user_turn))
            .filter(Message.session_id == session.id)
            .scalar()
        ) or 0

    msg = Message(
        session_id=session.id,
        turn_id=session.last_turn_id,
        role=role,
        content=content,
        created_at=_now(),
        meta_json=json.dumps(meta or {}, ensure_ascii=False),
        lang="mix",
        user_turn=msg_user_turn,
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

def chat_once(
    db: OrmSession,
    session_id: str | None,
    user_id: str,
    user_text: str,
    meta: dict | None = None,
    reply_text: str | None = None,
    assistant_meta: dict | None = None,
):
    """处理一次对话"""
    session = ensure_session(db, session_id, user_id)

    # 1. 存储用户消息
    user_msg = append_message(db, session, "user", user_text, meta=meta)
    new_user_turn = user_msg.user_turn

    # 2. （这里未来会触发 S4/S60 总结，并检索记忆，拼装 prompt，调用 LLM）
    #    现在先用简单回声
    reply = reply_text if reply_text is not None else naive_echo_reply(user_text)

    # 3. 存储助手回复
    asst_meta = {"generated_at": _now().isoformat()}
    if assistant_meta:
        asst_meta.update(assistant_meta)
    assistant_msg = append_message(db, session, "assistant", reply, meta=asst_meta)
    # 强制把 assistant 的 user_turn 修正为本轮 user_turn
    assistant_msg.user_turn = new_user_turn
    db.add(assistant_msg)
    db.commit()

    # 4. 触发 S4/S60 总结
    # 你要的节奏：用户说 4 句总结一次（短期）；用户说 30 句总结一次（长期）
    # 注意：summarizer 的 window 是“消息条数”，不是 user_turn 条数。
    if new_user_turn % 4 == 0:
        run_s4(db, session_id=session.id, to_turn=session.last_turn_id, window=8, model_name="summarizer_mvp")
    if new_user_turn % 30 == 0:
        run_s60(db, session_id=session.id, to_turn=session.last_turn_id, window=60, model_name="summarizer_mvp")


    return session.id, session.last_turn_id, reply
