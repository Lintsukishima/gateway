import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from .session import Base
from sqlalchemy import Boolean, Integer
from sqlalchemy import Integer

def gen_id() -> str:
    return str(uuid.uuid4())

class Session(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, default=gen_id)
    user_id = Column(String, index=True, default="default_user")
    title = Column(String, nullable=True)
    status = Column(String, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    last_turn_id = Column(Integer, default=0)
    meta_json = Column(Text, default="{}")
    proactive_enabled = Column(Boolean, nullable=True, default=False)
    silence_threshold_min = Column(Integer, nullable=True, default=240)
    silence_cooldown_min = Column(Integer, nullable=True, default=120)
    platform = Column(String, index=True, default="unknown")   # telegram / rikkahub / unknown
    external_id = Column(String, index=True, nullable=True)    # tg chat_id / rk thread_id (可选)



    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=gen_id)
    session_id = Column(String, ForeignKey("sessions.id"), index=True, nullable=False)
    turn_id = Column(Integer, index=True, nullable=False)
    user_turn = Column(Integer, index=True, nullable=True)
    role = Column(String, index=True, nullable=False)
    content = Column(Text, nullable=False)
    content_tokens = Column(Integer, nullable=True)
    lang = Column(String, default="mix")
    created_at = Column(DateTime, default=datetime.utcnow)
    platform = Column(String, index=True, default="unknown")

    meta_json = Column(Text, default="{}")

    session = relationship("Session", back_populates="messages")

class SummaryS4(Base):
    __tablename__ = "summaries_s4"
    id = Column(String, primary_key=True, default=gen_id)
    session_id = Column(String, index=True, nullable=False)

    from_turn = Column(Integer, index=True, nullable=False)
    to_turn = Column(Integer, index=True, nullable=False)

    summary_json = Column(Text, nullable=False)   # JSON 字符串
    model = Column(String, default="unknown")
    created_at = Column(DateTime, default=datetime.utcnow)
    meta_json = Column(Text, default="{}")

class SummaryS60(Base):
    __tablename__ = "summaries_s60"
    id = Column(String, primary_key=True, default=gen_id)
    session_id = Column(String, index=True, nullable=False)

    from_turn = Column(Integer, index=True, nullable=False)
    to_turn = Column(Integer, index=True, nullable=False)

    summary_json = Column(Text, nullable=False)
    model = Column(String, default="unknown")
    created_at = Column(DateTime, default=datetime.utcnow)
    meta_json = Column(Text, default="{}")

class TriggerJob(Base):
    __tablename__ = "trigger_jobs"
    id = Column(String, primary_key=True, default=gen_id)
    session_id = Column(String, index=True, nullable=False)
    trigger_type = Column(String, index=True, nullable=False)  # silence/morning/lunch/night
    trigger_payload_json = Column(Text, default="{}")

    status = Column(String, default="queued")  # queued/running/done/failed/skipped
    scheduled_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    attempts = Column(Integer, default=0)
    last_error = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    meta_json = Column(Text, default="{}")


class OutboxMessage(Base):
    __tablename__ = "outbox_messages"
    id = Column(String, primary_key=True, default=gen_id)
    job_id = Column(String, index=True, nullable=False)

    channel = Column(String, index=True, default="telegram")
    recipient = Column(String, nullable=True)     # chat_id / user id
    decision = Column(String, default="skip")     # send/skip
    message_text = Column(Text, nullable=True)

    status = Column(String, default="pending")    # pending/sent/failed
    sent_at = Column(DateTime, nullable=True)

    model_trace_json = Column(Text, default="{}") # 简短理由
    created_at = Column(DateTime, default=datetime.utcnow)
    meta_json = Column(Text, default="{}")

