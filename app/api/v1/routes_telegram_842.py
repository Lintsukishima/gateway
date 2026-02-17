import os
import json
import requests
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Request
from zoneinfo import ZoneInfo
from sqlalchemy import func
from app.services import summarizer

# ---------- 新增数据库相关导入 ----------
from app.db.session import SessionLocal
from app.db.models import Session as ChatSession, Message

router = APIRouter()

# ---------- 计算项目根目录绝对路径 ----------
BASE_DIR = Path(__file__).resolve().parents[3]
MEMORY_PATH = BASE_DIR / "app" / "memory" / "memory_seed.json"
PERSONA_PATH = BASE_DIR / "prompts" / "persona_telegram.txt"

# ---------- 加载记忆 ----------
_MEMORY_TEXT = ""
def _load_memory():
    global _MEMORY_TEXT
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            memories_data = json.load(f)
            memory_list = [f"- {item['text']}" for item in memories_data]
            _MEMORY_TEXT = "\n".join(memory_list)
    except Exception as e:
        print(f"[警告] 加载记忆文件失败: {e}")
        _MEMORY_TEXT = ""
_load_memory()

# ---------- 加载角色设定 ----------
def _load_persona() -> str:
    try:
        with open(PERSONA_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "你是一个在Telegram上聊天的助手，回复要像人类发消息，简洁自然，1-3句。"

# ===== Telegram sender =====
def send_telegram_message(text: str, chat_id: str):
    token = os.getenv("TG_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TG_BOT_TOKEN is missing")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

# ===== 调用 LLM =====
def call_chat_llm(user_text: str, time_context: str = "", context_text: str = "") -> str:
    persona = _load_persona()
    memory_text = _MEMORY_TEXT

    base_url = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("CHAT_MODEL", os.getenv("LLM_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")).strip()

    if not api_key:
        return f"收到啦：{user_text}"

    url = f"{base_url}/chat/completions"

    messages = [{"role": "system", "content": persona}]
    if time_context:
        messages.append({"role": "system", "content": f"[time]\n{time_context}"})
    if memory_text:
        messages.append({"role": "system", "content": f"[memory]\n{memory_text}"})
    if context_text:
        messages.append({"role": "system", "content": f"[context]\n{context_text}"})
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.8,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip() or "（我这边刚刚卡了一下，再说一遍？）"
    except Exception as e:
        return f"（暂时没法思考，直接回你：{user_text}）"

# ===== 构造上下文占位（待替换为真实 ContextPack）=====
def build_context_text_stub(chat_id: str) -> str:
    return f"session=tg:{chat_id} (no summaries yet)"

# ===== Telegram webhook 入口 =====
@router.post("/telegram/webhook")
async def telegram_webhook(req: Request):
    update = await req.json()
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return {"ok": True}

    chat = msg.get("chat", {})
    chat_id = str(chat.get("id", "")).strip()
    text = (msg.get("text") or "").strip()

    if not chat_id or not text:
        return {"ok": True}

    # 获取当前时间
    tz = ZoneInfo(os.getenv("USER_TZ", "Asia/Shanghai"))
    now = datetime.now(tz)
    time_context = f"现在时间：{now.isoformat()}（本地 {now.hour:02d}:{now.minute:02d}，{now.strftime('%A')}，时区 {tz.key}）"

    # 构造上下文（目前 stub）
    context_text = build_context_text_stub(chat_id)

    # ---------- 数据库操作 ----------
    db = SessionLocal()
    try:
        # 1. 生成 Telegram 专用 session_id
        session_id = f"tg:{chat_id}"

        # 2. 查找或创建 Session
        sess = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not sess:
            # 创建新 session，platform 固定为 "telegram"，external_id 存 chat_id
            sess = ChatSession(
                id=session_id,
                platform="telegram",
                external_id=chat_id,
                title="Telegram Chat",
                status="active",
                # user_id 使用默认值 "default_user" 或可设为 "telegram"
                # 其他字段如 proactive_enabled 等保持默认
            )
            db.add(sess)
            db.commit()

        # 3) next_turn：用 sess.last_turn_id
        next_turn = (sess.last_turn_id or 0) + 1

        # 4) new_user_turn：只数 user
        last_ut = (
            db.query(func.max(Message.user_turn))
            .filter(Message.session_id == session_id)
            .scalar()
        ) or 0
        new_user_turn = last_ut + 1

        # 5) 写 user message
        user_msg = Message(
            session_id=session_id,
            platform="telegram",
            role="user",
            content=text,
            turn_id=next_turn,
            user_turn=new_user_turn,
        )
        db.add(user_msg)
        

        # 6) 调用 LLM
        reply = call_chat_llm(text, time_context=time_context, context_text=context_text)

        # 7) 写 assistant message（turn_id +1，user_turn 同值）
        assistant_msg = Message(
            session_id=session_id,
            platform="telegram",
            role="assistant",
            content=reply,
            turn_id=next_turn + 1,
            user_turn=new_user_turn,
        )
        db.add(assistant_msg)
        

        # 8) 更新 session.last_turn_id（一次性更新到最新）
        sess.last_turn_id = next_turn + 1
        sess.updated_at = datetime.utcnow()

        db.commit()


        if new_user_turn % 4 == 0:
            summarizer.run_s4(db, session_id=session_id, to_user_turn=new_user_turn, window_user_turn=4)

        if new_user_turn % 60 == 0:
            summarizer.run_s60(db, session_id=session_id, to_user_turn=new_user_turn, window_user_turn=60)



        # 8. 发送回复
        send_telegram_message(reply, chat_id)

        return {"ok": True}



    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()