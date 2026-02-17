import os
import json
import requests
from datetime import datetime
from fastapi import APIRouter, Request
from zoneinfo import ZoneInfo

from app.db.session import SessionLocal
from app.services.chat_service import chat_once
from pathlib import Path


router = APIRouter()

# ---------- 启动时加载记忆（全局只加载一次） ----------
_MEMORY_TEXT = ""

def _load_memory():
    global _MEMORY_TEXT
    try:
        MEMORY_SEED_PATH = Path(__file__).resolve().parents[2] / "memory" / "memory_seed.json"
        with open(MEMORY_SEED_PATH, "r", encoding="utf-8") as f:
            memories_data = json.load(f)
            # 每条记忆是一个对象，提取 text 字段
            memory_list = [f"- {item['text']}" for item in memories_data]
            _MEMORY_TEXT = "\n".join(memory_list)
    except Exception as e:
        print(f"[警告] 加载记忆文件失败: {e}")
        _MEMORY_TEXT = ""

# 在模块加载时执行一次
_load_memory()

# ---------- 角色设定（每次调用读取，如果文件小也可以启动时加载，但为简单保持每次读）----------
def _load_persona() -> str:
    try:
        with open("prompts/persona_telegram.txt", "r", encoding="utf-8") as f:
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
def call_chat_llm(user_text: str, context_text: str = "") -> str:
    # 获取角色设定（每次读，便于修改后立即生效，也可以改为全局变量提高性能）
    persona = _load_persona()

    # 使用全局记忆文本
    memory_text = _MEMORY_TEXT

    base_url = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("CHAT_MODEL", os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")).strip()

    if not api_key:
        return f"收到啦：{user_text}"

    url = f"{base_url}/chat/completions"

    # 构建消息列表
    messages = [
        {"role": "system", "content": persona},
    ]

    if context_text:
        messages.append({"role": "system", "content": context_text})

    if memory_text:
        messages.append({"role": "system", "content": f"我们共同的记忆：\n{memory_text}"})

    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip() or "（我这边刚刚卡了一下，再说一遍？）"
    except Exception as e:
        # 模型调用失败，回退到简单回复
        return f"（暂时没法思考，直接回你：{user_text}）"

# ===== 构造上下文占位（后面替换为真实 ContextPack）=====
def build_context_text_stub(chat_id: str, now_iso: str, hour: int, minute: int, weekday: str, tz_key: str) -> str:
    # 先用一个轻量占位：给模型“时间感知”。
    # 你后面把这里替换成真正的 ContextPack（S4/S60 + recent）即可。
    return (
        f"现在时间：{now_iso}（本地 {hour:02d}:{minute:02d}，{weekday}，时区 {tz_key}）\n"
        f"渠道：Telegram\n"
        f"session=tg:{chat_id}"
    )

# ===== Telegram webhook 入口 =====
@router.post("/telegram/webhook")
async def telegram_webhook(req: Request):
    update = await req.json()

    # 只处理文本消息
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return {"ok": True}

    chat = msg.get("chat", {})
    chat_id = str(chat.get("id", "")).strip()
    text = (msg.get("text") or "").strip()

    if not chat_id or not text:
        return {"ok": True}

    # 获取当前时间（指定时区）
    tz = ZoneInfo(os.getenv("USER_TZ", "Asia/Shanghai"))
    now = datetime.now(tz)
    now_iso = now.isoformat()
    hour = now.hour
    minute = now.minute
    weekday = now.strftime("%A")

    # 构造完整上下文（时间 + 占位信息）
    full_context = build_context_text_stub(chat_id, now_iso, hour, minute, weekday, tz.key)

    # 调用 LLM 生成回复
    reply = call_chat_llm(text, context_text=full_context)

    # 把这一轮写入 DB（含 user_turn），并触发 S4(每4句)/S60(每30句) 自动摘要
    db = SessionLocal()
    try:
        chat_once(
            db,
            session_id=f"tg:{chat_id}",
            user_id=f"tg_user:{chat_id}",
            user_text=text,
            meta={
                "channel": "telegram",
                "telegram_message_id": msg.get("message_id"),
                "telegram_date": msg.get("date"),
            },
            reply_text=reply,
            assistant_meta={"channel": "telegram"},
        )
    finally:
        db.close()

    # 发送回 Telegram
    send_telegram_message(reply, chat_id)

    return {"ok": True}