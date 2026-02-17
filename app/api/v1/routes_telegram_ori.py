import os
import json
import requests
from datetime import datetime, timezone
from fastapi import APIRouter, Request
from zoneinfo import ZoneInfo 

router = APIRouter()

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

# ===== Minimal chat LLM (SiliconFlow/OpenAI-compatible) =====
def call_chat_llm(user_text: str, context_text: str = "") -> str:
    # 加载角色设定（从文件读取）
    try:
        with open("prompts/persona_telegram.txt", "r", encoding="utf-8") as f:
            persona = f.read().strip()
    except FileNotFoundError:
        persona = "你是一个在Telegram上聊天的助手，回复要像人类发消息，简洁自然，1-3句。"  # 降级默认值

    # 加载记忆种子（从 JSON 文件读取）
    try:
        with open("memory/memory_seed.json", "r", encoding="utf-8") as f:
            memories = json.load(f)
            memory_text = "\n".join([f"- {m}" for m in memories])
    except (FileNotFoundError, json.JSONDecodeError):
        memory_text = ""

    base_url = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("CHAT_MODEL", os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")).strip()

    if not api_key:
        return f"收到啦：{user_text}"

    url = f"{base_url}/chat/completions"

    # 构建消息列表，按顺序：角色设定、时间上下文、记忆、当前对话上下文、用户输入
    messages = [
        {"role": "system", "content": persona},
    ]

    # 如果有时间上下文（由 webhook 传入），也作为 system 消息
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
        # 如果模型调用失败，回退到简单回复
        return f"（暂时没法思考，直接回你：{user_text}）"
    
    
    """
    先用便宜模型跑通：只要能返回字符串即可。
    SiliconFlow 是 OpenAI-compatible 的 /v1/chat/completions。
    """
    base_url = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("CHAT_MODEL", os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")).strip()

    if not api_key:
        # 没 key 就先回 stub，保证通路
        return f"收到啦：{user_text}"

    url = f"{base_url}/chat/completions"
    messages = [
        {"role": "system", "content": "你是一个在Telegram上聊天的助手，回复要像人类发消息，简洁自然，1-3句。"},
    ]
    if context_text:
        messages.append({"role": "system", "content": f"[context]\n{context_text}"})
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    # OpenAI-compatible
    return (data["choices"][0]["message"]["content"] or "").strip() or "（我这边刚刚卡了一下，再说一遍？）"

# ===== (Optional) Build context text stub =====
def build_context_text_stub(chat_id: str) -> str:
    """
    先别接你的DB/summary，给个占位，后面我们再换成真实 ContextPack。
    """
    return f"session=tg:{chat_id} (no summaries yet)"

@router.post("/telegram/webhook")
async def telegram_webhook(req: Request):
    """
    Telegram -> Gateway
    """
    update = await req.json()

    # 只处理普通文本消息
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return {"ok": True}

    chat = msg.get("chat", {})
    chat_id = str(chat.get("id", "")).strip()
    text = (msg.get("text") or "").strip()

    if not chat_id or not text:
        return {"ok": True}

    # ===== 新增：获取当前时间（指定时区） =====
    tz = ZoneInfo(os.getenv("USER_TZ", "Asia/Shanghai"))  # 默认上海时区，可通过环境变量覆盖
    now = datetime.now(tz)
    now_iso = now.isoformat()
    hour = now.hour
    minute = now.minute
    weekday = now.strftime("%A")  # 英文星期，例如 "Monday"
    
    time_context = f"现在时间：{now_iso}（本地 {hour:02d}:{minute:02d}，{weekday}，时区 {tz.key}）"
    # ===== 结束 =====

    # --- MVP: 先不写DB，直接回一句，验证“回路” ---
    # 如果你要先马上写DB：下一步我们把这里换成调用你现有的 chat_service 写入即可
    context_text = build_context_text_stub(chat_id)
    reply = call_chat_llm(text, context_text=time_context + "\n" + context_text)

    # 发回 Telegram
    send_telegram_message(reply, chat_id)

    return {"ok": True}
