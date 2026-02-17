import os
import json
import requests
from datetime import datetime
from fastapi import APIRouter, Request
from zoneinfo import ZoneInfo

from app.db.session import SessionLocal
from app.services.context_builder import build_context_pack
from app.services.chat_service import append_user_and_assistant

router = APIRouter()

# ---------- 启动时加载记忆（全局只加载一次） ----------
_MEMORY_TEXT = ""


def _load_memory():
    global _MEMORY_TEXT
    try:
        with open("memory/memory_seed.json", "r", encoding="utf-8") as f:
            memories_data = json.load(f)
            memory_list = [f"- {item.get('text', '')}" for item in memories_data if item.get("text")]
            _MEMORY_TEXT = "\n".join(memory_list)
    except Exception as e:
        print(f"[warn] load memory_seed.json failed: {e}")
        _MEMORY_TEXT = ""


_load_memory()


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


# ===== 调用 LLM（OpenAI-compatible） =====

def _as_chat_messages_from_recent(recent: list[dict], *, max_pairs: int = 12) -> list[dict]:
    """
    把 pack.recent 变成真正的 chat messages（role=user/assistant）。
    recent 里已经是 turn 序列：[{role, content, ...}, ...]
    """
    msgs: list[dict] = []
    if not recent:
        return msgs

    # 只截最后 N 条（按消息条数，不是 user_turn）
    # max_pairs=12 约等于 24 条消息上限（视你的 recent 返回情况）
    max_msgs = max_pairs * 2
    tail = recent[-max_msgs:]

    for m in tail:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})
    return msgs


def call_chat_llm(*, user_text: str, system_blocks: list[str], recent_messages: list[dict] | None = None) -> str:
    persona = _load_persona()
    memory_text = _MEMORY_TEXT

    base_url = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "").strip()

    # 你说你刚才调了模型名——这里优先 CHAT_MODEL，其次 LLM_MODEL，其次默认
    model = os.getenv("CHAT_MODEL", os.getenv("LLM_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")).strip()

    # 温度也用你现在的为准：TG_TEMPERATURE > TEMPERATURE > 默认 0.8
    try:
        temperature = float(os.getenv("TG_TEMPERATURE", os.getenv("TEMPERATURE", "0.8")))
    except Exception:
        temperature = 0.8

    if not api_key:
        return f"收到啦：{user_text}"

    url = f"{base_url}/chat/completions"

    messages = [{"role": "system", "content": persona}]

    # S4/S60/time/recent(系统版) 仍然可以放 system（但真正的 recent 会作为 messages 再放一遍）
    for blk in system_blocks:
        if blk:
            messages.append({"role": "system", "content": blk})

    if memory_text:
        messages.append({"role": "system", "content": f"我们共同的记忆：\n{memory_text}"})

    # ✅关键：把 recent 作为真正历史消息喂给模型（比塞在 system 文本里强很多）
    if recent_messages:
        messages.extend(recent_messages)

    # 当前用户输入
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip() or "（我这边刚刚卡了一下，再说一遍？）"
    except Exception:
        return f"（我这边暂时没连上模型，先回你：{user_text}）"


def _format_context_for_llm(pack: dict, *, now_text: str) -> str:
    """把 ContextPack 转成“喂给 LLM 的 system 文本”。"""

    s4 = pack.get("s4", {})
    s60 = pack.get("s60", {})
    recent = pack.get("recent", [])

    # recent 仍保留一份在 system 里（让模型即便忽略 history 也能看到一点）
    recent_lines = []
    for m in recent:
        recent_lines.append(f"{m.get('role')}: {m.get('content')}")

    return "\n".join(
        [
            f"[time]\n{now_text}",
            "[s4_latest]" + ("\n" + json.dumps(s4.get("summary"), ensure_ascii=False) if s4 else "\n(暂无)"),
            "[s60_latest]" + ("\n" + json.dumps(s60.get("summary"), ensure_ascii=False) if s60 else "\n(暂无)"),
            "[recent]\n" + ("\n".join(recent_lines) if recent_lines else "(暂无)"),
        ]
    )


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

    session_id = f"tg:{chat_id}"

    # 1) 时间感知
    tz = ZoneInfo(os.getenv("USER_TZ", "Asia/Shanghai"))
    now = datetime.now(tz)
    now_text = f"现在是 {now.strftime('%Y-%m-%d %H:%M')}（{tz.key}）"

    # 2) 取上下文（S4/S60/recent），用于给模型
    db = SessionLocal()
    try:
        pack = build_context_pack(db=db, session_id=session_id, recent=int(os.getenv("TG_CONTEXT_RECENT", "16")))
    finally:
        db.close()

    ctx_for_llm = _format_context_for_llm(pack, now_text=now_text)

    # ✅把 recent 变成真正 messages
    recent_msgs = _as_chat_messages_from_recent(pack.get("recent", []), max_pairs=int(os.getenv("TG_CONTEXT_PAIRS", "12")))

    # 3) 调用模型生成回复（Telegram 只发 reply，不发 ctx）
    reply = call_chat_llm(
        user_text=text,
        system_blocks=[ctx_for_llm],
        recent_messages=recent_msgs,
    )

    # 4) 写入 DB（触发 S4 / S60）
    db = SessionLocal()
    try:
        append_user_and_assistant(
            db,
            session_id=session_id,
            user_text=text,
            assistant_text=reply,
            s4_every_user_turn=int(os.getenv("S4_EVERY", "4")),
            s60_every_user_turn=int(os.getenv("S60_EVERY", "30")),
            model_name=os.getenv("SUMMARIZER_MODEL_NAME", "summarizer_mvp"),
        )
    finally:
        db.close()

    # 5) 回 Telegram
    send_telegram_message(reply, chat_id)

    return {"ok": True}
