import os
import json
import requests
from typing import Any, Dict, Optional
from datetime import datetime
from fastapi import APIRouter, Request
from zoneinfo import ZoneInfo

from app.db.session import SessionLocal
from app.services.context_builder import build_context_pack
from app.services.chat_service import append_user_and_assistant
from app.services.anchor_rag import query_anchor_snippets, format_anchor_block
from app.services.fact_constraints import build_fact_constraint_block

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


def _load_tg_rk_session_map() -> Dict[str, str]:
    raw = os.getenv("TG_RK_SESSION_MAP_JSON", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception as e:
        print(f"[warn] parse TG_RK_SESSION_MAP_JSON failed: {e}")
        return {}
    if not isinstance(data, dict):
        return {}

    out: Dict[str, str] = {}
    for k, v in data.items():
        key = str(k).strip()
        val = str(v).strip()
        if not key or not val:
            continue
        out[key] = val if val.startswith("rk:") else f"rk:{val}"
    return out


def _resolve_rk_session_id(chat_id: str) -> Optional[str]:
    mapping = _load_tg_rk_session_map()
    if chat_id in mapping:
        return mapping[chat_id]

    # 可选前缀映射：设置 TG_RK_FALLBACK_PREFIX=uid: 则 chat_id=123 映射为 rk:uid:123
    fallback_prefix = os.getenv("TG_RK_FALLBACK_PREFIX", "").strip()
    if fallback_prefix:
        return f"rk:{fallback_prefix}{chat_id}"

    return None


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


# ===== 把 pack.recent 转成真正的 messages =====

def _as_chat_messages_from_recent(recent: list[dict], *, max_pairs: int = 12) -> list[dict]:
    """
    recent 里是 [{role, content, ...}, ...]
    这里把它变成 chat.completions 的 messages（role=user/assistant）。
    """
    if not recent:
        return []

    max_msgs = max_pairs * 2
    tail = recent[-max_msgs:]

    msgs: list[dict] = []
    for m in tail:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})
    return msgs


# ===== 调用 LLM（OpenAI-compatible） =====

def call_chat_llm(*, user_text: str, system_blocks: list[str], recent_messages: list[dict] | None = None) -> str:
    persona = _load_persona()
    memory_text = _MEMORY_TEXT

    base_url = os.getenv("LLM_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("CHAT_MODEL", os.getenv("LLM_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507")).strip()

    # 你说你刚刚调了温度/模型名：这里按 env 为准
    try:
        temperature = float(os.getenv("TG_TEMPERATURE", os.getenv("TEMPERATURE", "0.8")))
    except Exception:
        temperature = 0.8

    if not api_key:
        return f"收到啦：{user_text}"

    url = f"{base_url}/chat/completions"

    messages = [{"role": "system", "content": persona}]

    for blk in system_blocks:
        if blk:
            messages.append({"role": "system", "content": blk})

    if memory_text:
        messages.append({"role": "system", "content": f"我们共同的记忆：\n{memory_text}"})

    # ✅关键：recent 当作真正历史消息
    if recent_messages:
        messages.extend(recent_messages)

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
    except Exception as e:
        print(f"[warn] call_chat_llm failed: {e}")
        return f"（我这边暂时没连上模型，先回你：{user_text}）"


def _format_context_for_llm(pack: dict, *, now_text: str) -> str:
    """
    把 ContextPack 转成 system 文本（S4/S60 放这里最合适）
    recent 不需要塞太多，因为我们会用 recent_messages 作为真正历史消息。
    """
    s4 = pack.get("s4", {})
    s60 = pack.get("s60", {})

    return "\n".join(
        [
            f"[time]\n{now_text}",
            "[s4_latest]" + ("\n" + json.dumps(s4.get("summary"), ensure_ascii=False) if s4 else "\n(暂无)"),
            "[s60_latest]" + ("\n" + json.dumps(s60.get("summary"), ensure_ascii=False) if s60 else "\n(暂无)"),
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

    # 2) 取上下文（S4/S60/recent）
    db = SessionLocal()
    try:
        pack = build_context_pack(
            db=db,
            session_id=session_id,
            recent=int(os.getenv("TG_CONTEXT_RECENT", "16")),
        )
    finally:
        db.close()

    # 2.1) 单向读取 RikkaHub 摘要（Telegram -> 只读 rk:*；RikkaHub 不读 tg:*）
    rk_pack: Optional[Dict[str, Any]] = None
    rk_session_id = _resolve_rk_session_id(chat_id)
    if rk_session_id:
        db = SessionLocal()
        try:
            rk_pack = build_context_pack(
                db=db,
                session_id=rk_session_id,
                recent=0,
            )
        finally:
            db.close()

    ctx_for_llm = _format_context_for_llm(pack, now_text=now_text)

    # 2.5) Anchor RAG（语气锚点）
    anchor_block = ""
    snips: list[str] = []
    try:
        snips = query_anchor_snippets(
            user_text=text,
            allow_context=None,  # 先不做 Allow Context 过滤，稳定后再加
        )
        anchor_block = format_anchor_block(snips)
    except Exception as e:
        print(f"[anchor_rag warn] {e}")
        anchor_block = ""

    print(f"[anchor_rag] snippets={len(snips)}")
    if snips:
        print(f"[anchor_rag] first_snip={snips[0][:80]}")
    print(f"[anchor_rag] block_len={len(anchor_block)}")

    # 2.6) 与 RikkaHub 共用事实约束规则（summary 优先于普通 anchor）
    evidence: list[dict] = []
    if rk_pack and rk_pack.get("s4", {}).get("summary"):
        evidence.append({
            "type": "summary_s4",
            "W_fact": 1.35,
            "text": json.dumps(rk_pack["s4"]["summary"], ensure_ascii=False),
            "source": "rk",
        })
    if rk_pack and rk_pack.get("s60", {}).get("summary"):
        evidence.append({
            "type": "summary_s60",
            "W_fact": 1.35,
            "text": json.dumps(rk_pack["s60"]["summary"], ensure_ascii=False),
            "source": "rk",
        })
    if pack.get("s4", {}).get("summary"):
        evidence.append({
            "type": "summary_s4",
            "W_fact": 1.35,
            "text": json.dumps(pack["s4"]["summary"], ensure_ascii=False),
            "source": "tg",
        })
    if pack.get("s60", {}).get("summary"):
        evidence.append({
            "type": "summary_s60",
            "W_fact": 1.35,
            "text": json.dumps(pack["s60"]["summary"], ensure_ascii=False),
            "source": "tg",
        })
    for snip in snips[:2]:
        evidence.append({"type": "anchor", "W_fact": 1.0, "text": snip})

    grounding_mode = "strong" if evidence else "weak"
    fact_block = build_fact_constraint_block(evidence, grounding_mode)


    # 2.8) recent -> messages
    recent_msgs = _as_chat_messages_from_recent(
        pack.get("recent", []),
        max_pairs=int(os.getenv("TG_CONTEXT_PAIRS", "12")),
    )

    # 3) 调用模型生成回复
    reply = call_chat_llm(
        user_text=text,
        system_blocks=[ctx_for_llm, anchor_block, fact_block],
        recent_messages=recent_msgs,
    )

    # 4) 写入 DB（触发 S4/S60）
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
