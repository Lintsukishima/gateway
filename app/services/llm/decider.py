import os
import json
import re
import requests

JSON_RE = re.compile(r"\{.*\}", re.S)

def _extract_json(text: str) -> dict:
    """
    尽量从模型输出里抠出一个 JSON 对象（兼容它啰嗦几句）。
    """
    if not text:
        raise ValueError("empty model output")
    m = JSON_RE.search(text)
    if not m:
        raise ValueError(f"no json found in output: {text[:200]}")
    obj = json.loads(m.group(0))
    return obj

def call_openrouter(messages: list[dict], model: str) -> str:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing env LLM_API_KEY")

    url = "https://api.siliconflow.cn/v1/chat/completions" 
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP", "gateway"),
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 220,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def decide_message(trigger_type: str, session_id: str, context_text: str) -> dict:
    """
    返回格式（严格）:
    {
      "decision": "send" | "skip",
      "text": "给人看的1-3句短消息",
      "reason": "一句理由(用于存档)"
    }
    """
    model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-14B-Instruct")
    provider = os.getenv("LLM_PROVIDER", "siliconflow")

    system = (
        "你是一个贴心的伴侣，正在给你的恋人发送一条简短的消息。请根据对话摘要和最近聊天内容，决定是否发送一条温馨的问候或提醒，并生成消息文本。\n"
        "你只输出一个JSON对象，不能输出任何多余文字。\n"
        "规则：\n"
        "1) 决策 decision 只能是 \"send\" 或 \"skip\"。\n"
        "2) text 必须是中文口语化、非常简短（1-3句），语气亲近、温柔，像恋人之间自然的聊天。不要使用括号或动作描写，不要列点，不要像系统提示。\n"
        "3) 如果觉得对方可能不希望被打扰，或者没有合适的话题，就 decision=\"skip\"。\n"
        "4) 参考上下文（对话摘要和最近对话），用温柔自然的语气生成消息。\n"
        "示例（仅供参考，不要照抄）：\n"
        "  - 如果对方很久没说话，可以发：“在忙什么呢？想你了~”\n"
        "  - 如果刚刚聊过某个话题，可以延续：“刚才说到的那家店，周末一起去试试？”\n"
        "  - 如果是提醒类：“记得多喝水哦，爱你❤️”\n"
    )

    user = (
        f"触发类型: {trigger_type}\n"
        f"session_id: {session_id}\n"
        f"上下文(仅供你判断，不要原样复述):\n{context_text}\n\n"
        "现在输出JSON："
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    # 调用模型
    if provider in ("openrouter", "siliconflow"):
        raw = call_openrouter(messages, model=model)
    else:
        raise RuntimeError(f"Unsupported LLM_PROVIDER={provider}")

    # 解析 JSON
    obj = _extract_json(raw)

    # 兜底校验
    decision = obj.get("decision", "skip")
    text = (obj.get("text") or "").strip()
    reason = (obj.get("reason") or "").strip()

    if decision not in ("send", "skip"):
        decision = "skip"

    if decision == "send":
        if not text:
            decision = "skip"
            reason = reason or "empty text"
        elif len(text) > 300:
            text = text[:300] + "…"

    return {"decision": decision, "text": text, "reason": reason, "model": model}