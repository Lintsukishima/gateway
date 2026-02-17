import os
import requests

def send_telegram_message(text: str, chat_id: str | None = None) -> dict:
    token = os.getenv("TG_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing env TG_BOT_TOKEN")

    cid = (chat_id or os.getenv("TG_CHAT_ID", "")).strip()
    if not cid:
        raise RuntimeError("Missing env TG_CHAT_ID")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": cid,
        "text": text,
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    return r.json()
