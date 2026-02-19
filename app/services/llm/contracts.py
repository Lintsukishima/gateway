import httpx
import os

# 从环境变量读取配置
SUM_BASE_URL = os.getenv("SUM_BASE_URL", "https://api.openai.com")
SUM_API_KEY = os.getenv("SUM_API_KEY", "")
SUM_MODEL = os.getenv("SUM_MODEL", "gpt-4o-mini")

def call_llm(system: str, user: str, json_mode: bool = True) -> dict:
    """统一调用LLM的函数，返回解析后的JSON"""
    headers = {
        "Authorization": f"Bearer {SUM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": SUM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    with httpx.Client(timeout=30) as client:
        resp = client.post(f"{SUM_BASE_URL}/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return json.loads(content) if json_mode else {"text": content}