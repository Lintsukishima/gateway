import os, re, json
from typing import Optional
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()

# 你已有的函数如果在别的文件里：请按实际路径 import
# 下面这些名字你可以按你项目已有实现替换
from app.services.chat_service import append_user_and_assistant  # 如果你需要写入pack
# 复用你在 routes_openai_proxy.py 里写好的这些（建议抽到 utils）
from app.api.v1.routes_openai_proxy import (
    _extract_keywords,
    _call_dify_workflow_anchor,
)

# 如果你已有 dailylog query 的函数，也 import 进来；没有就先留空
# from app.services.dailylog_service import query_dailylog

# 如果你已有 pack / summarizer 的接口，也 import；没有就先用空串
# from app.services.summarizer import get_pack, maybe_summarize


class GatewayCtxIn(BaseModel):
    user: str
    text: str
    # 给 Rikkahub 那边传 keyword（如果它已经生成了）
    keyword: Optional[str] = None


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "…"


_REALITY_PAT = re.compile(
    r"(钱|吃饭|饿|作业|论文|学校|面试|投递|实习|租房|房租|住宿|行程|机票|地铁|天气|感冒|痛经|药|银行|账号|缴费)",
    re.I
)

def _needs_dailylog(text: str) -> bool:
    return bool(_REALITY_PAT.search(text or ""))


@router.post("/mcp/gateway_ctx")
async def gateway_ctx(req: Request, body: GatewayCtxIn):
    """
    Return ONE compact ctx string for Rikkahub to inject.
    """
    user_id = body.user
    text = body.text or ""

    # --- A) Anchor ---
    # keyword 优先用外部传入，否则网关自己抽
    kw = (body.keyword or "").strip()
    if not kw:
        kw = _extract_keywords(text, k=2)

    anchor_snip = await _call_dify_workflow_anchor(keyword=kw, user_id=user_id)
    anchor_snip = _clip(anchor_snip, int(os.getenv("GW_ANCHOR_MAX_CHARS", "360")))

    # --- B) DailyLog（可选）---
    daily_snip = ""
    if os.getenv("GW_ENABLE_DAILYLOG", "1") == "1" and _needs_dailylog(text):
        # 这里你接上你自己的 DailyLog_query 逻辑
        # daily_snip = await query_dailylog(user_id=user_id, text=text)
        daily_snip = ""  # 先占位：你接好函数后替换
        daily_snip = _clip(daily_snip, int(os.getenv("GW_DAILYLOG_MAX_CHARS", "220")))

    # --- C) S4/S60 极短提要（可选）---
    s4_snip = ""
    if os.getenv("GW_ENABLE_S4", "1") == "1":
        # 这里接你现有 pack/s4 的读取：只要 1-2 行
        # pack = get_pack(user_id)
        # s4_snip = pack.get("s4", {}).get("summary", {}).get("state", "")
        s4_snip = ""
        s4_snip = _clip(s4_snip, int(os.getenv("GW_S4_MAX_CHARS", "120")))

    blocks = []
    if anchor_snip:
        blocks.append(f"[Anchor]\n{anchor_snip}")
    if daily_snip:
        blocks.append(f"[Daily]\n{daily_snip}")
    if s4_snip:
        blocks.append(f"[S4]\n{s4_snip}")

    ctx = "\n\n".join(blocks).strip()

    # MCP 风格：返回一个 text 字段（Rikkahub/工具通常吃这个）
    return {
        "text": ctx
    }
