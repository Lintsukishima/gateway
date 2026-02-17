import json
from datetime import datetime, timezone
from app.db.session import SessionLocal
from app.db.models import TriggerJob

# 创建一个数据库会话
db = SessionLocal()

# 构造一个示例 payload（fake_decider 会用到）
payload = {
    "silence_minutes": 10.5,
    "threshold_minutes": 240,
    "cooldown_minutes": 120
}

# 创建 TriggerJob 对象
job = TriggerJob(
    session_id="09999",            # 替换成一个真实存在的 session_id
    trigger_type="silence",
    trigger_payload_json=json.dumps(payload, ensure_ascii=False),
    status="queued",
    # created_at 会自动生成，不用手动赋值
)

# 添加到数据库并提交
db.add(job)
db.commit()

# 查看刚刚创建的 job ID
print(f"Created job id: {job.id}")

# 关闭会话
db.close()