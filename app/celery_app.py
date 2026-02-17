import os
from dotenv import load_dotenv
from celery import Celery
from app.core.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

celery = Celery(
    "gateway",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["app.tasks"],
)

celery.conf.update(
    task_default_queue="default",
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # 先用 beat 的最小定时：每分钟跑一次“心跳任务”
    beat_schedule={
        "scan-triggers-every-60-seconds": {
            "task": "app.tasks.scan_triggers",
            "schedule": 60.0,
      },
        "process-trigger-jobs-every-30-seconds": {
            "task": "app.tasks.process_trigger_jobs",
            "schedule": 30.0,
      }

    }
)
