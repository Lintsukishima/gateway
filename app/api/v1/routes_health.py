from fastapi import APIRouter
from app.celery_app import celery

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/enqueue-test")
def enqueue_test():
    r = celery.send_task("app.tasks.tick")
    return {"enqueued": True, "task_id": r.id}
