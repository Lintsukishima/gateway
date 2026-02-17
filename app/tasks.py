
import json
import os
from app.integrations.telegram import send_telegram_message
from datetime import datetime, timezone
from app.celery_app import celery
from app.db.session import SessionLocal
from app.db.models import Message, TriggerJob
#from app.db.models import Session as ChatSession
from app.db.models import TriggerJob, OutboxMessage, SummaryS4, SummaryS60, Message
from app.db.models import Message, SummaryS4, SummaryS60
from app.integrations.decider_llm import decide_message



def utc_now():
    return datetime.now(timezone.utc)


@celery.task(name="app.tasks.tick")
def tick():
    now = utc_now().isoformat()
    print(f"[tick] {now}")
    return {"ok": True, "ts": now}

@celery.task(name="app.tasks.scan_triggers")
def scan_triggers():
    """
    MVP: 每分钟扫描一次
    规则：如果某 session 最后一次 user 消息距离现在 > threshold_minutes
         且最近 X 分钟内没创建过 silence job，则创建一个 TriggerJob
    """
    from app.db.models import Session as ChatSession  # 如果还没导入就加上

    db = SessionLocal()
    try:
        # 查询所有启用了 proactive 的 session
        sessions = db.query(ChatSession).filter(ChatSession.proactive_enabled == True).all()
        created = 0

        for s in sessions:
            sid = s.id
            threshold_minutes = s.silence_threshold_min or 240
            cooldown_minutes = s.silence_cooldown_min or 120

            # 1. 最后一次 user 消息
            last_user = (db.query(Message)
                         .filter(Message.session_id == sid, Message.role == "user")
                         .order_by(Message.turn_id.desc())
                         .first())
            if not last_user:
                continue

            # 2. 计算静默时长（分钟）
            now_naive = utc_now().replace(tzinfo=None)
            silence_minutes = (now_naive - last_user.created_at).total_seconds() / 60.0
            if silence_minutes < threshold_minutes:
                continue

            # 3. 检查冷却期（只看有效状态的任务）
            recent_job = (db.query(TriggerJob)
                          .filter(
                              TriggerJob.session_id == sid,
                              TriggerJob.trigger_type == "silence",
                              TriggerJob.status.in_(["queued", "running", "done"])
                          )
                          .order_by(TriggerJob.created_at.desc())
                          .first())
            if recent_job:
                delta = (now_naive - recent_job.created_at).total_seconds() / 60.0
                if delta < cooldown_minutes:
                    continue

            # 4. 创建新 job
            payload = {
                "silence_minutes": round(silence_minutes, 1),
                "threshold_minutes": threshold_minutes,
                "cooldown_minutes": cooldown_minutes,
            }
            job = TriggerJob(
                session_id=sid,
                trigger_type="silence",
                trigger_payload_json=json.dumps(payload, ensure_ascii=False),
                status="queued",
            )
            db.add(job)
            created += 1

        db.commit()
        if created:
            print(f"[scan_triggers] created {created} silence job(s)")
        return {"created": created}

    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()


def build_brief(db, session_id: str) -> dict:
    s4 = (db.query(SummaryS4).filter(SummaryS4.session_id == session_id)
          .order_by(SummaryS4.to_turn.desc()).first())
    s60 = (db.query(SummaryS60).filter(SummaryS60.session_id == session_id)
           .order_by(SummaryS60.to_turn.desc()).first())
    recent = (db.query(Message).filter(Message.session_id == session_id)
              .order_by(Message.turn_id.desc()).limit(12).all()[::-1])

    return {
        "s60": json.loads(s60.summary_json) if s60 else None,
        "s4": json.loads(s4.summary_json) if s4 else None,
        "recent": [{"role": m.role, "turn": m.turn_id, "content": m.content} for m in recent],
    }

def fake_decider(trigger_type: str, payload: dict, brief: dict) -> dict:
    """
    MVP：先写死逻辑，等你接模型时把它替换成 llm_decider()
    """
    if trigger_type == "silence":
        mins = payload.get("silence_minutes")
        text = f"（自动提醒测试）你已经沉默 {mins} 分钟了，我在。"
        return {"decision": "send", "message_text": text, "trace": {"mode": "fake"}}
    return {"decision": "skip", "message_text": None, "trace": {"mode": "fake"}}

import json
from app.db.models import Message, SummaryS4, SummaryS60


def build_telegram_text(db, session_id: str, trigger_type: str, recent: int = 6) -> str:
    # 只取窗口=16 的最新短期
    s4 = (db.query(SummaryS4)
          .filter(SummaryS4.session_id == session_id)
          .filter((SummaryS4.to_turn - SummaryS4.from_turn + 1) == 16)
          .order_by(SummaryS4.to_turn.desc())
          .first())

    # 只取窗口=60 的最新长期
    s60 = (db.query(SummaryS60)
           .filter(SummaryS60.session_id == session_id)
           .filter((SummaryS60.to_turn - SummaryS60.from_turn + 1) == 60)
           .order_by(SummaryS60.to_turn.desc())
           .first())

    # 最近几条原文（只取 user/assistant，避免 system 混进来）
    msgs = (db.query(Message)
            .filter(Message.session_id == session_id)
            .filter(Message.role.in_(["user", "assistant"]))
            .order_by(Message.turn_id.desc())
            .limit(recent)
            .all())[::-1]

    def _fmt_summary(row):
        if not row:
            return "（暂无）"
        try:
            obj = json.loads(row.summary_json)
        except Exception:
            obj = row.summary_json
        return json.dumps(obj, ensure_ascii=False)

    def _fmt_msgs(rows):
        lines = []
        for m in rows:
            role = "U" if m.role == "user" else "A"
            txt = (m.content or "").strip().replace("\n", " ")
            if len(txt) > 120:
                txt = txt[:120] + "…"
            lines.append(f"{role}{m.user_turn or ''}: {txt}")
        return "\n".join(lines) if lines else "（暂无）"

    text = (
        f"[gateway] trigger={trigger_type} session={session_id}\n"
        f"\n[S4 latest window=16]\n{_fmt_summary(s4)}\n"
        f"\n[S60 latest window=60]\n{_fmt_summary(s60)}\n"
        f"\n[recent]\n{_fmt_msgs(msgs)}"
    )

    # Telegram 单条消息长度上限约 4096，这里做个硬裁剪
    if len(text) > 3500:
        text = text[:3500] + "\n…(truncated)"
    return text


@celery.task(name="app.tasks.process_trigger_jobs")
def process_trigger_jobs(limit: int = 5):
    db = SessionLocal()
    processed = 0

    def _now_naive():
        return utc_now().replace(tzinfo=None)

    try:
        jobs = (
            db.query(TriggerJob)
            .filter(TriggerJob.status == "queued")
            .order_by(TriggerJob.scheduled_at.asc())
            .limit(limit)
            .all()
        )

        for job in jobs:
            processed += 1

            # ---- mark running ----
            job.status = "running"
            job.started_at = _now_naive()
            job.attempts = (job.attempts or 0) + 1
            db.add(job)
            db.commit()

            # ---- build context for decider ----
            context_text = build_telegram_text(db, job.session_id, job.trigger_type, recent=6)

            # ---- decide (LLM) ----
            try:
                decision_obj = decide_message(
                    trigger_type=job.trigger_type,
                    session_id=job.session_id,
                    context_text=context_text,
                )
                decision = decision_obj.get("decision", "skip")
                text = (decision_obj.get("text") or "").strip()
                reason = (decision_obj.get("reason") or "").strip()
                model = (decision_obj.get("model") or "unknown").strip()
            except Exception as e:
                # 决策失败也要落库，别让任务静默
                decision = "skip"
                text = ""
                reason = f"decider_error: {repr(e)}"
                model = "unknown"

            # ---- create outbox record ----
            out = OutboxMessage(
                job_id=job.id,
                channel="telegram",
                recipient=os.getenv("TG_CHAT_ID", "").strip(),
                decision=decision,
                message_text=text if decision == "send" else None,
                status="pending",
                created_at=_now_naive(),
                model_trace_json=json.dumps(
                    {"model": model, "reason": reason, "decision": decision},
                    ensure_ascii=False
                ),
            )
            db.add(out)
            db.commit()
            db.refresh(out)

            # ---- send if needed ----
            try:
                if decision == "send" and out.recipient and text:
                    send_telegram_message(text, chat_id=out.recipient)
                    out.status = "sent"
                    out.sent_at = _now_naive()
                else:
                    out.status = "sent" if decision == "skip" else "failed"
                    if decision != "skip":
                        # 走到这里通常是 chat_id 空 / text 空
                        out.model_trace_json = json.dumps(
                            {"model": model, "reason": reason, "decision": decision, "warn": "missing chat_id or text"},
                            ensure_ascii=False
                        )

                # job done
                job.status = "done" if out.status == "sent" else "failed"
                job.finished_at = _now_naive()
                job.last_error = None if out.status == "sent" else (out.model_trace_json or "send_failed")

                db.add(out)
                db.add(job)
                db.commit()

                print(f"[process_trigger_jobs] job={job.id} decision={decision} out_status={out.status}")

            except Exception as e:
                out.status = "failed"
                out.model_trace_json = json.dumps(
                    {"error": repr(e), "decision": decision, "model": model},
                    ensure_ascii=False
                )
                job.status = "failed"
                job.finished_at = _now_naive()
                job.last_error = repr(e)

                db.add(out)
                db.add(job)
                db.commit()

                print(f"[process_trigger_jobs] job={job.id} FAILED err={repr(e)}")

        return {"processed": processed}

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
























    
