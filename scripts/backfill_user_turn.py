import sys
import os
# 把项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import SessionLocal
from app.db.models import Message

def main():
    db = SessionLocal()
    try:
        session_ids = [r[0] for r in db.query(Message.session_id).distinct().all()]
        for sid in session_ids:
            msgs = (db.query(Message)
                      .filter(Message.session_id == sid)
                      .order_by(Message.turn_id.asc())
                      .all())
            ut = 0
            for m in msgs:
                if m.role == "user":
                    ut += 1
                    m.user_turn = ut
                else:
                    # assistant/system 也写一个值，方便排序/范围查询
                    m.user_turn = ut
            db.commit()
        print("backfill done")
    finally:
        db.close()

if __name__ == "__main__":
    main()
