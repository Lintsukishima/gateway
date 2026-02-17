from .session import engine, Base
from . import models  # noqa

def init_db():
    Base.metadata.create_all(bind=engine)
