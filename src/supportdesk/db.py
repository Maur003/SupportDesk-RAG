from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from supportdesk.config import get_settings


def make_engine(database_url: str | None = None) -> Engine:
    url = database_url if database_url is not None else get_settings().database_url
    return create_engine(url, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
