import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session


@pytest.fixture(scope="session")
def database_url() -> str:
    from supportdesk.config import get_settings

    url = make_url(get_settings().database_url).set(database="supportdesk_test")
    admin = create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as connection:
            connection.execute(text("SELECT 1"))
            exists = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = 'supportdesk_test'")
            ).scalar()
            if exists is None:
                connection.execute(text("CREATE DATABASE supportdesk_test"))
    except Exception as exc:
        pytest.skip(f"Postgres is not reachable: {exc}")
    finally:
        admin.dispose()
    return url.render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def migrated_engine(database_url: str):
    import os

    from alembic import command
    from alembic.config import Config

    from supportdesk.config import get_settings

    os.environ["DATABASE_URL"] = database_url
    get_settings.cache_clear()
    config = Config("alembic.ini")
    command.upgrade(config, "head")
    engine = create_engine(database_url, pool_pre_ping=True)
    yield engine
    engine.dispose()
    get_settings.cache_clear()


@pytest.fixture
def db_session(migrated_engine) -> Session:
    with Session(migrated_engine) as session:
        session.execute(
            text("TRUNCATE TABLE request_log, chunks, documents RESTART IDENTITY CASCADE")
        )
        session.commit()
        yield session
        session.rollback()
