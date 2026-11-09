from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from supportdesk.access import allowed_levels
from supportdesk.auth import TokenError, read_role
from supportdesk.config import get_settings
from supportdesk.db import make_engine, make_session_factory
from supportdesk.embedder import BgeEmbedder
from supportdesk.models import Chunk
from supportdesk.reranker import MiniLMReranker
from supportdesk.retrieve import search

_bearer = HTTPBearer(auto_error=False)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)


class Hit(BaseModel):
    id: str
    access_level: str
    text: str


class SearchResponse(BaseModel):
    role: str
    hits: list[Hit]


def create_app(
    *,
    session_factory: sessionmaker[Session] | None = None,
    embedder: object | None = None,
    reranker: object | None = None,
) -> FastAPI:
    app = FastAPI(
        title="SupportDesk RAG",
        version="0.1.0",
        summary="Synthetic Northline Telecom support agent.",
    )
    app.state.session_factory = session_factory
    app.state.embedder = embedder
    app.state.reranker = reranker

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/search")
    def search_route(
        body: SearchRequest,
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> SearchResponse:
        role = _role_from_header(credentials)
        settings = get_settings()
        factory = _session_factory(app)
        query_embedder = _embedder(app)
        query_reranker = _reranker(app)
        with factory() as session:
            ids = search(
                session,
                body.query,
                query_embedder.embed_query(body.query),
                allowed_levels(role),
                query_reranker,
                rerank_enabled=settings.rerank_enabled,
            )
            hits = _hits(session, ids)
        return SearchResponse(role=role, hits=hits)

    return app


def _role_from_header(credentials: HTTPAuthorizationCredentials | None) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="missing bearer token")
    try:
        return read_role(credentials.credentials, get_settings().jwt_secret)
    except TokenError as exc:
        raise HTTPException(status_code=401, detail="invalid bearer token") from exc


def _session_factory(app: FastAPI) -> sessionmaker[Session]:
    factory = app.state.session_factory
    if factory is None:
        factory = make_session_factory(make_engine())
        app.state.session_factory = factory
    return factory


def _embedder(app: FastAPI) -> BgeEmbedder:
    embedder = app.state.embedder
    if embedder is None:
        settings = get_settings()
        embedder = BgeEmbedder(settings.embedding_model, settings.embedding_revision)
        app.state.embedder = embedder
    return embedder


def _reranker(app: FastAPI) -> MiniLMReranker:
    reranker = app.state.reranker
    if reranker is None:
        settings = get_settings()
        reranker = MiniLMReranker(settings.rerank_model, settings.rerank_revision)
        app.state.reranker = reranker
    return reranker


def _hits(session: Session, ids: list[str]) -> list[Hit]:
    if not ids:
        return []
    rows = session.scalars(select(Chunk).where(Chunk.id.in_(ids))).all()
    order = {chunk_id: index for index, chunk_id in enumerate(ids)}
    rows = sorted(rows, key=lambda row: order[row.id])
    return [Hit(id=row.id, access_level=row.access_level, text=row.text) for row in rows]


app = create_app()
