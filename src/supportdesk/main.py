from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from supportdesk.access import allowed_levels
from supportdesk.agent import REFUSAL, run_agent
from supportdesk.auth import TokenError, read_role
from supportdesk.breaker import CircuitBreaker
from supportdesk.config import get_settings
from supportdesk.confirmations import ConfirmationStore
from supportdesk.crm_client import CrmClient, CrmError
from supportdesk.db import make_engine, make_session_factory
from supportdesk.embedder import BgeEmbedder
from supportdesk.llm import GeminiLLM, GuardedLLM
from supportdesk.models import Chunk
from supportdesk.reranker import MiniLMReranker
from supportdesk.retrieve import retrieve

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


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)


class AskResponse(BaseModel):
    role: str
    status: str
    steps: int
    answer: str | None = None
    confirmation_id: str | None = None
    summary: str | None = None
    account_id: str | None = None
    citations: list[Hit] = Field(default_factory=list)


def create_app(
    *,
    session_factory: sessionmaker[Session] | None = None,
    embedder: object | None = None,
    reranker: object | None = None,
    llm: object | None = None,
    crm: CrmClient | None = None,
    confirmations: ConfirmationStore | None = None,
    llm_breaker: CircuitBreaker | None = None,
    crm_breaker: CircuitBreaker | None = None,
    prefetch_retrieval: bool = True,
) -> FastAPI:
    app = FastAPI(
        title="SupportDesk RAG",
        version="0.1.0",
        summary="Synthetic Northline Telecom support agent.",
    )
    app.state.session_factory = session_factory
    app.state.embedder = embedder
    app.state.reranker = reranker
    settings = get_settings()
    app.state.llm = llm
    app.state.crm = crm
    app.state.confirmations = confirmations
    app.state.prefetch_retrieval = prefetch_retrieval
    app.state.llm_breaker = llm_breaker or CircuitBreaker(
        settings.circuit_failure_threshold,
        settings.circuit_reset_seconds,
    )
    app.state.crm_breaker = crm_breaker or CircuitBreaker(
        settings.circuit_failure_threshold,
        settings.circuit_reset_seconds,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/search")
    def search_route(
        body: SearchRequest,
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> SearchResponse:
        role = _role_from_header(credentials)
        hits = _search_hits(app, role, body.query)
        return SearchResponse(role=role, hits=[Hit(**hit) for hit in hits])

    @app.post("/ask")
    def ask_route(
        body: AskRequest,
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> AskResponse:
        role = _role_from_header(credentials)
        llm = _llm(app)
        citations = _search_hits(app, role, body.query) if app.state.prefetch_retrieval else []
        if app.state.prefetch_retrieval and not citations:
            return AskResponse(role=role, status="refused", steps=0, answer=REFUSAL)
        result = run_agent(
            body.query,
            role,
            llm=llm,
            search_fn=lambda query: _search_hits(app, role, query),
            crm=_crm(app),
            confirmations=_confirmations(app),
            citations=citations,
        )
        return AskResponse(
            role=role,
            status=result.status,
            steps=result.steps,
            answer=result.answer,
            confirmation_id=result.confirmation_id,
            summary=result.summary,
            account_id=result.account_id,
            citations=[Hit(**hit) for hit in result.citations],
        )

    @app.post("/confirmations/{confirmation_id}")
    def confirm_route(
        confirmation_id: str,
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> dict[str, str]:
        role = _role_from_header(credentials)
        store = _confirmations(app)
        try:
            item = store.get(confirmation_id, role)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="unknown confirmation") from exc
        except PermissionError as exc:
            raise HTTPException(
                status_code=403, detail="confirmation belongs to another role"
            ) from exc
        try:
            ticket = _crm(app).create_ticket(item["summary"], item["account_id"])
        except CrmError as exc:
            if str(exc) == "circuit open":
                raise HTTPException(status_code=503, detail="crm unavailable") from exc
            if str(exc) == "timeout":
                raise HTTPException(status_code=504, detail="crm timeout") from exc
            raise HTTPException(status_code=502, detail="crm create failed") from exc
        store.discard(confirmation_id)
        return ticket

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


def _llm(app: FastAPI) -> GuardedLLM:
    llm = app.state.llm
    settings = get_settings()
    if llm is not None:
        return GuardedLLM(
            llm,
            app.state.llm_breaker,
            attempts=settings.llm_retry_attempts,
            backoff=settings.llm_retry_backoff_seconds,
        )
    if not settings.gemini_api_key.strip():
        raise HTTPException(status_code=503, detail="generator is not configured")
    llm = GeminiLLM(
        api_key=settings.gemini_api_key,
        generator_model=settings.generator_model,
        judge_model=settings.judge_model,
        temperature=settings.generator_temperature,
    )
    app.state.llm = llm
    return GuardedLLM(
        llm,
        app.state.llm_breaker,
        attempts=settings.llm_retry_attempts,
        backoff=settings.llm_retry_backoff_seconds,
    )


def _crm(app: FastAPI) -> CrmClient:
    crm = app.state.crm
    if crm is None:
        settings = get_settings()
        crm = CrmClient(
            settings.crm_base_url,
            settings.crm_api_key,
            timeout=settings.crm_timeout_seconds,
            attempts=settings.crm_retry_attempts,
            backoff=settings.crm_retry_backoff_seconds,
            breaker=app.state.crm_breaker,
        )
        app.state.crm = crm
    return crm


def _confirmations(app: FastAPI) -> ConfirmationStore:
    store = app.state.confirmations
    if store is None:
        store = ConfirmationStore()
        app.state.confirmations = store
    return store


def _search_hits(app: FastAPI, role: str, query: str) -> list[dict[str, str]]:
    settings = get_settings()
    factory = _session_factory(app)
    with factory() as session:
        ids = retrieve(
            session,
            query,
            _embedder(app),
            allowed_levels(role),
            _reranker(app),
            rerank_enabled=settings.rerank_enabled,
        )
        return [hit.model_dump() for hit in _hits(session, ids)]


def _hits(session: Session, ids: list[str]) -> list[Hit]:
    if not ids:
        return []
    rows = session.scalars(select(Chunk).where(Chunk.id.in_(ids))).all()
    order = {chunk_id: index for index, chunk_id in enumerate(ids)}
    rows = sorted(rows, key=lambda row: order[row.id])
    return [Hit(id=row.id, access_level=row.access_level, text=row.text) for row in rows]


app = create_app()
