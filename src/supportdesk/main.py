import json
import time
import uuid
from datetime import date
from decimal import Decimal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from supportdesk.access import allowed_levels
from supportdesk.agent import REFUSAL, run_agent
from supportdesk.auth import TokenError, read_role
from supportdesk.breaker import CircuitBreaker
from supportdesk.cache import (
    CacheHit,
    access_scope,
    exact_cache_key,
    lookup_exact,
    lookup_semantic,
    store_answer,
)
from supportdesk.compress import ContextChunk, compress_context
from supportdesk.config import get_settings
from supportdesk.confirmations import ConfirmationStore
from supportdesk.crm_client import CrmClient, CrmError
from supportdesk.db import make_engine, make_session_factory
from supportdesk.embedder import BgeEmbedder, EmbedderError
from supportdesk.llm import GeminiLLM, GuardedLLM, ModelTurn
from supportdesk.models import Chunk
from supportdesk.observe import (
    METRICS_MEDIA_TYPE,
    AppMetrics,
    Trace,
    add_embed,
    add_retrieval,
    configure_logging,
    current_trace,
    emit,
    note_cache,
    note_cost,
    note_error,
    note_model,
    note_role,
    note_route,
    note_status,
    save_request,
)
from supportdesk.pricing import cost_usd
from supportdesk.reranker import MiniLMReranker
from supportdesk.retrieve import Retrieval, timed_retrieve
from supportdesk.routing import choose_model

_UNSET = object()

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
    model: str | None = None
    cache_hit: str | None = None


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
    max_context_tokens: int | None = None,
    semantic_cache_distance: float | None = None,
    min_context_score: object = _UNSET,
    price_as_of: date | None = None,
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
    app.state.max_context_tokens = (
        settings.max_context_tokens if max_context_tokens is None else max_context_tokens
    )
    app.state.semantic_cache_distance = (
        settings.semantic_cache_distance
        if semantic_cache_distance is None
        else semantic_cache_distance
    )
    app.state.min_context_score = (
        settings.min_context_score if min_context_score is _UNSET else min_context_score
    )
    app.state.price_as_of = price_as_of
    app.state.llm_breaker = llm_breaker or CircuitBreaker(
        settings.circuit_failure_threshold,
        settings.circuit_reset_seconds,
    )
    app.state.crm_breaker = crm_breaker or CircuitBreaker(
        settings.circuit_failure_threshold,
        settings.circuit_reset_seconds,
    )
    app.state.metrics = AppMetrics()
    configure_logging()

    @app.middleware("http")
    async def trace_request(request, call_next):
        if request.url.path in {"/health", "/metrics"}:
            return await call_next(request)
        header = request.headers.get("x-trace-id", "")
        trace = Trace(trace_id=header.strip()[:128] or uuid.uuid4().hex)
        token = current_trace.set(trace)
        try:
            response = await call_next(request)
        except Exception as exc:
            trace.error_class = trace.error_class or type(exc).__name__
            trace.status = trace.status or "error"
            _finish(app, trace)
            current_trace.reset(token)
            raise
        if not trace.status:
            trace.status = str(response.status_code)
        if response.status_code >= 400 and not trace.error_class:
            trace.error_class = f"http_{response.status_code}"
        _finish(app, trace)
        response.headers["x-trace-id"] = trace.trace_id
        current_trace.reset(token)
        return response

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(content=app.state.metrics.render(), media_type=METRICS_MEDIA_TYPE)

    @app.post("/search")
    def search_route(
        body: SearchRequest,
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> SearchResponse:
        note_route("search")
        role = _role_from_header(credentials)
        note_role(role)
        hits = _search_hits(app, role, body.query)
        note_status("ok")
        return SearchResponse(role=role, hits=[Hit(**hit) for hit in hits])

    @app.post("/ask")
    def ask_route(
        body: AskRequest,
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> AskResponse:
        note_route("ask")
        role = _role_from_header(credentials)
        note_role(role)
        llm = _llm(app)
        settings = get_settings()
        model_id = choose_model(
            body.query,
            simple_max_chars=settings.simple_query_max_chars,
            cheap_model=settings.cheap_model,
            generator_model=settings.generator_model,
        )
        note_model(model_id)
        cached, query_vector = _read_cache(app, role, body.query, model_id)
        if cached is not None:
            note_status("answered")
            note_cache(cached.kind)
            note_cost(Decimal(0))
            return AskResponse(
                role=role,
                status="answered",
                steps=0,
                answer=cached.answer,
                model=model_id,
                cache_hit=cached.kind,
            )
        citations = (
            _search_hits(app, role, body.query, compress=True, query_vector=query_vector)
            if app.state.prefetch_retrieval
            else []
        )
        if app.state.prefetch_retrieval and not citations:
            note_status("refused")
            return AskResponse(role=role, status="refused", steps=0, answer=REFUSAL, model=model_id)
        result = run_agent(
            body.query,
            role,
            llm=_bound_model(llm, model_id),
            search_fn=lambda query: _search_hits(app, role, query, compress=True),
            crm=_crm(app),
            confirmations=_confirmations(app),
            citations=citations,
        )
        note_status(result.status)
        if result.status == "unavailable":
            note_error("LLMUnavailable")
        _note_generation_cost(app, model_id)
        if result.status == "answered" and result.answer:
            _store_cache(app, role, body.query, model_id, query_vector, result.answer, result.steps)
        return AskResponse(
            role=role,
            status=result.status,
            steps=result.steps,
            answer=result.answer,
            confirmation_id=result.confirmation_id,
            summary=result.summary,
            account_id=result.account_id,
            citations=[Hit(**hit) for hit in result.citations],
            model=model_id,
        )

    @app.post("/confirmations/{confirmation_id}")
    def confirm_route(
        confirmation_id: str,
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> dict[str, str]:
        note_route("confirm")
        role = _role_from_header(credentials)
        note_role(role)
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
            note_error("CrmError")
            if str(exc) == "circuit open":
                raise HTTPException(status_code=503, detail="crm unavailable") from exc
            if str(exc) == "timeout":
                raise HTTPException(status_code=504, detail="crm timeout") from exc
            raise HTTPException(status_code=502, detail="crm create failed") from exc
        store.discard(confirmation_id)
        note_status("created")
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


def _search_hits(
    app: FastAPI,
    role: str,
    query: str,
    *,
    compress: bool = False,
    query_vector: list[float] | None = None,
) -> list[dict[str, str]]:
    settings = get_settings()
    factory = _session_factory(app)
    with factory() as session:
        found = timed_retrieve(
            session,
            query,
            _embedder(app),
            allowed_levels(role),
            _reranker(app),
            rerank_enabled=settings.rerank_enabled,
            query_vector=query_vector,
        )
        context = _context(session, found)
        add_retrieval(
            ids=found.ids,
            scores=found.scores,
            embed_seconds=found.embed_seconds,
            search_seconds=found.search_seconds,
            rerank_seconds=found.rerank_seconds,
            retrieved_tokens=sum(item.tokens for item in context),
        )
        chosen = context
        if compress:
            chosen = compress_context(
                context,
                max_tokens=app.state.max_context_tokens,
                min_score=app.state.min_context_score,
            )
        return [
            {"id": item.id, "access_level": item.access_level, "text": item.text} for item in chosen
        ]


def _context(session: Session, found: Retrieval) -> list[ContextChunk]:
    if not found.ids:
        return []
    rows = session.scalars(select(Chunk).where(Chunk.id.in_(found.ids))).all()
    order = {chunk_id: index for index, chunk_id in enumerate(found.ids)}
    scores = dict(zip(found.ids, found.scores, strict=True))
    rows = sorted(rows, key=lambda row: order[row.id])
    return [
        ContextChunk(
            id=row.id,
            access_level=row.access_level,
            text=row.text,
            score=float(scores[row.id]),
            tokens=row.token_count,
        )
        for row in rows
    ]


def _cache_enabled(app: FastAPI) -> bool:
    return app.state.session_factory is not None or app.state.prefetch_retrieval


def _read_cache(
    app: FastAPI, role: str, query: str, model_id: str
) -> tuple[CacheHit | None, list[float] | None]:
    if not _cache_enabled(app):
        return None, None
    scope = access_scope(role)
    factory = _session_factory(app)
    key = exact_cache_key(query, scope, model_id)
    with factory() as session:
        hit = lookup_exact(session, key, scope, model_id)
    if hit is not None:
        return hit, None
    vector = _query_vector(app, query)
    if vector is None:
        return None, None
    with factory() as session:
        hit = lookup_semantic(session, vector, scope, model_id, app.state.semantic_cache_distance)
    return hit, vector


def _query_vector(app: FastAPI, query: str) -> list[float] | None:
    started = time.perf_counter()
    try:
        vector = _embedder(app).embed_query(query)
    except EmbedderError:
        add_embed(time.perf_counter() - started)
        return None
    add_embed(time.perf_counter() - started)
    return vector


def _store_cache(
    app: FastAPI,
    role: str,
    query: str,
    model_id: str,
    vector: list[float] | None,
    answer: str,
    steps: int,
) -> None:
    if not _cache_enabled(app):
        return
    factory = _session_factory(app)
    with factory() as session:
        store_answer(
            session,
            query=query,
            scope=access_scope(role),
            model=model_id,
            embedding=vector,
            answer=answer,
            steps=steps,
        )


def _note_generation_cost(app: FastAPI, model_id: str) -> None:
    trace = current_trace.get()
    if trace is None:
        return
    if trace.llm_input_tokens is None and trace.llm_output_tokens is None:
        return
    on = app.state.price_as_of or date.today()
    try:
        amount = cost_usd(
            model_id,
            trace.llm_input_tokens or 0,
            trace.llm_output_tokens or 0,
            on,
        )
    except ValueError:
        return
    note_cost(amount)


class _PinnedModel:
    def __init__(self, guarded: GuardedLLM, model_id: str) -> None:
        self._guarded = guarded
        self.model_id = model_id

    def generate(self, messages: list[dict]) -> ModelTurn:
        return self._guarded.generate(messages, model=self.model_id)


def _bound_model(llm: GuardedLLM, model_id: str) -> object:
    if isinstance(llm._inner, GeminiLLM):
        return _PinnedModel(llm, model_id)
    return llm


def _finish(app: FastAPI, trace: Trace) -> None:
    emit(app.state.metrics, trace)
    factory = app.state.session_factory
    if factory is None or not trace.route:
        return
    try:
        with factory() as session:
            save_request(session, trace)
    except Exception:
        from supportdesk.observe import logger

        logger.warning(json.dumps({"trace_id": trace.trace_id, "error_class": "request_log_write"}))


app = create_app()
