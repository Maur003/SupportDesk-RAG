"""One JSON log line and Prometheus counters for each API request.

The line carries the trace id, stage times, retrieved ids, scores, and token
counts. It does not carry chunk text.
"""

import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from sqlalchemy.orm import Session

logger = logging.getLogger("supportdesk")
current_trace: ContextVar["Trace | None"] = ContextVar("current_trace", default=None)


@dataclass
class Trace:
    trace_id: str
    route: str = ""
    role: str = ""
    status: str = ""
    embed_seconds: float = 0.0
    search_seconds: float = 0.0
    rerank_seconds: float = 0.0
    llm_seconds: float = 0.0
    retrieved_ids: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    retrieved_tokens: int | None = None
    llm_input_tokens: int | None = None
    llm_output_tokens: int | None = None
    error_class: str | None = None


class AppMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "supportdesk_requests_total",
            "Logged API requests.",
            ["route", "status"],
            registry=self.registry,
        )
        self.errors = Counter(
            "supportdesk_errors_total",
            "Logged API requests that recorded an error class.",
            ["error_class"],
            registry=self.registry,
        )
        self.stage = Histogram(
            "supportdesk_stage_seconds",
            "Seconds spent in one stage of a logged request.",
            ["stage"],
            registry=self.registry,
        )

    def observe(self, trace: Trace) -> None:
        self.requests.labels(trace.route or "unknown", trace.status or "unknown").inc()
        if trace.error_class:
            self.errors.labels(trace.error_class).inc()
        self.stage.labels("embed").observe(trace.embed_seconds)
        self.stage.labels("search").observe(trace.search_seconds)
        self.stage.labels("rerank").observe(trace.rerank_seconds)
        self.stage.labels("llm").observe(trace.llm_seconds)

    def render(self) -> bytes:
        return generate_latest(self.registry)


def configure_logging() -> None:
    logger.setLevel(logging.INFO)
    if any(getattr(handler, "_supportdesk_json", False) for handler in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._supportdesk_json = True  # type: ignore[attr-defined]
    logger.addHandler(handler)


def note_route(route: str) -> None:
    trace = current_trace.get()
    if trace is not None:
        trace.route = route


def note_role(role: str) -> None:
    trace = current_trace.get()
    if trace is not None:
        trace.role = role


def note_status(status: str) -> None:
    trace = current_trace.get()
    if trace is not None:
        trace.status = status


def note_error(error_class: str) -> None:
    trace = current_trace.get()
    if trace is not None:
        trace.error_class = error_class


def add_retrieval(
    *,
    ids: list[str],
    scores: list[float],
    embed_seconds: float,
    search_seconds: float,
    rerank_seconds: float,
    retrieved_tokens: int,
) -> None:
    trace = current_trace.get()
    if trace is None:
        return
    trace.embed_seconds += embed_seconds
    trace.search_seconds += search_seconds
    trace.rerank_seconds += rerank_seconds
    trace.retrieved_ids = list(ids)
    trace.scores = list(scores)
    trace.retrieved_tokens = retrieved_tokens


def add_llm(seconds: float, input_tokens: int | None, output_tokens: int | None) -> None:
    trace = current_trace.get()
    if trace is None:
        return
    trace.llm_seconds += seconds
    if input_tokens is not None:
        trace.llm_input_tokens = (trace.llm_input_tokens or 0) + input_tokens
    if output_tokens is not None:
        trace.llm_output_tokens = (trace.llm_output_tokens or 0) + output_tokens


def payload(trace: Trace) -> dict:
    return {
        "trace_id": trace.trace_id,
        "route": trace.route,
        "role": trace.role,
        "status": trace.status,
        "embed_ms": _ms(trace.embed_seconds),
        "search_ms": _ms(trace.search_seconds),
        "rerank_ms": _ms(trace.rerank_seconds),
        "llm_ms": _ms(trace.llm_seconds),
        "retrieved_ids": list(trace.retrieved_ids),
        "scores": [round(score, 6) for score in trace.scores],
        "retrieved_tokens": trace.retrieved_tokens,
        "llm_input_tokens": trace.llm_input_tokens,
        "llm_output_tokens": trace.llm_output_tokens,
        "error_class": trace.error_class,
    }


def emit(metrics: AppMetrics, trace: Trace) -> None:
    metrics.observe(trace)
    logger.info(json.dumps(payload(trace), separators=(",", ":")))


def save_request(session: Session, trace: Trace) -> None:
    from supportdesk.models import RequestLog

    body = payload(trace)
    session.add(
        RequestLog(
            trace_id=body["trace_id"],
            route=body["route"],
            role=body["role"],
            status=body["status"],
            embed_ms=body["embed_ms"],
            search_ms=body["search_ms"],
            rerank_ms=body["rerank_ms"],
            llm_ms=body["llm_ms"],
            retrieved_tokens=body["retrieved_tokens"],
            llm_input_tokens=body["llm_input_tokens"],
            llm_output_tokens=body["llm_output_tokens"],
            retrieved_ids=body["retrieved_ids"],
            scores=body["scores"],
            error_class=body["error_class"],
        )
    )
    session.commit()


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


METRICS_MEDIA_TYPE = CONTENT_TYPE_LATEST
