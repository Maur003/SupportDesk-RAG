"""Generator client. The agent loop calls `generate`. It does not use the SDK tool loop.

`GeminiLLM` checks `client.models.list()` once before the first `generate_content`.
Both the generator id and the judge id must be present. Automatic function calling
is turned off, so a tool runs only when this process executes it.

`GuardedLLM` retries 429 and 5xx with the same delay as the CRM client. A timeout
is not retried. Those failures, and `LLMUnavailable`, open the circuit breaker.
`generate(..., model=)` selects one listed model id for that call. The default
is the generator.
"""

import random
import time
from collections.abc import Callable
from dataclasses import dataclass

from supportdesk.breaker import RETRYABLE_STATUS, CircuitBreaker, retry_delay
from supportdesk.observe import add_llm

TOOL_PARAMETERS = {
    "search_knowledge_base": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    "get_ticket_status": {
        "type": "object",
        "properties": {"ticket_id": {"type": "string"}},
        "required": ["ticket_id"],
    },
    "create_ticket": {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    },
    "calculate": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
}

TOOL_DESCRIPTIONS = {
    "search_knowledge_base": (
        "Search the Northline knowledge base. Results are already limited to the caller's role."
    ),
    "get_ticket_status": (
        "Read one ticket from the mock CRM by its id. The ticket must belong to the caller."
    ),
    "create_ticket": (
        "Ask the server to prepare a ticket for the caller's account. This does not write "
        "the ticket. The server asks the user to confirm. A confirmed flag is ignored."
    ),
    "calculate": "Evaluate + - * / on numbers and parentheses. No other syntax.",
}


class ModelError(Exception):
    pass


class LLMUnavailable(Exception):
    """The generator could not be called. The API returns citations instead of an answer."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict
    call_id: str | None = None
    thought_signature: bytes | None = None


@dataclass(frozen=True)
class ModelTurn:
    text: str
    tool_call: ToolCall | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class GeminiLLM:
    def __init__(
        self,
        *,
        api_key: str,
        generator_model: str,
        judge_model: str,
        temperature: float = 0.0,
        client: object | None = None,
    ) -> None:
        self._generator = generator_model
        self._judge = judge_model
        self._temperature = temperature
        self._models_ok = False
        self._names: list[str] = []
        if client is None:
            from google import genai

            client = genai.Client(api_key=api_key)
        self._client = client

    def generate(self, messages: list[dict], model: str | None = None) -> ModelTurn:
        self._ensure_models()
        chosen = model or self._generator
        if not _listed(self._names, chosen):
            raise ModelError(f"models.list() did not include {chosen}")
        from google.genai import types

        response = self._client.models.generate_content(
            model=chosen,
            contents=_contents(messages),
            config=types.GenerateContentConfig(
                temperature=self._temperature,
                tools=[types.Tool(function_declarations=_declarations())],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        return _turn(response)

    def _ensure_models(self) -> None:
        if self._models_ok:
            return
        names = _model_names(self._client.models.list())
        missing = [
            model_id for model_id in (self._generator, self._judge) if not _listed(names, model_id)
        ]
        if missing:
            joined = ", ".join(missing)
            raise ModelError(f"models.list() did not include {joined}")
        self._names = names
        self._models_ok = True


class GuardedLLM:
    def __init__(
        self,
        inner: object,
        breaker: CircuitBreaker,
        *,
        attempts: int = 3,
        backoff: float = 0.2,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self._inner = inner
        self._breaker = breaker
        self._attempts = attempts
        self._backoff = backoff
        self._sleep = sleep
        self._rng = rng

    def generate(self, messages: list[dict], model: str | None = None) -> ModelTurn:
        started = time.perf_counter()
        try:
            return self._generate(messages, model)
        finally:
            add_llm(time.perf_counter() - started, None, None)

    def _generate(self, messages: list[dict], model: str | None) -> ModelTurn:
        if not self._breaker.allow():
            raise LLMUnavailable("circuit open")
        last: Exception | None = None
        for attempt in range(self._attempts):
            try:
                if model is None:
                    turn = self._inner.generate(messages)
                else:
                    turn = self._inner.generate(messages, model=model)
            except ModelError:
                raise
            except LLMUnavailable:
                self._breaker.record_failure()
                raise
            except Exception as exc:
                status = _status_code(exc)
                if status in RETRYABLE_STATUS and attempt + 1 < self._attempts:
                    last = exc
                    self._sleep(retry_delay(self._backoff, attempt, self._rng))
                    continue
                if status in RETRYABLE_STATUS or _is_timeout(exc):
                    self._breaker.record_failure()
                    detail = "timeout" if _is_timeout(exc) else "unavailable"
                    raise LLMUnavailable(detail) from exc
                raise
            else:
                if not isinstance(turn, ModelTurn):
                    raise TypeError("llm.generate must return a ModelTurn")
                self._breaker.record_success()
                add_llm(0.0, turn.input_tokens, turn.output_tokens)
                return turn
        self._breaker.record_failure()
        raise LLMUnavailable("unavailable") from last


def _status_code(exc: Exception) -> int | None:
    for name in ("status_code", "code"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            return value
    return None


def _is_timeout(exc: Exception) -> bool:
    return isinstance(exc, TimeoutError) or exc.__class__.__name__.endswith("TimeoutException")


def _declarations() -> list:
    from google.genai import types

    return [
        types.FunctionDeclaration(
            name=name,
            description=TOOL_DESCRIPTIONS[name],
            parameters_json_schema=TOOL_PARAMETERS[name],
        )
        for name in TOOL_PARAMETERS
    ]


def _contents(messages: list[dict]) -> list:
    from google.genai import types

    contents = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            payload = message["content"]
            if isinstance(payload, dict) and "error" in payload:
                response = {"error": payload["error"]}
            else:
                response = {"output": payload}
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=message["name"],
                                id=message.get("call_id"),
                                response=response,
                            )
                        )
                    ],
                )
            )
            continue
        parts = []
        if message.get("text"):
            parts.append(types.Part(text=message["text"]))
        tool_call = message.get("tool_call")
        if tool_call is not None:
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        name=tool_call.name,
                        args=tool_call.arguments,
                        id=tool_call.call_id,
                    ),
                    thought_signature=tool_call.thought_signature,
                )
            )
        contents.append(types.Content(role="model" if role == "model" else "user", parts=parts))
    return contents


def _turn(response: object) -> ModelTurn:
    calls = getattr(response, "function_calls", None) or []
    text = getattr(response, "text", None) or ""
    usage = getattr(response, "usage_metadata", None)
    input_tokens = _usage_count(usage, "prompt_token_count")
    output_tokens = _usage_count(usage, "candidates_token_count")
    if not calls:
        return ModelTurn(text=text, input_tokens=input_tokens, output_tokens=output_tokens)
    call = calls[0]
    return ModelTurn(
        text=text,
        tool_call=ToolCall(
            name=call.name or "",
            arguments=dict(call.args or {}),
            call_id=call.id,
            thought_signature=_thought_signature(response),
        ),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _thought_signature(response: object) -> bytes | None:
    """Gemini 3 requires this opaque blob on the function-call part we send back."""

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    signed = [part for part in parts if isinstance(getattr(part, "thought_signature", None), bytes)]
    for part in signed:
        if getattr(part, "function_call", None) is not None and part.thought_signature:
            return part.thought_signature
    for part in signed:
        if part.thought_signature:
            return part.thought_signature
    return None


def _usage_count(usage: object, name: str) -> int | None:
    if usage is None:
        return None
    value = getattr(usage, name, None)
    if isinstance(value, int):
        return value
    return None


def _model_names(listed: object) -> list[str]:
    names = []
    for model in listed:
        name = getattr(model, "name", None)
        if isinstance(name, str):
            names.append(name)
    return names


def _listed(names: list[str], model_id: str) -> bool:
    return any(name == model_id or name.endswith("/" + model_id) for name in names)
