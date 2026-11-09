"""Manual tool loop. At most five model calls. The process executes each tool.

`create_ticket` does not write. It stores a confirmation and stops. A later request
from the user, not another model call, performs the CRM write.
"""

from dataclasses import dataclass, field

from supportdesk.arithmetic import calculate
from supportdesk.crm_client import CrmClient, CrmError
from supportdesk.llm import TOOL_DESCRIPTIONS, LLMUnavailable, ModelTurn

MAX_STEPS = 5
REFUSAL = "I don't know"
UNAVAILABLE = "No answer was generated. The model is unavailable."


@dataclass(frozen=True)
class AgentResult:
    status: str
    steps: int
    answer: str | None = None
    confirmation_id: str | None = None
    summary: str | None = None
    account_id: str | None = None
    citations: list[dict[str, str]] = field(default_factory=list)


def run_agent(
    query: str,
    role: str,
    *,
    llm: object,
    search_fn,
    crm: CrmClient,
    confirmations: object,
    citations: list[dict[str, str]] | None = None,
) -> AgentResult:
    held = list(citations or [])
    messages: list[dict] = [{"role": "user", "text": query}]
    for step in range(1, MAX_STEPS + 1):
        try:
            turn = llm.generate(messages)
        except LLMUnavailable:
            return AgentResult(status="unavailable", steps=step, answer=UNAVAILABLE, citations=held)
        if not isinstance(turn, ModelTurn):
            raise TypeError("llm.generate must return a ModelTurn")
        if turn.tool_call is None:
            return AgentResult(status="answered", steps=step, answer=turn.text)
        call = turn.tool_call
        if call.name == "create_ticket":
            summary = _text_arg(call.arguments, "summary")
            account_id = _text_arg(call.arguments, "account_id")
            if summary and account_id:
                confirmation_id = confirmations.create(
                    role=role, summary=summary, account_id=account_id
                )
                return AgentResult(
                    status="confirmation_required",
                    steps=step,
                    confirmation_id=confirmation_id,
                    summary=summary,
                    account_id=account_id,
                )
            result = {"error": "summary and account_id are required"}
        else:
            result = _execute(call.name, call.arguments, search_fn=search_fn, crm=crm)
        if result.get("hits") == []:
            return AgentResult(status="refused", steps=step, answer=REFUSAL)
        messages.append({"role": "model", "text": turn.text, "tool_call": call})
        messages.append(
            {
                "role": "tool",
                "name": call.name,
                "call_id": call.call_id,
                "content": result,
            }
        )
    return AgentResult(status="step_limit", steps=MAX_STEPS)


def _execute(name: str, arguments: dict, *, search_fn, crm: CrmClient) -> dict:
    if name not in TOOL_DESCRIPTIONS:
        return {"error": "unknown tool"}
    if name == "search_knowledge_base":
        query = _text_arg(arguments, "query")
        if not query:
            return {"error": "query is required"}
        return {"hits": search_fn(query)}
    if name == "get_ticket_status":
        ticket_id = _text_arg(arguments, "ticket_id")
        if not ticket_id:
            return {"error": "ticket_id is required"}
        try:
            return {"ticket": crm.get_ticket(ticket_id)}
        except CrmError as exc:
            return {"error": _crm_error(exc)}
    if name == "calculate":
        expression = arguments.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            return {"error": "expression is required"}
        try:
            return {"value": calculate(expression)}
        except ArithmeticError as exc:
            return {"error": str(exc)}
    return {"error": "unknown tool"}


def _text_arg(arguments: dict, key: str) -> str:
    value = arguments.get(key, "")
    if not isinstance(value, str):
        return ""
    return value.strip()


def _crm_error(exc: CrmError) -> str:
    if str(exc) == "circuit open":
        return "crm circuit open"
    if exc.status_code is None:
        return "crm timeout"
    return f"crm status {exc.status_code}"
