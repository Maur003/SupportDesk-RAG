import pytest

from supportdesk.llm import TOOL_PARAMETERS, GeminiLLM, ModelError, ToolCall


class _Model:
    def __init__(self, name: str) -> None:
        self.name = name


class _Call:
    def __init__(self) -> None:
        self.name = "calculate"
        self.args = {"expression": "1+1"}
        self.id = "call-1"


class _Response:
    def __init__(self, calls: list | None) -> None:
        self.text = ""
        self.function_calls = calls


class _Models:
    def __init__(self, names: list[str]) -> None:
        self._names = names
        self.list_calls = 0
        self.generate_calls = 0
        self.last_model = ""
        self.last_config = None
        self.last_contents = None

    def list(self):
        self.list_calls += 1
        return [_Model(name) for name in self._names]

    def generate_content(self, *, model, contents, config):
        self.generate_calls += 1
        self.last_model = model
        self.last_config = config
        self.last_contents = contents
        if self.generate_calls == 1:
            return _Response([_Call()])
        return _Response(None)


class _Client:
    def __init__(self, names: list[str]) -> None:
        self.models = _Models(names)


def test_list_runs_once_and_automatic_function_calling_is_off() -> None:
    client = _Client(["models/gemini-3.8-flash", "models/gemini-2.5-flash"])
    llm = GeminiLLM(
        api_key="unused",
        generator_model="gemini-3.8-flash",
        judge_model="gemini-2.5-flash",
        temperature=0,
        client=client,
    )
    messages = [{"role": "user", "text": "what is 1 + 1"}]
    first = llm.generate(messages)
    assert first.tool_call is not None
    assert first.tool_call.name == "calculate"
    assert first.tool_call.call_id == "call-1"
    messages.append({"role": "model", "text": "", "tool_call": first.tool_call})
    messages.append(
        {
            "role": "tool",
            "name": "calculate",
            "call_id": "call-1",
            "content": {"value": 2},
        }
    )
    second = llm.generate(messages)
    assert second.tool_call is None
    assert client.models.list_calls == 1
    assert client.models.generate_calls == 2
    assert client.models.last_model == "gemini-3.8-flash"
    config = client.models.last_config
    assert config.temperature == 0
    assert config.automatic_function_calling.disable is True
    names = [item.name for item in config.tools[0].function_declarations]
    assert names == list(TOOL_PARAMETERS)
    response = client.models.last_contents[-1].parts[0].function_response
    assert response.name == "calculate"
    assert response.response == {"output": {"value": 2}}


def test_generate_refuses_when_list_omits_a_configured_model() -> None:
    client = _Client(["models/gemini-3.8-flash"])
    llm = GeminiLLM(
        api_key="unused",
        generator_model="gemini-3.8-flash",
        judge_model="gemini-2.5-flash",
        client=client,
    )
    with pytest.raises(ModelError, match="gemini-2.5-flash"):
        llm.generate([{"role": "user", "text": "hello"}])
    assert client.models.generate_calls == 0


def test_function_call_keeps_the_thought_signature() -> None:
    client = _Client(["models/gemini-3.8-flash", "models/gemini-2.5-flash"])
    signature = b"thought-signature"
    client.models.generate_content = _signed_generate(client.models, signature)
    llm = GeminiLLM(
        api_key="unused",
        generator_model="gemini-3.8-flash",
        judge_model="gemini-2.5-flash",
        client=client,
    )
    first = llm.generate([{"role": "user", "text": "look up the outage credit"}])
    assert first.tool_call is not None
    assert first.tool_call.thought_signature == signature
    llm.generate(
        [
            {"role": "user", "text": "look up the outage credit"},
            {"role": "model", "text": "", "tool_call": first.tool_call},
            {
                "role": "tool",
                "name": first.tool_call.name,
                "call_id": first.tool_call.call_id,
                "content": {"hits": []},
            },
        ]
    )
    model_part = client.models.last_contents[1].parts[0]
    assert model_part.thought_signature == signature
    assert model_part.function_call.name == "search_knowledge_base"


def _signed_generate(models, signature: bytes):
    def generate_content(*, model, contents, config):
        models.generate_calls += 1
        models.last_model = model
        models.last_config = config
        models.last_contents = contents
        if models.generate_calls == 1:
            call = _Call()
            call.name = "search_knowledge_base"
            call.args = {"query": "outage credit"}
            part = _SignedPart(call, signature)
            return _SignedResponse(part)
        return _Response(None)

    return generate_content


class _SignedPart:
    def __init__(self, call: _Call, signature: bytes) -> None:
        self.function_call = call
        self.thought_signature = signature
        self.text = None


class _SignedContent:
    def __init__(self, part: _SignedPart) -> None:
        self.parts = [part]


class _SignedCandidate:
    def __init__(self, part: _SignedPart) -> None:
        self.content = _SignedContent(part)


class _SignedResponse:
    def __init__(self, part: _SignedPart) -> None:
        self.text = ""
        self.function_calls = [part.function_call]
        self.candidates = [_SignedCandidate(part)]


def test_generate_can_target_the_cheap_model() -> None:
    client = _Client(["models/gemini-3.8-flash", "models/gemini-2.5-flash"])
    llm = GeminiLLM(
        api_key="unused",
        generator_model="gemini-3.8-flash",
        judge_model="gemini-2.5-flash",
        client=client,
    )
    llm.generate([{"role": "user", "text": "hi"}], model="gemini-2.5-flash")
    assert client.models.last_model == "gemini-2.5-flash"
    assert client.models.list_calls == 1
    with pytest.raises(ModelError, match="gemini-missing"):
        llm.generate([{"role": "user", "text": "hi"}], model="gemini-missing")
    assert client.models.generate_calls == 1


def test_tool_call_round_trip_keeps_the_call_id() -> None:
    call = ToolCall("calculate", {"expression": "1+1"}, "call-1")
    assert call.arguments["expression"] == "1+1"
