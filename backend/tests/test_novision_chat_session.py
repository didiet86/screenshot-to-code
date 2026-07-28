"""Unit tests for NovisionChatSession Chat Completions delta assembly (spec §3.3).

The riskiest new logic is parsing streamed Chat Completions deltas into a
ProviderTurn: text arrives as ``delta.content`` fragments, tool calls arrive
incrementally as ``delta.tool_calls`` (keyed by index, with id/name on the
first fragment and arguments appended across fragments). These tests mock the
streaming response and assert correct assembly.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, AsyncIterator, List

import pytest

from agent.providers.novision import (
    NovisionChatSession,
    create_novision_session,
    serialize_chat_completions_tools,
)
from agent.tools.novision_definitions import canonical_novision_tools


# ---------------------------------------------------------------------------
# Helpers: build mock streaming chunks
# ---------------------------------------------------------------------------

def _text_chunk(text: str) -> SimpleNamespace:
    """A chunk carrying only a text delta."""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))],
        usage=None,
    )


def _tool_call_start_chunk(index: int, call_id: str, name: str) -> SimpleNamespace:
    """First fragment of a tool call: carries id + name, empty arguments."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=index,
                            id=call_id,
                            function=SimpleNamespace(name=name, arguments=""),
                        )
                    ],
                )
            )
        ],
        usage=None,
    )


def _tool_call_args_chunk(index: int, args_fragment: str) -> SimpleNamespace:
    """A subsequent fragment: appends to arguments."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=index,
                            id=None,
                            function=SimpleNamespace(name=None, arguments=args_fragment),
                        )
                    ],
                )
            )
        ],
        usage=None,
    )


def _usage_chunk(input_tokens: int, output_tokens: int) -> SimpleNamespace:
    """Final chunk with usage stats and empty choices."""
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        ),
    )


class _MockStream:
    """Async-iterable over a list of chunks."""

    def __init__(self, chunks: List[SimpleNamespace]) -> None:
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self) -> AsyncIterator[SimpleNamespace]:
        for c in self._chunks:
            yield c


class _MockChatCompletions:
    def __init__(self, chunks: List[SimpleNamespace]) -> None:
        self._chunks = chunks

    async def create(self, **kwargs: Any) -> _MockStream:
        self.last_kwargs = kwargs
        return _MockStream(self._chunks)


class _MockClient:
    def __init__(self, chunks: List[SimpleNamespace]) -> None:
        self.chat = SimpleNamespace(completions=_MockChatCompletions(chunks))

    async def close(self) -> None:
        pass


def _make_session(chunks: List[SimpleNamespace]) -> NovisionChatSession:
    client = _MockClient(chunks)
    session = NovisionChatSession.__new__(NovisionChatSession)
    session._client = client
    session._model = "test-model"
    session._tools = []
    session._messages = [{"role": "system", "content": "sys"}]
    session._total_input_tokens = 0
    session._total_output_tokens = 0
    return session


async def _noop_event(_event: Any) -> None:
    """Async no-op sink for tests that don't care about events."""
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_text_only_turn_assembles() -> None:
    """Pure text deltas concatenate into assistant_text, no tool calls."""
    chunks = [
        _text_chunk("Hello"),
        _text_chunk(", "),
        _text_chunk("world!"),
        _usage_chunk(10, 5),
    ]
    session = _make_session(chunks)
    turn = await session.stream_turn(_noop_event)
    assert turn.assistant_text == "Hello, world!"
    assert turn.tool_calls == []
    assert turn.assistant_turn["role"] == "assistant"
    assert turn.assistant_turn["content"] == "Hello, world!"
    assert "tool_calls" not in turn.assistant_turn


@pytest.mark.asyncio
async def test_single_tool_call_assembles_across_fragments() -> None:
    """A tool call split across id/name + argument fragments assembles correctly."""
    args_obj = {"path": "src/Button.tsx", "content": "export const X = 1;"}
    args_str = json.dumps(args_obj)
    # Split args into two fragments to test incremental assembly.
    mid = len(args_str) // 2
    chunks = [
        _tool_call_start_chunk(0, "call_abc", "create_file"),
        _tool_call_args_chunk(0, args_str[:mid]),
        _tool_call_args_chunk(0, args_str[mid:]),
        _usage_chunk(20, 30),
    ]
    session = _make_session(chunks)
    turn = await session.stream_turn(_noop_event)
    assert len(turn.tool_calls) == 1
    tc = turn.tool_calls[0]
    assert tc.id == "call_abc"
    assert tc.name == "create_file"
    assert tc.arguments == args_obj
    # assistant_turn carries the raw-arguments string for the history.
    assert turn.assistant_turn["tool_calls"][0]["function"]["arguments"] == args_str


@pytest.mark.asyncio
async def test_multiple_tool_calls_assemble_in_order() -> None:
    """Two interleaved tool calls (by index) assemble in index order."""
    chunks = [
        _tool_call_start_chunk(0, "call_1", "create_file"),
        _tool_call_start_chunk(1, "call_2", "list_files"),
        _tool_call_args_chunk(0, '{"path":"a.tsx"}'),
        _tool_call_args_chunk(1, "{}"),
        _usage_chunk(15, 25),
    ]
    session = _make_session(chunks)
    turn = await session.stream_turn(_noop_event)
    assert len(turn.tool_calls) == 2
    assert turn.tool_calls[0].id == "call_1"
    assert turn.tool_calls[0].name == "create_file"
    assert turn.tool_calls[0].arguments == {"path": "a.tsx"}
    assert turn.tool_calls[1].id == "call_2"
    assert turn.tool_calls[1].name == "list_files"
    assert turn.tool_calls[1].arguments == {}


@pytest.mark.asyncio
async def test_malformed_tool_arguments_captured() -> None:
    """Malformed JSON arguments are captured (spec §7.6 gate counts these)."""
    chunks = [
        _tool_call_start_chunk(0, "call_x", "edit_file"),
        _tool_call_args_chunk(0, "{not valid json"),
        _usage_chunk(5, 5),
    ]
    session = _make_session(chunks)
    turn = await session.stream_turn(_noop_event)
    assert len(turn.tool_calls) == 1
    # The INVALID_JSON marker lets the engine/runtime count malformed calls.
    assert "INVALID_JSON" in turn.tool_calls[0].arguments


@pytest.mark.asyncio
async def test_text_and_tool_call_together() -> None:
    """Text before a tool call is preserved alongside the tool call."""
    chunks = [
        _text_chunk("Creating the button now."),
        _tool_call_start_chunk(0, "call_t", "create_file"),
        _tool_call_args_chunk(0, '{"path":"b.tsx","content":"x"}'),
        _usage_chunk(12, 18),
    ]
    session = _make_session(chunks)
    turn = await session.stream_turn(_noop_event)
    assert turn.assistant_text == "Creating the button now."
    assert len(turn.tool_calls) == 1
    assert turn.assistant_turn["content"] == "Creating the button now."
    assert len(turn.assistant_turn["tool_calls"]) == 1


@pytest.mark.asyncio
async def test_append_tool_results_extends_messages() -> None:
    """append_tool_results adds the assistant turn + tool result messages."""
    chunks = [
        _tool_call_start_chunk(0, "call_r", "list_files"),
        _tool_call_args_chunk(0, "{}"),
        _usage_chunk(8, 4),
    ]
    session = _make_session(chunks)
    assert len(session._messages) == 1  # just system prompt
    turn = await session.stream_turn(_noop_event)

    from agent.providers.base import ExecutedToolCall
    from agent.tools.types import ToolExecutionResult

    executed = [
        ExecutedToolCall(
            tool_call=turn.tool_calls[0],
            result=ToolExecutionResult(
                ok=True, result={"files": ["a.tsx"]}, summary={"fileCount": 1}
            ),
        )
    ]
    await session.append_tool_results(turn, executed)
    # system + assistant turn + tool result = 3
    assert len(session._messages) == 3
    assert session._messages[1]["role"] == "assistant"
    assert session._messages[2]["role"] == "tool"
    assert session._messages[2]["tool_call_id"] == "call_r"


def test_serialize_tools_produces_chat_completions_format() -> None:
    """Tool serialization wraps each function in {type:function, function:{}}."""
    tools = canonical_novision_tools()
    serialized = serialize_chat_completions_tools(tools)
    assert len(serialized) == 7
    for entry in serialized:
        assert entry["type"] == "function"
        assert "name" in entry["function"]
        assert "parameters" in entry["function"]


def test_create_novision_session_builds_client() -> None:
    """Factory builds a session without importing the vision factory."""
    import sys
    before = set(sys.modules)
    session = create_novision_session(
        model="m",
        prompt_messages=[{"role": "user", "content": "hi"}],
        tools=canonical_novision_tools(),
        base_url="http://localhost:4000/v1",
        api_key="sk-test",
    )
    assert session._model == "m"
    assert str(session._client.base_url).rstrip("/") == "http://localhost:4000/v1"
    # No vision factory leaked in.
    assert "agent.providers.factory" not in sys.modules
    assert "agent.providers.anthropic" not in sys.modules
