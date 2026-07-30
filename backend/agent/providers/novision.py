"""No-vision Chat Completions provider session (spec §3.3).

Speaks the OpenAI **Chat Completions** API (``/v1/chat/completions``), which is
what LiteLLM exposes. This is deliberately separate from
``OpenAIProviderSession`` (``providers/openai.py``), which uses the
**Responses API** (``/v1/responses``) — OpenAI-direct-only and incompatible
with a LiteLLM gateway.

Import-graph rule (spec §3.3 rule 1): this module must NOT import
``providers/factory.py`` or ``providers/openai.py`` — both pull in vision
providers (Anthropic/Gemini) and ``is_screenshot_preview_available``. The
no-vision path's import graph must be vision-free. We import only:
  - ``providers/base``  — the ProviderSession protocol + StreamEvent/ProviderTurn
  - ``tools``           — CanonicalToolDefinition, ToolCall, parse_json_arguments
  - ``state``           — ensure_str

The session is text-only by construction: there is no code path that attaches
an image to a message.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from agent.providers.base import (
    EventSink,
    ExecutedToolCall,
    ProviderSession,
    ProviderTurn,
    StreamEvent,
)
from agent.state import ensure_str
from agent.tools import CanonicalToolDefinition, ToolCall, parse_json_arguments


# ---------------------------------------------------------------------------
# Tool serialization — Chat Completions format
# ---------------------------------------------------------------------------

def serialize_chat_completions_tools(
    tools: List[CanonicalToolDefinition],
) -> List[ChatCompletionToolParam]:
    """Convert canonical tool defs to Chat Completions ``tools`` format.

    Unlike the Responses API, Chat Completions wraps each function in
    ``{"type": "function", "function": {...}}`` and does not support
    ``strict`` mode the same way, so we pass the schema through as-is.
    """
    serialized: List[ChatCompletionToolParam] = []
    for tool in tools:
        entry: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        serialized.append(entry)  # type: ignore[arg-type]
    return serialized


# ---------------------------------------------------------------------------
# Chat Completions streaming parse state
# ---------------------------------------------------------------------------

class _ChatCompletionsParseState:
    """Accumulates a single streamed Chat Completions turn.

    Chat Completions streams a list of ``choices``, each carrying a ``delta``.
    Text arrives as ``delta.content``; tool calls arrive incrementally as
    ``delta.tool_calls`` (a list keyed by ``index``, with ``id``/``name`` on
    the first fragment and ``arguments`` fragments appended across deltas).
    """

    def __init__(self) -> None:
        self.assistant_text: str = ""
        # tool calls keyed by their stream index → accumulated dict
        self.tool_calls: Dict[int, Dict[str, Any]] = {}

    @property
    def ordered_tool_calls(self) -> List[Dict[str, Any]]:
        return [self.tool_calls[i] for i in sorted(self.tool_calls)]


def _build_provider_turn(state: _ChatCompletionsParseState) -> ProviderTurn:
    """Build a ProviderTurn from accumulated parse state.

    ``assistant_turn`` is the Chat Completions assistant message dict (with
    ``role``, ``content``, and ``tool_calls``). The engine hands it back via
    ``append_tool_results``, where we append it verbatim to ``_messages``.
    """
    tool_calls: List[ToolCall] = []
    assistant_tool_calls: List[Dict[str, Any]] = []

    for entry in state.ordered_tool_calls:
        raw_args = entry.get("arguments", "")
        args, error = parse_json_arguments(raw_args)
        if error:
            args = {"INVALID_JSON": ensure_str(raw_args)}
        call_id = entry.get("id") or f"call-{uuid.uuid4().hex[:6]}"
        name = entry.get("name") or "unknown_tool"
        tool_calls.append(ToolCall(id=call_id, name=name, arguments=args))
        assistant_tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": raw_args},
            }
        )

    # The assistant message that gets appended to the conversation history.
    assistant_message: Dict[str, Any] = {"role": "assistant"}
    if state.assistant_text:
        assistant_message["content"] = state.assistant_text
    else:
        # OpenAI requires either content or tool_calls; null content is valid
        # when tool_calls are present.
        assistant_message["content"] = None
    if assistant_tool_calls:
        assistant_message["tool_calls"] = assistant_tool_calls

    return ProviderTurn(
        assistant_text=state.assistant_text,
        tool_calls=tool_calls,
        assistant_turn=assistant_message,
    )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class NovisionChatSession(ProviderSession):
    """A text-only Chat Completions session pointed at a LiteLLM gateway.

    Implements the same ``ProviderSession`` protocol as
    ``OpenAIProviderSession`` but via ``chat.completions.create`` instead of
    ``responses.create``. No image content is ever attached.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt_messages: List[ChatCompletionMessageParam],
        tools: List[ChatCompletionToolParam],
    ):
        self._client = client
        self._model = model
        self._tools = tools
        # Working message list — mutated as turns proceed. We keep the original
        # prompt messages, then append assistant turns + tool results.
        self._messages: List[ChatCompletionMessageParam] = list(prompt_messages)
        # LiteLLM reports usage on the final chunk; accumulate for logging.
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    async def stream_turn(self, on_event: Optional[EventSink] = None) -> ProviderTurn:
        params: Dict[str, Any] = {
            "model": self._model,
            "messages": self._messages,
            "tools": self._tools,
            "tool_choice": "auto",
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        async def _emit(event: StreamEvent) -> None:
            if on_event is not None:
                await on_event(event)

        state = _ChatCompletionsParseState()
        stream = await self._client.chat.completions.create(**params)  # type: ignore[call-overload]

        async for chunk in stream:  # type: ignore[union-attr]
            # Usage arrives on a final chunk with empty choices.
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self._total_input_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self._total_output_tokens += getattr(usage, "completion_tokens", 0) or 0

            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            # --- text ---
            content = getattr(delta, "content", None)
            if content:
                state.assistant_text += content
                await _emit(StreamEvent(type="assistant_delta", text=content))

            # --- tool calls (incremental) ---
            delta_tool_calls = getattr(delta, "tool_calls", None) or []
            for tc in delta_tool_calls:
                index = getattr(tc, "index", 0)
                entry = state.tool_calls.setdefault(
                    index,
                    {"id": "", "name": "", "arguments": ""},
                )
                tc_id = getattr(tc, "id", None)
                if tc_id:
                    entry["id"] = tc_id
                function = getattr(tc, "function", None)
                if function is not None:
                    fname = getattr(function, "name", None)
                    if fname:
                        entry["name"] = fname
                    fargs = getattr(function, "arguments", None)
                    if fargs:
                        entry["arguments"] += fargs
                        await _emit(
                            StreamEvent(
                                type="tool_call_delta",
                                tool_call_id=entry["id"] or f"idx-{index}",
                                tool_name=entry["name"],
                                tool_arguments=entry["arguments"],
                            )
                        )

        turn = _build_provider_turn(state)
        return turn

    async def append_tool_results(
        self,
        turn: ProviderTurn,
        executed_tool_calls: List[ExecutedToolCall],
    ) -> None:
        # 1. Append the assistant turn that produced the tool calls.
        assistant_message = turn.assistant_turn
        if assistant_message:
            self._messages.append(
                assistant_message  # type: ignore[arg-type]
            )

        # 2. Append each tool result as a ``role=tool`` message.
        for executed in executed_tool_calls:
            result_json = json.dumps(executed.result.result)
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": executed.tool_call.id,
                    "content": result_json,
                }  # type: ignore[arg-type]
            )

    async def append_user_message(
        self,
        content: str,
    ) -> None:
        """Append a standalone user message (e.g. a nudge)."""
        self._messages.append(
            {"role": "user", "content": content}  # type: ignore[arg-type]
        )

    def total_cost_usd(self) -> Optional[float]:
        """Cost tracking is delegated to LiteLLM (which has the pricing table).

        Returning None marks the model as unpriced from this session's
        perspective; the engine treats None as unbounded (spec §4.4 — budget is
        enforced via LiteLLM-side limits / the NO_VISION_BUDGET_USD gate at the
        engine level, not via per-token pricing here).
        """
        return None

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    async def close(self) -> None:
        print(
            f"[TOKEN USAGE] provider=novision model={self._model} | "
            f"input={self._total_input_tokens} output={self._total_output_tokens} "
            f"cost tracked by LiteLLM"
        )
        await self._client.close()


# ---------------------------------------------------------------------------
# Factory entry point (does NOT use providers/factory.py)
# ---------------------------------------------------------------------------

def create_novision_session(
    model: str,
    prompt_messages: List[ChatCompletionMessageParam],
    tools: List[CanonicalToolDefinition],
    base_url: str,
    api_key: str,
) -> NovisionChatSession:
    """Construct a NovisionChatSession wired to a LiteLLM gateway.

    Builds the AsyncOpenAI client directly (spec §3.3 rule 1) — no factory,
    no vision providers.
    """
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    chat_tools = serialize_chat_completions_tools(tools)
    return NovisionChatSession(
        client=client,
        model=model,
        prompt_messages=prompt_messages,
        tools=chat_tools,
    )
