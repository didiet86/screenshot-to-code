"""No-vision agent engine (spec §4).

A clean agent loop that consumes a design spec and produces a multi-file
project. Simpler than ``AgentEngine``: no image extraction, no screenshot
preview, no single-file ``file_state``, no frontend ``setCode`` streaming.

Loop:
  1. Build prompt (system + spec-as-user) and no-vision tools.
  2. Create a ``NovisionChatSession`` pointed at LiteLLM.
  3. Stream turns; for each turn, execute any tool calls against
     ``NovisionToolRuntime``; append results; repeat.
  4. Stop when the agent calls ``finish``, or hits the iteration / budget cap.

Vision-free import graph: imports only the no-vision session, tools, prompt,
and base types. Does NOT import ``agent.engine`` (which pulls in the factory,
image-gen, screenshot_preview, extract_assets).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from openai.types.chat import ChatCompletionMessageParam

from agent.providers.base import EventSink, StreamEvent
from agent.providers.novision import create_novision_session
from agent.prompts.novision_system import build_novision_system_prompt
from agent.tools import CanonicalToolDefinition
from agent.tools.novision_definitions import canonical_novision_tools
from agent.tools.novision_runtime import NovisionToolRuntime
from agent.tools.types import ToolExecutionResult

logger = logging.getLogger(__name__)

# Safety caps (spec §4.4). Budget is enforced at the LiteLLM/engine level;
# iteration cap prevents a runaway loop regardless of cost.
DEFAULT_MAX_ITERATIONS = 40


class NovisionEngine:
    """Runs a single no-vision generation variant."""

    def __init__(
        self,
        spec: Dict[str, Any],
        framework: str,
        stack: str,
        model: str,
        base_url: str,
        api_key: str,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        on_event: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ):
        self.spec = spec
        self.framework = framework
        self.stack = stack
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_iterations = max_iterations
        self.on_event = on_event or _noop_event
        self.runtime = NovisionToolRuntime(spec=spec)
        self._input_tokens = 0
        self._output_tokens = 0

    async def run(self) -> Dict[str, Any]:
        """Run the agent loop to completion. Returns a result dict."""
        tools = canonical_novision_tools()
        prompt_messages = self._build_prompt_messages()
        session = create_novision_session(
            model=self.model,
            prompt_messages=prompt_messages,
            tools=tools,
            base_url=self.base_url,
            api_key=self.api_key,
        )

        finished = False
        iterations = 0
        tool_call_count = 0
        malformed_tool_calls = 0

        try:
            while iterations < self.max_iterations:
                iterations += 1

                async def on_stream_event(event: StreamEvent) -> None:
                    await self.on_event(
                        {
                            "type": event.type,
                            "iteration": iterations,
                            "text": event.text,
                        }
                    )

                turn = await session.stream_turn(on_stream_event)  # type: ignore[arg-type]

                # Track malformed tool calls (spec §7.6 gate).
                for tc in turn.tool_calls:
                    tool_call_count += 1
                    if "INVALID_JSON" in tc.arguments:
                        malformed_tool_calls += 1

                if not turn.tool_calls:
                    # No tool calls — if there's no text either, the model is
                    # done; if it called finish, we stop. Either way, a turn
                    # with no tool calls ends the loop.
                    finished = True
                    break

                # Execute each tool call and collect results.
                executed = []
                for tc in turn.tool_calls:
                    if tc.name == "finish":
                        finished = True
                        executed.append(
                            _make_executed(
                                tc,
                                ToolExecutionResult(
                                    ok=True,
                                    result={"content": "done"},
                                    summary={"finished": True},
                                ),
                            )
                        )
                        await self.on_event(
                            {"type": "finish", "iteration": iterations}
                        )
                        continue
                    result = await self.runtime.execute(tc)
                    executed.append(_make_executed(tc, result))
                    await self.on_event(
                        {
                            "type": "tool_result",
                            "iteration": iterations,
                            "tool": tc.name,
                            "ok": result.ok,
                        }
                    )

                await session.append_tool_results(turn, executed)

                if finished:
                    break
        finally:
            # Read token counts from the session before close() discards them.
            self._input_tokens = session.total_input_tokens
            self._output_tokens = session.total_output_tokens
            await session.close()

        return {
            "files": self.runtime.files,
            "finished": finished,
            "iterations": iterations,
            "tool_call_count": tool_call_count,
            "malformed_tool_calls": malformed_tool_calls,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
        }

    def _build_prompt_messages(self) -> List[ChatCompletionMessageParam]:
        system_prompt = build_novision_system_prompt(self.framework, self.stack)
        # The user message is the spec JSON (spec §4.2). We send the full spec
        # so the model has global context (section order, token overview); the
        # read_spec_section / read_spec_tokens tools let it pull deep views.
        spec_json = json.dumps(self.spec, ensure_ascii=False, indent=2)
        user_content = (
            f"Reproduce this site from the spec below.\n\n"
            f"Target framework: {self.framework}\n"
            f"CSS stack: {self.stack}\n\n"
            f"```json\n{spec_json}\n```"
        )
        return [
            {"role": "system", "content": system_prompt},  # type: ignore[list-item]
            {"role": "user", "content": user_content},  # type: ignore[list-item]
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_executed(tool_call, result: ToolExecutionResult):
    from agent.providers.base import ExecutedToolCall

    return ExecutedToolCall(tool_call=tool_call, result=result)


async def _noop_event(_event: Dict[str, Any]) -> None:
    pass
