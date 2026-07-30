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

# Phase-3 mitigation gates (spec §7.6). The no-vision session reports
# total_cost_usd() as None (pricing is delegated to LiteLLM), so a USD cap
# can't fire inside the loop. We proxy it with a token ceiling derived from
# NO_VISION_BUDGET_USD and a rough blended $/M-token rate. This is a backstop,
# not a precise meter — LiteLLM-side limits remain the authoritative cap.
_BLENDED_USD_PER_MILLION_TOKENS = 3.0  # conservative blended in+out rate

# Abort if the model emits this many malformed tool calls in a row — a sign
# it has lost tool-calling coherence and will burn budget without progress.
_MALFORMED_STREAK_ABORT = 5

# How many consecutive turns with no tool calls we tolerate before aborting.
# The model sometimes "thinks out loud" — a nudge gets it back on track.
_MAX_IDLE_TURNS = 2


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
        budget_usd: Optional[float] = None,
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
        # Spec size for coverage guard.
        self.spec_section_count = len(spec.get("sections", []))
        self.spec_component_count = len(spec.get("components", []))
        # Phase-3 mitigation: token-based proxy for the USD budget (spec §7.6).
        # The session can't price itself, so we derive a token ceiling from the
        # USD budget and a blended rate. None disables the cap.
        if budget_usd is not None and budget_usd > 0:
            self._token_budget = int(
                (budget_usd / _BLENDED_USD_PER_MILLION_TOKENS) * 1_000_000
            )
        else:
            self._token_budget = None

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
        malformed_streak = 0
        idle_turns = 0

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

                # Log what the model did this turn for diagnostics.
                tool_summary = ", ".join(
                    f"{tc.name}({tc.arguments[:80]}{'…' if len(tc.arguments) > 80 else ''})"
                    for tc in turn.tool_calls
                ) or "(no tool calls)"
                logger.info(
                    "Turn %d: %d tool call(s) [%s] | files so far: %d | "
                    "tokens in=%d out=%d",
                    iterations,
                    len(turn.tool_calls),
                    tool_summary,
                    len(self.runtime.files),
                    session.total_input_tokens,
                    session.total_output_tokens,
                )

                # Track malformed tool calls (spec §7.6 gate).
                for tc in turn.tool_calls:
                    tool_call_count += 1
                    if "INVALID_JSON" in tc.arguments:
                        malformed_tool_calls += 1

                # Phase-3 mitigation: abort on a streak of malformed tool calls.
                # If the model emits _MALFORMED_STREAK_ABORT bad calls in a row
                # it has lost tool-calling coherence — continuing burns budget
                # without progress. We count the trailing streak in this turn.
                turn_malformed = sum(
                    1 for tc in turn.tool_calls if "INVALID_JSON" in tc.arguments
                )
                if turn_malformed == len(turn.tool_calls) and turn_malformed > 0:
                    malformed_streak += 1
                else:
                    malformed_streak = 0
                if malformed_streak >= _MALFORMED_STREAK_ABORT:
                    await self.on_event(
                        {
                            "type": "abort",
                            "reason": "malformed_streak",
                            "streak": malformed_streak,
                            "iteration": iterations,
                        }
                    )
                    break

                # Phase-3 mitigation: token-based budget proxy (spec §7.6).
                # The session can't price itself, so we cap on tokens derived
                # from NO_VISION_BUDGET_USD. LiteLLM-side limits remain primary.
                if self._token_budget is not None:
                    spent = session.total_input_tokens + session.total_output_tokens
                    if spent > self._token_budget:
                        await self.on_event(
                            {
                                "type": "abort",
                                "reason": "budget",
                                "tokens": spent,
                                "budget": self._token_budget,
                                "iteration": iterations,
                            }
                        )
                        break

                if not turn.tool_calls:
                    # Model produced text without a tool call.
                    idle_turns += 1
                    if idle_turns >= _MAX_IDLE_TURNS:
                        logger.warning(
                            "Turn %d: %d consecutive idle turns — aborting",
                            iterations, idle_turns,
                        )
                        finished = True
                        break
                    # Nudge the model to continue generating files.
                    logger.info(
                        "Turn %d: no tool calls (idle %d/%d) — nudging",
                        iterations, idle_turns, _MAX_IDLE_TURNS,
                    )
                    await session.append_user_message(
                        "You haven't generated all required files yet. "
                        "Continue by calling write_file for the remaining "
                        "components and pages, then call finish()."
                    )
                    continue

                idle_turns = 0  # reset on any tool call

                # Execute each tool call and collect results.
                executed = []
                for tc in turn.tool_calls:
                    if tc.name == "finish":
                        # Coverage guard: reject premature finish when the model
                        # has barely started generating files.
                        non_config_files = [
                            p for p in self.runtime.files
                            if not p.endswith((".json", ".config.js", ".config.ts"))
                        ]
                        expected = max(
                            self.spec_section_count,
                            self.spec_component_count,
                        )
                        if (
                            expected > 3
                            and len(non_config_files) < max(3, expected // 4)
                            and iterations < self.max_iterations - 2
                        ):
                            logger.warning(
                                "Turn %d: finish() rejected — only %d non-config "
                                "file(s) for %d sections/%d components",
                                iterations,
                                len(non_config_files),
                                self.spec_section_count,
                                self.spec_component_count,
                            )
                            executed.append(
                                _make_executed(
                                    tc,
                                    ToolExecutionResult(
                                        ok=False,
                                        result={"content": (
                                            f"FINISH REJECTED: You have only generated "
                                            f"{len(non_config_files)} source file(s) but "
                                            f"the spec has {self.spec_section_count} sections "
                                            f"and {self.spec_component_count} components. "
                                            f"You MUST generate app/page.tsx, all section "
                                            f"components, and a layout before calling finish(). "
                                            f"Continue writing files now."
                                        )},
                                        summary={"finished": False, "rejected": True},
                                    ),
                                )
                            )
                            continue
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
            "idle_turns": idle_turns,
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
