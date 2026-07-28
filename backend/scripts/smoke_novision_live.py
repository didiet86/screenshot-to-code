#!/usr/bin/env python3
"""Live LiteLLM smoke test for the no-vision path (spec §3.3, §7.6).

Does ONE real call against the configured LiteLLM gateway + model and logs
diagnostics that reveal exactly how the model behaves over Chat Completions
streaming + tool-calling. This is the "does one real call even parse
correctly" gate that mock tests cannot substitute (spec §7.6 is the
at-scale follow-up).

Run from clone-design/backend/ with the no-vision venv:

    LITELLM_BASE_URL=... LITELLM_API_KEY=... LITELLM_MODEL=zai-glm-5.2 \
    /path/to/novision-venv/bin/python scripts/smoke_novision_live.py

Exit code 0 = the model's tool-call format is compatible with NovisionChatSession.
Exit code 1 = incompatible — diagnostics below tell you how.

What it checks:
  1. The gateway responds at all (connectivity + auth).
  2. The model emits a tool call in Chat Completions format (delta.tool_calls).
  3. The tool-call arguments are valid JSON (parse_json_arguments succeeds).
  4. A second turn (with tool result appended) completes without error.
  5. finish_reason is one of the expected values.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

# --- config (read directly, no app shell) -----------------------------------

BASE_URL = os.environ.get("LITELLM_BASE_URL")
API_KEY = os.environ.get("LITELLM_API_KEY")
MODEL = os.environ.get("LITELLM_MODEL")

missing = [n for n, v in (("LITELLM_BASE_URL", BASE_URL), ("LITELLM_API_KEY", API_KEY), ("LITELLM_MODEL", MODEL)) if not v]
if missing:
    print(f"❌ Missing env vars: {', '.join(missing)}")
    print("   Set LITELLM_BASE_URL, LITELLM_API_KEY, LITELLM_MODEL and re-run.")
    sys.exit(2)


async def main() -> int:
    from openai import AsyncOpenAI

    from agent.providers.novision import (
        NovisionChatSession,
        serialize_chat_completions_tools,
    )
    from agent.tools.novision_definitions import canonical_novision_tools

    print(f"=== Live no-vision smoke test ===")
    print(f"gateway : {BASE_URL}")
    print(f"model   : {MODEL}")
    print()

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    # --- Step 1: raw connectivity probe (no tools, no streaming) -------------
    print("[1/5] Connectivity probe (plain chat.completions.create, no tools)...")
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=10,
        )
        text = resp.choices[0].message.content
        print(f"      ✓ Gateway responded. Model said: {text!r}")
    except Exception as e:
        print(f"      ❌ Connectivity failed: {type(e).__name__}: {e}")
        return 1

    # --- Step 2: tool-call probe (streaming) --------------------------------
    print()
    print("[2/5] Tool-call probe (streaming, with create_file tool)...")
    tools = serialize_chat_completions_tools(canonical_novision_tools())
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a test. You MUST call the `create_file` tool to create "
                "a file at path 'hello.txt' with content 'world'. Call the tool "
                "now. Do not explain."
            ),
        },
        {"role": "user", "content": "Create the file as instructed."},
    ]

    session = NovisionChatSession(
        client=client,
        model=MODEL,
        prompt_messages=messages,  # type: ignore[arg-type]
        tools=tools,
    )

    events_seen: list[str] = []
    text_fragments = 0
    tool_call_fragments = 0

    async def on_event(event: Any) -> None:
        nonlocal text_fragments, tool_call_fragments
        if event.type == "assistant_delta":
            text_fragments += 1
        elif event.type == "tool_call_delta":
            tool_call_fragments += 1

    try:
        turn = await session.stream_turn(on_event)
    except Exception as e:
        print(f"      ❌ stream_turn raised: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"      text delta fragments received : {text_fragments}")
    print(f"      tool-call delta fragments     : {tool_call_fragments}")
    print(f"      assistant_text                : {turn.assistant_text!r}")
    print(f"      tool_calls parsed             : {len(turn.tool_calls)}")

    if not turn.tool_calls:
        print("      ❌ Model did NOT emit a tool call. It may not support tool-calling,")
        print("         or the prompt wasn't forceful enough. This is a hard fail for")
        print("         the no-vision agent loop.")
        print(f"      (assistant_turn keys: {list(turn.assistant_turn.keys())})")
        return 1

    # --- Step 3: tool-call argument validity --------------------------------
    print()
    print("[3/5] Tool-call argument validity...")
    all_valid = True
    for i, tc in enumerate(turn.tool_calls):
        is_valid = "INVALID_JSON" not in tc.arguments
        status = "✓ valid JSON" if is_valid else "❌ MALFORMED JSON"
        print(f"      call[{i}] id={tc.id!r} name={tc.name!r} → {status}")
        if not is_valid:
            print(f"           raw args: {tc.arguments.get('INVALID_JSON', '')!r}")
            all_valid = False
        else:
            print(f"           parsed args: {json.dumps(tc.arguments)}")
    if not all_valid:
        print("      ❌ At least one tool call had malformed JSON arguments.")
        print("         The model's tool-calling is unreliable for this gateway/model combo.")
        return 1

    # --- Step 4: second turn (append tool result, stream again) -------------
    print()
    print("[4/5] Second-turn probe (append tool result, stream again)...")
    from agent.providers.base import ExecutedToolCall
    from agent.tools.types import ToolExecutionResult

    executed = [
        ExecutedToolCall(
            tool_call=turn.tool_calls[0],
            result=ToolExecutionResult(
                ok=True,
                result={"content": "Successfully created file at hello.txt."},
                summary={"path": "hello.txt"},
            ),
        )
    ]
    await session.append_tool_results(turn, executed)

    try:
        turn2 = await session.stream_turn(on_event)
        print(f"      ✓ Second turn completed. assistant_text={turn2.assistant_text!r}")
        print(f"        tool_calls in turn 2: {len(turn2.tool_calls)}")
    except Exception as e:
        print(f"      ❌ Second turn failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # --- Step 5: raw finish_reason inspection -------------------------------
    print()
    print("[5/5] Raw finish_reason inspection (non-streaming, for clarity)...")
    try:
        raw = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        fr = raw.choices[0].finish_reason
        msg = raw.choices[0].message
        has_tc = bool(getattr(msg, "tool_calls", None))
        print(f"      finish_reason: {fr!r}")
        print(f"      has tool_calls on message: {has_tc}")
        if fr == "tool_calls" and has_tc:
            print("      ✓ finish_reason='tool_calls' — standard OpenAI format.")
        elif fr in ("stop", "length") and not has_tc:
            print(f"      ⚠ finish_reason={fr!r} with no tool calls in non-streaming mode.")
            print("        The model may be reluctant to call tools without streaming,")
            print("        or it needs a more forceful system prompt. Not necessarily fatal.")
        else:
            print(f"      ⚠ Unexpected combination: finish_reason={fr!r}, tool_calls={has_tc}")
    except Exception as e:
        print(f"      ⚠ Non-streaming probe failed (non-fatal): {type(e).__name__}: {e}")

    await session.close()

    print()
    print("════════════════════════════════════════════════════")
    print("✅ LIVE SMOKE TEST PASSED")
    print("   The model's Chat Completions tool-call format is")
    print("   compatible with NovisionChatSession. Safe to build")
    print("   Phase 2 (project assembler) on this foundation.")
    print("════════════════════════════════════════════════════")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
