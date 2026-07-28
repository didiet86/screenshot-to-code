#!/usr/bin/env python3
"""Live end-to-end no-vision generation (spec §4, §5, §7).

Runs the FULL NovisionEngine against a real model via LiteLLM, assembles the
project, and prints coverage metrics. This is the "does the whole loop produce
a real multi-file project" gate.

Run from clone-design/backend/ on the lan-proxy network:

    docker run --rm --network lan-proxy \
      -v $PWD:/work -w /work -e PYTHONPATH=/work \
      -e LITELLM_BASE_URL=http://gw-litellm:4000/v1 \
      -e LITELLM_API_KEY=... -e LITELLM_MODEL=zai-glm-5.2 \
      python:3.12-slim sh -c 'pip install --quiet openai && python scripts/e2e_novision_live.py'
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict

BASE_URL = os.environ.get("LITELLM_BASE_URL")
API_KEY = os.environ.get("LITELLM_API_KEY")
MODEL = os.environ.get("LITELLM_MODEL")
FRAMEWORK = os.environ.get("FRAMEWORK", "html")
STACK = os.environ.get("STACK", "html_css")

if not all([BASE_URL, API_KEY, MODEL]):
    print("Missing LITELLM_BASE_URL / LITELLM_API_KEY / LITELLM_MODEL")
    sys.exit(2)


# A small but realistic spec — exercises tokens, sections, components.
SPEC: Dict[str, Any] = {
    "version": "1.0",
    "url": "https://example.com",
    "title": "Acme Landing",
    "tokens": {
        "colors": {
            "palette": [
                {"hex": "#0b0f19", "role": "bg"},
                {"hex": "#6366f1", "role": "accent"},
                {"hex": "#e5e7eb", "role": "text"},
            ]
        },
        "typography": {"font_families": ["Inter"]},
        "spacing": {"scale": [{"name": "md", "value": "16px"}]},
    },
    "sections": [
        {
            "id": "sec-header",
            "role": "header",
            "layout": "flex-row",
            "components": ["logo", "nav"],
            "html_hint": "<header>logo + nav links</header>",
        },
        {
            "id": "sec-hero",
            "role": "hero",
            "layout": "flex-col",
            "components": ["cta-button"],
            "html_hint": "<section><h1>Build faster</h1><button>Get started</button></section>",
        },
        {
            "id": "sec-footer",
            "role": "footer",
            "layout": "flex-row",
            "components": [],
            "html_hint": "<footer>copyright</footer>",
        },
    ],
    "components": [
        {
            "name": "cta-button",
            "type": "button",
            "reusable": True,
            "styles": {"background": "#6366f1", "border_radius": "8px"},
            "key_elements": ["Get started"],
        },
        {"name": "logo", "type": "image", "reusable": True, "key_elements": ["Acme"]},
        {"name": "nav", "type": "nav", "reusable": True, "key_elements": ["Features", "Pricing"]},
    ],
}


async def main() -> int:
    from agent.novision_engine import NovisionEngine
    from codegen.project_assembler import assemble_project, compute_coverage

    print(f"=== Live end-to-end no-vision generation ===")
    print(f"gateway   : {BASE_URL}")
    print(f"model     : {MODEL}")
    print(f"framework : {FRAMEWORK} / stack: {STACK}")
    print(f"spec      : {len(SPEC['sections'])} sections, {len(SPEC['components'])} components")
    print()

    events: list[Dict[str, Any]] = []

    async def on_event(event: Dict[str, Any]) -> None:
        events.append(event)
        etype = event.get("type")
        if etype == "tool_result":
            print(f"  [iter {event['iteration']}] {event['tool']} → {'ok' if event['ok'] else 'FAIL'}")
        elif etype == "finish":
            print(f"  [iter {event['iteration']}] finish")

    engine = NovisionEngine(
        spec=SPEC,
        framework=FRAMEWORK,
        stack=STACK,
        model=MODEL,
        base_url=BASE_URL,  # type: ignore[arg-type]
        api_key=API_KEY,  # type: ignore[arg-type]
        max_iterations=30,
        on_event=on_event,
    )

    print("[running agent loop...]")
    result = await engine.run()

    print()
    print(f"=== Agent result ===")
    print(f"finished             : {result['finished']}")
    print(f"iterations           : {result['iterations']}")
    print(f"tool calls           : {result['tool_call_count']}")
    print(f"malformed tool calls : {result['malformed_tool_calls']}")
    print(f"files written        : {len(result['files'])}")
    for path in sorted(result["files"]):
        print(f"  • {path} ({len(result['files'][path])} chars)")

    print()
    print(f"=== Coverage metrics ===")
    cov = compute_coverage(result["files"], SPEC)
    print(f"section coverage   : {cov['section_covered']}/{cov['section_total']} ({cov['section_coverage_pct']}%)")
    print(f"reusable components: {cov['reusable_component_covered']}/{cov['reusable_component_total']} ({cov['reusable_component_coverage_pct']}%)")
    print(f"token usage        : {cov['token_usage_pct']}% (palette {cov['palette_referenced']}/{cov['palette_total']}, fonts {cov['fonts_referenced']}/{cov['fonts_total']})")

    # Assemble + save zip.
    zip_bytes = assemble_project(
        files=result["files"],
        spec=SPEC,
        framework=FRAMEWORK,
        stack=STACK,
        generation_meta={
            "finished": result["finished"],
            "iterations": result["iterations"],
            "tool_call_count": result["tool_call_count"],
            "malformed_tool_calls": result["malformed_tool_calls"],
        },
    )
    out_path = "/tmp/novision-e2e-output.zip"
    with open(out_path, "wb") as f:
        f.write(zip_bytes)
    print(f"\nzip saved → {out_path} ({len(zip_bytes)} bytes)")

    # Verdict
    print()
    ok = result["finished"] and result["malformed_tool_calls"] == 0 and len(result["files"]) >= 3
    if ok:
        print("════════════════════════════════════════════════════")
        print("✅ END-TO-END GENERATION SUCCEEDED")
        print("════════════════════════════════════════════════════")
        return 0
    else:
        print("════════════════════════════════════════════════════")
        print("⚠  GENERATION COMPLETED WITH ISSUES")
        print(f"   finished={result['finished']} malformed={result['malformed_tool_calls']} files={len(result['files'])}")
        print("════════════════════════════════════════════════════")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
