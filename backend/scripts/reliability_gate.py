#!/usr/bin/env python3
"""Reliability gate for the no-vision path (spec §7.6).

Runs 15 generations (5 diverse specs × 3 runs each) against a real model via
LiteLLM and measures tool-calling reliability. This is the formal gate that
the 2-run live smoke test could NOT substitute.

Pass criterion (spec §7.6): ≥90% of runs complete to finish() with ZERO
malformed tool calls. The run may still produce imperfect code — the gate is
about loop integrity, not output quality.

Run from clone-design/backend/ on the lan-proxy network:

    docker run --rm --network lan-proxy \
      -v $PWD:/work -w /work -e PYTHONPATH=/work \
      -e LITELLM_BASE_URL=http://gw-litellm:4000/v1 \
      -e LITELLM_API_KEY=... -e LITELLM_MODEL=auto/deepseek-v4-flash \
      python:3.12-slim sh -c 'pip install --quiet openai && python scripts/reliability_gate.py'

Output: per-run diagnostics + aggregate verdict. Results also saved to
/tmp/reliability_gate_results.json for the budget re-benchmark (Phase 3.2).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List

BASE_URL = os.environ.get("LITELLM_BASE_URL")
API_KEY = os.environ.get("LITELLM_API_KEY")
MODEL = os.environ.get("LITELLM_MODEL", "auto/deepseek-v4-flash")
FRAMEWORK = os.environ.get("FRAMEWORK", "next")
STACK = os.environ.get("STACK", "tailwind")
RUNS_PER_SPEC = int(os.environ.get("RUNS_PER_SPEC", "3"))

if not all([BASE_URL, API_KEY]):
    print("Missing LITELLM_BASE_URL / LITELLM_API_KEY")
    sys.exit(2)


# ---------------------------------------------------------------------------
# 5 diverse specs — different domains/complexity to catch variance
# ---------------------------------------------------------------------------

def _spec(
    title: str,
    sections: List[Dict[str, Any]],
    components: List[Dict[str, Any]],
    palette: List[Dict[str, str]],
    fonts: List[str],
) -> Dict[str, Any]:
    return {
        "version": "1.0",
        "url": "https://example.com",
        "title": title,
        "tokens": {
            "colors": {"palette": palette},
            "typography": {"font_families": fonts},
            "spacing": {"scale": [{"name": "md", "value": "16px"}]},
        },
        "sections": sections,
        "components": components,
    }


SPECS: Dict[str, Dict[str, Any]] = {
    # 1. Landing page — simple, 3 sections (the smoke-test spec)
    "landing": _spec(
        "Acme Landing",
        sections=[
            {"id": "sec-header", "role": "header", "layout": "flex-row", "components": ["logo", "nav"]},
            {"id": "sec-hero", "role": "hero", "layout": "flex-col", "components": ["cta-button"]},
            {"id": "sec-footer", "role": "footer", "layout": "flex-row", "components": []},
        ],
        components=[
            {"name": "cta-button", "type": "button", "reusable": True, "styles": {"background": "#6366f1"}},
            {"name": "logo", "type": "image", "reusable": True},
            {"name": "nav", "type": "nav", "reusable": True},
        ],
        palette=[{"hex": "#0b0f19", "role": "bg"}, {"hex": "#6366f1", "role": "accent"}, {"hex": "#e5e7eb", "role": "text"}],
        fonts=["Inter"],
    ),
    # 2. Dashboard — denser, grid layouts, sidebar
    "dashboard": _spec(
        "Analytics Dashboard",
        sections=[
            {"id": "sec-sidebar", "role": "nav", "layout": "sidebar", "components": ["sidebar-nav"]},
            {"id": "sec-topbar", "role": "header", "layout": "flex-row", "components": ["search-bar", "user-menu"]},
            {"id": "sec-stats", "role": "features", "layout": "grid-4", "components": ["stat-card", "stat-card-2"]},
            {"id": "sec-chart", "role": "content", "layout": "flex-col", "components": ["chart"]},
            {"id": "sec-table", "role": "content", "layout": "flex-col", "components": ["data-table"]},
        ],
        components=[
            {"name": "sidebar-nav", "type": "nav", "reusable": True},
            {"name": "search-bar", "type": "form", "reusable": True},
            {"name": "user-menu", "type": "button", "reusable": True},
            {"name": "stat-card", "type": "card", "reusable": True},
            {"name": "stat-card-2", "type": "card", "reusable": True},
            {"name": "chart", "type": "card", "reusable": False},
            {"name": "data-table", "type": "card", "reusable": False},
        ],
        palette=[{"hex": "#ffffff", "role": "bg"}, {"hex": "#f3f4f6", "role": "surface"}, {"hex": "#111827", "role": "text"}, {"hex": "#3b82f6", "role": "accent"}],
        fonts=["Inter", "JetBrains Mono"],
    ),
    # 3. Docs site — long-form text, code blocks
    "docs": _spec(
        "DevDocs",
        sections=[
            {"id": "sec-header", "role": "header", "layout": "flex-row", "components": ["logo", "search"]},
            {"id": "sec-sidebar", "role": "nav", "layout": "sidebar", "components": ["doc-nav"]},
            {"id": "sec-content", "role": "content", "layout": "flex-col", "components": ["article", "code-block"]},
            {"id": "sec-footer", "role": "footer", "layout": "flex-row", "components": []},
        ],
        components=[
            {"name": "logo", "type": "image", "reusable": True},
            {"name": "search", "type": "form", "reusable": True},
            {"name": "doc-nav", "type": "nav", "reusable": True},
            {"name": "article", "type": "card", "reusable": False},
            {"name": "code-block", "type": "card", "reusable": True},
        ],
        palette=[{"hex": "#1e293b", "role": "bg"}, {"hex": "#334155", "role": "surface"}, {"hex": "#f8fafc", "role": "text"}, {"hex": "#22d3ee", "role": "accent"}],
        fonts=["Inter", "Fira Code"],
    ),
    # 4. E-commerce product — cards, pricing, CTA-heavy
    "ecommerce": _spec(
        "ShopHub",
        sections=[
            {"id": "sec-header", "role": "header", "layout": "flex-row", "components": ["logo", "cart", "search"]},
            {"id": "sec-hero", "role": "hero", "layout": "flex-col", "components": ["hero-banner"]},
            {"id": "sec-products", "role": "features", "layout": "grid-3", "components": ["product-card", "product-card-2", "product-card-3"]},
            {"id": "sec-cta", "role": "cta", "layout": "flex-row", "components": ["cta-button"]},
            {"id": "sec-footer", "role": "footer", "layout": "flex-row", "components": []},
        ],
        components=[
            {"name": "logo", "type": "image", "reusable": True},
            {"name": "cart", "type": "button", "reusable": True},
            {"name": "search", "type": "form", "reusable": True},
            {"name": "hero-banner", "type": "card", "reusable": False},
            {"name": "product-card", "type": "card", "reusable": True},
            {"name": "product-card-2", "type": "card", "reusable": True},
            {"name": "product-card-3", "type": "card", "reusable": True},
            {"name": "cta-button", "type": "button", "reusable": True},
        ],
        palette=[{"hex": "#fff7ed", "role": "bg"}, {"hex": "#ffffff", "role": "surface"}, {"hex": "#1c1917", "role": "text"}, {"hex": "#ea580c", "role": "accent"}],
        fonts=["Inter"],
    ),
    # 5. Portfolio — asymmetric, image-forward
    "portfolio": _spec(
        "Studio Folio",
        sections=[
            {"id": "sec-header", "role": "header", "layout": "flex-row", "components": ["logo", "menu"]},
            {"id": "sec-hero", "role": "hero", "layout": "grid-2", "components": ["hero-image", "hero-text"]},
            {"id": "sec-work", "role": "features", "layout": "grid-2", "components": ["project-card", "project-card-2"]},
            {"id": "sec-about", "role": "content", "layout": "flex-col", "components": ["bio"]},
            {"id": "sec-contact", "role": "cta", "layout": "flex-row", "components": ["contact-form"]},
            {"id": "sec-footer", "role": "footer", "layout": "flex-row", "components": []},
        ],
        components=[
            {"name": "logo", "type": "image", "reusable": True},
            {"name": "menu", "type": "nav", "reusable": True},
            {"name": "hero-image", "type": "image", "reusable": False},
            {"name": "hero-text", "type": "card", "reusable": False},
            {"name": "project-card", "type": "card", "reusable": True},
            {"name": "project-card-2", "type": "card", "reusable": True},
            {"name": "bio", "type": "card", "reusable": False},
            {"name": "contact-form", "type": "form", "reusable": True},
        ],
        palette=[{"hex": "#0a0a0a", "role": "bg"}, {"hex": "#171717", "role": "surface"}, {"hex": "#fafafa", "role": "text"}, {"hex": "#f59e0b", "role": "accent"}],
        fonts=["Inter", "Playfair Display"],
    ),
}


# ---------------------------------------------------------------------------
# Gate runner
# ---------------------------------------------------------------------------

async def run_single(
    spec_name: str,
    run_idx: int,
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one generation, return diagnostics."""
    from agent.novision_engine import NovisionEngine

    async def on_event(_event: Dict[str, Any]) -> None:
        pass

    engine = NovisionEngine(
        spec=spec,
        framework=FRAMEWORK,
        stack=STACK,
        model=MODEL,
        base_url=BASE_URL,  # type: ignore[arg-type]
        api_key=API_KEY,  # type: ignore[arg-type]
        max_iterations=30,
        on_event=on_event,
    )

    t0 = time.time()
    try:
        result = await engine.run()
        elapsed = round(time.time() - t0, 1)
        return {
            "spec": spec_name,
            "run": run_idx,
            "ok": True,
            "finished": result["finished"],
            "iterations": result["iterations"],
            "tool_calls": result["tool_call_count"],
            "malformed": result["malformed_tool_calls"],
            "files": len(result["files"]),
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "elapsed_s": elapsed,
            "error": None,
        }
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        return {
            "spec": spec_name,
            "run": run_idx,
            "ok": False,
            "finished": False,
            "iterations": 0,
            "tool_calls": 0,
            "malformed": 0,
            "files": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_s": elapsed,
            "error": f"{type(e).__name__}: {e}",
        }


async def main() -> int:
    print(f"═══════════════════════════════════════════════════════════════")
    print(f"  RELIABILITY GATE (spec §7.6)")
    print(f"  model: {MODEL}  |  {len(SPECS)} specs × {RUNS_PER_SPEC} runs = {len(SPECS)*RUNS_PER_SPEC} total")
    print(f"  framework: {FRAMEWORK}/{STACK}  |  pass = ≥90% finish w/ 0 malformed")
    print(f"═══════════════════════════════════════════════════════════════")
    print()

    results: List[Dict[str, Any]] = []
    total = len(SPECS) * RUNS_PER_SPEC
    i = 0
    for spec_name, spec in SPECS.items():
        for run_idx in range(1, RUNS_PER_SPEC + 1):
            i += 1
            print(f"[{i}/{total}] {spec_name} run {run_idx}...", end=" ", flush=True)
            r = await run_single(spec_name, run_idx, spec)
            results.append(r)
            if r["ok"]:
                verdict = "✓" if (r["finished"] and r["malformed"] == 0) else "⚠"
                print(
                    f"{verdict} finish={r['finished']} malformed={r['malformed']} "
                    f"tools={r['tool_calls']} files={r['files']} "
                    f"tok(in={r['input_tokens']},out={r['output_tokens']}) {r['elapsed_s']}s"
                )
            else:
                print(f"✗ ERROR: {r['error']}")
        print()

    # --- Aggregate ---
    completed = [r for r in results if r["ok"] and r["finished"]]
    clean = [r for r in completed if r["malformed"] == 0]
    errored = [r for r in results if not r["ok"]]
    success_rate = round(100 * len(clean) / total, 1)

    print(f"═══════════════════════════════════════════════════════════════")
    print(f"  AGGREGATE")
    print(f"═══════════════════════════════════════════════════════════════")
    print(f"  total runs          : {total}")
    print(f"  completed to finish : {len(completed)}/{total} ({round(100*len(completed)/total,1)}%)")
    print(f"  clean (0 malformed) : {len(clean)}/{total} ({success_rate}%)")
    print(f"  errored (exception) : {len(errored)}")
    print()

    # Per-spec breakdown
    print("  per-spec:")
    for spec_name in SPECS:
        spec_runs = [r for r in results if r["spec"] == spec_name]
        spec_clean = [r for r in spec_runs if r["ok"] and r["finished"] and r["malformed"] == 0]
        print(f"    {spec_name:12s}: {len(spec_clean)}/{len(spec_runs)} clean")
    print()

    # Token stats (for budget re-benchmark)
    ok_runs = [r for r in results if r["ok"]]
    if ok_runs:
        in_tokens = [r["input_tokens"] for r in ok_runs]
        out_tokens = [r["output_tokens"] for r in ok_runs]
        elapsed = [r["elapsed_s"] for r in ok_runs]
        in_tokens.sort(); out_tokens.sort(); elapsed.sort()
        n = len(ok_runs)
        print("  token usage (for budget re-benchmark, Phase 3.2):")
        print(f"    input  tokens: median={in_tokens[n//2]} min={in_tokens[0]} max={in_tokens[-1]}")
        print(f"    output tokens: median={out_tokens[n//2]} min={out_tokens[0]} max={out_tokens[-1]}")
        print(f"    elapsed (s)  : median={elapsed[n//2]} min={elapsed[0]} max={elapsed[-1]}")
    print()

    # Verdict
    PASS_THRESHOLD = 90.0
    passed = success_rate >= PASS_THRESHOLD
    print(f"═══════════════════════════════════════════════════════════════")
    if passed:
        print(f"  ✅ GATE PASSED — {success_rate}% clean (≥{PASS_THRESHOLD}% required)")
        print(f"     Model {MODEL} is reliable for tool-calling over Chat Completions.")
    else:
        print(f"  ❌ GATE FAILED — {success_rate}% clean (< {PASS_THRESHOLD}% required)")
        print(f"     Model {MODEL} is unreliable. See failures above.")
        if errored:
            print(f"     Errored runs:")
            for r in errored:
                print(f"       {r['spec']} run {r['run']}: {r['error']}")
    print(f"═══════════════════════════════════════════════════════════════")

    # Save results for budget re-benchmark
    out = {
        "model": MODEL,
        "framework": FRAMEWORK,
        "stack": STACK,
        "total_runs": total,
        "success_rate": success_rate,
        "passed": passed,
        "runs": results,
    }
    with open("/tmp/reliability_gate_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → /tmp/reliability_gate_results.json")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
