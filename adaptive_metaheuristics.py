#!/usr/bin/env python3
"""
Full pipeline for LLM-guided metaheuristic operator optimization.

Usage:
    python adaptive_metaheuristics.py --code base_algorithm.py --function crossover \\
        --experiment-name crossover_opt \\
        --variant-provider anthropic --adaptive-provider anthropic

Requires:
    - llm_keys.json with API keys (gitignore this!)
    - GNBG_Runners/ benchmark suite with run_gnbg_II_parallel.py
    - replace_functions.py extraction utility (attached)
    - pip install anthropic numpy requests
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import statistics
import sys
import textwrap
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import sys
# Force UTF-8 for stdout/stderr printing too (the ═ ✔ ⚠ ★ in your prints)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ═══════════════════════════════════════════════════════════════════════
# Imports from companion modules
# ═══════════════════════════════════════════════════════════════════════

from replace_functions import parse_ideas, replace_function_in_source

# Import is deferred for GNBG so the script can be syntax-checked standalone
_GNBG_IMPORTED = False


def _ensure_gnbg():
    global _GNBG_IMPORTED, evaluateGNGB
    if not _GNBG_IMPORTED:
        from GNBG_runners.run_gnbg_II_parallel import evaluateGNGB as _eval
        evaluateGNGB = _eval
        _GNBG_IMPORTED = True


# ═══════════════════════════════════════════════════════════════════════
# LLM Client Abstraction
# ═══════════════════════════════════════════════════════════════════════

def load_keys(path: str = "llm_keys.json") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{path} not found. Create it with:\n"
            '  {"anthropic": "sk-ant-...", "ollama_url": "http://localhost:11434",\n'
            '   "llamacpp_url": "http://localhost:8080"}'
        )
    return json.loads(p.read_text(encoding="utf-8"))


def call_anthropic(
    prompt: str,
    system_prompt: str,
    keys: dict,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 16384,
) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")
    client = anthropic.Anthropic(api_key=keys["anthropic"])
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def call_ollama(
    prompt: str,
    system_prompt: str,
    keys: dict,
    model: str = "llama3.1:70b",
) -> str:
    import requests

    url = keys.get("ollama_url", "http://localhost:11434")
    resp = requests.post(
        f"{url}/api/generate",
        json={
            "model": model,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
        },
        timeout=900,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def call_llamacpp(
    prompt: str,
    system_prompt: str,
    keys: dict,
    model: str | None = None,
    max_tokens: int = 16384,
    temperature: float = 0.7,
    timeout: int = 1800,
) -> str:
    """
    Call a llama.cpp server (started with `llama-server`).

    Uses the OpenAI-compatible /v1/chat/completions endpoint, which is the
    best-supported surface on recent llama.cpp builds and handles the
    system / user role split natively.

    keys fields:
      - llamacpp_url   : base URL, default http://localhost:8080
      - llamacpp_model : optional model name (most llama.cpp servers ignore
                         this and just serve whatever weights they loaded,
                         but the OpenAI schema requires the field)
      - llamacpp_api_key : optional bearer token if the server was started
                           with --api-key
    """
    import requests

    base_url = keys.get("llamacpp_url", "http://localhost:8080").rstrip("/")
    # Accept both "http://host:port" and "http://host:port/v1" in config.
    if base_url.endswith("/v1"):
        endpoint = f"{base_url}/chat/completions"
    else:
        endpoint = f"{base_url}/v1/chat/completions"

    model_name = model or keys.get("llamacpp_model", "local-model")

    headers = {"Content-Type": "application/json"}
    api_key = keys.get("llamacpp_api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # Standard OpenAI-compatible shape.
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            f"Unexpected llama.cpp response shape: {data!r}"
        ) from e


def call_llm(
    prompt: str,
    system_prompt: str,
    provider: str,
    keys: dict,
    model: str | None = None,
) -> str:
    if provider == "anthropic":
        m = model or keys.get("anthropic_model", "claude-sonnet-4-20250514")
        return call_anthropic(prompt, system_prompt, keys, model=m)
    elif provider == "ollama":
        m = model or keys.get("ollama_model", "qwen3-coder-next:q8_0")
        return call_ollama(prompt, system_prompt, keys, model=m)
    elif provider == "llamacpp":
        m = model or keys.get("llamacpp_model")  # may be None; call_llamacpp handles it
        return call_llamacpp(prompt, system_prompt, keys, model=m)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


# ═══════════════════════════════════════════════════════════════════════
# Source Code Helpers
# ═══════════════════════════════════════════════════════════════════════

def extract_function_signature(code: str, func_name: str) -> str:
    """Return the 'def ...(...)' line of the named function."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                lines = code.splitlines()
                sig_parts = []
                for i in range(node.lineno - 1, min(node.lineno + 5, len(lines))):
                    sig_parts.append(lines[i])
                    if ")" in lines[i] and ":" in lines[i]:
                        break
                return "\n".join(sig_parts).strip()
    raise KeyError(f"Function '{func_name}' not found in source")


def extract_full_function(code: str, func_name: str) -> str | None:
    """Return the full source of a function from a code string."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                seg = ast.get_source_segment(code, node)
                if seg:
                    return seg
    return None


def extract_full_class_code(response_text: str) -> str | None:
    """Pull the longest python code block from an LLM response."""
    blocks = re.findall(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
    if not blocks:
        return None
    longest = max(blocks, key=len)
    # Quick syntax check
    try:
        ast.parse(longest)
    except SyntaxError:
        return None
    return longest


# ═══════════════════════════════════════════════════════════════════════
# Prompt Construction
# ═══════════════════════════════════════════════════════════════════════

VARIANT_SYSTEM = (
    "You are an expert in metaheuristic optimization, evolutionary computation, "
    "and differential evolution. You produce concise, correct, numerically robust "
    "Python code. Format every proposal exactly as:\n"
    "**Idea N: Short Name**\n"
    "One-line description.\n"
    "```python\n"
    "def func_name(self, ...):\n"
    "    ...\n"
    "```\n"
    "Do NOT include anything outside the function in the code block."
)


def build_variant_prompt(code: str, func_name: str, n_ideas: int = 8) -> str:
    sig = extract_function_signature(code, func_name)
    func_body = extract_full_function(code, func_name) or ""
    return f"""\
Below is a metaheuristic optimizer. Propose {n_ideas} DISTINCT replacement \
implementations for `{func_name}`.

Requirements:
- Keep the EXACT function signature: `{sig}`
- Each idea must use a fundamentally different strategy/operator
- Include well-known operators (exponential crossover, arithmetic blend, \
segment-based, eigenvector-rotated) AND creative / unconventional ones
- You may read any `self` attribute but do NOT modify `__init__` or other methods
- Each fenced code block must contain ONLY the single replacement function
- Ensure numerical robustness (no division by zero, handle edge dims, clip to bounds)

Current implementation:
```python
{func_body}
```

Full algorithm for context:
```python
{code}
```

Respond with exactly {n_ideas} ideas using the Idea N format."""


ADAPTIVE_SYSTEM = (
    "You are an expert in metaheuristic optimization and adaptive operator selection. "
    "Respond ONLY with the full, runnable Python class inside a single ```python``` "
    "code block.  Include `import numpy as np` at the top. "
    "No explanatory text outside the code block. "
    "The class must be callable as: algorithm(func, stopping_condition) and return (f_opt, x_opt)."
)


def build_adaptive_prompt(
    code: str,
    func_name: str,
    analysis: dict,
    variant_codes: str,
) -> str:
    return f"""\
Given benchmark results on 24 optimization test functions, create an ADAPTIVE \
version of the algorithm that automatically selects the best `{func_name}` \
strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
{analysis['results_table']}

BEST VARIANT PER FUNCTION (non-trivial tasks only):
{analysis['best_per_task']}

WIN COUNTS:
{analysis['win_counts_str']}

WINNING VARIANT IMPLEMENTATIONS:
{variant_codes}

Requirements:
- Respond with the COMPLETE class code inside a single ```python``` block
- Include `import numpy as np` at the top
- Use an adaptive mechanism (Thompson Sampling, Multi-Armed Bandit, or sliding \
window credit assignment) to learn which operator works best during a run
- You may add new attributes in __init__ and new private helper methods
- Do NOT change signatures of existing public methods (__call__, crossover, \
mutation, selection, etc.)
- The class MUST be callable as: `obj(func, stopping_condition)` → (f_opt, x_opt)
- Handle ALL edge cases: no -inf, no crashes, no NaN, clip to bounds
- Use integer-index-based operator tracking (not id()-based)
- Cast all reward/fitness values to float scalars before storing

Original algorithm:
```python
{code}
```"""


# ═══════════════════════════════════════════════════════════════════════
# Experiment Directory Management
# ═══════════════════════════════════════════════════════════════════════

def create_experiment_dir(name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"{name}-{ts}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "variants").mkdir(exist_ok=True)
    (exp_dir / "prompts").mkdir(exist_ok=True)
    return exp_dir


# ═══════════════════════════════════════════════════════════════════════
# Pipeline Steps
# ═══════════════════════════════════════════════════════════════════════

def step_generate_variants(
    base_code: str,
    func_name: str,
    provider: str,
    keys: dict,
    model: str | None,
    exp_dir: Path,
    n_ideas: int = 8,
) -> list[tuple[str, Path, str]]:
    """Phase 1: Call LLM to generate variant operator functions."""
    print("\n" + "=" * 60)
    print("PHASE 1: Generating variant operators via LLM")
    print("=" * 60)

    prompt = build_variant_prompt(base_code, func_name, n_ideas)
    (exp_dir / "prompts" / "variant_prompt.md").write_text(prompt, encoding="utf-8")

    print(f"  Calling {provider} for {n_ideas} variants...")
    response = call_llm(prompt, VARIANT_SYSTEM, provider, keys, model)
    (exp_dir / "prompts" / "variant_response.md").write_text(response, encoding="utf-8")

    ideas = parse_ideas(response)
    print(f"  Parsed {len(ideas)} idea(s) from LLM response")

    # Save original as a variant too
    orig_path = exp_dir / "variants" / "original.py"
    orig_path.write_text(base_code, encoding="utf-8")
    variant_paths: list[tuple[str, Path, str]] = [
        ("original.py", orig_path, "original")
    ]

    for idea in ideas:
        if not idea.snippets:
            continue
        src = base_code
        applied = []
        for snip in idea.snippets:
            try:
                src = replace_function_in_source(src, snip.name, snip.source)
                applied.append(snip.name)
            except KeyError as e:
                print(f"    ⚠  {idea.label}: {e} – skipped")
        if not applied:
            continue

        slug = idea.label.lower().replace(" ", "_")
        fname = f"variant_{slug}.py"
        fpath = exp_dir / "variants" / fname
        fpath.write_text(src, encoding="utf-8")
        variant_paths.append((fname, fpath, "variant"))
        print(f"    ✔  {fname}  (replaced: {', '.join(applied)})")

    return variant_paths


# ═══════════════════════════════════════════════════════════════════════
# Iterative variant generation (one-at-a-time with feedback)
# ═══════════════════════════════════════════════════════════════════════

ITERATIVE_VARIANT_SYSTEM = (
    "You are an expert in metaheuristic optimization, evolutionary computation, "
    "and differential evolution. You will propose ONE replacement implementation "
    "at a time for a target function, using feedback on how well previously "
    "generated variants performed on each of 24 benchmark tasks. "
    "Your explicit goal is to find an operator that WINS on tasks where "
    "previous variants are losing. "
    "IMPORTANT CONTEXT: After all variants are generated, an ADAPTIVE version "
    "of the algorithm will be built that selects the best operator PER TASK "
    "at runtime. This means we need AT LEAST ONE good variant for EVERY task — "
    "a variant that wins on 10 tasks but fails on 14 is still valuable, but "
    "tasks with NO good variant are a critical gap that you must fill. "
    "Format your proposal exactly as:\n"
    "**Idea: Short Name**\n"
    "One-line description.\n"
    "```python\n"
    "def func_name(self, ...):\n"
    "    ...\n"
    "```\n"
    "Do NOT include anything outside the function in the code block."
)


def _format_per_task_feedback_table(
    results: dict[str, dict],
    variant_order: list[str],
    max_cols: int = 8,
) -> str:
    """
    Build a compact per-task error table across (at most) the most recent
    `max_cols` variants plus the original. Rows are tasks 0..23, columns
    are algorithm names. Values are formatted in scientific notation.
    """
    cols: list[str] = []
    if "original.py" in results:
        cols.append("original.py")
    # Keep the most recent variants (iterative order is what the caller passed).
    extras = [n for n in variant_order if n != "original.py"]
    cols.extend(extras[-max_cols:])

    header = f"{'Task':<5}" + "".join(f"{c[:20]:<22}" for c in cols)
    sep = "-" * len(header)
    rows = [header, sep]
    for t in range(24):
        row = f"{t:<5}"
        for c in cols:
            res = results.get(c, {}).get("results")
            if res is None or len(res) != 24:
                row += f"{'FAILED':<22}"
            else:
                v = res[t]
                row += f"{v:<22.4e}"
        rows.append(row)
    return "\n".join(rows)


def _identify_weak_tasks(
    results: dict[str, dict],
    variant_order: list[str],
) -> list[int]:
    """
    Return task indices where NO generated variant clearly improves on the
    original — i.e. tasks still lacking a good operator. Non-trivial tasks
    only (skip tasks where everybody is essentially at the optimum).
    """
    orig = results.get("original.py", {}).get("results")
    if orig is None or len(orig) != 24:
        return []
    variants = [n for n in variant_order if n != "original.py"]
    if not variants:
        return []

    weak: list[int] = []
    for t in range(24):
        orig_v = orig[t]
        if not np.isfinite(orig_v):
            continue
        # Trivial: original already at optimum → skip.
        if abs(orig_v) < 1e-7:
            continue
        best_variant = np.inf
        for n in variants:
            res = results.get(n, {}).get("results")
            if res is None or len(res) != 24:
                continue
            v = res[t]
            if np.isfinite(v) and v < best_variant:
                best_variant = v
        # Weak: no variant beat original by a meaningful margin.
        if best_variant >= orig_v * 0.9:
            weak.append(t)
    return weak


def _format_task_coverage(
    results: dict[str, dict],
    variant_order: list[str],
) -> str:
    """
    Build a per-task coverage summary showing:
      - Which tasks are COVERED (at least one variant beats the original)
      - Which tasks are UNCOVERED (no variant improves on original)
      - Which variant is the current best for each covered task
      - The error gap on uncovered tasks

    This gives the LLM a clear map of where to focus next.
    """
    orig = results.get("original.py", {}).get("results")
    if orig is None or len(orig) != 24:
        return "(original not yet benchmarked)"
    variants = [n for n in variant_order if n != "original.py"]

    lines: list[str] = []
    covered = 0
    uncovered = 0
    trivial = 0

    for t in range(24):
        orig_v = orig[t]
        if not np.isfinite(orig_v):
            lines.append(f"  Task {t:2d}: CRASHED in original (error=-inf)")
            uncovered += 1
            continue
        if abs(orig_v) < 1e-7:
            lines.append(f"  Task {t:2d}: TRIVIAL (already at optimum, error={orig_v:.2e})")
            trivial += 1
            continue

        # Find best variant for this task
        best_name = None
        best_val = orig_v
        for n in variants:
            res = results.get(n, {}).get("results")
            if res is None or len(res) != 24:
                continue
            v = res[t]
            if np.isfinite(v) and v < best_val:
                best_val = v
                best_name = n

        if best_name is not None:
            improvement = (orig_v - best_val) / orig_v * 100
            lines.append(
                f"  Task {t:2d}: COVERED — best={best_name[:25]:<26} "
                f"error={best_val:.3e} (orig={orig_v:.3e}, -{improvement:.0f}%)"
            )
            covered += 1
        else:
            lines.append(
                f"  Task {t:2d}: *** UNCOVERED *** — orig error={orig_v:.3e}, "
                f"no variant improves on this"
            )
            uncovered += 1

    summary = (
        f"TASK COVERAGE SUMMARY: {covered} covered, {uncovered} UNCOVERED, "
        f"{trivial} trivial (out of 24)\n"
        f"Goal: get at least one winning variant for every non-trivial task.\n\n"
    )
    return summary + "\n".join(lines)


def build_iterative_variant_prompt(
    code: str,
    func_name: str,
    iteration: int,
    total: int,
    results_so_far: dict[str, dict],
    variant_order: list[str],
    prior_labels: list[str],
) -> str:
    sig = extract_function_signature(code, func_name)
    func_body = extract_full_function(code, func_name) or ""

    # First iteration: no feedback yet.
    if iteration == 1 or not any(n != "original.py" for n in variant_order):
        feedback_section = (
            "This is the FIRST proposal, so no feedback is available yet. "
            "Propose a strong, well-known baseline operator that is known to "
            "work across many problem types."
        )
    else:
        table = _format_per_task_feedback_table(results_so_far, variant_order)
        coverage = _format_task_coverage(results_so_far, variant_order)
        weak = _identify_weak_tasks(results_so_far, variant_order)
        weak_str = (
            ", ".join(str(t) for t in weak)
            if weak else "(every non-trivial task already has an improving variant)"
        )
        prior_names = ", ".join(prior_labels) if prior_labels else "(none)"
        feedback_section = f"""\
WHY PER-TASK COVERAGE MATTERS:
After all variants are generated, an ADAPTIVE algorithm will be built that
selects the best operator FOR EACH TASK at runtime (e.g. via Thompson Sampling
or multi-armed bandit). This means:
- We do NOT need a single variant that wins everywhere.
- We DO need at least one variant that wins on EACH task.
- Tasks marked UNCOVERED below are critical gaps — your variant should
  specifically target those.

{coverage}

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
{table}

TASKS WHERE NO VARIANT CURRENTLY BEATS THE ORIGINAL (priority targets):
{weak_str}

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
{prior_names}

Your new proposal MUST be designed to perform WELL on the UNCOVERED tasks \
listed above, even at the cost of being worse on tasks where other variants \
already succeed. Reason explicitly:
1. What property of those uncovered tasks (multimodality, ill-conditioning, \
separability, ruggedness, noise, deceptive local optima, etc.) are the \
existing operators failing to handle?
2. What specific mechanism in your proposed operator addresses that property?
3. Why is this approach fundamentally different from the prior variants?"""

    return f"""\
This is iteration {iteration} of {total}. Propose ONE replacement implementation \
for `{func_name}`.

Requirements:
- Keep the EXACT function signature: `{sig}`
- Propose a SINGLE, FUNDAMENTALLY DIFFERENT strategy vs. the variants below
- You may read any `self` attribute but do NOT modify `__init__` or other methods
- The fenced code block must contain ONLY the single replacement function
- Ensure numerical robustness (no division by zero, handle edge dims, clip to bounds)

{feedback_section}

Current implementation:
```python
{func_body}
```

Full algorithm for context:
```python
{code}
```

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description.
```python
def {func_name}(self, ...):
    ...
```"""


def step_generate_variants_iterative(
    base_code: str,
    func_name: str,
    provider: str,
    keys: dict,
    model: str | None,
    exp_dir: Path,
    n_ideas: int,
    budget: int,
    repetitions: int,
) -> tuple[list[tuple[str, Path, str]], dict[str, dict]]:
    """
    Phase 1 (iterative): generate variants ONE AT A TIME, benchmarking each
    immediately and feeding per-task results back to the LLM so it can target
    tasks where prior variants are weak.

    Returns (variant_paths, variant_results) — the results dict is returned
    so callers can skip re-benchmarking in Phase 2.
    """
    print("\n" + "=" * 60)
    print(f"PHASE 1 (iterative): generating up to {n_ideas} variant(s) "
          f"one-at-a-time with per-task feedback")
    print("=" * 60)

    # Save original and benchmark it first so we have a baseline to beat.
    orig_path = exp_dir / "variants" / "original.py"
    orig_path.write_text(base_code, encoding="utf-8")
    variant_paths: list[tuple[str, Path, str]] = [("original.py", orig_path, "original")]

    print("\n  Benchmarking ORIGINAL (baseline) ...")
    results: dict[str, dict] = step_run_benchmarks(
        variant_paths, budget, repetitions, label="ORIGINAL (baseline)"
    )

    variant_order = ["original.py"]
    prior_labels: list[str] = []

    for i in range(1, n_ideas + 1):
        # Check coverage — if all non-trivial tasks are covered, we can stop early
        weak = _identify_weak_tasks(results, variant_order)
        if i > 3 and not weak:
            print(f"\n  All non-trivial tasks have a winning variant — "
                  f"stopping early at iteration {i-1}/{n_ideas}.")
            break

        print(f"\n  --- Iteration {i}/{n_ideas} "
              f"(uncovered tasks: {len(weak) if weak else 'checking...'}) ---")
        prompt = build_iterative_variant_prompt(
            base_code, func_name, i, n_ideas,
            results, variant_order, prior_labels,
        )
        prompt_file = exp_dir / "prompts" / f"iter_{i:02d}_prompt.md"
        prompt_file.write_text(prompt, encoding="utf-8")

        print(f"  Calling {provider} for variant {i}...")
        try:
            response = call_llm(prompt, ITERATIVE_VARIANT_SYSTEM, provider, keys, model)
        except Exception as e:
            print(f"    ⚠  LLM call failed: {e}; skipping iteration.")
            (exp_dir / "prompts" / f"iter_{i:02d}_response.md").write_text(
                f"ERROR: {e}", encoding="utf-8"
            )
            continue
        (exp_dir / "prompts" / f"iter_{i:02d}_response.md").write_text(
            response, encoding="utf-8"
        )

        ideas = parse_ideas(response)
        if not ideas:
            print("    ⚠  No idea parsed from response; skipping.")
            continue
        idea = ideas[0]
        if not idea.snippets:
            print("    ⚠  Idea contained no code snippets; skipping.")
            continue

        src = base_code
        applied: list[str] = []
        for snip in idea.snippets:
            try:
                src = replace_function_in_source(src, snip.name, snip.source)
                applied.append(snip.name)
            except KeyError as e:
                print(f"    ⚠  {idea.label}: {e} – skipped")
        if not applied:
            print("    ⚠  No snippet applied; skipping.")
            continue

        slug = idea.label.lower().replace(" ", "_")
        fname = f"variant_{i:02d}_{slug}.py"
        fpath = exp_dir / "variants" / fname
        fpath.write_text(src, encoding="utf-8")
        print(f"    ✔  Saved {fname}  (replaced: {', '.join(applied)})")

        # Benchmark this single variant and merge into cumulative results.
        single = [(fname, fpath, "variant")]
        new_res = step_run_benchmarks(
            single, budget, repetitions, label=f"variant {i} ({idea.label})"
        )
        results.update(new_res)
        variant_paths.append((fname, fpath, "variant"))
        variant_order.append(fname)
        prior_labels.append(idea.label)

    # Final coverage report
    coverage = _format_task_coverage(results, variant_order)
    print(f"\n  FINAL COVERAGE:\n{coverage}")

    return variant_paths, results


def step_run_benchmarks(
    algo_list: list[tuple[str, Path, str]],
    budget: int,
    repetitions: int,
    label: str = "",
) -> dict[str, dict]:
    """Run GNBG benchmarks on a list of algorithms."""
    _ensure_gnbg()

    print(f"\n{'=' * 60}")
    print(f"BENCHMARKING: {label}")
    print(f"{'=' * 60}")

    results: dict[str, dict] = {}

    for fname, fpath, method in algo_list:
        code = fpath.read_text(encoding="utf-8")
        print(f"  Running {fname} ...", end=" ", flush=True)

        start = time.perf_counter()
        try:
            raw = evaluateGNGB(code, iterations=budget, repetitions_per_fid=repetitions)
            elapsed = time.perf_counter() - start

            # evaluateGNGB may return a list or a string representation
            if isinstance(raw, str):
                raw = eval(raw.replace("inf", "np.inf"))
            result_list = [float(np.ravel(v)[0]) if hasattr(v, "__len__") else float(v)
                           for v in raw]

            has_bad = any(v == -np.inf or np.isnan(v) for v in result_list)
            if has_bad:
                safe_vals = [v for v in result_list if v > 0 and np.isfinite(v)]
                avg_str = f"geomean={statistics.geometric_mean(safe_vals):.4e}" if safe_vals else "ALL-BAD"
            else:
                safe_vals = [max(v, 1e-30) for v in result_list]
                avg_str = f"geomean={statistics.geometric_mean(safe_vals):.4e}"

            print(f"done ({elapsed:.1f}s) {avg_str}")

            results[fname] = {
                "path": str(fpath),
                "method": method,
                "results": result_list,
                "time": elapsed,
                "error": None,
            }
        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"FAILED ({elapsed:.1f}s) — {e}")
            results[fname] = {
                "path": str(fpath),
                "method": method,
                "results": None,
                "time": elapsed,
                "error": str(e),
            }

    return results


def step_analyze(results: dict[str, dict]) -> dict | None:
    """Analyze benchmark results: best per task, win counts, tables."""
    print(f"\n{'=' * 60}")
    print("ANALYSIS")
    print("=" * 60)

    N_TASKS = 24
    valid = {
        k: v
        for k, v in results.items()
        if v["results"] is not None and len(v["results"]) == N_TASKS
    }
    if not valid:
        print("  No valid results to analyze!")
        return None

    # ── best per task ────────────────────────────────────────────────
    best_per_task: list[tuple[int, str | None, float | None]] = []
    active_tasks: list[int] = []

    for t in range(N_TASKS):
        task_vals = {}
        for name, data in valid.items():
            v = data["results"][t]
            if np.isfinite(v):
                task_vals[name] = v

        if not task_vals:
            best_per_task.append((t, None, None))
            continue

        # Skip trivial tasks (>50% variants below 1e-7)
        below = sum(1 for v in task_vals.values() if abs(v) < 1e-7)
        if below > len(task_vals) / 2:
            best_per_task.append((t, "SKIP", None))
            continue

        best_name = min(task_vals, key=task_vals.get)
        best_per_task.append((t, best_name, task_vals[best_name]))
        active_tasks.append(t)

    # ── win counts ───────────────────────────────────────────────────
    win_counts: dict[str, int] = {}
    for t, name, val in best_per_task:
        if name and name not in (None, "SKIP"):
            win_counts[name] = win_counts.get(name, 0) + 1

    win_lines = sorted(win_counts.items(), key=lambda x: -x[1])
    win_counts_str = "\n".join(f"  {name}: {cnt} wins" for name, cnt in win_lines)

    # ── full results table (compact) ─────────────────────────────────
    names = list(valid.keys())
    header = f"{'Task':<7}" + "".join(f"{n[:22]:<24}" for n in names)
    table_rows = [header, "-" * len(header)]
    for t in range(N_TASKS):
        row = f"{t:<7}"
        for n in names:
            v = valid[n]["results"][t]
            row += f"{v:<24.6e}"
        table_rows.append(row)
    results_table = "\n".join(table_rows)

    # ── best-per-task summary string ─────────────────────────────────
    bpt_lines = []
    for t, name, val in best_per_task:
        if name == "SKIP":
            bpt_lines.append(f"Task {t:2d}: SKIPPED (trivial)")
        elif name is None:
            bpt_lines.append(f"Task {t:2d}: NO VALID RESULT")
        else:
            bpt_lines.append(f"Task {t:2d}: {name}  (error={val:.6e})")
    best_per_task_str = "\n".join(bpt_lines)

    # ── print summary ────────────────────────────────────────────────
    skipped = sum(1 for _, n, _ in best_per_task if n == "SKIP")
    print(f"  {len(valid)} valid algorithms, {N_TASKS} tasks, {skipped} trivial-skipped")
    print(f"\n  Win counts:")
    for name, cnt in win_lines:
        print(f"    {name}: {cnt}")
    print()
    for line in bpt_lines:
        if "SKIP" not in line:
            print(f"  {line}")

    return {
        "results_table": results_table,
        "best_per_task": best_per_task_str,
        "best_per_task_list": best_per_task,
        "win_counts": win_counts,
        "win_counts_str": win_counts_str,
        "active_tasks": active_tasks,
        "valid_results": valid,
    }


def step_generate_adaptive(
    base_code: str,
    func_name: str,
    analysis: dict,
    all_results: dict[str, dict],
    provider: str,
    keys: dict,
    model: str | None,
    exp_dir: Path,
    n_tries: int = 3,
) -> list[tuple[str, Path, str]]:
    """Phase 3: Generate adaptive operator code via LLM (multiple tries)."""
    print(f"\n{'=' * 60}")
    print("PHASE 3: Generating adaptive operators via LLM")
    print("=" * 60)

    # Collect winning variant function implementations
    winners: set[str] = set()
    for _, name, _ in analysis["best_per_task_list"]:
        if name and name not in (None, "SKIP", "original.py"):
            winners.add(name)

    variant_code_parts = []
    for name in sorted(winners):
        fpath = Path(all_results[name]["path"])
        code = fpath.read_text(encoding="utf-8")
        func_code = extract_full_function(code, func_name)
        if func_code:
            variant_code_parts.append(
                f"# --- From {name} ({analysis['win_counts'].get(name, 0)} wins) ---\n"
                f"```python\n{func_code}\n```"
            )

    variant_codes_str = "\n\n".join(variant_code_parts) if variant_code_parts else "(no non-original winners)"
    prompt = build_adaptive_prompt(base_code, func_name, analysis, variant_codes_str)

    adaptive_paths: list[tuple[str, Path, str]] = []

    for attempt in range(1, n_tries + 1):
        print(f"\n  Attempt {attempt}/{n_tries} ...")
        prompt_file = exp_dir / "prompts" / f"adaptive_prompt_{attempt}.md"
        response_file = exp_dir / "prompts" / f"adaptive_response_{attempt}.md"
        prompt_file.write_text(prompt, encoding="utf-8")

        try:
            response = call_llm(prompt, ADAPTIVE_SYSTEM, provider, keys, model)
        except Exception as e:
            print(f"    ⚠  LLM call failed: {e}")
            response_file.write_text(f"ERROR: {e}", encoding="utf-8")
            continue

        response_file.write_text(response, encoding="utf-8")
        adaptive_code = extract_full_class_code(response)

        if adaptive_code is None:
            print("    ⚠  No valid Python code block found in response")
            continue

        # Ensure numpy import
        if "import numpy" not in adaptive_code:
            adaptive_code = "import numpy as np\n\n" + adaptive_code

        fname = f"adaptive_attempt_{attempt}.py"
        fpath = exp_dir / "variants" / fname
        fpath.write_text(adaptive_code, encoding="utf-8")
        adaptive_paths.append((fname, fpath, "adaptive"))
        print(f"    ✔  Saved {fname} ({len(adaptive_code)} chars)")

    return adaptive_paths


# ═══════════════════════════════════════════════════════════════════════
# Results Persistence
# ═══════════════════════════════════════════════════════════════════════

def save_results_csv(all_results: dict[str, dict], exp_dir: Path) -> Path:
    """Write comprehensive results.csv into the experiment directory."""
    N_TASKS = 24
    csv_path = exp_dir / "results.csv"

    fieldnames = [
        "algorithm",
        "generation_method",
        "avg_result",
        "time_seconds",
        "error",
    ] + [f"task_{i}" for i in range(N_TASKS)] + ["prompt_file"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name, data in all_results.items():
            row: dict = {
                "algorithm": name,
                "generation_method": data["method"],
                "time_seconds": f"{data['time']:.2f}",
                "error": data.get("error") or "",
            }

            res = data.get("results")
            if res and len(res) == N_TASKS:
                good = [v for v in res if np.isfinite(v) and v > 0]
                row["avg_result"] = (
                    f"{statistics.geometric_mean(good):.6e}" if good else "N/A"
                )
                for i in range(N_TASKS):
                    row[f"task_{i}"] = res[i]
            else:
                row["avg_result"] = "FAILED"
                for i in range(N_TASKS):
                    row[f"task_{i}"] = ""

            # Link to the prompt that generated this algorithm
            if data["method"] == "original":
                row["prompt_file"] = ""
            elif data["method"] == "variant":
                row["prompt_file"] = "prompts/variant_prompt.md"
            elif data["method"] == "adaptive":
                # Extract attempt number from filename like adaptive_attempt_2.py
                m = re.search(r"(\d+)", name)
                n = m.group(1) if m else "1"
                row["prompt_file"] = f"prompts/adaptive_prompt_{n}.md"
            else:
                row["prompt_file"] = ""

            writer.writerow(row)

    print(f"\n  Results CSV saved to {csv_path}")
    return csv_path


def save_summary(all_results: dict[str, dict], analysis: dict | None, exp_dir: Path):
    """Write a human-readable summary.txt."""
    lines = [
        f"Experiment: {exp_dir.name}",
        f"Timestamp: {datetime.now().isoformat()}",
        "",
        "=" * 60,
        "ALGORITHM RANKINGS (by geometric mean of finite positive tasks)",
        "=" * 60,
        "",
    ]

    ranked = []
    for name, data in all_results.items():
        res = data.get("results")
        if res and len(res) == 24:
            good = [v for v in res if np.isfinite(v) and v > 0]
            if good:
                gm = statistics.geometric_mean(good)
                ranked.append((name, data["method"], gm, len(good)))

    ranked.sort(key=lambda x: x[2])
    for i, (name, method, gm, n_good) in enumerate(ranked, 1):
        lines.append(f"  {i:2d}. {name:<40} [{method:<10}]  geomean={gm:.6e}  ({n_good}/24 valid)")

    failed = [
        name
        for name, data in all_results.items()
        if data.get("results") is None or len(data.get("results", [])) != 24
    ]
    if failed:
        lines += ["", "FAILED ALGORITHMS:", ""]
        for name in failed:
            err = all_results[name].get("error", "unknown")
            lines.append(f"  - {name}: {err}")

    if analysis:
        lines += [
            "",
            "=" * 60,
            "PER-TASK BEST VARIANTS",
            "=" * 60,
            "",
            analysis["best_per_task"],
            "",
            "WIN COUNTS:",
            analysis["win_counts_str"],
        ]

    summary_path = exp_dir / "summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Summary saved to {summary_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main Pipeline Orchestration
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LLM-guided metaheuristic operator optimization pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example:
  python adaptive_metaheuristics.py --code my_algorithm.py --function crossover \\
      --experiment-name xover_test --variant-provider anthropic \\
      --adaptive-provider anthropic --budget 500000 --n-ideas 8
""",
    )
    parser.add_argument("--code", required=True, help="Path to base algorithm .py file")
    parser.add_argument("--function", required=True, help="Name of function to optimize")
    parser.add_argument("--experiment-name", default="experiment", help="Experiment name prefix")
    parser.add_argument(
        "--variant-provider",
        default="anthropic",
        choices=["anthropic", "ollama", "llamacpp"],
        help="LLM provider for variant generation",
    )
    parser.add_argument("--variant-model", default=None, help="Override model for variants")
    parser.add_argument(
        "--adaptive-provider",
        default="anthropic",
        choices=["anthropic", "ollama", "llamacpp"],
        help="LLM provider for adaptive generation",
    )
    parser.add_argument("--adaptive-model", default=None, help="Override model for adaptive")
    parser.add_argument("--budget", type=int, default=1_000_000, help="Evaluation budget per benchmark")
    parser.add_argument("--repetitions", type=int, default=6, help="Repetitions per function ID")
    parser.add_argument("--n-ideas", type=int, default=8, help="Number of variant ideas to request")
    parser.add_argument("--n-adaptive-tries", type=int, default=3, help="Number of adaptive attempts")
    parser.add_argument(
        "--variant-mode",
        default="batch",
        choices=["batch", "iterative"],
        help=(
            "How to generate variants. 'batch': one LLM call asking for "
            "N variants at once (original behavior). 'iterative': generate "
            "one at a time, benchmarking each and feeding per-task errors "
            "back to the LLM so it can target weak tasks. Best used with "
            "loop-friendly models like MiniMax."
        ),
    )
    parser.add_argument("--keys", default="llm_keys.json", help="Path to LLM API keys file")
    parser.add_argument(
        "--skip-variant-benchmark",
        action="store_true",
        help="Skip variant benchmarking (use existing results)",
    )
    parser.add_argument(
        "--skip-adaptive-benchmark",
        action="store_true",
        help="Skip adaptive benchmarking",
    )
    args = parser.parse_args()

    # ── Load inputs ──────────────────────────────────────────────────
    keys = load_keys(args.keys)
    base_code = Path(args.code).read_text(encoding="utf-8")

    try:
        sig = extract_function_signature(base_code, args.function)
    except KeyError:
        print(f"ERROR: function '{args.function}' not found in {args.code}")
        sys.exit(1)

    print(f"Target function: {args.function}")
    print(f"  Signature: {sig.splitlines()[0].strip()}")

    # ── Create experiment directory ──────────────────────────────────
    exp_dir = create_experiment_dir(args.experiment_name)
    print(f"Experiment directory: {exp_dir}/")

    config = {**vars(args), "timestamp": datetime.now().isoformat()}
    (exp_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1  —  Generate variant functions
    # ══════════════════════════════════════════════════════════════════
    if args.variant_mode == "iterative":
        # Iterative mode generates AND benchmarks each variant in the loop,
        # so --skip-variant-benchmark is meaningless here.
        variant_paths, variant_results = step_generate_variants_iterative(
            base_code,
            args.function,
            args.variant_provider,
            keys,
            args.variant_model,
            exp_dir,
            args.n_ideas,
            args.budget,
            args.repetitions,
        )
        if len(variant_paths) <= 1:
            print("\n  ⚠  No usable variants generated. Check prompts/.")
            print("  Aborting.")
            sys.exit(1)
    else:
        variant_paths = step_generate_variants(
            base_code,
            args.function,
            args.variant_provider,
            keys,
            args.variant_model,
            exp_dir,
            args.n_ideas,
        )

        if len(variant_paths) <= 1:
            print("\n  ⚠  No usable variants extracted. Check the LLM response in prompts/.")
            print("  Aborting.")
            sys.exit(1)

        # ══════════════════════════════════════════════════════════════════
        # PHASE 2  —  Benchmark variants
        # ══════════════════════════════════════════════════════════════════
        if args.skip_variant_benchmark:
            print("\n  Skipping variant benchmarks (--skip-variant-benchmark)")
            variant_results = {}
        else:
            variant_results = step_run_benchmarks(
                variant_paths, args.budget, args.repetitions, label="VARIANTS"
            )

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2b — Analyze
    # ══════════════════════════════════════════════════════════════════
    analysis = step_analyze(variant_results)
    if analysis is None:
        print("\n  No valid results to build adaptive prompt from. Saving what we have.")
        save_results_csv(variant_results, exp_dir)
        save_summary(variant_results, None, exp_dir)
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3  —  Generate adaptive operators
    # ══════════════════════════════════════════════════════════════════
    adaptive_paths = step_generate_adaptive(
        base_code,
        args.function,
        analysis,
        variant_results,
        args.adaptive_provider,
        keys,
        args.adaptive_model,
        exp_dir,
        args.n_adaptive_tries,
    )

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4  —  Benchmark adaptive operators
    # ══════════════════════════════════════════════════════════════════
    adaptive_results: dict[str, dict] = {}
    if adaptive_paths and not args.skip_adaptive_benchmark:
        adaptive_results = step_run_benchmarks(
            adaptive_paths, args.budget, args.repetitions, label="ADAPTIVE"
        )
    elif not adaptive_paths:
        print("\n  No valid adaptive variants were generated.")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5  —  Save everything
    # ══════════════════════════════════════════════════════════════════
    all_results = {**variant_results, **adaptive_results}
    save_results_csv(all_results, exp_dir)
    save_summary(all_results, analysis, exp_dir)

    # ── Final leaderboard ────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("FINAL LEADERBOARD")
    print("=" * 60)

    ranked = []
    for name, data in all_results.items():
        res = data.get("results")
        if res and len(res) == 24:
            good = [v for v in res if np.isfinite(v) and v > 0]
            if good:
                gm = statistics.geometric_mean(good)
                ranked.append((name, data["method"], gm))

    ranked.sort(key=lambda x: x[2])
    for i, (name, method, gm) in enumerate(ranked, 1):
        marker = " ★" if method == "adaptive" else ""
        print(f"  {i:2d}. {name:<40} [{method:<10}] geomean={gm:.6e}{marker}")

    failed_count = sum(1 for d in all_results.values() if d.get("results") is None)
    if failed_count:
        print(f"\n  ({failed_count} algorithm(s) failed — see summary.txt)")

    print(f"\n  All data saved in: {exp_dir}/")
    print("  Done.")


if __name__ == "__main__":
    main()