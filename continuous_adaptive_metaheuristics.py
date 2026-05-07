#!/usr/bin/env python3
"""
Continuous LLM-guided metaheuristic evolution.

High-level loop:
  PHASE 0 (once, at start of a new lineage)
    - Ask the LLM for N fresh, full-class metaheuristics (multi-method classes),
      informed by per-task results from previous lineages/cycles.
    - Benchmark them, pick the best as the "champion".
    - FALLBACK: if no seed produces good results, promote the global best from
      prior cycles as the new champion (avoids losing progress).

  CYCLE (repeats forever)
    1. Pick a not-yet-adapted method of the current champion, using one of
       three strategies (--component-selector):
         * smallest : deterministic; always pick the method with the fewest LOC
         * random   : uniform random pick from remaining methods
         * llm      : ask the LLM for its top-3 picks (with full algorithm
                      context + history of what's already been tried) and
                      pick one of those three uniformly at random
    2. Run the component-level variant + adaptive pipeline on that method.
       Methods that have ALREADY been made adaptive (in a previous cycle of
       this lineage) are excluded from selection — they won't be adapted twice.
    3. Compare the new adapted algorithm to the current champion and decide:
         - If it improved: accept it as new champion.
         - If not, keep trying remaining components.
         - If no components left OR too many consecutive failures:
           start a FRESH lineage (back to PHASE 0).

All experiments are stored in a run_root/ directory, each cycle in its own
subfolder, with a top-level history.json that ties everything together.

Usage:
  python continuous_adaptive_metaheuristics.py \\
      --run-root runs/de_evolution \\
      --seed-provider anthropic --seed-model claude-opus-4-20250514 \\
      --variant-provider anthropic --adaptive-provider anthropic \\
      --component-selector llm --selector-provider anthropic \\
      --seed-class-dim 30

Requires the same environment as adaptive_metaheuristics.py (llm_keys.json,
GNBG_runners, replace_functions, numpy, anthropic/requests).
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Stdout UTF-8 (mirrors adaptive_metaheuristics.py) ────────────────────
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ── Reuse the heavy lifting from adaptive_metaheuristics.py ──────────────
from adaptive_metaheuristics import (
    load_keys,
    call_llm,
    extract_full_class_code,
    extract_full_function,
    create_experiment_dir,
    step_generate_variants,
    step_generate_variants_iterative,
    step_run_benchmarks,
    step_analyze,
    step_generate_adaptive,
    save_results_csv,
    save_summary,
    _ensure_gnbg,
)

PROVIDERS = ("anthropic", "ollama", "llamacpp")
SELECTORS = ("smallest", "random", "llm")


# ═══════════════════════════════════════════════════════════════════════
# Phase 0 — seed metaheuristic generation (with feedback)
# ═══════════════════════════════════════════════════════════════════════

SEED_SYSTEM = (
    "You are an expert in metaheuristic optimization and evolutionary computation. "
    "You design COMPLETE, runnable Python classes that implement population-based "
    "black-box optimizers. Your classes are ALWAYS structured as many small, single-"
    "responsibility methods (e.g. _initialize_population, _mutate, _crossover, "
    "_select, _adapt_parameters) so that individual operators can later be swapped "
    "out and tuned independently. "
    "Respond ONLY with the class in a single ```python``` code block. "
    "Include `import numpy as np` at the top. No text outside the code block."
)


SEED_USER_TEMPLATE = """\
Propose a NEW population-based metaheuristic optimizer as a complete Python class.

HARD REQUIREMENTS
- Start with `import numpy as np`.
- One ```python``` code block only, nothing else.
- The class must be instantiable with no required args and callable as:
      obj(func, stopping_condition)  ->  (f_opt, x_opt)
  where `func` is a callable taking a 2D array (pop x dim) of candidates and
  returning a 1D array of fitness values, and `stopping_condition` is a
  zero-argument callable — call it as `stopping_condition()` with no
  arguments — that returns True when the budget has been exhausted and the
  algorithm should stop.
- Problem dimension is {dim}; search bounds are [-100, 100]^{dim}.
- Clip every candidate to the bounds before evaluation.
- Robust to edge cases: no NaN, no division by zero, no runaway population size.

STRUCTURAL REQUIREMENTS (very important)
- Decompose the algorithm into MANY SMALL METHODS, each with a single
  responsibility. Good examples:
      _initialize_population, _mutate, _crossover, _select_survivors,
      _adapt_step_size, _restart_if_stagnant, _compute_diversity, ...
  Each such method should be short (ideally < 25 lines) so it can later be
  swapped out in isolation.
- Do NOT inline all logic inside __call__. __call__ should be a thin driver
  that orchestrates calls to the helper methods.
- Use clear, snake_case method names describing the operator.

CREATIVE DIRECTION
{creative_hint}

{prior_results_section}

Previously proposed seed classes in this run — propose something STRATEGICALLY
DIFFERENT from these (different operator family, different adaptation scheme,
different population topology, etc.):
{prior_summary}
"""


CREATIVE_HINTS = [
    "Use a Differential Evolution backbone with a non-standard mutation scheme.",
    "Use a CMA-ES style covariance-adaptation scheme but with a simplified update.",
    "Use a Particle Swarm backbone with topology-aware neighborhoods.",
    "Use a hybrid: DE-style mutation + local search (e.g. Nelder–Mead-like) refinement.",
    "Use an estimation-of-distribution approach (sample from a fitted Gaussian / GMM).",
    "Use a restart-heavy scheme with aggressive diversity preservation.",
    "Use an island / multi-population model with migration between sub-populations.",
    "Use a self-adaptive parameter-control scheme (parameters evolve with candidates).",
]


def _format_prior_results_for_seed(
    history: list,  # list[CycleRecord]
    run_root: Path,
) -> str:
    """
    Build a feedback section for seed generation from prior cycle results.
    Shows per-task performance of the best algorithm found so far, plus
    common failure patterns, so new seeds can be designed to address gaps.
    """
    if not history:
        return ""

    # Find the globally best cycle and load its results
    gbest = global_best(history)
    if gbest is None:
        return ""

    # Try to load the best algorithm's per-task results from its cycle dir
    best_results = _load_cycle_best_results(gbest, run_root)

    lines = [
        "FEEDBACK FROM PREVIOUS RUNS (use this to design a better algorithm):",
        f"Best algorithm so far: geomean = {gbest.best_geomean:.6e}",
        f"  from lineage {gbest.lineage_id}, cycle {gbest.cycle_index}",
        "",
    ]

    if best_results is not None:
        lines.append("Per-task errors of the best algorithm (lower = better):")
        lines.append("  Tasks with high error are where you should focus improvement.")
        for t in range(24):
            if t < len(best_results):
                v = best_results[t]
                if not np.isfinite(v):
                    lines.append(f"  Task {t:2d}: CRASHED (error=-inf) *** CRITICAL ***")
                elif abs(v) < 1e-7:
                    lines.append(f"  Task {t:2d}: SOLVED  (error={v:.2e})")
                elif v > 1.0:
                    lines.append(f"  Task {t:2d}: error={v:.3e}  *** NEEDS IMPROVEMENT ***")
                else:
                    lines.append(f"  Task {t:2d}: error={v:.3e}")
        lines.append("")

    # Summarize common error patterns from history
    crash_count = sum(1 for r in history if r.best_geomean is None)
    total = len(history)
    accept_count = sum(1 for r in history if r.accepted_as_champion)
    lines.append(f"History stats: {total} cycles, {accept_count} improvements, "
                 f"{crash_count} crashes")

    # Show which methods have been successfully adapted
    adapted_methods: dict[str, int] = {}
    for r in history:
        if r.accepted_as_champion and r.targeted_method:
            adapted_methods[r.targeted_method] = adapted_methods.get(r.targeted_method, 0) + 1
    if adapted_methods:
        lines.append("Methods that responded well to adaptation:")
        for m, cnt in sorted(adapted_methods.items(), key=lambda x: -x[1]):
            lines.append(f"  - {m} ({cnt} successful adaptation(s))")
        lines.append("")
        lines.append("Design your algorithm so it has methods similar to the "
                     "responsive ones above — they are more likely to benefit "
                     "from later operator-level tuning.")

    return "\n".join(lines)


def _load_cycle_best_results(
    record,  # CycleRecord
    run_root: Path,
) -> list[float] | None:
    """Try to load per-task results for a CycleRecord's best algorithm."""
    # Check if the cycle dir has a results.csv we can parse
    cycle_dir = Path(record.cycle_dir)
    csv_path = cycle_dir / "results.csv"
    if not csv_path.exists():
        return None

    import csv
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("algorithm") == record.best_algo_name:
                    results = []
                    for t in range(24):
                        val_str = row.get(f"task_{t}", "")
                        if val_str:
                            results.append(float(val_str))
                        else:
                            results.append(float("-inf"))
                    return results
    except Exception:
        pass

    # Fallback: try to load and benchmark the best algo's code
    # (too expensive — just return None)
    return None


def build_seed_prompt(
    dim: int,
    prior_classes: list[str],
    hint_idx: int,
    prior_results_section: str = "",
) -> str:
    if prior_classes:
        prior_summary = "\n".join(f"- {c}" for c in prior_classes)
    else:
        prior_summary = "(none yet)"
    return SEED_USER_TEMPLATE.format(
        dim=dim,
        creative_hint=CREATIVE_HINTS[hint_idx % len(CREATIVE_HINTS)],
        prior_summary=prior_summary,
        prior_results_section=prior_results_section,
    )


def extract_class_name(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            return node.name
    return None


def list_class_methods(code: str, class_name: str | None = None) -> list[tuple[str, int]]:
    """Return [(method_name, loc)] for all methods of the (first) class,
    sorted shortest→longest. Excludes dunder methods."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    target: ast.ClassDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if class_name is None or node.name == class_name:
                target = node
                break
    if target is None:
        return []

    out: list[tuple[str, int]] = []
    for item in target.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name.startswith("__") and item.name.endswith("__"):
                continue
            loc = (getattr(item, "end_lineno", item.lineno) or item.lineno) - item.lineno + 1
            out.append((item.name, loc))
    out.sort(key=lambda x: x[1])
    return out


def step_generate_seeds(
    n_seeds: int,
    dim: int,
    provider: str,
    keys: dict,
    model: str | None,
    lineage_dir: Path,
    prior_results_section: str = "",
) -> list[tuple[str, Path, str]]:
    """Phase 0: ask the LLM for N fresh full-class metaheuristics,
    informed by per-task feedback from prior lineages."""
    print("\n" + "=" * 60)
    print(f"PHASE 0: Generating {n_seeds} seed metaheuristic(s) via {provider}"
          + (f" / {model}" if model else ""))
    if prior_results_section:
        print("  (with feedback from previous runs)")
    print("=" * 60)

    seeds_dir = lineage_dir / "seeds"
    seeds_dir.mkdir(exist_ok=True)
    prompts_dir = lineage_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    saved: list[tuple[str, Path, str]] = []
    prior_classes: list[str] = []

    for i in range(n_seeds):
        prompt = build_seed_prompt(dim, prior_classes, hint_idx=i,
                                   prior_results_section=prior_results_section)
        (prompts_dir / f"seed_prompt_{i+1}.md").write_text(prompt, encoding="utf-8")
        print(f"  Seed {i+1}/{n_seeds}: asking {provider}...", end=" ", flush=True)

        try:
            response = call_llm(prompt, SEED_SYSTEM, provider, keys, model)
        except Exception as e:
            print(f"LLM call failed: {e}")
            continue
        (prompts_dir / f"seed_response_{i+1}.md").write_text(response, encoding="utf-8")

        code = extract_full_class_code(response)
        if code is None:
            print("no valid code block")
            continue
        if "import numpy" not in code:
            code = "import numpy as np\n\n" + code

        cname = extract_class_name(code) or f"Seed{i+1}"
        fname = f"seed_{i+1}_{cname}.py"
        fpath = seeds_dir / fname
        fpath.write_text(code, encoding="utf-8")
        saved.append((fname, fpath, "seed"))
        prior_classes.append(cname)
        print(f"ok → {fname}")

    return saved


# ═══════════════════════════════════════════════════════════════════════
# Component selection strategies
# ═══════════════════════════════════════════════════════════════════════

SELECTOR_SYSTEM = (
    "You are an expert in metaheuristic optimization. You will be given a "
    "complete population-based optimizer and a list of its methods that are "
    "candidates for being replaced by an adaptive, learned version. "
    "Your job is to pick the 3 methods whose adaptation is MOST LIKELY to "
    "improve overall benchmark performance. "
    "NOTE: Some methods may already have been made adaptive in previous cycles "
    "and are therefore NOT in the candidate list — do not suggest them. "
    "Reply in strict JSON: "
    '{\"picks\": [\"method_a\", \"method_b\", \"method_c\"], \"reason\": \"...\"}. '
    "No commentary outside the JSON block. The three picks MUST be chosen "
    "from the provided candidate list and must be distinct."
)


def _render_history_for_selector(
    history: list,            # list[CycleRecord]
    lineage_id: int | None,
    adaptive_methods: set[str] | None = None,
) -> str:
    """Compact textual summary of past cycles in the current lineage."""
    if not history or lineage_id is None:
        return "(no prior cycles in this lineage)"
    lines = []
    for r in history:
        if r.lineage_id != lineage_id:
            continue
        mark = "✔" if r.accepted_as_champion else "✗"
        gm = f"{r.best_geomean:.3e}" if r.best_geomean is not None else "  —   "
        lines.append(f"  {mark} cycle {r.cycle_index:>3}: targeted={r.targeted_method:<28} "
                     f"best_gm={gm}  ({r.reason})")
    if adaptive_methods:
        lines.append("")
        lines.append(f"  Methods already made adaptive (excluded from candidates): "
                     f"{', '.join(sorted(adaptive_methods))}")
    return "\n".join(lines) if lines else "(no prior cycles in this lineage)"


def build_selector_prompt(
    champion_code: str,
    candidates: list[tuple[str, int]],
    history_text: str,
) -> str:
    methods_list = "\n".join(f"  - {n}  ({loc} LOC)" for n, loc in candidates)
    return f"""\
Below is the current champion metaheuristic. Some of its methods have already
been replaced by adaptive versions in previous cycles (see history). You must
choose the THREE methods — from the CANDIDATE list below — whose adaptive
replacement is most likely to yield the biggest benchmark improvement.

Consider:
- Which operators are typically the most performance-sensitive in this family
  of algorithms? (e.g. mutation strategies in DE, recombination in CMA-ES)
- Which methods embody a choice that is known to matter (discrete strategy
  picks, per-problem-dependent heuristics) rather than mere bookkeeping?
- What has already been tried — don't suggest anything already in the history
  of exhausted methods (it is not in the CANDIDATE list anyway).

CANDIDATE methods (you MUST pick exactly 3 distinct names from this list):
{methods_list}

HISTORY of cycles in this lineage so far (most recent last):
{history_text}

Current champion source:
```python
{champion_code}
```

Reply with JSON only:
{{\"picks\": [\"...\", \"...\", \"...\"], \"reason\": \"...\"}}
"""


def _parse_selector_json(response: str, valid_names: set[str]) -> list[str]:
    """Extract a clean list of 3 valid method names from the LLM's JSON reply."""
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in selector response")
    payload = json.loads(m.group(0))
    picks = payload.get("picks", [])
    if not isinstance(picks, list):
        raise ValueError("'picks' is not a list")
    clean = []
    for p in picks:
        if isinstance(p, str) and p in valid_names and p not in clean:
            clean.append(p)
    if not clean:
        raise ValueError(f"no valid method names in picks={picks!r}")
    return clean[:3]


def select_component_llm(
    champion_code: str,
    candidates: list[tuple[str, int]],
    history: list,
    lineage_id: int | None,
    provider: str,
    keys: dict,
    model: str | None,
    cycle_dir: Path,
    adaptive_methods: set[str] | None = None,
) -> tuple[str, int, list[str]]:
    """
    Ask the LLM for its top-3 candidate methods, then pick one uniformly at
    random from those three. Returns (chosen_name, chosen_loc, top3_list).
    """
    history_text = _render_history_for_selector(
        history, lineage_id, adaptive_methods
    )
    prompt = build_selector_prompt(champion_code, candidates, history_text)
    cycle_dir.mkdir(parents=True, exist_ok=True)
    (cycle_dir / "prompts").mkdir(exist_ok=True)
    (cycle_dir / "prompts" / "selector_prompt.md").write_text(prompt, encoding="utf-8")

    valid_names = {n for n, _ in candidates}
    try:
        response = call_llm(prompt, SELECTOR_SYSTEM, provider, keys, model)
        (cycle_dir / "prompts" / "selector_response.md").write_text(response, encoding="utf-8")
        picks = _parse_selector_json(response, valid_names)
    except Exception as e:
        print(f"  ⚠  selector LLM failed ({e}); falling back to uniform random.")
        chosen_name, chosen_loc = random.choice(candidates)
        return chosen_name, chosen_loc, [chosen_name]

    if not picks:
        print("  ⚠  selector returned no valid picks; falling back to uniform random.")
        chosen_name, chosen_loc = random.choice(candidates)
        return chosen_name, chosen_loc, [chosen_name]

    chosen_name = random.choice(picks)
    loc_by_name = dict(candidates)
    chosen_loc = loc_by_name.get(chosen_name, -1)
    return chosen_name, chosen_loc, picks


def select_component(
    strategy: str,
    candidates: list[tuple[str, int]],
    champion_code: str,
    history: list,
    lineage_id: int | None,
    provider: str,
    keys: dict,
    model: str | None,
    cycle_dir: Path,
    adaptive_methods: set[str] | None = None,
) -> tuple[str, int, list[str]]:
    """
    Dispatch over selection strategies. Returns (chosen_name, chosen_loc,
    top_candidates_considered).
    """
    if not candidates:
        raise ValueError("no candidates to choose from")
    if strategy == "smallest":
        name, loc = candidates[0]
        return name, loc, [name]
    if strategy == "random":
        name, loc = random.choice(candidates)
        return name, loc, [name]
    if strategy == "llm":
        return select_component_llm(
            champion_code, candidates, history, lineage_id,
            provider, keys, model, cycle_dir, adaptive_methods,
        )
    raise ValueError(f"unknown component selector strategy: {strategy}")


# ═══════════════════════════════════════════════════════════════════════
# Scoring & history
# ═══════════════════════════════════════════════════════════════════════

N_TASKS = 24


def geomean_of(result_list: list[float] | None) -> float | None:
    """Geometric mean of strictly-positive finite entries, or None."""
    if not result_list or len(result_list) != N_TASKS:
        return None
    good = [v for v in result_list if np.isfinite(v) and v > 0]
    if not good:
        return None
    return statistics.geometric_mean(good)


def pick_best(results: dict[str, dict]) -> tuple[str, dict, float] | None:
    """Return (name, data, geomean) of the best algorithm, lower-is-better."""
    ranked: list[tuple[str, dict, float]] = []
    for name, data in results.items():
        gm = geomean_of(data.get("results"))
        if gm is not None:
            ranked.append((name, data, gm))
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[2])
    return ranked[0]


@dataclass
class CycleRecord:
    """One entry in the top-level history.json."""
    cycle_index: int
    lineage_id: int
    cycle_dir: str
    parent_algo_path: str
    parent_geomean: float | None
    targeted_method: str | None
    selector_strategy: str = ""
    selector_candidates: list[str] = field(default_factory=list)
    best_algo_name: str = ""
    best_algo_path: str = ""
    best_geomean: float | None = None
    accepted_as_champion: bool = False
    reason: str = ""
    # Track which methods are now adaptive in the champion after this cycle
    adaptive_methods: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def load_history(run_root: Path) -> list[CycleRecord]:
    hp = run_root / "history.json"
    if not hp.exists():
        return []
    raw = json.loads(hp.read_text(encoding="utf-8"))
    records = []
    for r in raw:
        r.setdefault("selector_strategy", "")
        r.setdefault("selector_candidates", [])
        r.setdefault("adaptive_methods", [])
        records.append(CycleRecord(**r))
    return records


def save_history(run_root: Path, history: list[CycleRecord]):
    hp = run_root / "history.json"
    hp.write_text(
        json.dumps([asdict(r) for r in history], indent=2),
        encoding="utf-8",
    )


def global_best(history: list[CycleRecord]) -> CycleRecord | None:
    best: CycleRecord | None = None
    for r in history:
        if r.best_geomean is None:
            continue
        if best is None or r.best_geomean < best.best_geomean:
            best = r
    return best


def _rebuild_adaptive_methods(history: list[CycleRecord], lineage_id: int) -> set[str]:
    """
    Reconstruct the set of methods that have been successfully made adaptive
    in this lineage. A method counts as adaptive if it was targeted in a cycle
    that was accepted as champion.
    """
    adaptive = set()
    for r in history:
        if r.lineage_id != lineage_id:
            continue
        if r.accepted_as_champion and r.targeted_method:
            adaptive.add(r.targeted_method)
    return adaptive


# ═══════════════════════════════════════════════════════════════════════
# Cycle orchestration
# ═══════════════════════════════════════════════════════════════════════

def run_component_cycle(
    champion_code: str,
    champion_path: Path,
    target_method: str,
    cycle_dir: Path,
    args,
    keys: dict,
) -> dict[str, dict]:
    """
    Apply variant → benchmark → adaptive → benchmark to one method of the
    current champion. Returns the merged results dict.
    """
    (cycle_dir / "variants").mkdir(exist_ok=True)
    (cycle_dir / "prompts").mkdir(exist_ok=True)

    parent_copy = cycle_dir / "parent_algorithm.py"
    parent_copy.write_text(champion_code, encoding="utf-8")

    config = {
        **vars(args),
        "cycle_dir": str(cycle_dir),
        "parent_algorithm": str(champion_path),
        "target_method": target_method,
        "timestamp": datetime.now().isoformat(),
    }
    (cycle_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # --- Phase 1: generate variants of the chosen method -----------------
    if getattr(args, "variant_mode", "batch") == "iterative":
        variant_paths, variant_results = step_generate_variants_iterative(
            champion_code,
            target_method,
            args.variant_provider,
            keys,
            args.variant_model,
            cycle_dir,
            args.n_ideas,
            args.budget,
            args.repetitions,
        )
        if len(variant_paths) <= 1:
            print("  ⚠  No usable variants generated — cycle aborted.")
            return {}
    else:
        variant_paths = step_generate_variants(
            champion_code,
            target_method,
            args.variant_provider,
            keys,
            args.variant_model,
            cycle_dir,
            args.n_ideas,
        )

        if len(variant_paths) <= 1:
            print("  ⚠  No usable variants extracted — cycle aborted.")
            return {}

        variant_results = step_run_benchmarks(
            variant_paths, args.budget, args.repetitions, label="VARIANTS"
        )

    analysis = step_analyze(variant_results)
    if analysis is None:
        print("  ⚠  No valid variant results — skipping adaptive step.")
        save_results_csv(variant_results, cycle_dir)
        save_summary(variant_results, None, cycle_dir)
        return variant_results

    # --- Phase 3: adaptive ----------------------------------------------
    adaptive_paths = step_generate_adaptive(
        champion_code,
        target_method,
        analysis,
        variant_results,
        args.adaptive_provider,
        keys,
        args.adaptive_model,
        cycle_dir,
        args.n_adaptive_tries,
    )

    adaptive_results: dict[str, dict] = {}
    if adaptive_paths:
        adaptive_results = step_run_benchmarks(
            adaptive_paths, args.budget, args.repetitions, label="ADAPTIVE"
        )

    all_results = {**variant_results, **adaptive_results}
    save_results_csv(all_results, cycle_dir)
    save_summary(all_results, analysis, cycle_dir)
    return all_results


# ═══════════════════════════════════════════════════════════════════════
# Lineage bootstrap (with fallback to history best)
# ═══════════════════════════════════════════════════════════════════════

def start_new_lineage(
    lineage_id: int,
    run_root: Path,
    args,
    keys: dict,
    history: list[CycleRecord],
) -> tuple[Path, dict] | None:
    """Run Phase 0 + initial benchmark to pick a champion for a new lineage.
    Returns (champion_path, champion_data) or None on total failure.

    KEY BEHAVIOR:
    - Seed prompts include per-task feedback from prior lineages so the LLM
      can design algorithms that address known weak spots.
    - If no seed produces valid results, the global best from prior cycles is
      promoted as the new champion (avoids losing all progress).
    """
    lineage_dir = run_root / f"lineage_{lineage_id:03d}"
    lineage_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n╔══════════════════════════════════════════════════════════╗")
    print(f"║  STARTING NEW LINEAGE {lineage_id}")
    print(f"║  Directory: {lineage_dir}")
    print(f"╚══════════════════════════════════════════════════════════╝")

    # Build feedback from prior runs for seed prompts
    prior_results_section = _format_prior_results_for_seed(history, run_root)

    seed_paths = step_generate_seeds(
        args.n_seeds,
        args.seed_class_dim,
        args.seed_provider,
        keys,
        args.seed_model,
        lineage_dir,
        prior_results_section=prior_results_section,
    )

    # Benchmark seeds (if any were generated)
    seed_results: dict[str, dict] = {}
    if seed_paths:
        seed_results = step_run_benchmarks(
            seed_paths, args.budget, args.repetitions,
            label=f"SEEDS (lineage {lineage_id})"
        )
        save_results_csv(seed_results, lineage_dir)
        save_summary(seed_results, None, lineage_dir)

    best = pick_best(seed_results) if seed_results else None

    # ── FALLBACK: if no seed is good, use the global best from history ──
    if best is None:
        gbest = global_best(history)
        if gbest is not None and gbest.best_algo_path:
            gbest_path = Path(gbest.best_algo_path)
            if gbest_path.exists():
                print(f"\n  ⚠  No seed produced valid results.")
                print(f"  ★ Falling back to global best from history:")
                print(f"    lineage {gbest.lineage_id}, cycle {gbest.cycle_index}, "
                      f"geomean={gbest.best_geomean:.6e}")
                champ_code = gbest_path.read_text(encoding="utf-8")
                champ_copy = lineage_dir / f"champion_gen0__fallback_from_L{gbest.lineage_id}"
                champ_copy.write_text(champ_code, encoding="utf-8")
                data = {
                    "path": str(champ_copy),
                    "results": None,  # will be re-benchmarked in the first cycle
                }
                return champ_copy, data
            else:
                print(f"  ⚠  Global best path {gbest_path} no longer exists.")

        print("  ⚠  No seeds generated and no prior champion to fall back on.")
        return None

    name, data, gm = best
    print(f"\n  ★ Lineage {lineage_id} champion: {name}  (geomean={gm:.6e})")

    # Check if a seed actually beats the global best
    gbest = global_best(history)
    if gbest is not None and gbest.best_geomean is not None:
        if gm > gbest.best_geomean:
            print(f"  ⚠  Best seed (geomean={gm:.6e}) is WORSE than global best "
                  f"(geomean={gbest.best_geomean:.6e}).")
            print(f"  ★ Using global best as champion instead.")
            gbest_path = Path(gbest.best_algo_path)
            if gbest_path.exists():
                champ_code = gbest_path.read_text(encoding="utf-8")
                champ_copy = lineage_dir / f"champion_gen0__fallback_from_L{gbest.lineage_id}"
                champ_copy.write_text(champ_code, encoding="utf-8")
                return champ_copy, {
                    "path": str(champ_copy),
                    "results": None,
                }
            # If path doesn't exist, fall through to use the seed

    champ_code = Path(data["path"]).read_text(encoding="utf-8")
    champ_copy = lineage_dir / f"champion_gen0__{name}"
    champ_copy.write_text(champ_code, encoding="utf-8")
    data = {**data, "path": str(champ_copy)}
    return champ_copy, data


# ═══════════════════════════════════════════════════════════════════════
# Decision logic
# ═══════════════════════════════════════════════════════════════════════

def decide_next_step(
    history: list[CycleRecord],
    lineage_id: int,
    consecutive_failures: int,
    exhausted_methods: set[str],
    remaining_methods: list[str],
    max_failures_per_lineage: int,
) -> str:
    """
    Decide what to do next. Returns 'continue' or 'new_lineage'.
    """
    if not remaining_methods:
        return "new_lineage"
    if consecutive_failures >= max_failures_per_lineage:
        return "new_lineage"
    return "continue"


# ═══════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Continuous LLM-guided metaheuristic evolution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-root", required=True,
                        help="Root directory for the continuous run.")
    parser.add_argument("--max-cycles", type=int, default=1_000_000,
                        help="Hard cap on total cycles (default: effectively infinite).")
    parser.add_argument("--max-lineages", type=int, default=1_000_000,
                        help="Hard cap on lineages (default: effectively infinite).")
    parser.add_argument("--max-failures-per-lineage", type=int, default=3,
                        help="After this many non-improving cycles, abandon lineage.")
    parser.add_argument("--random-seed", type=int, default=None,
                        help="Seed for Python's `random` module.")

    # Seed / Phase-0 generation
    parser.add_argument("--n-seeds", type=int, default=4,
                        help="How many seed classes to generate per new lineage.")
    parser.add_argument("--seed-class-dim", type=int, default=30,
                        help="Dimensionality communicated to the seed LLM prompt.")
    parser.add_argument("--seed-provider", default="anthropic",
                        choices=PROVIDERS)
    parser.add_argument("--seed-model", default=None)

    # Component selection
    parser.add_argument("--component-selector", default="smallest",
                        choices=SELECTORS)
    parser.add_argument("--selector-provider", default="anthropic",
                        choices=PROVIDERS)
    parser.add_argument("--selector-model", default=None)

    # Variant / adaptive operator LLMs
    parser.add_argument("--variant-provider", default="anthropic",
                        choices=PROVIDERS)
    parser.add_argument("--variant-model", default=None)
    parser.add_argument("--variant-mode", default="batch",
                        choices=["batch", "iterative"])
    parser.add_argument("--adaptive-provider", default="anthropic",
                        choices=PROVIDERS)
    parser.add_argument("--adaptive-model", default=None)

    # Benchmark knobs
    parser.add_argument("--budget", type=int, default=1_000_000)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--n-ideas", type=int, default=8)
    parser.add_argument("--n-adaptive-tries", type=int, default=3)
    parser.add_argument("--keys", default="llm_keys.json")

    args = parser.parse_args()

    if args.random_seed is not None:
        random.seed(args.random_seed)

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "args.json").write_text(
        json.dumps({**vars(args), "start_time": datetime.now().isoformat()}, indent=2),
        encoding="utf-8",
    )

    keys = load_keys(args.keys)
    _ensure_gnbg()

    history = load_history(run_root)
    existing_lineages = sorted(
        int(p.name.split("_")[-1])
        for p in run_root.glob("lineage_*")
        if p.is_dir() and p.name.split("_")[-1].isdigit()
    )
    next_lineage_id = (existing_lineages[-1] + 1) if existing_lineages else 0
    cycle_index = len(history)

    # State within the currently-active lineage:
    current_champion_path: Path | None = None
    current_champion_data: dict | None = None
    exhausted_methods: set[str] = set()      # methods tried (success or not)
    adaptive_methods: set[str] = set()        # methods successfully made adaptive
    consecutive_failures = 0
    active_lineage_id: int | None = None
    lineages_started = 0

    # If we're resuming, attempt to rehydrate from last history entry.
    if history:
        last = history[-1]
        active_lineage_id = last.lineage_id
        if last.accepted_as_champion:
            current_champion_path = Path(last.best_algo_path)
        else:
            current_champion_path = Path(last.parent_algo_path)
        if current_champion_path.exists():
            current_champion_data = {
                "path": str(current_champion_path),
                "results": None,
            }
            # Rebuild exhausted-methods set from history for this lineage.
            exhausted_methods = {
                r.targeted_method
                for r in history
                if r.lineage_id == active_lineage_id and r.targeted_method
            }
            # Rebuild adaptive methods (only successfully adapted ones)
            adaptive_methods = _rebuild_adaptive_methods(history, active_lineage_id)
            # Rebuild consecutive-failure streak.
            streak = 0
            for r in reversed(history):
                if r.lineage_id != active_lineage_id:
                    break
                if r.accepted_as_champion:
                    break
                streak += 1
            consecutive_failures = streak
            print(f"  Resumed lineage {active_lineage_id}: "
                  f"champion={current_champion_path.name}, "
                  f"exhausted={len(exhausted_methods)}, "
                  f"adaptive={len(adaptive_methods)} ({', '.join(sorted(adaptive_methods)) or 'none'}), "
                  f"failures={consecutive_failures}")
        else:
            print(f"  ⚠  Could not rehydrate champion from {current_champion_path} — starting fresh.")
            current_champion_path = None
            current_champion_data = None
            active_lineage_id = None

    # ── Main continuous loop ─────────────────────────────────────────────
    while cycle_index < args.max_cycles and lineages_started < args.max_lineages:

        # ── Bootstrap a lineage if we don't have one ─────────────────────
        if current_champion_path is None:
            if next_lineage_id >= args.max_lineages:
                print("\n  Reached --max-lineages; stopping.")
                break
            lineages_started += 1
            boot = start_new_lineage(
                next_lineage_id, run_root, args, keys, history
            )
            if boot is None:
                next_lineage_id += 1
                continue
            current_champion_path, current_champion_data = boot
            active_lineage_id = next_lineage_id
            next_lineage_id += 1
            exhausted_methods = set()
            adaptive_methods = set()
            consecutive_failures = 0

        # ── Pick un-tried method (excluding already-adaptive ones) ───────
        champion_code = current_champion_path.read_text(encoding="utf-8")
        methods = list_class_methods(champion_code)

        # Filter out: exhausted (tried already) AND adaptive (already adapted
        # in a previous successful cycle — don't adapt twice)
        skip = exhausted_methods | adaptive_methods
        remaining = [(n, loc) for (n, loc) in methods if n not in skip]

        if not remaining:
            total_methods = len(methods)
            print(f"\n  All {total_methods} method(s) of the current champion "
                  f"have been tried or are already adaptive.")
            if adaptive_methods:
                print(f"  Already-adaptive methods: {', '.join(sorted(adaptive_methods))}")
            current_champion_path = None
            current_champion_data = None
            continue

        decision = decide_next_step(
            history, active_lineage_id, consecutive_failures,
            exhausted_methods, [n for n, _ in remaining],
            args.max_failures_per_lineage,
        )
        if decision == "new_lineage":
            print(f"\n  Decision: start a new lineage "
                  f"(failures={consecutive_failures}, remaining={len(remaining)}).")
            current_champion_path = None
            current_champion_data = None
            continue

        # Pre-create the cycle dir
        cycle_dir = run_root / f"cycle_{cycle_index:04d}__lineage_{active_lineage_id:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        target_method, target_loc, selector_candidates = select_component(
            strategy=args.component_selector,
            candidates=remaining,
            champion_code=champion_code,
            history=history,
            lineage_id=active_lineage_id,
            provider=args.selector_provider,
            keys=keys,
            model=args.selector_model,
            cycle_dir=cycle_dir,
            adaptive_methods=adaptive_methods,
        )

        # Re-tag the cycle dir with the chosen method
        final_cycle_dir = run_root / (
            f"cycle_{cycle_index:04d}__lineage_{active_lineage_id:03d}__{target_method}"
        )
        if final_cycle_dir != cycle_dir:
            cycle_dir.rename(final_cycle_dir)
            cycle_dir = final_cycle_dir

        print(f"\n╔══════════════════════════════════════════════════════════╗")
        print(f"║  CYCLE {cycle_index}  (lineage {active_lineage_id})")
        print(f"║  Champion : {current_champion_path.name}")
        print(f"║  Selector : {args.component_selector}"
              + (f"  → top-3: {selector_candidates}"
                 if args.component_selector == 'llm' else ""))
        print(f"║  Target   : {target_method}  ({target_loc} LOC)")
        if adaptive_methods:
            print(f"║  Already adaptive: {', '.join(sorted(adaptive_methods))}")
        print(f"║  Dir      : {cycle_dir.name}")
        print(f"╚══════════════════════════════════════════════════════════╝")

        parent_gm = geomean_of(
            current_champion_data.get("results")
        ) if current_champion_data else None

        t0 = time.perf_counter()
        cycle_results = run_component_cycle(
            champion_code=champion_code,
            champion_path=current_champion_path,
            target_method=target_method,
            cycle_dir=cycle_dir,
            args=args,
            keys=keys,
        )
        elapsed = time.perf_counter() - t0
        print(f"\n  Cycle finished in {elapsed/60:.1f} min "
              f"({len(cycle_results)} algorithm(s) benchmarked).")

        # ── Evaluate the cycle's output ──────────────────────────────────
        best = pick_best(cycle_results)
        gbest = global_best(history)
        if gbest is not None:
            print(f"  Global best so far: lineage {gbest.lineage_id} / "
                  f"cycle {gbest.cycle_index}, geomean={gbest.best_geomean:.6e}")

        accepted = False
        reason = ""
        if best is None:
            reason = "no valid benchmark results from cycle"
            new_champion_name = "(none)"
            new_champion_path = str(current_champion_path)
            new_gm = None
        else:
            new_champion_name, new_data, new_gm = best
            new_champion_path = new_data["path"]

            # Re-score the incumbent from within this cycle
            incumbent_gm_in_cycle: float | None = None
            for nm, dt in cycle_results.items():
                if nm == "original.py":
                    incumbent_gm_in_cycle = geomean_of(dt.get("results"))
                    break
            if incumbent_gm_in_cycle is None:
                incumbent_gm_in_cycle = parent_gm

            if incumbent_gm_in_cycle is None:
                accepted = True
                reason = "no incumbent score to compare; accepting improver"
            elif new_gm < incumbent_gm_in_cycle:
                accepted = True
                reason = (f"new geomean {new_gm:.3e} < incumbent "
                          f"{incumbent_gm_in_cycle:.3e}")
            else:
                accepted = False
                reason = (f"new geomean {new_gm:.3e} >= incumbent "
                          f"{incumbent_gm_in_cycle:.3e}")

        # ── Update state & history ───────────────────────────────────────
        if accepted and best is not None:
            new_src = Path(new_champion_path).read_text(encoding="utf-8")
            generation = sum(
                1 for r in history
                if r.lineage_id == active_lineage_id and r.accepted_as_champion
            ) + 1
            champ_copy = (run_root / f"lineage_{active_lineage_id:03d}" /
                          f"champion_gen{generation}__{new_champion_name}")
            champ_copy.parent.mkdir(parents=True, exist_ok=True)
            champ_copy.write_text(new_src, encoding="utf-8")

            current_champion_path = champ_copy
            current_champion_data = {
                "path": str(champ_copy),
                "results": new_data.get("results"),
            }
            # Mark this method as adaptive — won't be targeted again
            adaptive_methods.add(target_method)
            consecutive_failures = 0
            print(f"  ✔  Accepted new champion: {champ_copy.name}")
            print(f"     Method '{target_method}' is now marked as ADAPTIVE "
                  f"(won't be re-adapted)")
        else:
            consecutive_failures += 1
            print(f"  ✗  Rejected — {reason}")

        exhausted_methods.add(target_method)

        history.append(CycleRecord(
            cycle_index=cycle_index,
            lineage_id=active_lineage_id,
            cycle_dir=str(cycle_dir),
            parent_algo_path=str(current_champion_path),
            parent_geomean=parent_gm,
            targeted_method=target_method,
            selector_strategy=args.component_selector,
            selector_candidates=selector_candidates,
            best_algo_name=new_champion_name if best else "(none)",
            best_algo_path=str(new_champion_path) if best else str(current_champion_path),
            best_geomean=new_gm,
            accepted_as_champion=accepted,
            reason=reason,
            adaptive_methods=sorted(adaptive_methods),
        ))
        save_history(run_root, history)
        cycle_index += 1

    # ── End of loop ──────────────────────────────────────────────────────
    gbest = global_best(history)
    print("\n" + "=" * 60)
    print("CONTINUOUS RUN FINISHED")
    print("=" * 60)
    if gbest is not None:
        print(f"  Best overall: lineage {gbest.lineage_id} / cycle {gbest.cycle_index}")
        print(f"               {gbest.best_algo_path}")
        print(f"               geomean = {gbest.best_geomean:.6e}")
    print(f"  Total cycles : {cycle_index}")
    print(f"  Total lineages : {lineages_started}")
    print(f"  Run root     : {run_root}")


if __name__ == "__main__":
    main()