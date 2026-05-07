#!/usr/bin/env python3
"""
Generate / curate LLM-written metaheuristic operators for rl_algorithm.py.

Three categories, 5 each by default:
  good         — solid, reliable, based on established metaheuristic families
  specialized  — tuned for ONE kind of landscape (multimodal, ill-conditioned, …)
  weird        — unconventional, creative, unusual

Output file: generated_operators.py (imported automatically by rl_algorithm.py)

Commands:
  show-prompt [--category CAT] [--flavor F]
                              Print the prompt(s). Default: one sample per category.
  list-models                 List models available on local Ollama
  generate [--categories ...] [--count N] [--models ...]
                              Query local Ollama, validate, append to file
  add-manual <file> --category CAT [--flavor F]
                              Ingest a model's reply from disk, validate, append
  test                        Quick-test generated_operators.py on sphere

Examples:
  python operator_generator.py show-prompt
  python operator_generator.py show-prompt --category specialized
  python operator_generator.py generate                       # all 15
  python operator_generator.py generate --categories good     # just 5 good
  python operator_generator.py add-manual reply.txt --category weird
"""

import argparse
import re
import sys
import textwrap
from pathlib import Path

DEFAULT_MODELS = ["qwen3-coder-next:q8_0"]
OUT_FILE = Path("generated_operators.py")

CATEGORIES = ("good", "specialized", "weird")
PREFIX = {"good": "op_good_", "specialized": "op_spec_", "weird": "op_weird_"}

# ──────────────────────────────────────────────────────────────────────
#  PROMPTS
# ──────────────────────────────────────────────────────────────────────

_INTERFACE = textwrap.dedent("""\

    ─── STRICT INTERFACE ───────────────────────────────────────────────
    Your function MUST follow this signature EXACTLY:

        def {func_name}(pop, fit, f, rng, bounds):
            # pop     : np.ndarray, shape (N, D), current population
            # fit     : np.ndarray, shape (N,),   fitness (LOWER is better)
            # f       : callable, f(x: np.ndarray (D,)) -> float (the objective)
            # rng     : numpy.random.Generator
            # bounds  : tuple(float, float) — (low, high), same for every dim
            # returns : (new_pop, new_fit) — np.ndarray of shapes (N, D), (N,)

    ─── HARD RULES ─────────────────────────────────────────────────────
    1. Only numpy (as `np`) and the passed arguments. No imports beyond numpy.
    2. Call f AT MOST N times, where N = len(pop). Never more.
    3. Clip every candidate you evaluate to bounds via np.clip.
    4. Do NOT mutate `pop` or `fit` in place. Copy first.
    5. Return float64 arrays with exact shapes (N, D) and (N,).
    6. Keep it self-contained — no helper functions outside the body.
    7. Be numerically safe: no NaN, no div-by-zero, no infinite loops.

    Respond with ONLY a single Python code block. No prose, no explanation:

    ```python
    import numpy as np

    def {func_name}(pop, fit, f, rng, bounds):
        ...
    ```
    """)

_GOOD_INTRO = textwrap.dedent("""\
    You are an experienced optimization researcher. Design ONE RELIABLE, SOLID
    population-level operator for continuous black-box minimization, based on the
    established metaheuristic family:

        {flavor}

    It should be the kind of operator that works well across a broad range of
    problems — the backbone of a good solver, not a gimmick. Use proven principles
    from the literature. Pick sensible, well-tuned hyperparameters.
    """)

_SPEC_INTRO = textwrap.dedent("""\
    You are an optimization researcher. Design ONE SPECIALIZED population-level
    operator for continuous black-box minimization, tuned specifically for
    landscapes that are:

        {flavor}

    It should EXCEL on that kind of problem, even if it is mediocre elsewhere.
    Pick techniques that directly exploit the specialty. Be deliberate about the
    mechanism and why it fits the landscape.
    """)

_WEIRD_INTRO = textwrap.dedent("""\
    You are a creative optimization researcher. Invent ONE WEIRD, unconventional
    population-level operator for continuous black-box minimization. It will plug
    into a reinforcement-learning-driven metaheuristic hub that chooses between
    many operators each step via a UCB1 bandit.

    {flavor}

    Be WEIRD. Think: fractal jumps, quantum tunneling, gossip dynamics, cellular
    automata, magnetic attractors, dream logic, reaction-diffusion, predator-prey,
    memetic hallucination — anything unusual, but valid.
    """)

FLAVORS = {
    "good": [
        "differential evolution (DE)",
        "particle swarm optimization (PSO)",
        "evolution strategies (ES / CMA-ES-style rank-1 updates)",
        "genetic algorithm with crossover and mutation",
        "simulated annealing / adaptive local search",
    ],
    "specialized": [
        "highly multimodal with many deceptive local optima",
        "ill-conditioned with narrow elongated valleys (large eigenvalue spread)",
        "separable (each coordinate contributes independently)",
        "plateau-heavy with large flat regions and sparse gradient information",
        "noisy or deceptive, with misleading fitness signals near the optimum",
    ],
    "weird": [
        "(free creative direction)",
        "(free creative direction)",
        "(free creative direction)",
        "(free creative direction)",
        "(free creative direction)",
    ],
}

_INTROS = {"good": _GOOD_INTRO, "specialized": _SPEC_INTRO, "weird": _WEIRD_INTRO}


def build_prompt(category: str, func_name: str, flavor: str) -> str:
    return _INTROS[category].format(flavor=flavor) + _INTERFACE.format(func_name=func_name)


# ──────────────────────────────────────────────────────────────────────
#  CODE EXTRACTION + VALIDATION
# ──────────────────────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def validate(code: str, func_name: str) -> tuple[bool, str]:
    if f"def {func_name}" not in code:
        return False, f"function `{func_name}` not defined"

    import numpy as np
    ns: dict = {"np": np, "__builtins__": __builtins__}
    try:
        exec(code, ns)
    except Exception as e:
        return False, f"exec failed: {e}"

    fn = ns.get(func_name)
    if fn is None or not callable(fn):
        return False, f"`{func_name}` not callable after exec"

    rng = np.random.default_rng(0)
    pop = rng.uniform(-10, 10, (8, 5)).astype(np.float64)
    fit = np.array([float(np.sum(x * x)) for x in pop])
    call_count = [0]

    def f(x):
        call_count[0] += 1
        return float(np.sum(x * x))

    try:
        new_pop, new_fit = fn(pop.copy(), fit.copy(), f, rng, (-10.0, 10.0))
    except Exception as e:
        return False, f"runtime error: {e}"

    if not (hasattr(new_pop, "shape") and hasattr(new_fit, "shape")):
        return False, "returns not numpy arrays"
    if new_pop.shape != pop.shape:
        return False, f"new_pop shape {new_pop.shape} != {pop.shape}"
    if new_fit.shape != fit.shape:
        return False, f"new_fit shape {new_fit.shape} != {fit.shape}"
    if call_count[0] > len(pop):
        return False, f"too many f() calls: {call_count[0]} > {len(pop)}"
    if not np.all(np.isfinite(new_pop)) or not np.all(np.isfinite(new_fit)):
        return False, "NaN or inf in output"
    if np.any(new_pop < -10.0 - 1e-9) or np.any(new_pop > 10.0 + 1e-9):
        return False, "not clipped to bounds"
    return True, "ok"


# ──────────────────────────────────────────────────────────────────────
#  OLLAMA
# ──────────────────────────────────────────────────────────────────────

def call_ollama(model: str, prompt: str, timeout: int = 500) -> str:
    import urllib.request
    import json as _json
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=_json.dumps({"model": model, "prompt": prompt, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _json.loads(resp.read())["response"]


def list_ollama_models() -> list[str]:
    import urllib.request
    import json as _json
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as resp:
        data = _json.loads(resp.read())
    return [m["name"] for m in data.get("models", [])]


# ──────────────────────────────────────────────────────────────────────
#  FILE I/O
# ──────────────────────────────────────────────────────────────────────

HEADER = '"""Auto-generated operators (see operator_generator.py)."""\nimport numpy as np\n\n'

# Each entry: (category, source, func_name, flavor, code_block_starting_with_def)
Entry = tuple[str, str, str, str, str]


def _strip_imports(code: str) -> str:
    return "\n".join(
        ln for ln in code.splitlines()
        if not ln.strip().startswith(("import ", "from "))
    ).strip()


def read_existing() -> list[Entry]:
    if not OUT_FILE.exists():
        return []
    text = OUT_FILE.read_text()
    pattern = re.compile(
        r"# category: (good|specialized|weird) \| source: ([^|\n]+?)"
        r"(?:\s*\|\s*flavor:\s*([^\n]+))?\n"
        r"(def (op_(?:good|spec|weird)_\d+)\b.*?)"
        r"(?=\n# category:|\n# ══|\nGOOD_OPERATORS|\nSPECIALIZED_OPERATORS|\nWEIRD_OPERATORS|\Z)",
        re.DOTALL,
    )
    out: list[Entry] = []
    for m in pattern.finditer(text):
        out.append((
            m.group(1),
            m.group(2).strip(),
            m.group(5),
            (m.group(3) or "").strip(),
            m.group(4).rstrip(),
        ))
    return out


def write_file(entries: list[Entry]):
    buckets: dict[str, list[Entry]] = {c: [] for c in CATEGORIES}
    for e in entries:
        buckets[e[0]].append(e)

    labels = {"good": "GOOD", "specialized": "SPECIALIZED", "weird": "WEIRD"}
    list_names = {
        "good": "GOOD_OPERATORS",
        "specialized": "SPECIALIZED_OPERATORS",
        "weird": "WEIRD_OPERATORS",
    }

    with OUT_FILE.open("w") as f:
        f.write(HEADER)
        for cat in CATEGORIES:
            if not buckets[cat]:
                continue
            f.write(f"# ══════════ {labels[cat]} ══════════\n\n")
            for (_c, source, name, flavor, code) in buckets[cat]:
                body = _strip_imports(code)
                flavor_str = f" | flavor: {flavor}" if flavor else ""
                f.write(f"# category: {cat} | source: {source}{flavor_str}\n{body}\n\n")
        for cat in CATEGORIES:
            names = ", ".join(e[2] for e in buckets[cat])
            f.write(f"{list_names[cat]} = [{names}]\n")
    totals = ", ".join(f"{len(buckets[c])} {c}" for c in CATEGORIES)
    print(f"→ wrote {len(entries)} operators to {OUT_FILE} ({totals})")


def next_func_name(entries: list[Entry], category: str) -> str:
    prefix = PREFIX[category]
    used = {e[2] for e in entries if e[0] == category}
    i = 1
    while f"{prefix}{i}" in used:
        i += 1
    return f"{prefix}{i}"


# ──────────────────────────────────────────────────────────────────────
#  COMMANDS
# ──────────────────────────────────────────────────────────────────────

def cmd_show_prompt(args):
    cats = [args.category] if args.category else list(CATEGORIES)
    for cat in cats:
        flavor = args.flavor or FLAVORS[cat][0]
        name = f"{PREFIX[cat]}1"
        header = f"  CATEGORY: {cat.upper()}   FLAVOR: {flavor}   FUNC: {name}  "
        print("═" * max(64, len(header)))
        print(header)
        print("═" * max(64, len(header)))
        print()
        print(build_prompt(cat, name, flavor))
    print("# Save the reply to a file, then:")
    print("#   python operator_generator.py add-manual <reply.txt> --category <cat>")


def cmd_list_models(args):
    try:
        for m in list_ollama_models():
            print(m)
    except Exception as e:
        print(f"Could not reach Ollama at localhost:11434 — {e}", file=sys.stderr)
        sys.exit(1)


def cmd_generate(args):
    entries = read_existing()
    for cat in args.categories:
        have = sum(1 for e in entries if e[0] == cat)
        needed = args.count - have
        if needed <= 0:
            print(f"[{cat}] already have {have}/{args.count} — skipping")
            continue
        print(f"[{cat}] generating {needed} operator(s)...")
        attempts = 0
        flavor_idx = have
        while sum(1 for e in entries if e[0] == cat) < args.count and attempts < needed * 4:
            model = args.models[attempts % len(args.models)]
            flavor = FLAVORS[cat][flavor_idx % len(FLAVORS[cat])]
            name = next_func_name(entries, cat)
            prompt = build_prompt(cat, name, flavor)
            print(f"  [{cat} {name}] model={model} flavor={flavor[:40]!r}")
            try:
                raw = call_ollama(model, prompt)
            except Exception as e:
                print(f"    ollama error: {e}")
                attempts += 1
                continue
            code = extract_code(raw)
            ok, msg = validate(code, name)
            if ok:
                entries.append((cat, model, name, flavor, code))
                print(f"    ✓ accepted")
                flavor_idx += 1
            else:
                print(f"    ✗ rejected: {msg}")
            attempts += 1

    if not entries:
        print("no operators accepted", file=sys.stderr)
        sys.exit(1)
    write_file(entries)


def cmd_add_manual(args):
    path = Path(args.file)
    raw = path.read_text()
    entries = read_existing()
    name = next_func_name(entries, args.category)
    code = extract_code(raw)
    m = re.search(r"def\s+(op_\w+)\s*\(", code)
    if m and m.group(1) != name:
        print(f"  renaming {m.group(1)} → {name}")
        code = re.sub(rf"def\s+{m.group(1)}\s*\(", f"def {name}(", code)
    ok, msg = validate(code, name)
    if not ok:
        print(f"rejected: {msg}", file=sys.stderr)
        sys.exit(1)
    entries.append((args.category, f"manual:{path.name}", name, args.flavor, code))
    write_file(entries)
    print(f"  ✓ added {name} as {args.category}")


def cmd_test(args):
    sys.path.insert(0, str(Path.cwd()))
    try:
        import generated_operators as g
    except Exception as e:
        print(f"cannot import generated_operators: {e}")
        sys.exit(1)

    import numpy as np
    rng = np.random.default_rng(0)
    for list_name in ("GOOD_OPERATORS", "SPECIALIZED_OPERATORS", "WEIRD_OPERATORS"):
        ops = getattr(g, list_name, [])
        if not ops:
            print(f"{list_name}: (empty)")
            continue
        print(f"{list_name}:")
        for op in ops:
            pop = rng.uniform(-100, 100, (20, 10))
            fit = np.array([float(np.sum(x * x)) for x in pop])
            try:
                np_new, nf_new = op(
                    pop.copy(), fit.copy(),
                    lambda x: float(np.sum(x * x)),
                    rng, (-100.0, 100.0),
                )
                improved = int((nf_new < fit).sum())
                print(f"  {op.__name__:20s}  improved {improved:2d}/{len(pop)} on sphere")
            except Exception as e:
                print(f"  {op.__name__:20s}  FAILED: {e}")


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("show-prompt")
    sp.add_argument("--category", choices=CATEGORIES)
    sp.add_argument("--flavor", help="Override the flavor text")
    sp.set_defaults(func=cmd_show_prompt)

    sub.add_parser("list-models").set_defaults(func=cmd_list_models)

    g = sub.add_parser("generate")
    g.add_argument("--count", type=int, default=5, help="operators per category (default 5)")
    g.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    g.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    g.set_defaults(func=cmd_generate)

    am = sub.add_parser("add-manual")
    am.add_argument("file")
    am.add_argument("--category", choices=CATEGORIES, required=True)
    am.add_argument("--flavor", default="")
    am.set_defaults(func=cmd_add_manual)

    sub.add_parser("test").set_defaults(func=cmd_test)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()