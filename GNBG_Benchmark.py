# ─────────────────────────────────────────────────────────────────────────
# CRITICAL: cap BLAS threads BEFORE numpy is imported anywhere.
# With 32 workers × 32 BLAS threads each, 1024 threads fight 32 cores,
# memory balloons from thread-local scratch, and htop shows 0% CPU.
# These env vars must be set before the first `import numpy`.
# ─────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import datetime
import re
import signal
import time
import traceback
from statistics import geometric_mean, harmonic_mean

import numpy as np
import multiprocessing as mp
from gnbg_loader import load_gnbg

# ── Benchmark config ─────────────────────────────────────────────
REPETITIONS_PER_FID = 31
HALF_BUDGET_PACK_SIZE = 3
ACCEPTANCE_THRESHOLD = 1e-8

# Per-task wall-clock caps. These are hard upper bounds; a correctly-
# behaving task should finish in a small fraction of these.
TIMEOUT_HALF = 10 * 60      # f1-f15 with half budget
TIMEOUT_FULL = 20 * 60      # f16-f24 with full budget

# Recycle workers periodically to prevent slow memory growth from
# closure-cycle garbage and per-worker caches.
MAX_TASKS_PER_CHILD = 16

# Emit a "still alive" line this often, even if no task completes.
PROGRESS_HEARTBEAT_SEC = 30


# ── Per-worker global state (initialised once per process) ───────
_worker_metaheuristic = None
_worker_algorithm_name = None


def _worker_init(code, algorithm_name):
    """Called once when each worker process starts."""
    global _worker_metaheuristic, _worker_algorithm_name
    _worker_algorithm_name = algorithm_name
    shared_env = {"__builtins__": __builtins__}
    exec(code, shared_env)
    _worker_metaheuristic = shared_env[algorithm_name]


class EarlyStoppingError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError(
        f"Timed out at line {frame.f_lineno} in {frame.f_code.co_filename}."
    )


# ── Single-run core ──────────────────────────────────────────────

def _run_one(fid, budget):
    """Run a single (fid) evaluation. Returns (best_value, best_params)."""
    problem = load_gnbg(fid)

    # If the GNBG instance's MaxEvals is below the requested budget, clamp.
    # Without this, FE saturates at MaxEvals and fitness returns all-NaN
    # without incrementing FE → stopping_condition() never triggers.
    effective_budget = min(int(budget), int(problem.MaxEvals))
    optimum_value = float(problem.OptimumValue)

    best_x = None
    best_f = np.inf

    # FE-watchdog state: [last_fe, stall_count]
    _fe_watch = [0, 0]

    def fitness_wrapper(x, *args, **kwargs):
        nonlocal best_x, best_f
        f = problem.fitness(x, *args, **kwargs)

        f_vals = np.atleast_1d(f)
        x_vals = np.atleast_2d(x)
        n = min(len(f_vals), len(x_vals))
        if n == 0:
            return f

        sub_f = f_vals[:n]
        valid = np.isfinite(sub_f)
        if not valid.any():
            return f

        # Single vectorised argmin instead of Python for-loop per call
        masked = np.where(valid, sub_f, np.inf)
        idx = int(np.argmin(masked))
        fi = float(masked[idx])
        if fi < best_f:
            best_f = fi
            best_x = x_vals[idx].copy()
        return f

    def stopping_condition():
        fe = problem.FE

        # Watchdog: if FE hasn't moved for 200 consecutive checks, bail.
        # Catches the pathological case where fitness silently returns NaN.
        if fe == _fe_watch[0]:
            _fe_watch[1] += 1
            if _fe_watch[1] > 200:
                return True
        else:
            _fe_watch[0] = fe
            _fe_watch[1] = 0

        if fe >= effective_budget:
            return True
        if best_f - optimum_value <= ACCEPTANCE_THRESHOLD:
            return True
        return False

    algorithm = _worker_metaheuristic(dim=problem.Dimension)
    try:
        algorithm(fitness_wrapper, stopping_condition)
    finally:
        # Break reference cycle (gnbg.fitness closure captures gnbg itself)
        # and drop the ~N-float FEhistory list so GC can reclaim memory
        # immediately rather than waiting for cycle collection.
        try:
            problem.fitness = None
            problem.FEhistory = None
        except Exception:
            pass

    if problem.FE < effective_budget and best_f - optimum_value > ACCEPTANCE_THRESHOLD:
        raise EarlyStoppingError(
            f"Algorithm stopped early: FE={problem.FE}/{effective_budget}, "
            f"best_f={best_f}, optimum={optimum_value}"
        )

    return best_f, best_x


# ── Task wrappers (run inside workers) ───────────────────────────

def run_packed_gnbg(args):
    """Run a PACK of N reps of the same fid sequentially in one worker task."""
    fid, num_reps, budget, timeout_seconds = args
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    t0 = time.perf_counter()
    try:
        results_list = []
        for _ in range(num_reps):
            results_list.append(_run_one(fid, budget))
        return ("ok", fid, results_list, time.perf_counter() - t0)
    except Exception as excp:
        tb_str = traceback.format_exc()
        return ("error", fid, excp, tb_str)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


def run_single_gnbg(args):
    """Run one (fid, rep) evaluation."""
    fid, rep_index, budget, timeout_seconds = args
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    t0 = time.perf_counter()
    try:
        best_f, best_x = _run_one(fid, budget)
        return ("ok", fid, [(best_f, best_x)], time.perf_counter() - t0)
    except Exception as excp:
        tb_str = traceback.format_exc()
        return ("error", fid, excp, tb_str)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


# ── Utilities ────────────────────────────────────────────────────

def get_first_non_enum_class(code: str) -> str:
    pattern = r"class\s+(\w+)(?:\s*\([^)]*\))?:"
    for match in re.finditer(pattern, code):
        if not re.search(r"\(\s*\w*Enum\w*\s*\)", match.group(0)):
            return match.group(1)
    return "Not-Identified"


def _fmt_time(s: float) -> str:
    s = int(max(0, s))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:02d}m{s:02d}s"
    return f"{s}s"


def _fid_summary(fid: int, results, optimum_value: float) -> str:
    errs = np.array([max(r[0] - optimum_value, 1e-30) for r in results])
    return (f"fid={fid:<2} n={len(results)} "
            f"best={errs.min():.3e} median={np.median(errs):.3e} "
            f"worst={errs.max():.3e}")


# ── Main orchestrator ────────────────────────────────────────────

def evaluateGNGB(code, explogger=None, details=True,
                 iterations=1_000_000, repetitions_per_fid=31):
    algorithm_name = get_first_non_enum_class(code)
    half_iterations = iterations // 2

    # ── Build task list ──────────────────────────────────────────
    packed_tasks, single_tasks = [], []

    # f1-f15: fast, pack HALF_BUDGET_PACK_SIZE reps per task
    for fid in range(1, 16):
        num_packs = repetitions_per_fid // HALF_BUDGET_PACK_SIZE
        remainder = repetitions_per_fid % HALF_BUDGET_PACK_SIZE
        for _ in range(num_packs):
            packed_tasks.append((run_packed_gnbg,
                (fid, HALF_BUDGET_PACK_SIZE, half_iterations, TIMEOUT_HALF)))
        for rep in range(remainder):
            single_tasks.append((run_single_gnbg,
                (fid, rep, half_iterations, TIMEOUT_HALF)))

    # f16-f24: slow, one rep per task
    for fid in range(16, 25):
        for rep in range(repetitions_per_fid):
            single_tasks.append((run_single_gnbg,
                (fid, rep, iterations, TIMEOUT_FULL)))

    all_tasks = packed_tasks + single_tasks
    total_reps = 24 * repetitions_per_fid
    num_workers = mp.cpu_count()

    # Cache optimum values once in the main process (avoids 24 re-loads later)
    optimum_values = {fid: load_gnbg(fid).OptimumValue for fid in range(1, 25)}

    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
          f"Dispatching {len(all_tasks)} tasks "
          f"({len(packed_tasks)} packed + {len(single_tasks)} single = "
          f"{total_reps} reps) across {num_workers} workers "
          f"(max {MAX_TASKS_PER_CHILD} tasks/child, "
          f"BLAS threads pinned to 1)", flush=True)

    # ── Execute ─────────────────────────────────────────────────
    results_dict: dict[int, list[tuple]] = {}
    fid_done_reported: set[int] = set()
    errors = []

    t_start = time.perf_counter()
    last_heartbeat = t_start

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(code, algorithm_name),
        max_tasks_per_child=MAX_TASKS_PER_CHILD,
    ) as executor:
        futures = {executor.submit(func, args): args for func, args in all_tasks}
        total_tasks = len(futures)
        tasks_done = 0
        reps_done = 0

        # Use wait() with a timeout so the heartbeat fires on wall-clock
        # even when no future completes for a long time. (as_completed
        # blocks until something returns, which hides real hangs.)
        pending = set(futures.keys())
        while pending:
            done_set, pending = wait(
                pending,
                timeout=PROGRESS_HEARTBEAT_SEC,
                return_when=FIRST_COMPLETED,
            )
            now = time.perf_counter()
            elapsed = now - t_start

            if not done_set:
                # Timeout hit with zero completions: emit a loud heartbeat
                # so it's clear the parent is alive and what's pending.
                fid_counts: dict[int, int] = {}
                for f in pending:
                    args = futures[f]
                    fid = args[1][0]   # (func, args) -> args[0] is fid
                    fid_counts[fid] = fid_counts.get(fid, 0) + 1
                breakdown = ", ".join(f"f{fid}×{n}"
                    for fid, n in sorted(fid_counts.items()))
                print(f"[{_fmt_time(elapsed)}] ⟳ heartbeat: "
                      f"{len(pending)} pending ({breakdown}); "
                      f"{reps_done}/{total_reps} reps done; "
                      f"rate={reps_done/max(elapsed,1e-9):.2f} reps/s",
                      flush=True)
                last_heartbeat = now
                continue

            for future in done_set:
                elem = future.result()
                tasks_done += 1

                if elem[0] == "error":
                    _, fid, excp, tb_str = elem
                    errors.append(elem)
                    print(f"[{_fmt_time(elapsed)}] ✗ ERROR fid={fid}: "
                          f"{type(excp).__name__}: {excp}", flush=True)
                    continue

                _, fid, run_results, task_time = elem
                results_dict.setdefault(fid, []).extend(run_results)
                reps_done += len(run_results)

                rate = reps_done / max(elapsed, 1e-9)
                eta_s = (total_reps - reps_done) / max(rate, 1e-9)
                best_err = max(min(r[0] for r in run_results)
                               - optimum_values[fid], 0.0)

                print(
                    f"[{_fmt_time(elapsed)}] "
                    f"task {tasks_done:>3}/{total_tasks}  "
                    f"reps {reps_done:>3}/{total_reps}  "
                    f"fid={fid:<2} (+{len(run_results)})  "
                    f"err={best_err:.2e}  "
                    f"t={task_time:5.1f}s  "
                    f"ETA={_fmt_time(eta_s)}",
                    flush=True,
                )

                if (fid not in fid_done_reported
                        and len(results_dict[fid]) >= repetitions_per_fid):
                    fid_done_reported.add(fid)
                    print(f"    ★ {_fid_summary(fid, results_dict[fid], optimum_values[fid])}",
                          flush=True)

            # Periodic low-frequency heartbeat even when completions are
            # rolling in (useful for very long runs)
            if now - last_heartbeat > PROGRESS_HEARTBEAT_SEC * 10:
                print(f"[{_fmt_time(elapsed)}] ⟳ still running: "
                      f"{len(pending)} pending, "
                      f"{reps_done}/{total_reps} reps done",
                      flush=True)
                last_heartbeat = now

    total_elapsed = time.perf_counter() - t_start
    print(f"\n[{_fmt_time(total_elapsed)}] All tasks complete. "
          f"Errors: {len(errors)}/{total_tasks}", flush=True)

    if errors:
        print("First error traceback:")
        print(errors[0][3])
        return [-np.inf] * 24

    # ── Validate counts ────────────────────────────────────────
    for fid in range(1, 25):
        n_found = len(results_dict.get(fid, []))
        if n_found != repetitions_per_fid:
            print(f"Warning: fid={fid} has {n_found} results, "
                  f"expected {repetitions_per_fid}", flush=True)

    # ── Save per-function result files ─────────────────────────
    for fid in range(1, 25):
        run_results = sorted(results_dict[fid], key=lambda x: x[0])
        with open(f"f_{fid}_value.txt", "w") as vf, \
             open(f"f_{fid}_params.txt", "w") as pf:
            for best_f, best_x in run_results:
                vf.write(f"{best_f}\n")
                pf.write(",".join(str(x) for x in best_x.flatten()) + "\n")

    print(f"Saved f_1..f_24 value/params files", flush=True)

    # ── Per-fid score + overall summary ────────────────────────
    results_list = [0.0] * 24
    print("\nFinal per-fid summary:")
    for i in range(24):
        fid = i + 1
        values = [r[0] for r in results_dict[fid]]
        fid_errors = sorted([max(v - optimum_values[fid], 1e-8) for v in values])
        results_list[i] = harmonic_mean(fid_errors[1:5])
        print(f"  {_fid_summary(fid, results_dict[fid], optimum_values[fid])}"
              f"  score={results_list[i]:.3e}", flush=True)

    worst_idx = int(np.argmax(np.array(results_list)))
    worst = results_list[worst_idx]
    mean_score = geometric_mean(results_list)
    print(f"\nAlgorithm {algorithm_name}: "
          f"avg error {mean_score:.2e}, "
          f"worst {worst:.2e} at f{worst_idx + 1} (target: 1e-8)", flush=True)

    return results_list


if __name__ == "__main__":
    with open("best_algorithm.py", "r") as f:
        code = f.read()
    budgets = [1_000_000]
    with open("time_comparison.csv", "w") as file:
        file.write("budget,time\n")
        for budget in budgets:
            start = time.perf_counter()
            print(f"Start time: {datetime.datetime.now()}", flush=True)
            result = evaluateGNGB(code, iterations=budget,
                                  repetitions_per_fid=REPETITIONS_PER_FID)
            print(result)
            elapsed = time.perf_counter() - start
            print(f"Elapsed: {elapsed:.2f} seconds for budget={budget}", flush=True)
            file.write(f"{budget},{elapsed}\n")