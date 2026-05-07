# Optimized GNBG-III evaluator (same mechanic as the GNBG-I/II evaluator).
#
# Key design points (identical to the GNBG-I evaluator):
#   1. exec() runs ONCE per worker process (via initializer)
#   2. Half-budget fids (1-15) batched in packs of 3 reps to reduce overhead
#   3. Full-budget fids (16-24) stay as individual tasks (slower, benefit from fine scheduling)
#   4. as_completed() streams results without waiting for ordering
#
# What changed vs. GNBG-I:
#   - Imports load_gnbg from gnbg_iii_loader (wraps the .mat/batch GNBG-III interface
#     into the GNBG-I object-style interface).
#   - Default full budget is 500_000 to match the GNBG-III competition spec
#     (MaxEvals = 5e5 per problem); half budget is therefore 250_000.
#   - Scoring unchanged: harmonic_mean of sorted[1:5] per fid,
#     then geometric_mean across fids.
#   - Cross-platform timeout: SIGALRM on Unix, threading.Timer + ctypes on Windows.

from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from statistics import geometric_mean, harmonic_mean

import numpy as np

from gnbg_iii_loader import load_gnbg

# GNBG-III competition defaults
DEFAULT_FULL_BUDGET = 500_000
REPETITIONS_PER_FID = 6
HALF_BUDGET_PACK_SIZE = 3  # pack fast tasks in groups of 3
time_limit_s = 4500

_IS_WINDOWS = sys.platform == "win32"

# ── Cross-platform timeout guard ──────────────────────────────────
# SIGALRM is Unix-only. On Windows we emulate it with a daemon Timer
# thread that asynchronously raises TimeoutError in the worker's main
# thread via the CPython PyThreadState_SetAsyncExc API.
if not _IS_WINDOWS:
    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError(
            f"Timed out at line {frame.f_lineno} in {frame.f_code.co_filename}."
        )

    class _TimeoutGuard:
        """Unix: uses SIGALRM for a low-overhead preemptive timeout."""

        def __init__(self, seconds):
            self.seconds = int(seconds)
            self._old_handler = None

        def __enter__(self):
            self._old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self.seconds)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            signal.alarm(0)
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)
            return False
else:
    import ctypes
    import threading

    def _async_raise(thread_id, exception):
        """Asynchronously raise `exception` in the thread with the given id."""
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(thread_id), ctypes.py_object(exception)
        )
        if res > 1:
            # Should never affect more than one thread; roll back if it did.
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(thread_id), None
            )

    class _TimeoutGuard:
        """Windows: Timer-thread based timeout (ctypes async-exc injection).

        Note: pure C-extension code blocks the async exception until it
        returns to the Python interpreter, so extremely long NumPy calls
        may only terminate when they yield back. This is the same
        limitation as using `future.result(timeout=...)` and is fine for
        the 10-hour watchdog used here.
        """

        def __init__(self, seconds):
            self.seconds = float(seconds)
            self._timer = None

        def __enter__(self):
            tid = threading.get_ident()
            self._timer = threading.Timer(
                self.seconds, _async_raise, args=(tid, TimeoutError)
            )
            self._timer.daemon = True
            self._timer.start()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            return False


# ── Per-worker global state (initialized once per process) ────────
_worker_metaheuristic = None
_worker_algorithm_name = None


def _worker_init(code, algorithm_name):
    """Called once when each worker process starts. Exec's code only once."""
    # Avoid BLAS thread contention: we already have process-level parallelism
    # across `cpu_count()` workers, so letting each one spawn BLAS threads
    # oversubscribes the machine on small (D=30) matmuls.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass

    global _worker_metaheuristic, _worker_algorithm_name
    _worker_algorithm_name = algorithm_name
    shared_env = {"__builtins__": __builtins__}
    exec(code, shared_env)
    _worker_metaheuristic = shared_env[algorithm_name]


class EarlyStoppingError(Exception):
    pass


def _run_one(fid, budget):
    """Run a single (fid) evaluation, return absolute error."""
    problem = load_gnbg(fid)
    algorithm = _worker_metaheuristic(dim=problem.Dimension)

    deadline = time.monotonic() + time_limit_s
    stopping_condition = lambda: problem.FE >= budget or time.monotonic() >= deadline
    algorithm(problem.fitness, stopping_condition)

    if problem.FE < budget:
        raise EarlyStoppingError(
            "The algorithm stopped before stopping_condition() returned True"
        )

    best_value = float(np.min(problem.FEhistory[:budget]))
    return max(best_value - problem.OptimumValue, 1e-8)


def run_packed_gnbg(args):
    """Run a PACK of (fid, rep) evaluations sequentially. Returns list of results."""
    fid, num_reps, budget, timeout_seconds = args
    try:
        with _TimeoutGuard(timeout_seconds):
            errors_list = []
            for _ in range(num_reps):
                errors_list.append(_run_one(fid, budget))
        return ("ok", fid, errors_list)
    except Exception as excp:
        tb_str = traceback.format_exc()
        return ("error", fid, excp, tb_str)
 
 
def run_single_gnbg(args):
    """Run ONE (fid, rep) evaluation. For full-budget tasks."""
    fid, rep_index, budget, timeout_seconds = args
    try:
        with _TimeoutGuard(timeout_seconds):
            abs_error = _run_one(fid, budget)
        return ("ok", fid, [abs_error])
    except Exception as excp:
        tb_str = traceback.format_exc()
        return ("error", fid, excp, tb_str)


def get_first_non_enum_class(code: str) -> str:
    pattern = r"class\s+(\w+)(?:\s*\([^)]*\))?:"
    for match in re.finditer(pattern, code):
        class_name = match.group(1)
        full_match = match.group(0)
        if not re.search(r"\(\s*\w*Enum\w*\s*\)", full_match):
            return class_name
    return "Not-Identified"


def evaluateGNBG3(
    code,
    explogger=None,
    details=True,
    iterations=DEFAULT_FULL_BUDGET,
    repetitions_per_fid=6,
):
    """Evaluate `code` (a string containing a metaheuristic class definition) on GNBG-III.

    Returns a list of 24 per-fid scores (harmonic mean of the 2nd..5th best
    absolute errors across reps). Lower is better; 1e-8 is the acceptance target.
    """
    algorithm_name = get_first_non_enum_class(code)
    half_iterations = iterations // 2

    # # ── Build tasks ───────────────────────────────────────────────
    # # Fids 1-15 (half budget, fast): `repetitions_per_fid` reps packed in groups of 3
    # # Fids 16-24 (full budget, slow): individual tasks for fine-grained scheduling
    packed_tasks = []
    # for fid in range(1, 16):
    #     for pack in range(repetitions_per_fid // HALF_BUDGET_PACK_SIZE):
    #         packed_tasks.append(
    #             (run_packed_gnbg,
    #              (fid, HALF_BUDGET_PACK_SIZE, half_iterations, 3 * 3600))
    #         )

    single_tasks = []
    for fid in range(1, 25):
        for rep in range(repetitions_per_fid):
            single_tasks.append(
                (run_single_gnbg, (fid, rep, iterations, 3 * 3600))
            )

    all_tasks = packed_tasks + single_tasks
    num_workers = min(mp.cpu_count(), 48)
    print(
        f"Dispatching {len(all_tasks)} tasks "
        f"({len(packed_tasks)} packed + {len(single_tasks)} single) "
        f"across {num_workers} workers..."
    )

    # ── Execute ───────────────────────────────────────────────────
    results_dict: dict[int, list[float]] = {}
    errors = []

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(code, algorithm_name),
    ) as executor:
        futures = {executor.submit(func, args): args for func, args in all_tasks}
        for future in as_completed(futures):
            elem = future.result()
            if elem[0] == "error":
                errors.append(elem)
            else:
                _, fid, abs_errors = elem
                results_dict.setdefault(fid, []).extend(abs_errors)

    if errors:
        print(f"{len(errors)} task(s) failed. First error:")
        _, fid, excp, tb_str = errors[0]
        print(f"  fid={fid}: {tb_str}")
        return [-np.inf for _ in range(24)]

    # ── Score each fid ────────────────────────────────────────────
    results_list = [0.0] * 24
    for i in range(24):
        fid_errors = sorted(results_dict[i + 1])
        # Harmonic mean of the 2nd..5th smallest errors (needs >= 5 reps)
        results_list[i] = harmonic_mean(fid_errors[1:5])

    worst_index = int(np.argmax(np.array(results_list)))
    worst_absolute_error = results_list[worst_index]
    absolute_error_mean = geometric_mean(results_list)

    print(
        f"Algorithm {algorithm_name}: avg error {absolute_error_mean:.2e}, "
        f"worst {worst_absolute_error:.2e} at F{worst_index + 1} (target: 1e-8)"
    )

    return results_list


if __name__ == "__main__":
    with open("runs/mixed/lineage_000/seeds/seed_2_CovarianceRingEllipsoidOptimizer.py", "r") as f:
        code = f.read()
    budgets = [DEFAULT_FULL_BUDGET]
    with open("time_comparison.csv", "w") as file:
        file.write("budget,time\n")
        for budget in budgets:
            start = time.perf_counter()
            print("Start time", str(datetime.datetime.now()))
            print(evaluateGNBG3(
                code,
                iterations=budget,
                repetitions_per_fid=REPETITIONS_PER_FID,
            ))
            end = time.perf_counter()
            elapsed = end - start
            print(f"Elapsed time: {elapsed:.6f} seconds for {budget} iterations")
            file.write(f"{budget},{elapsed}\n")