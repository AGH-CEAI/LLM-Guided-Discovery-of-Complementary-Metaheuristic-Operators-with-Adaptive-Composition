"""
Minimal diagnostic to isolate the hang.

Run this standalone in the same environment as your pipeline. It does, in order:
  1. One smoke test (uses mp.Process with 'spawn')
  2. One evaluateGNBG3 call on the same algorithm
  3. Reports timing for each

If step 2 hangs or takes dramatically longer than running evaluate_gnbg3.py
standalone, the issue is cross-contamination between the two multiprocessing
contexts. If step 2 is normal, the issue is elsewhere (watchdog, stdout
buffering, or something in the full pipeline that this harness doesn't
exercise).

Usage:
    python diagnose_hang.py path/to/seed_1_AdaptiveMultiStrategyDE.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")


def main():
    if len(sys.argv) != 2:
        print("usage: python diagnose_hang.py <seed.py>")
        sys.exit(1)

    seed_path = Path(sys.argv[1])
    if not seed_path.exists():
        print(f"Seed file not found: {seed_path}")
        sys.exit(1)

    print(f"Testing with seed: {seed_path.name}")
    print()

    # --- Step 1: smoke test ---
    print("Step 1: smoke test")
    t0 = time.perf_counter()
    from smoke_test import smoke_test_seed
    ok, msg = smoke_test_seed(seed_path, timeout=60.0)
    t1 = time.perf_counter() - t0
    print(f"  result: {'ok' if ok else 'FAIL'}")
    print(f"  message: {msg}")
    print(f"  elapsed: {t1:.1f}s")
    print()

    if not ok:
        print("Smoke test failed — stopping here.")
        sys.exit(1)

    # --- Step 2: full GNBG benchmark on the same algorithm ---
    print("Step 2: evaluateGNBG3 (this is the call that's hanging in the pipeline)")
    print("  If this hangs too, it's a multiprocessing context interaction.")
    print("  If this completes normally, the hang is elsewhere in the pipeline.")
    print()

    from evaluate_gnbg3 import evaluateGNBG3
    code = seed_path.read_text(encoding="utf-8")
    
    t0 = time.perf_counter()
    print(f"  Starting evaluateGNBG3 at {time.strftime('%H:%M:%S')}...")
    sys.stdout.flush()
    
    # Use a small budget so this finishes fast if it's going to work at all
    result = evaluateGNBG3(code, iterations=50_000, repetitions_per_fid=6)
    t2 = time.perf_counter() - t0
    
    print(f"  elapsed: {t2:.1f}s")
    print(f"  result: {result[:3]}... (first 3 of 24)")
    print()

    # --- Step 3: Second GNBG call, to test back-to-back ---
    print("Step 3: second evaluateGNBG3 call (same algo, immediately after)")
    print("  If this hangs but step 2 worked, it's a leaked-handle issue")
    print("  between successive ProcessPoolExecutor invocations.")
    print()
    
    t0 = time.perf_counter()
    print(f"  Starting second evaluateGNBG3 at {time.strftime('%H:%M:%S')}...")
    sys.stdout.flush()
    result = evaluateGNBG3(code, iterations=50_000, repetitions_per_fid=6)
    t3 = time.perf_counter() - t0
    print(f"  elapsed: {t3:.1f}s")
    print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Smoke test:       {t1:6.1f}s")
    print(f"  First GNBG call:  {t2:6.1f}s")
    print(f"  Second GNBG call: {t3:6.1f}s")
    print()

    if t3 > t2 * 2.5:
        print("⚠  Second GNBG call was much slower than the first — suggests")
        print("   resource leak between runs.")
    elif abs(t3 - t2) / max(t2, 1) < 0.3:
        print("✔  Both GNBG calls ran in similar time — multiprocessing is clean.")
        print("   The pipeline hang is likely elsewhere (watchdog, stdout")
        print("   buffering, or orchestration-layer issue, not the benchmark).")


if __name__ == "__main__":
    main()