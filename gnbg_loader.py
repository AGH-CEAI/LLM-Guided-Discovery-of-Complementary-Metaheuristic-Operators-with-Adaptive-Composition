"""
gnbg_loader.py
--------------
Loads GNBG problem instances from .mat files and applies a vectorised
performance patch to gnbg.fitness() so the benchmark runs significantly
faster without changing any results.

Usage:
    from gnbg_loader import load_gnbg
    gnbg = load_gnbg(func_id=1, gnbg_dir="GNBG")
"""

import contextlib
import io
import sys
import numpy as np
from pathlib import Path
from scipy.io import loadmat

# Per-process cache: avoids re-reading the same .mat file for every run of the
# same function (31 runs × 24 functions = 744 reads reduced to 24 reads).
_mat_cache: dict = {}


def _add_gnbg_to_path(gnbg_dir: str):
    gnbg_path = str(Path(gnbg_dir).resolve())
    if gnbg_path not in sys.path:
        sys.path.insert(0, gnbg_path)


# ══════════════════════════════════════════════════════════════════════════════
# Performance patch
# ══════════════════════════════════════════════════════════════════════════════
#
# The original GNBG.fitness() has two expensive patterns:
#
#   1. Python for-loop over every solution in the population (SolutionNumber).
#      For pop_size = 40 that is 40 Python iterations every generation.
#      We replace this with fully-vectorised NumPy operations.
#
#   2. np.diag(CompH[k, :]) — builds a full dim×dim matrix every call just
#      to do a weighted dot product.  We use element-wise multiply instead:
#
#        a @ diag(H) @ b  ≡  Σ_j H[j] · a[j] · b[j]
#
#      Moreover, because b = a.T (they are transposes of the same values —
#      see proof in comments below), this simplifies further to:
#
#        Σ_j H[j] · a[j]²  =  np.sum(H * a**2, axis=1)   (vectorised)
#
#      This lets us skip computing b entirely.
#
# Why b = a.T (one-solution proof):
#   a_arg = (x − μ)ᵀ @ Rᵀ  →  shape (1, dim),  element [0,j] = Σ_i (x−μ)_i · R_{ji}
#   b_arg = R @ (x − μ)     →  shape (dim, 1),  element [j,0] = Σ_i R_{ji} · (x−μ)_i
#   ∴ a_arg[0,j] = b_arg[j,0]  ∀j,  so transform(a_arg)[0,j] = transform(b_arg)[j,0]
#   i.e. a = bᵀ  element-wise.
#
# ══════════════════════════════════════════════════════════════════════════════

def _transform_batch(X: np.ndarray, Alpha, Beta) -> np.ndarray:
    """
    The GNBG transform applied to an (N, dim) or (1, dim) array.
    Identical element-wise logic to GNBG.transform(); extracted here so the
    patched fitness closure does not need a bound-method lookup every call.
    """
    Y = X.copy()
    pos = X > 0
    if pos.any():
        lp = np.log(X[pos])
        Y[pos] = np.exp(lp + Alpha[0] * (np.sin(Beta[0] * lp) + np.sin(Beta[1] * lp)))
    neg = X < 0
    if neg.any():
        ln = np.log(-X[neg])
        Y[neg] = -np.exp(ln + Alpha[1] * (np.sin(Beta[2] * ln) + np.sin(Beta[3] * ln)))
    return Y


def _apply_speed_patch(gnbg) -> None:
    """
    Replace gnbg.fitness with a vectorised version that evaluates all N
    solutions with NumPy batch ops instead of a Python loop over solutions.

    State fields updated identically to the original:
        gnbg.FE                  — incremented by to_eval
        gnbg.BestFoundResult     — updated to batch minimum
        gnbg.AcceptanceReachPoint— set to 1-indexed FE of first threshold hit
        gnbg.FEhistory           — extended with the batch values (list.extend)
    """
    g = gnbg  # short alias kept in closure

    # ── Pre-compute per-component data (done once at load time) ────────── #
    if len(g.RotationMatrix.shape) == 3:
        R_list = [np.ascontiguousarray(g.RotationMatrix[:, :, k])
                  for k in range(g.CompNum)]
    else:
        R_list = [np.ascontiguousarray(g.RotationMatrix)] * g.CompNum

    # H diagonal values as 1-D arrays — avoids np.diag every call
    H_list = [g.CompH[k, :] for k in range(g.CompNum)]

    # Capture scalar constants in closure for speed (avoids attribute lookup)
    CompNum   = g.CompNum
    CompMinPos = g.CompMinPos
    CompSigma  = g.CompSigma
    Lambda_    = g.Lambda
    Mu_        = g.Mu
    Omega_     = g.Omega
    OptVal     = g.OptimumValue
    Threshold  = g.AcceptanceThreshold

    def fast_fitness(X: np.ndarray) -> np.ndarray:
        if X.ndim < 2:
            X = X.reshape(1, -1)
        N = X.shape[0]
        result = np.full(N, np.nan)

        # How many solutions can we still evaluate?
        remaining = g.MaxEvals - g.FE
        if remaining <= 0:
            return result
        to_eval = min(N, remaining)
        X_b = X[:to_eval]                          # (to_eval, dim)

        # ── Vectorised component loop (typically 1-10 iterations) ──────── #
        f_comp = np.empty((to_eval, CompNum))
        for k in range(CompNum):
            dx = X_b - CompMinPos[k]               # (to_eval, dim)  broadcast
            # a_raw = dx @ R.T  shape (to_eval, dim)
            # (b_raw = R @ dx.T = same values transposed → b = a, so skip it)
            a = _transform_batch(dx @ R_list[k].T, Mu_[k], Omega_[k])
            # inner[i] = Σ_j H[j] · a[i,j]²  ≡  a @ diag(H) @ b
            inner = np.sum(H_list[k] * (a * a), axis=1)   # (to_eval,)
            f_comp[:, k] = CompSigma[k] + inner ** Lambda_[k]

        batch_vals = np.min(f_comp, axis=1)        # (to_eval,)
        result[:to_eval] = batch_vals

        # ── Update GNBG state (all vectorised) ──────────────────────────── #
        start_fe = g.FE
        g.FE += to_eval
        g.FEhistory.extend(batch_vals.tolist())    # list.extend is O(to_eval)

        min_val = float(batch_vals.min())
        if min_val < g.BestFoundResult:
            g.BestFoundResult = min_val

        # AcceptanceReachPoint: 1-indexed FE of the FIRST solution in this
        # batch whose absolute error falls below the threshold.
        if np.isinf(g.AcceptanceReachPoint):
            errors = np.abs(batch_vals - OptVal)
            hits   = np.nonzero(errors < Threshold)[0]
            if len(hits):
                g.AcceptanceReachPoint = float(start_fe + int(hits[0]) + 1)

        return result

    gnbg.fitness = fast_fitness   # monkey-patch the instance (not the class)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def load_gnbg(func_id: int, gnbg_dir: str = "GNBG-Python", fast: bool = True):
    """
    Load a GNBG problem instance from a .mat file.

    Parameters
    ----------
    func_id  : int  — function index (1..24)
    gnbg_dir : str  — folder containing GNBG_instances.py and f*.mat files
    fast     : bool — apply vectorised performance patch (default True)
                      Set False only if you need byte-identical original behaviour.

    Returns
    -------
    gnbg : GNBG instance ready to use  (call gnbg.fitness(X) to evaluate)

    Key attributes on the returned object
    --------------------------------------
    gnbg.Dimension          — problem dimensionality
    gnbg.MinCoordinate      — lower search bound (scalar, same for all dims)
    gnbg.MaxCoordinate      — upper search bound (scalar, same for all dims)
    gnbg.MaxEvals           — evaluation budget
    gnbg.AcceptanceThreshold— early-stop threshold  (1e-8)
    gnbg.OptimumValue       — true global optimum value
    gnbg.FE                 — current function-evaluation counter (starts at 0)
    gnbg.BestFoundResult    — best fitness found so far  (starts at +inf)
    gnbg.AcceptanceReachPoint — FE at which threshold was met (inf if not yet)
    """
    _add_gnbg_to_path(gnbg_dir)

    # Import GNBG class — suppress the module-level print spam that
    # GNBG_instances.py emits (it prints all 24 optimum values on import).
    # contextlib.redirect_stdout only affects the first import per process;
    # subsequent calls find the module already cached in sys.modules.
    with contextlib.redirect_stdout(io.StringIO()):
        from GNBG_instances import GNBG  # noqa: PLC0415

    if not (1 <= func_id <= 24):
        raise ValueError(f"func_id must be 1..24, got {func_id}")

    mat_path = Path(gnbg_dir) / f"f{func_id}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(
            f"Could not find {mat_path}\n"
            f"  Copy GNBG_instances.py and f1.mat-f24.mat into '{gnbg_dir}/'"
        )

    # ── Unpack the nested MATLAB struct (cached per process) ─────────────── #
    cache_key = (func_id, str(Path(gnbg_dir).resolve()))
    if cache_key not in _mat_cache:
        _mat_cache[cache_key] = loadmat(str(mat_path))["GNBG"]
    raw = _mat_cache[cache_key]

    def _scalar(field):
        return np.array([item[0] for item in raw[field].flatten()])[0, 0]

    def _matrix(field):
        return np.array(raw[field][0, 0])

    gnbg = GNBG(
        MaxEvals            = int(_scalar("MaxEvals")),
        AcceptanceThreshold = float(_scalar("AcceptanceThreshold")),
        Dimension           = int(_scalar("Dimension")),
        CompNum             = int(np.array([item[0] for item in raw["o"].flatten()])[0, 0]),
        MinCoordinate       = float(_scalar("MinCoordinate")),
        MaxCoordinate       = float(_scalar("MaxCoordinate")),
        CompMinPos          = _matrix("Component_MinimumPosition"),
        CompSigma           = np.array(raw["ComponentSigma"][0, 0], dtype=np.float64),
        CompH               = _matrix("Component_H"),
        Mu                  = _matrix("Mu"),
        Omega               = _matrix("Omega"),
        Lambda              = _matrix("lambda"),
        RotationMatrix      = _matrix("RotationMatrix"),
        OptimumValue        = float(_scalar("OptimumValue")),
        OptimumPosition     = _matrix("OptimumPosition"),
    )

    if fast:
        _apply_speed_patch(gnbg)

    return gnbg
