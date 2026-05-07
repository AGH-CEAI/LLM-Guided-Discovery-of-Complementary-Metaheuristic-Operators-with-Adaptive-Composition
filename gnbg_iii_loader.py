"""
GNBG-III loader — optimized, GNBG-I/II style interface.

Preserves the numerical semantics of the shipped GNBG-III fitness exactly
(modulo reductions that are mathematically identical but reorder floating-
point sums, causing ULP-level drift). No approximation, no surrogate, no
skipped evaluations. Specifically:

  1. Per-process caching of parsed .mat data (deep-copy only mutable fields).
  2. Vectorised fitness:
       - physics vectorised across the N solutions in a batch
       - physics vectorised across the o mixture components via einsum
       - `a @ diag(H) @ b` collapsed to Σ H·v²  (because the shipped code's
         `a` and `b` are literally the same rotated+transformed vector, just
         reshaped row vs column; see verification in your transcript)
       - F23 noise vectorised (randn(N) is bit-identical to N × randn()
         from the same state)
  3. Per-FE bookkeeping loop preserved unchanged (FE counter, FEhistory,
     BestFoundResult, AcceptanceReachPoint, FirstPoint/SecondPoint
     checkpoints, F24 dynamic shift at call-start).
  4. Two bugs in the shipped fitness.py still fixed:
       - ComponentSigma / lambda reshaped from (o, 1) to (o,)
       - result of the dot-product sandwich extracted as scalar
"""

from __future__ import annotations

import os
import numpy as np
from scipy.io import loadmat


# --------------------------------------------------------------------------
# Benchmark directory resolution
# --------------------------------------------------------------------------
_DEFAULT_DIR_NAME = "GNBG_III_Benchmarks_Python"


def _resolve_benchmark_dir(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("GNBG_III_BENCHMARK_DIR")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, _DEFAULT_DIR_NAME)
    if os.path.isdir(candidate):
        return candidate
    return os.path.join(os.getcwd(), _DEFAULT_DIR_NAME)


PROBLEM_FILES = {
    1:  "F1_Unimodal_Separable.mat",
    2:  "F2_Unimodal_FullyCoupled.mat",
    3:  "F3_IllConditioned_Separable.mat",
    4:  "F4_IllConditioned_Coupled.mat",
    5:  "F5_Chain_Deceptive.mat",
    6:  "F6_SuperLinear.mat",
    7:  "F7_PartialSeparable.mat",
    8:  "F8_Sparse50.mat",
    9:  "F9_Dense90.mat",
    10: "F10_MixedConditioning.mat",
    11: "F11_Multimodal_Symmetric_Sep.mat",
    12: "F12_Multimodal_Symmetric_Coupled.mat",
    13: "F13_Multimodal_Asymmetric_Sep.mat",
    14: "F14_Multimodal_Asymmetric_Coupled.mat",
    15: "F15_HighlyMultimodal_IllConditioned.mat",
    16: "F16_Deceptive.mat",
    17: "F17_ThreeComponents_Overlapping.mat",
    18: "F18_ThreeComponents_MixedCond.mat",
    19: "F19_FiveComponents_HighAsym.mat",
    20: "F20_MixedBasin.mat",
    21: "F21_PartialSep_MultiComp.mat",
    22: "F22_ExtremeHybrid.mat",
    23: "F23_Noisy.mat",
    24: "F24_Dynamic.mat",
}


# --------------------------------------------------------------------------
# MATLAB struct -> Python dict
# --------------------------------------------------------------------------
def _matobj_to_py(obj):
    if hasattr(obj, "_fieldnames"):
        return {n: _matobj_to_py(getattr(obj, n)) for n in obj._fieldnames}
    if isinstance(obj, np.void) and obj.dtype.names is not None:
        return {n: _matobj_to_py(obj[n]) for n in obj.dtype.names}
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return np.array([_matobj_to_py(x) for x in obj.ravel()],
                            dtype=object).reshape(obj.shape)
        if obj.shape == ():
            return _matobj_to_py(obj.item())
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _load_gnbg_dict(mat_path: str) -> dict:
    data = loadmat(mat_path, struct_as_record=False, squeeze_me=False)
    if "GNBG" not in data:
        raise ValueError(f"MAT-file does not contain 'GNBG' variable: {mat_path}")
    gnbg_obj = data["GNBG"]
    if isinstance(gnbg_obj, np.ndarray) and gnbg_obj.size == 1:
        gnbg_obj = gnbg_obj.reshape(-1)[0]
    G = _matobj_to_py(gnbg_obj)
    if not isinstance(G, dict):
        raise TypeError("Failed to convert GNBG struct to dict")

    keep_arrays = {
        "ComponentSigma", "Component_H", "Mu", "Omega", "lambda",
        "RotationMatrix", "Component_MinimumPosition", "OptimumPosition",
        "FEhistory", "H_Values", "SigmaPattern", "H_pattern",
    }
    for k, v in list(G.items()):
        if k in keep_arrays:
            continue
        if isinstance(v, np.ndarray) and v.size == 1:
            G[k] = v.reshape(-1)[0].item()
        elif isinstance(v, np.generic):
            G[k] = v.item()

    for k in ["Dimension", "MaxEvals", "o", "FE",
              "FirstPoint", "SecondPoint", "DynamicPeriod"]:
        if k in G and G[k] is not None and not isinstance(G[k], np.ndarray):
            try:
                G[k] = int(G[k])
            except Exception:
                pass

    if "RotationMatrix" in G and isinstance(G["RotationMatrix"], np.ndarray):
        if G["RotationMatrix"].ndim == 2:
            G["RotationMatrix"] = G["RotationMatrix"][:, :, None]
        G["RotationMatrix"] = np.asarray(G["RotationMatrix"], dtype=np.float64)

    for k in ["Component_MinimumPosition", "Mu", "Omega", "Component_H"]:
        if k in G and isinstance(G[k], np.ndarray):
            if G[k].ndim == 1:
                G[k] = G[k][None, :]
            G[k] = np.asarray(G[k], dtype=np.float64)

    for k in ["ComponentSigma", "lambda"]:
        if k in G and isinstance(G[k], np.ndarray):
            G[k] = np.asarray(G[k], dtype=np.float64).reshape(-1)

    if "FEhistory" in G and isinstance(G["FEhistory"], np.ndarray):
        G["FEhistory"] = np.array(G["FEhistory"], dtype=np.float64)

    # ---- pre-computed layouts for the vectorised fitness ---------------
    # R has shape (D, D, o); we want (o, D, D) so einsum can batch over o.
    G["_R_T"] = np.ascontiguousarray(G["RotationMatrix"].transpose(2, 0, 1))
    # Pre-cache the -H matrix we actually need in the sum
    G["_CH"] = G["Component_H"]             # alias, (o, D)
    G["_noise_level"] = float(G.get("NoiseLevel", 0) or 0)
    G["_dyn_period"] = int(G.get("DynamicPeriod", 0) or 0)
    G["_dyn_shift"] = float(G.get("DynamicShift", 0) or 0)
    G["_MinRandOptimaPos"] = G.get("MinRandOptimaPos", -np.inf)
    G["_MaxRandOptimaPos"] = G.get("MaxRandOptimaPos",  np.inf)
    G["_FirstPoint"] = int(G.get("FirstPoint", 0) or 0)
    G["_SecondPoint"] = int(G.get("SecondPoint", 0) or 0)
    return G


# --------------------------------------------------------------------------
# Per-process cache
# --------------------------------------------------------------------------
# Maps (bench_dir, fid) -> a fully-parsed base dict. Immutable fields are
# shared by reference across reps; mutable fields are deep-copied per rep.

_PROBLEM_CACHE: dict[tuple[str, int], dict] = {}

_MUTABLE_ARRAYS = ("Component_MinimumPosition", "OptimumPosition")
_MUTABLE_SCALARS = ("OptimumValue", "CurrentShift")


def _fresh_state(base: dict) -> dict:
    """Produce a fresh problem dict for a new rep, without re-parsing the .mat.

    Shares immutable fields (rotation, H, Mu, Omega, ...) by reference;
    deep-copies anything the fitness mutates (FEhistory, Component_MinimumPosition
    for F24, OptimumPosition/OptimumValue for F24, plus all scalar counters).
    """
    G = dict(base)  # shallow copy of dict
    G["FEhistory"] = np.full(int(base["MaxEvals"]), np.inf, dtype=np.float64)
    # Copy fields that F24's dynamic shift rewrites (cheap — tiny arrays)
    for k in _MUTABLE_ARRAYS:
        if k in base and isinstance(base[k], np.ndarray):
            G[k] = base[k].copy()
    # Reset per-run scalar state
    G["FE"] = 0
    G["BestFoundResult"] = float("inf")
    G["AcceptanceReachPoint"] = float("inf")
    G["BestAtFirstLine"] = float("nan")
    G["BestAtSecondLine"] = float("nan")
    # OptimumValue can be rewritten by F24 dynamic shift; restore original
    G["OptimumValue"] = float(base["OptimumValue"])
    return G


def _load_cached(fid: int, bench_dir: str) -> dict:
    key = (bench_dir, fid)
    base = _PROBLEM_CACHE.get(key)
    if base is None:
        mat_path = os.path.join(bench_dir, PROBLEM_FILES[fid])
        if not os.path.isfile(mat_path):
            raise FileNotFoundError(
                f"Benchmark file not found: {mat_path}\n"
                f"Set GNBG_III_BENCHMARK_DIR or pass benchmark_dir= explicitly."
            )
        base = _load_gnbg_dict(mat_path)
        _PROBLEM_CACHE[key] = base
    return _fresh_state(base)


# --------------------------------------------------------------------------
# Vectorised transform  (elementwise over the last axis)
# --------------------------------------------------------------------------
def _transform_batched(X, Mu, Om):
    """Vectorised equivalent of the shipped transform().

    X   : ndarray of any shape, transformed along every element.
    Mu  : (o, 2)
    Om  : (o, 4)
    The Mu/Om broadcast rule is: X's second-to-last axis is `o`, so
    Mu[:, 0].reshape(-1, 1) broadcasts into X[..., o, D].
    """
    # Mask and safe-log trick: log(|X|) where X != 0, else dummy value.
    # We only USE logX where the mask is True, so the dummy doesn't matter.
    absX = np.abs(X)
    safeX = np.where(absX > 0, absX, 1.0)
    logX = np.log(safeX)

    Mu0 = Mu[:, 0].reshape(-1, 1)
    Mu1 = Mu[:, 1].reshape(-1, 1)
    Om0 = Om[:, 0].reshape(-1, 1)
    Om1 = Om[:, 1].reshape(-1, 1)
    Om2 = Om[:, 2].reshape(-1, 1)
    Om3 = Om[:, 3].reshape(-1, 1)

    pos_val = np.exp(logX + Mu0 * (np.sin(Om0 * logX) + np.sin(Om1 * logX)))
    neg_val = -np.exp(logX + Mu1 * (np.sin(Om2 * logX) + np.sin(Om3 * logX)))

    Y = np.where(X > 0, pos_val, np.where(X < 0, neg_val, X))
    return Y


# --------------------------------------------------------------------------
# Vectorised fitness
# --------------------------------------------------------------------------
def _gnbg_fitness(X, G):
    """Batch fitness with full MaxEvals / FEhistory / F23 / F24 bookkeeping."""
    N = X.shape[0]
    MaxEvals = int(G["MaxEvals"])
    o = int(G["o"])
    D = int(G["Dimension"])

    # ------------------------ F24 dynamic shift ---------------------------
    dyn_period = G["_dyn_period"]
    dyn_shift = G["_dyn_shift"]
    if dyn_period and dyn_shift:
        fe_now = int(G["FE"])
        if fe_now > 0 and fe_now % dyn_period == 0:
            shift = dyn_shift * np.random.randn(o, D)
            new_Cmin = np.clip(
                G["Component_MinimumPosition"] + shift,
                G["_MinRandOptimaPos"],
                G["_MaxRandOptimaPos"],
            )
            G["Component_MinimumPosition"] = new_Cmin
            comp_sigma = G["ComponentSigma"]
            G["OptimumValue"] = float(np.min(comp_sigma))
            gid = int(np.argmin(comp_sigma))
            G["OptimumPosition"] = new_Cmin[gid, :].copy()
            # invalidate the R_T-cached copy? R is unchanged, so no.

    # --------------------- Vectorised physics -----------------------------
    Cmin = G["Component_MinimumPosition"]        # (o, D)
    R_T  = G["_R_T"]                             # (o, D, D); R_T[k] == R[:,:,k]
    CH   = G["_CH"]                              # (o, D)
    Mu   = G["Mu"]                               # (o, >=2)
    Om   = G["Omega"]                            # (o, >=4)
    sigma = G["ComponentSigma"]                  # (o,)
    lam   = G["lambda"]                          # (o,)

    # diff[n, k, d] = X[n, d] - Cmin[k, d]
    diff = X[:, None, :] - Cmin[None, :, :]                       # (N, o, D)
    # rotated[n, k, i] = Σ_d R_T[k, i, d] * diff[n, k, d] = (R[:,:,k] @ diff[n,k])[i]
    rotated = np.einsum("kid,nkd->nki", R_T, diff, optimize=True)  # (N, o, D)
    v = _transform_batched(rotated, Mu, Om)                        # (N, o, D)
    # inner[n, k] = Σ_d CH[k, d] * v[n, k, d]^2
    inner = np.einsum("kd,nkd->nk", CH, v * v, optimize=True)      # (N, o)
    comp = sigma[None, :] + inner ** lam[None, :]                  # (N, o)
    result = comp.min(axis=1)                                      # (N,)

    # --------------------- F23 noise (vectorised) -------------------------
    noise_level = G["_noise_level"]
    if noise_level:
        # Bit-identical to a per-jj scalar randn() loop from the same state.
        result = result + noise_level * np.random.randn(N)

    # --------------------- Per-FE bookkeeping -----------------------------
    # Kept in a Python loop to preserve exact per-FE semantics:
    # BestFoundResult, AcceptanceReachPoint, and FirstPoint/SecondPoint
    # checkpoints all depend on the exact FE value at which each write lands.
    FE          = int(G["FE"])
    best_found  = float(G["BestFoundResult"])
    acc_reach   = float(G["AcceptanceReachPoint"])
    acc_thr     = float(G["AcceptanceThreshold"])
    opt_val     = float(G["OptimumValue"])
    first_pt    = G["_FirstPoint"]
    second_pt   = G["_SecondPoint"]
    best_first  = float(G["BestAtFirstLine"])
    best_second = float(G["BestAtSecondLine"])
    FEhist_flat = G["FEhistory"].reshape(-1)

    for jj in range(N):
        if FE >= MaxEvals:
            break
        FE += 1
        fj = float(result[jj])
        FEhist_flat[FE - 1] = fj
        if fj < best_found:
            best_found = fj
        if abs(fj - opt_val) < acc_thr and acc_reach == float("inf"):
            acc_reach = float(FE)
        if FE == first_pt:
            best_first = float(FEhist_flat[:FE].min())
        if FE == second_pt:
            best_second = float(FEhist_flat[:FE].min())

    G["FE"] = FE
    G["BestFoundResult"] = best_found
    G["AcceptanceReachPoint"] = acc_reach
    G["BestAtFirstLine"] = best_first
    G["BestAtSecondLine"] = best_second

    return result, G


# --------------------------------------------------------------------------
# GNBG-I/II style problem object
# --------------------------------------------------------------------------
class GNBGProblem:
    __slots__ = ("_G", "Dimension", "MaxEvals", "OptimumValue",
                 "MinCoordinate", "MaxCoordinate", "fid")

    def __init__(self, fid: int, benchmark_dir: str | None = None):
        if fid not in PROBLEM_FILES:
            raise ValueError(f"Unknown fid={fid}; expected 1..24")
        bench_dir = _resolve_benchmark_dir(benchmark_dir)
        self._G = _load_cached(fid, bench_dir)
        self.fid = fid
        self.Dimension = int(self._G["Dimension"])
        self.MaxEvals = int(self._G["MaxEvals"])
        self.OptimumValue = float(self._G["OptimumValue"])
        self.MinCoordinate = self._G["MinCoordinate"]
        self.MaxCoordinate = self._G["MaxCoordinate"]

    @property
    def FE(self) -> int:
        return int(self._G["FE"])

    @property
    def FEhistory(self) -> np.ndarray:
        return self._G["FEhistory"].reshape(-1)

    @property
    def gnbg_dict(self) -> dict:
        return self._G

    def fitness(self, x):
        arr = np.asarray(x, dtype=np.float64)
        single = (arr.ndim == 1)
        if single:
            arr = arr.reshape(1, -1)
        elif arr.ndim != 2:
            raise ValueError(f"fitness(): expected 1-D or 2-D input, got ndim={arr.ndim}")

        remaining = self.MaxEvals - int(self._G["FE"])
        if remaining <= 0:
            out = np.full(arr.shape[0], np.nan, dtype=np.float64)
            return float(out[0]) if single else out
        if arr.shape[0] > remaining:
            arr = arr[:remaining]

        result, self._G = _gnbg_fitness(arr, self._G)
        if single:
            return float(result[0])
        return result


def load_gnbg(fid: int, benchmark_dir: str | None = None) -> GNBGProblem:
    return GNBGProblem(fid, benchmark_dir=benchmark_dir)