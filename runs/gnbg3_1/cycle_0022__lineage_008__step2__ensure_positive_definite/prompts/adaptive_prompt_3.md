Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_ensure_positive_definite` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.957150e-02            5.629712e-02            5.177750e+00            5.930329e-02            5.700972e-02            5.740701e-02            5.869522e-02            5.642593e-02            5.789019e-02            5.850862e-02            5.718853e-02            
1      4.521667e-02            4.484182e-02            5.296896e+00            4.545728e-02            4.671790e-02            4.417879e-02            4.489021e-02            4.590179e-02            4.625983e-02            4.563123e-02            4.449514e-02            
2      1.563511e-01            1.497446e-01            7.243806e+00            4.961205e+00            1.881483e-01            1.695528e-01            3.442688e+00            2.914125e+00            2.830418e+00            3.861641e+00            1.502065e-01            
3      1.308743e-01            1.357399e-01            1.238467e+01            2.359261e+00            2.082902e+00            1.390239e-01            2.785240e+00            2.830145e+00            2.815766e+00            1.948593e+00            1.338799e-01            
4      5.290559e-01            5.503959e-01            2.489941e+00            5.202404e-01            5.254844e-01            5.204068e-01            5.397789e-01            6.841394e-01            5.174089e-01            5.173159e-01            5.100052e-01            
5      8.374466e-05            7.305885e-05            9.776934e+01            6.403821e-04            8.452874e-05            1.310835e-04            1.809289e-01            1.828293e-01            5.529618e-04            2.017058e-04            1.064128e-04            
6      1.468631e-02            1.525306e-02            2.327831e+03            2.478892e+01            8.259477e+00            2.030194e-02            2.940625e+01            2.946667e+01            2.937076e+01            2.885850e+01            1.530180e-02            
7      8.010645e-02            8.211033e-02            9.240739e+00            1.252336e-01            8.142553e-02            8.044835e-02            6.368069e-01            2.764004e+00            1.245676e-01            9.344154e-02            7.873612e-02            
8      2.498045e-01            2.654417e-01            5.683768e+00            1.783633e+00            2.938693e-01            2.603951e-01            1.753080e+00            1.995416e+00            2.026622e+00            1.314760e+00            2.623427e-01            
9      6.947953e-02            6.976119e-02            1.238368e+02            7.483636e+00            7.660347e+00            6.845041e-02            7.726754e+00            8.125277e+00            7.873218e+00            7.519414e+00            2.800367e-01            
10     1.074929e+01            8.237656e+00            5.089793e+01            1.108045e+01            7.372998e+00            8.894270e+00            7.538769e+00            9.238476e+00            8.522079e+00            7.438027e+00            8.782789e+00            
11     5.671458e+00            6.333666e+00            1.279241e+02            8.736174e+00            7.043215e+00            5.223067e+00            1.055992e+01            1.117937e+01            1.014135e+01            7.719641e+00            8.208306e+00            
12     6.976409e+00            8.777250e+00            3.870039e+01            1.040368e+01            5.634222e+00            1.098920e+01            7.732670e+00            8.812330e+00            1.057124e+01            8.989884e+00            8.789014e+00            
13     1.788624e+00            1.938208e+00            1.305916e+01            1.945755e+00            2.524328e+00            2.010564e+00            2.240309e+00            1.815508e+00            1.978492e+00            2.131943e+00            1.842693e+00            
14     2.702223e+00            2.604559e+00            4.998678e+00            2.773976e+00            2.673107e+00            2.739160e+00            2.855707e+00            2.904869e+00            2.827580e+00            2.781204e+00            2.776649e+00            
15     2.531735e+00            2.662371e+00            3.273055e+00            2.554604e+00            2.486639e+00            2.656724e+00            2.264128e+00            2.461622e+00            2.557659e+00            2.713768e+00            2.506442e+00            
16     7.043052e+01            7.124900e+01            2.684957e+03            8.017347e+01            9.562967e+01            8.514958e+01            1.455988e+02            8.631533e+01            9.930312e+01            8.004195e+01            6.211382e+01            
17     4.016573e+01            3.641342e+01            3.770534e+03            3.748330e+01            3.589330e+01            3.020383e+01            3.152024e+01            3.071392e+01            3.262079e+01            3.186681e+01            3.810674e+01            
18     1.037510e+01            1.143712e+01            2.806670e+01            1.069171e+01            1.344209e+01            1.210877e+01            1.239940e+01            1.263856e+01            1.247376e+01            1.028831e+01            1.266814e+01            
19     1.306437e+01            1.582809e+01            1.268009e+02            1.620813e+01            1.380573e+01            9.305721e+00            1.226237e+01            1.637002e+01            1.326937e+01            1.055449e+01            1.265567e+01            
20     1.830514e+01            1.913791e+01            2.910874e+01            1.749700e+01            1.850423e+01            1.822197e+01            1.827496e+01            1.864560e+01            1.856904e+01            1.838396e+01            1.885486e+01            
21     4.374806e+00            4.364156e+00            4.836029e+00            4.460685e+00            4.479939e+00            4.413609e+00            4.420383e+00            4.422665e+00            4.558532e+00            4.470348e+00            4.267406e+00            
22     4.685324e+00            7.281489e+00            1.295915e+01            5.651039e+00            8.399497e+00            6.451345e+00            6.146350e+00            3.675367e+00            7.852534e+00            8.446352e+00            7.479479e+00            
23     1.979292e+01            2.061533e+01            5.074922e+01            1.593555e+01            1.994550e+01            1.657222e+01            1.817968e+01            2.254737e+01            1.844042e+01            1.643879e+01            1.879373e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_01_idea_0.py  (error=5.629712e-02)
Task  1: variant_05_idea_0.py  (error=4.417879e-02)
Task  2: variant_01_idea_0.py  (error=1.497446e-01)
Task  3: original.py  (error=1.308743e-01)
Task  4: variant_10_idea_0.py  (error=5.100052e-01)
Task  5: variant_01_idea_0.py  (error=7.305885e-05)
Task  6: original.py  (error=1.468631e-02)
Task  7: variant_10_idea_0.py  (error=7.873612e-02)
Task  8: original.py  (error=2.498045e-01)
Task  9: variant_05_idea_0.py  (error=6.845041e-02)
Task 10: variant_04_idea_0.py  (error=7.372998e+00)
Task 11: variant_05_idea_0.py  (error=5.223067e+00)
Task 12: variant_04_idea_0.py  (error=5.634222e+00)
Task 13: original.py  (error=1.788624e+00)
Task 14: variant_01_idea_0.py  (error=2.604559e+00)
Task 15: variant_06_idea_0.py  (error=2.264128e+00)
Task 16: variant_10_idea_0.py  (error=6.211382e+01)
Task 17: variant_05_idea_0.py  (error=3.020383e+01)
Task 18: variant_09_idea_0.py  (error=1.028831e+01)
Task 19: variant_05_idea_0.py  (error=9.305721e+00)
Task 20: variant_03_idea_0.py  (error=1.749700e+01)
Task 21: variant_10_idea_0.py  (error=4.267406e+00)
Task 22: variant_07_idea_0.py  (error=3.675367e+00)
Task 23: variant_03_idea_0.py  (error=1.593555e+01)

WIN COUNTS:
  variant_05_idea_0.py: 5 wins
  variant_01_idea_0.py: 4 wins
  original.py: 4 wins
  variant_10_idea_0.py: 4 wins
  variant_04_idea_0.py: 2 wins
  variant_03_idea_0.py: 2 wins
  variant_06_idea_0.py: 1 wins
  variant_09_idea_0.py: 1 wins
  variant_07_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (4 wins) ---
```python
def _ensure_positive_definite(self, C):
        """Ensure matrix is symmetric and positive definite via eigenvalue clipping."""
        # Symmetrize the matrix
        C = 0.5 * (C + C.T)

        # Perform eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(C)

        # Clip all eigenvalues to minimum threshold
        eigvals = np.maximum(eigvals, 1e-10)

        # Reconstruct the matrix
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return C
```

# --- From variant_03_idea_0.py (2 wins) ---
```python
def _ensure_positive_definite(self, C):
        """Ensure matrix is symmetric and positive definite via eigendecomposition."""
        C = 0.5 * (C + C.T)

        # Eigendecomposition - directly handle eigenvalue issues
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        # Clip eigenvalues to ensure positive definiteness (key difference from diagonal shift)
        eig_min = np.min(eigenvalues)
        if eig_min < 1e-12:
            eigenvalues = np.clip(eigenvalues, 1e-12, None)
            C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Enforce maximum condition number to prevent ill-conditioning
        eig_max = np.max(eigenvalues)
        cond_threshold = 1e6
        if eig_max / max(eig_min, 1e-12) > cond_threshold:
            eigenvalues = np.clip(eigenvalues, eig_max / cond_threshold, None)
            C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        return C
```

# --- From variant_04_idea_0.py (2 wins) ---
```python
def _ensure_positive_definite(self, C):
        """Ensure matrix is symmetric and positive definite via adaptive eigenvalue flooring."""
        C = 0.5 * (C + C.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        min_eig = np.min(eigvals)
        max_eig = np.max(eigvals)
        if max_eig == 0:
            max_eig = 1.0
        # Adaptive threshold based on max eigenvalue and dimension
        eig_thresh = max(1e-10, 1e-6 * max_eig / self.dim)
        if min_eig < eig_thresh:
            eigvals = np.maximum(eigvals, eig_thresh)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return C
```

# --- From variant_05_idea_0.py (5 wins) ---
```python
def _ensure_positive_definite(self, C):
        """Ensure matrix is symmetric and positive definite using eigendecomposition with adaptive flooring."""
        C = 0.5 * (C + C.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        min_eig = np.min(eigvals)
        max_eig = np.max(eigvals)

        if min_eig < 1e-10:
            # Compute condition number to determine adaptive floor
            cond = max_eig / max(min_eig, 1e-30)

            # Adaptive floor: larger correction for ill-conditioned matrices
            # This helps the hardest tasks where covariance becomes elongated
            if cond > 1e6:
                floor = max(min_eig, max_eig * 1e-6)
            elif cond > 1e4:
                floor = max(min_eig, max_eig * 1e-5)
            elif cond > 1e2:
                floor = max(min_eig, max_eig * 1e-4)
            else:
                floor = max(min_eig, 1e-10)

            eigvals = np.maximum(eigvals, floor)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return C
```

# --- From variant_06_idea_0.py (1 wins) ---
```python
def _ensure_positive_definite(self, C):
        """Ensure symmetric positive definite with controlled condition number."""
        C = 0.5 * (C + C.T)

        # Full eigendecomposition for direct eigenvalue control
        eigvals, eigvecs = np.linalg.eigh(C)

        # Enforce minimum eigenvalue threshold
        eigvals = np.maximum(eigvals, 1e-10)

        # Control condition number via logarithmic rescaling
        cond = eigvals[-1] / eigvals[0]
        max_cond = 1e6
        if cond > max_cond:
            log_eigvals = np.log(eigvals)
            log_min = log_eigvals[0]
            log_max = log_eigvals[-1]
            target_range = np.log(max_cond * eigvals[0]) - log_min
            current_range = log_max - log_min
            if current_range > 1e-10:
                scale = target_range / current_range
                log_eigvals[1:] = log_min + scale * (log_eigvals[1:] - log_min)
            eigvals = np.exp(np.clip(log_eigvals, -50, 50))

        # Reconstruct with controlled spectrum
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Final safety clip
        C = 0.5 * (C + C.T)
        return C
```

# --- From variant_07_idea_0.py (1 wins) ---
```python
def _ensure_positive_definite(self, C):
        """Ensure matrix is symmetric and positive definite using adaptive eigenvalue rescaling."""
        C = 0.5 * (C + C.T)

        # Check for pathological values
        if np.any(np.isnan(C)) or np.any(np.isinf(C)):
            return np.eye(self.dim) * 1e-6

        try:
            eigvals, eigvecs = np.linalg.eigh(C)

            # Handle numerical issues
            eigvals = np.nan_to_num(eigvals, nan=1e-10, posinf=1e10, neginf=-1e10)

            eig_min = np.min(eigvals)
            eig_max = np.max(eigvals)

            # Condition number check
            cond = eig_max / max(abs(eig_min), 1e-30)

            if cond > 1e6 or eig_min < 1e-10:
                # Adaptive rescaling: reduce large eigenvalues, lift small ones
                target_min = max(1e-10, eig_min)

                if cond > 1e6:
                    # Aggressive rescaling for extreme ill-conditioning (targets worst tasks)
                    log_scale = np.log1p(eigvals - eig_min + 1e-10)
                    max_log = np.max(log_scale) + 1e-10
                    eigvals = target_min * np.exp(log_scale / max_log * np.log(1e6))
                else:
                    # Gentle rescaling
                    eigvals = np.maximum(eigvals, target_min)

                # Reconstruct with rescaled eigenvalues
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Final safety: ensure strict positive definiteness
            min_eig = np.min(np.linalg.eigvalsh(C))
            if min_eig < 1e-12:
                # Use mean eigenvalue for more robust offset
                mean_eig = np.mean(np.abs(eigvals))
                offset = max(1e-10, 1e-8 * mean_eig - min_eig)
                C = C + offset * np.eye(self.dim)

            return C

        except np.linalg.LinAlgError:
            # Fallback to diagonal loading with adaptive strength
            diag_C = np.diag(C)
            diag_C = np.maximum(np.abs(diag_C), 1e-10)
            return np.diag(diag_C)
```

# --- From variant_09_idea_0.py (1 wins) ---
```python
def _ensure_positive_definite(self, C):
        """Ensure matrix is symmetric and positive definite via eigenvalue clipping and conditioning."""
        C = 0.5 * (C + C.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        min_eig = np.min(eigvals)
        max_eig = np.max(eigvals)
        cond = max_eig / max(min_eig, 1e-15)
        dim_scale = max(1.0, self.dim / 10.0)
        eig_thresh = 1e-10 * dim_scale
        max_cond = 1e6 * dim_scale
        if cond > max_cond:
            target_min = max_eig / max_cond
            eigvals = np.clip(eigvals, target_min, None)
        elif min_eig < eig_thresh:
            eigvals = np.clip(eigvals, eig_thresh, None)
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        C = 0.5 * (C + C.T)
        return C
```

# --- From variant_10_idea_0.py (4 wins) ---
```python
def _ensure_positive_definite(self, C):
        """Ensure matrix is symmetric and positive definite via eigenvalue clipping with trace preservation."""
        # Symmetrize first
        C = 0.5 * (C + C.T)

        # Eigendecomposition for direct eigenvalue control
        eigvals, eigvecs = np.linalg.eigh(C)

        # Clip eigenvalues: ensure minimum and cap maximum for conditioning
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)

        # Minimum eigenvalue threshold (adaptive based on dimension)
        min_thresh = max(1e-12, abs(eig_min) * 0.1 + 1e-12)
        eigvals_clipped = np.clip(eigvals, min_thresh, None)

        # Maximum eigenvalue cap to prevent runaway conditioning
        max_thresh = max(eigvals_clipped) * 1e8
        eigvals_clipped = np.clip(eigvals_clipped, None, max_thresh)

        # Preserve trace (total variance) for numerical stability
        original_trace = np.sum(eigvals)
        clipped_trace = np.sum(eigvals_clipped)
        if clipped_trace > 0 and abs(original_trace - clipped_trace) > 1e-10:
            eigvals_clipped *= (original_trace / clipped_trace)

        # Reconstruct matrix
        C = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

        return C
```

Requirements:
- Respond with the COMPLETE class code inside a single ```python``` block
- Include `import numpy as np` at the top
- Use an adaptive mechanism (Thompson Sampling, Multi-Armed Bandit, or sliding window credit assignment) to learn which operator works best during a run
- You may add new attributes in __init__ and new private helper methods
- Do NOT change signatures of existing public methods (__call__, crossover, mutation, selection, etc.)
- The class MUST be callable as: `obj(func, stopping_condition)` → (f_opt, x_opt)
- Handle ALL edge cases: no -inf, no crashes, no NaN, clip to bounds
- Use integer-index-based operator tracking (not id()-based)
- Cast all reward/fitness values to float scalars before storing

Original algorithm:
```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 5 covariance adaptation strategies (original + 4 variants)
    - Thompson Sampling for operator selection
    - Sliding window credit assignment
    - Automatic restart on stagnation
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(240, max(80, 8 * dim))
        self.bounds = (-100.0, 100.0)
        self.lb = np.full(dim, -100.0)
        self.ub = np.full(dim, 100.0)
        
        # CMA-ES inspired parameters
        self.mu = self.NP // 4
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        
        # Learning rates
        self.cs = (self.mueff + 2.0) / (self.dim + self.mueff + 5.0)
        self.cc = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        self.ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        self.damping = 1.0 + np.maximum(0.0, np.sqrt(self.mueff) - 1.0)
        
        # Restart parameters
        self.max_stagnation = 50 + self.dim * 3
        self.min_diversity = 1e-6 * (self.ub[0] - self.lb[0])
        
        # Adaptive operator selection parameters
        self.num_operators = 5
        self.operator_names = ['original', 'variant_01', 'variant_06', 'variant_08', 'variant_09']
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_operators)
        self.beta = np.ones(self.num_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 10
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators)
        self.selection_counts = np.zeros(self.num_operators)
        
        # Runtime state
        self.generation = 0
        self.current_operator = 0
        self.L = None
    
    def _clip_to_bounds(self, x):
        """Clip solution to bounds."""
        return np.clip(x, self.lb, self.ub)
    
    def _ensure_positive_definite(self, C):
        """Ensure matrix is symmetric and positive definite."""
        C = 0.5 * (C + C.T)
        min_eig = np.min(np.linalg.eigvalsh(C))
        if min_eig < 1e-10:
            C += (1e-7 - min_eig) * np.eye(self.dim)
        return C
    
    def __call__(self, func, stopping_condition):
        self.func = func
        self._initialize_population()
        
        while not stopping_condition():
            self._sample_trials_batch()
            
            if stopping_condition():
                break
                
            self._evaluate_batch()
            
            if len(self.trial_fitness) < len(self.trials):
                self.trials = self.trials[:len(self.trial_fitness)]
                
            if len(self.trial_fitness) < self.NP:
                break
                
            self._update_best()
            
            if stopping_condition():
                break
                
            self._select_survivors_batch()
            self._adapt_step_size()
            
            # Select and apply covariance adaptation operator
            self._select_operator_thompson()
            self._adapt_covariance()
            
            # Update operator rewards based on improvement
            self._update_operator_rewards()
            
            self._check_stagnation()
            self._restart_if_needed()
        
        return self.f_opt, self.x_opt
    
    def _initialize_population(self):
        """Initialize population using Latin Hypercube sampling."""
        samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        
        for d in range(self.dim):
            bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
            perm = np.random.permutation(self.NP)
            samples[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(self.NP)
        
        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        
        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        
        self.fitness = self.func(self.population)
        
        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        
        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0
        
        self._reset_operator_state()
    
    def _reset_operator_state(self):
        """Reset all operator-specific state variables."""
        self.L = None
        if hasattr(self, 'pc_weighted'):
            self.pc_weighted = np.zeros(self.dim)
        if hasattr(self, 'p_cross'):
            self.p_cross = np.zeros(self.dim)
        if hasattr(self, 'prev_y_mean'):
            delattr(self, 'prev_y_mean')
        if hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.improvement_ema = 1.0
    
    def _sample_trials_batch(self):
        """Sample new trial population using Cholesky decomposition."""
        # Ensure positive definiteness before Cholesky
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        # Compute Cholesky factor L where C = L @ L.T
        # More efficient than eigendecomposition for sampling
        try:
            L = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            # Fallback: use diagonal approximation
            diag_C = np.diag(self.C)
            diag_C = np.maximum(diag_C, 1e-10)
            L = np.diag(np.sqrt(diag_C))

        # Sample from standard normal and transform
        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)

        # Clip to bounds
        self.trials = self._clip_to_bounds(self.trials)
    
    def _evaluate_batch(self):
        """Evaluate all trial candidates in single batch call."""
        self.trial_fitness = self.func(self.trials)
    
    def _update_best(self):
        """Update best solution if trial population contains improvement."""
        trial_best_idx = np.argmin(self.trial_fitness)
        trial_best_fit = float(np.asarray(self.trial_fitness[trial_best_idx]).flatten()[0])
        if trial_best_fit < self.f_opt:
            self.f_opt = trial_best_fit
            self.x_opt = self.trials[trial_best_idx].copy()
    
    def _select_survivors_batch(self):
        """Select survivors via elitist (mu, lambda)-selection."""
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])
        
        sorted_indices = np.argsort(combined_fit)
        self.population = combined_pop[sorted_indices[:self.NP]]
        self.fitness = combined_fit[sorted_indices[:self.NP]]
        
        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
    
    def _adapt_step_size(self):
        """Adapt step-size using momentum-enhanced improvement tracking with diversity-based damping."""
        recent_improved = 0
        recent_total = 0
        total_improvement = 0.0

        for i in range(self.NP):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                if self.trial_fitness[i] < self.fitness[i]:
                    recent_improved += 1
                    diff = self.fitness[i] - self.trial_fitness[i]
                    total_improvement += max(diff, 0.0)
                recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = recent_improved / recent_total
        success_rate = np.clip(success_rate, 0.0, 1.0)

        avg_improvement = total_improvement / max(recent_improved, 1)
        improvement_magnitude = np.log1p(max(avg_improvement, 1e-15))

        if not hasattr(self, 'improvement_ema'):
            self.improvement_ema = improvement_magnitude
        self.improvement_ema = 0.7 * self.improvement_ema + 0.3 * improvement_magnitude

        target_rate = 0.25
        adaptation = (success_rate - target_rate) / target_rate
        adaptation += 0.3 * np.tanh(self.improvement_ema - 1.0)
        adaptation = np.clip(adaptation, -0.8, 0.8)

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        diversity_boost = 1.0 + 2.0 * (1.0 - diversity)
        diversity_boost = np.clip(diversity_boost, 0.5, 3.0)

        damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim)) / diversity_boost
        damping_adaptive = np.clip(damping_adaptive, 0.05, 200.0)

        self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _update_operator_rewards(self):
        """Credit assignment via exponentially-weighted cumulative improvement with stagnation awareness."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)

        # Initialize cumulative tracking if needed
        if not hasattr(self, 'operator_cumulative_reward'):
            self.operator_cumulative_reward = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum'):
            self.operator_decay_sum = np.zeros(self.num_operators)

        # Apply exponential decay to accumulated reward (recent rewards weighted more)
        decay = 0.95
        self.operator_cumulative_reward *= decay
        self.operator_decay_sum *= decay

        # Log-scaled improvement reward (more sensitive to small improvements)
        if improvement > 0:
            reward = float(np.log1p(improvement * 1e10) / 10.0)
        else:
            reward = 0.0

        # Explicit stagnation detection
        is_stagnant = self.stagnation_counter > self.max_stagnation // 2

        # Diversity metric
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        # Exploration bonus when stagnant and diversity is low
        if is_stagnant and diversity < 0.1:
            reward *= 2.0

        # Update cumulative weighted reward
        self.operator_cumulative_reward[self.current_operator] += reward
        self.operator_decay_sum[self.current_operator] += 1.0

        # Normalize by decay sum to get comparable reward values
        norm = max(self.operator_decay_sum[self.current_operator], 1.0)
        normalized_reward = self.operator_cumulative_reward[self.current_operator] / norm
        normalized_reward = float(np.clip(normalized_reward, -10.0, 10.0))

        op = self.current_operator
        self.operator_rewards[op].append(normalized_reward)

        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)

        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            var_reward = float(np.var(self.operator_rewards[op]))
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.beta[op] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = 1.0 + sum_reward
            else:
                self.beta[op] = 1.0 - sum_reward
    
    def _adapt_covariance(self):
        """Dispatch to the selected covariance adaptation strategy."""
        if self.current_operator == 0:
            self._adapt_covariance_original()
        elif self.current_operator == 1:
            self._adapt_covariance_variant_01()
        elif self.current_operator == 2:
            self._adapt_covariance_variant_06()
        elif self.current_operator == 3:
            self._adapt_covariance_variant_08()
        else:
            self._adapt_covariance_variant_09()
    
    def _adapt_covariance_original(self):
        """Original covariance adaptation (baseline)."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_01(self):
        """Archive-guided covariance perturbation for escaping deceptive local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Initialize archive for tracking distinct best solutions
        if not hasattr(self, 'archive_positions'):
            self.archive_positions = []
            self.archive_fitness = []
            self.archive_generations = []

        # Add current best to archive if distinct enough
        min_dist_threshold = 1e-3 * (self.ub[0] - self.lb[0])
        is_distinct = True
        for arch_pos in self.archive_positions:
            dist = np.linalg.norm(self.x_opt - arch_pos)
            if dist < min_dist_threshold:
                is_distinct = False
                break

        if is_distinct and len(self.archive_positions) < 10:
            self.archive_positions.append(self.x_opt.copy())
            self.archive_fitness.append(self.f_opt)
            self.archive_generations.append(self.generation)
        elif is_distinct and len(self.archive_positions) >= 10:
            # Replace worst entry
            worst_idx = np.argmax(self.archive_fitness)
            self.archive_positions[worst_idx] = self.x_opt.copy()
            self.archive_fitness[worst_idx] = self.f_opt
            self.archive_generations[worst_idx] = self.generation

        # Compute diversity and condition metrics
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        eig_spread = eig_min / (eig_max + 1e-10)
        cond = eig_max / (max(eig_min, 1e-10))

        # Detect trapping: low variance OR ill-conditioned OR collapsed spectrum
        is_trapped = (rel_var < 1e-3) or (cond > 1e5) or (eig_spread < 1e-5)

        # Compute archive-based escape direction
        escape_perturb = np.zeros((self.dim, self.dim))
        if is_trapped and len(self.archive_positions) >= 3:
            # Direction from mean toward archive centroid, weighted by diversity
            centroid = np.mean(self.archive_positions, axis=0)
            toward_centroid = centroid - self.mean
            norm_toward = np.linalg.norm(toward_centroid)
            if norm_toward > 1e-10:
                toward_centroid /= norm_toward
                # Also consider directions to individual archive members
                for arch_pos in self.archive_positions:
                    dir_to_arch = arch_pos - self.mean
                    norm_dir = np.linalg.norm(dir_to_arch)
                    if norm_dir > 1e-10:
                        dir_to_arch /= norm_dir
                        # Perturb along diverse directions
                        escape_perturb += 0.3 * np.outer(dir_to_arch, dir_to_arch)
            # Add global exploration along principal axes
            escape_perturb += 0.1 * np.eye(self.dim)

        # Adaptive learning rates with exploration boost when trapped
        if is_trapped:
            ccov_scale = 0.8
            cc_scale = 0.8
        else:
            ccov_scale = min(1.0, 0.2 + 2.0 * rel_var)
            cc_scale = 1.0

        ccov_1 = ccov_scale * 1.0 / (self.dim + 2.0)
        ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        cc_adapt = cc_scale * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)

        # Evolution path update
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Combine updates with archive-based perturbation
        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
                  ccov_1 * rank_one +
                  ccov_mu * rank_mu)

        # Inject escape perturbation when trapped
        if is_trapped and len(self.archive_positions) >= 3:
            self.C = self.C + escape_perturb

        self.C = self._ensure_positive_definite(self.C)

        # Full restart on extreme ill-conditioning
        if cond > 1e8 or eig_spread < 1e-8:
            self.C = np.eye(self.dim) * np.mean(eigvals)
            self.pc = np.zeros(self.dim)
            self.L = None
    
    def _adapt_covariance_variant_06(self):
        """Diversity-sensitive learning rate scaling."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)
        
        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
        
        is_converging = (rel_var < 0.01) and (eig_spread < 1e-4)
        
        if is_converging:
            scale = 10.0
        else:
            scale = 1.0 + 5.0 * min(rel_var, 0.1)
        
        ccov_scaled = min(self.ccov * scale, 0.5)
        cc_scaled = min(self.cc * (1.0 + scale * 0.5), 0.3)
        
        self.pc = (1.0 - cc_scaled) * self.pc + np.sqrt(cc_scaled * (2.0 - cc_scaled)) * y_mean
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - ccov_scaled) * self.C + 
                  ccov_scaled * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_scaled * 2.0 * rank_mu))
        
        if is_converging:
            self.C += 0.05 * np.eye(self.dim)
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_08(self):
        """Active CMA-ES: Reduce learning rate along unfavorable eigendirections."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Evolution path update (unchanged)
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update with positive/negative decomposition
        rank_mu_pos = np.zeros((self.dim, self.dim))
        rank_mu_neg = np.zeros((self.dim, self.dim))

        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            weight = self.weights[i]
            outer = np.outer(diff, diff)

            if weight > 0:
                rank_mu_pos += weight * outer
            else:
                rank_mu_neg += abs(weight) * outer

        # Normalize by sum of positive and negative weights separately
        pos_sum = max(np.sum(self.weights[:self.mu][self.weights[:self.mu] > 0]), 1e-10)
        neg_sum = max(np.sum(np.abs(self.weights[:self.mu][self.weights[:self.mu] < 0])), 1e-10)

        rank_mu_pos /= pos_sum
        rank_mu_neg /= neg_sum

        # Combine positive and negative updates (active CMA-ES core)
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * rank_one +
                  (1.0 - 1.0 / self.mueff) * self.ccov * rank_mu_pos -
                  self.ccov * 0.4 * rank_mu_neg)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_09(self):
        """Exploration temperature with fitness gradient tracking for escaping local optima."""
        if not hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.exploration_temp = 1.0

        delta_f_opt = self.prev_f_opt - self.f_opt
        self.prev_f_opt = self.f_opt

        fitness_gradient = max(abs(delta_f_opt), 1e-15)
        temp_target = np.clip(1.0 / (1.0 + np.log1p(fitness_gradient * 1e5)), 0.05, 5.0)
        self.exploration_temp = 0.95 * self.exploration_temp + 0.05 * temp_target

        base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        ccov_adaptive = base_ccov * self.exploration_temp
        ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 0.5)

        cc_adaptive = self.cc * (1.0 / max(self.exploration_temp, 0.5))
        cc_adaptive = np.clip(cc_adaptive, 0.01, 0.3)

        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - cc_adaptive) * self.pc + np.sqrt(cc_adaptive * (2.0 - cc_adaptive)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        total_w = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_w += abs(w)

        if total_w > 0:
            rank_mu /= total_w

        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)
    
    def _compute_diversity(self):
        """Compute population diversity as mean per-dimension std."""
        return np.mean(np.std(self.population, axis=0))
    
    def _check_stagnation(self):
        """Track stagnation counter for restart logic."""
        if self.f_opt < self.f_opt_prev - 1e-12:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        self.f_opt_prev = self.f_opt
    
    def _restart_if_needed(self):
        """Restart population if stagnated or diversity lost."""
        diversity = self._compute_diversity()
        
        if (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            np.any(np.isnan(self.C))):
            
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]
            
            self._initialize_population()
            
            self.population[0] = elite
            self.fitness[0] = elite_fit
            self.f_opt = float(np.asarray(elite_fit).flatten()[0])
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0

```