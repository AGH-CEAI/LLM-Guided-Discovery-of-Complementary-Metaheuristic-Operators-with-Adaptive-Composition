Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_adapt_covariance` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.665524e-02            6.323274e-02            7.626308e-02            -inf                    4.629345e+00            3.156079e-01            5.641054e-02            7.377911e-02            7.970604e-02            2.788967e-01            7.401198e-02            
1      4.402381e-02            5.067008e-02            1.136084e-01            -inf                    3.880361e+00            2.110827e-01            3.548238e-01            1.045256e-01            9.934598e-01            6.955056e-02            3.309376e+00            
2      1.599018e-01            1.519315e-01            4.626278e+00            -inf                    1.675741e+01            3.176574e+00            1.431745e+01            8.355736e-01            4.326679e+00            4.261517e+00            4.970111e-01            
3      1.330377e-01            1.387504e-01            2.993929e+00            -inf                    1.327029e+01            3.034626e+00            1.132269e+01            3.060052e+00            2.562952e+00            2.898106e+00            4.505463e+00            
4      5.132094e-01            5.251271e-01            1.289812e+00            -inf                    5.772849e+00            1.061622e+00            1.688162e+00            6.865410e-01            1.051667e+00            6.335693e-01            1.436105e+00            
5      8.994948e-05            1.298623e-04            5.356923e+01            -inf                    5.379757e+09            1.061942e+01            4.351469e+04            4.070068e-02            5.390305e+01            1.128213e+01            4.609481e+01            
6      1.544828e-02            1.613955e-02            1.019323e+02            -inf                    9.726750e+03            4.227765e+02            1.035777e+03            1.212777e+02            5.763928e+01            9.860934e+01            4.289328e+02            
7      7.777278e-02            7.825385e-02            2.126796e+00            -inf                    7.778644e+00            3.049025e+00            1.037699e+01            3.613425e-01            1.894198e+00            4.200018e+00            3.648998e+00            
8      2.628168e-01            2.567760e-01            1.678830e+00            -inf                    8.254001e+00            1.985211e+00            4.123574e+00            7.132179e-01            1.860028e+00            2.014630e+00            2.315712e+00            
9      1.293048e-01            7.021879e-02            9.116840e+00            -inf                    1.052500e+01            2.330692e+01            1.517210e+02            7.633724e+00            1.016795e+01            7.527824e+00            8.386846e+00            
10     1.259035e+01            4.826029e+01            6.676809e+00            -inf                    1.811765e+01            8.659858e+00            6.678881e+01            8.116224e+00            6.756629e+01            1.270732e+01            6.717851e+01            
11     1.776846e+01            1.486271e+02            6.327739e+00            -inf                    1.843999e+03            5.453118e+00            9.955441e+01            5.751620e+00            2.013873e+02            2.594048e+01            6.553149e+01            
12     1.375762e+01            4.585608e+01            3.751308e+00            -inf                    3.265515e+01            4.415530e+00            8.439957e+01            6.717945e+00            7.454031e+01            1.295382e+01            8.592578e+01            
13     2.749330e+00            8.258270e+00            2.175963e+00            -inf                    1.036856e+01            1.886177e+00            9.595751e+00            2.020639e+00            1.630977e+01            2.752781e+00            2.305591e+01            
14     2.719233e+00            2.661984e+00            3.339101e+00            -inf                    6.313229e+00            2.997497e+00            1.145346e+01            2.582389e+00            3.688506e+00            2.723421e+00            5.250301e+00            
15     2.666066e+00            2.881905e+00            2.777447e+00            -inf                    4.400954e+00            2.725686e+00            3.190530e+00            2.812537e+00            3.070585e+00            2.686748e+00            3.841493e+00            
16     8.012821e+01            6.739054e+01            3.933331e+02            -inf                    3.304017e+04            3.741514e+02            2.241350e+03            2.754879e+02            1.886552e+02            3.134873e+02            6.059156e+02            
17     3.286199e+01            8.197471e+03            3.036447e+01            -inf                    3.955500e+05            3.174691e+01            2.312109e+03            3.209850e+01            2.171114e+03            1.627546e+03            1.445194e+04            
18     1.627988e+01            2.283731e+01            1.694245e+01            -inf                    2.527782e+01            1.510738e+01            2.143655e+01            1.805885e+01            3.590888e+01            2.037766e+01            4.054402e+01            
19     1.556031e+01            2.865980e+01            3.268203e+01            -inf                    2.326541e+01            3.629360e+01            1.738720e+02            9.499057e+00            2.625601e+01            1.453273e+01            9.569266e+01            
20     2.207555e+01            1.800386e+01            2.365049e+01            -inf                    2.542365e+01            2.473241e+01            2.027145e+01            2.366985e+01            2.271911e+01            2.417446e+01            3.049013e+01            
21     4.372075e+00            4.569545e+00            4.538417e+00            -inf                    6.071028e+00            4.544227e+00            5.170080e+00            4.574198e+00            4.754243e+00            4.470998e+00            4.999610e+00            
22     8.629916e+00            7.506800e+00            1.108044e+01            -inf                    1.148928e+01            1.096866e+01            8.199866e+00            1.113930e+01            1.166546e+01            1.138876e+01            1.339158e+01            
23     2.340256e+01            1.998000e+01            2.700340e+01            -inf                    2.302256e+01            2.977642e+01            4.731874e+01            2.716273e+01            2.877844e+01            2.477959e+01            3.426734e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_06_idea_0.py  (error=5.641054e-02)
Task  1: original.py  (error=4.402381e-02)
Task  2: variant_01_idea_0.py  (error=1.519315e-01)
Task  3: original.py  (error=1.330377e-01)
Task  4: original.py  (error=5.132094e-01)
Task  5: original.py  (error=8.994948e-05)
Task  6: original.py  (error=1.544828e-02)
Task  7: original.py  (error=7.777278e-02)
Task  8: variant_01_idea_0.py  (error=2.567760e-01)
Task  9: variant_01_idea_0.py  (error=7.021879e-02)
Task 10: variant_02_idea_0.py  (error=6.676809e+00)
Task 11: variant_05_idea_0.py  (error=5.453118e+00)
Task 12: variant_02_idea_0.py  (error=3.751308e+00)
Task 13: variant_05_idea_0.py  (error=1.886177e+00)
Task 14: variant_07_idea_0.py  (error=2.582389e+00)
Task 15: original.py  (error=2.666066e+00)
Task 16: variant_01_idea_0.py  (error=6.739054e+01)
Task 17: variant_02_idea_0.py  (error=3.036447e+01)
Task 18: variant_05_idea_0.py  (error=1.510738e+01)
Task 19: variant_07_idea_0.py  (error=9.499057e+00)
Task 20: variant_01_idea_0.py  (error=1.800386e+01)
Task 21: original.py  (error=4.372075e+00)
Task 22: variant_01_idea_0.py  (error=7.506800e+00)
Task 23: variant_01_idea_0.py  (error=1.998000e+01)

WIN COUNTS:
  original.py: 8 wins
  variant_01_idea_0.py: 7 wins
  variant_02_idea_0.py: 3 wins
  variant_05_idea_0.py: 3 wins
  variant_07_idea_0.py: 2 wins
  variant_06_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (7 wins) ---
```python
def _adapt_covariance(self):
        """Mirrored sampling with temperature-modulated covariance adaptation."""
        if not hasattr(self, 'mirror_improvement_history'):
            self.mirror_improvement_history = []

        mirror_improvement = 0.0
        if hasattr(self, 'trial_fitness') and len(self.trial_fitness) >= 2:
            half = len(self.trial_fitness) // 2
            for i in range(half):
                if i + half < len(self.trial_fitness):
                    diff_fit = self.trial_fitness[i] - self.trial_fitness[i + half]
                    mirror_improvement += max(0.0, diff_fit)

        self.mirror_improvement_history.append(mirror_improvement)
        if len(self.mirror_improvement_history) > 10:
            self.mirror_improvement_history.pop(0)

        avg_mirror = np.mean(self.mirror_improvement_history) if self.mirror_improvement_history else 0.0
        target_temp = np.clip(1.0 + 0.3 * np.tanh(avg_mirror * 0.01), 0.3, 3.0)

        if not hasattr(self, 'exploration_temp'):
            self.exploration_temp = 1.0
        self.exploration_temp = 0.9 * self.exploration_temp + 0.1 * target_temp

        y_mean = (self.mean - self.old_mean) / self.sigma
        cc_adaptive = np.clip(self.cc * max(self.exploration_temp, 0.5), 0.01, 0.3)

        self.pc = (1.0 - cc_adaptive) * self.pc + np.sqrt(cc_adaptive * (2.0 - cc_adaptive)) * y_mean
        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        ccov_adaptive = np.clip(self.ccov * self.exploration_temp, 1e-10, 0.5)

        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_02_idea_0.py (3 wins) ---
```python
def _adapt_covariance(self):
        """Eigenvalue floor with orthogonal perturbation for escaping deceptive local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        # Force exploration along underexplored eigendirections
        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
        except np.linalg.LinAlgError:
            eigvals = np.full(self.dim, np.mean(np.diag(self.C)))
            eigvecs = np.eye(self.dim)

        # Enforce eigenvalue floor to prevent spectrum collapse
        min_eig_target = max(1e-8, np.median(eigvals) * 1e-6)
        eigvals_clipped = np.maximum(eigvals, min_eig_target)

        # Detect under-explored eigendirections (smallest eigenvalues)
        median_eig = np.median(eigvals_clipped)
        small_mask = eigvals_clipped < (median_eig * 0.1)

        if np.any(small_mask):
            # Add structured perturbation along underexplored orthogonal directions
            small_indices = np.where(small_mask)[0]
            for idx in small_indices:
                # Perturb along this eigenvector direction
                perturbation = min_eig_target * 10.0 * np.outer(eigvecs[:, idx], eigvecs[:, idx])
                self.C = self.C + perturbation

        # Also add small isotropic perturbation to prevent full collapse
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = pop_variance / (expected_var + 1e-10)

        if diversity < 0.05:
            # Severe diversity loss - force exploration
            self.C = self.C + 0.01 * np.mean(np.diag(self.C)) * np.eye(self.dim)

        # If covariance is extremely ill-conditioned, reset along random orthogonal basis
        cond = np.max(eigvals_clipped) / (np.min(eigvals_clipped) + 1e-10)
        if cond > 1e7 or np.any(np.isnan(self.C)):
            # Reset to diagonal with floor
            diag_val = max(np.mean(np.diag(self.C)), min_eig_target)
            self.C = np.eye(self.dim) * diag_val
            self.pc = np.zeros(self.dim)
            self.L = None

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_05_idea_0.py (3 wins) ---
```python
def _adapt_covariance(self):
        """Eigendecomposition-based condition number control for ill-conditioned tasks."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Eigendecomposition-based condition number control
        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.clip(eigvals, 1e-30, None)

            cond = np.max(eigvals) / (np.min(eigvals) + 1e-30)

            # Exponential rescaling of eigenvalues based on condition number
            # When cond is high, eigenvalues are too spread - rescale toward geometric mean
            if cond > 1e2:
                log_eig = np.log(eigvals + 1e-30)
                log_mean = np.mean(log_eig)
                # Exponential rescaling: push eigenvalues back toward geometric mean
                alpha = min(1.0, np.log(cond) / 20.0)
                log_eig_scaled = log_mean + (log_eig - log_mean) * (1.0 - alpha)
                eigvals = np.exp(np.clip(log_eig_scaled, -30, 30))

            # Reconstruct covariance with (potentially) rescaled eigenvalues
            self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            self.C = 0.5 * (self.C + self.C.T)

        except np.linalg.LinAlgError:
            pass

        # Standard CMA-ES update on (possibly rescaled) covariance
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        # Clip to prevent numerical explosion
        self.C = np.clip(self.C, -1e10, 1e10)
        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_06_idea_0.py (1 wins) ---
```python
def _adapt_covariance(self):
        """Orthogonal Random Exploration Injection for catastrophic stagnation escape."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Detect severe stagnation (stuck for many generations)
        stagnation_ratio = self.stagnation_counter / max(self.max_stagnation, 1)
        is_severely_stuck = stagnation_ratio > 0.5

        # Compute condition number of current covariance
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min, eig_max = np.min(eigvals), np.max(eigvals)
        cond = eig_max / max(eig_min, 1e-10)
        is_ill_conditioned = cond > 1e4

        # Build orthogonal exploration directions using QR decomposition
        num_dirs = min(self.dim, 10)
        R = np.random.randn(self.dim, num_dirs)
        Q, _ = np.linalg.qr(R)

        # Base evolution path
        cc_adapt = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        cc_adapt = np.clip(cc_adapt, 0.001, 0.3)
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Adaptive learning rate: boost when stuck or ill-conditioned
        if is_severely_stuck or is_ill_conditioned:
            ccov_boost = 5.0
            exploration_scale = 0.3
        elif stagnation_ratio > 0.2:
            ccov_boost = 2.0
            exploration_scale = 0.1
        else:
            ccov_boost = 1.0
            exploration_scale = 0.0

        ccov_adapt = min(self.ccov * ccov_boost, 0.5)

        # Standard CMA-ES update
        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

        # Inject orthogonal random exploration when severely stuck
        if exploration_scale > 0.01:
            # Compute average eigenvalue as scale reference
            avg_eig = np.mean(eigvals)
            for j in range(num_dirs):
                dir_outer = np.outer(Q[:, j], Q[:, j])
                self.C += exploration_scale * avg_eig * dir_outer

        # Force regularization on ill-conditioned matrices
        if is_ill_conditioned:
            self.C += 0.1 * np.mean(np.diag(self.C)) * np.eye(self.dim)

        self.C = self._ensure_positive_definite(self.C)
```

# --- From variant_07_idea_0.py (2 wins) ---
```python
def _adapt_covariance(self):
        """Population-aware covariance reconstruction with empirical estimation."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min, eig_max = np.min(eigvals), np.max(eigvals)
        cond = eig_max / max(eig_min, 1e-10)
        eig_spread = eig_min / (eig_max + 1e-10)

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        if not hasattr(self, 'reconstruction_count'):
            self.reconstruction_count = 0
            self.last_reconstruction_gen = -1

        generations_since_recon = self.generation - self.last_reconstruction_gen

        needs_reconstruction = (
            cond > 1e6 or 
            eig_spread < 1e-6 or 
            diversity < 0.05 or
            self.stagnation_counter > self.max_stagnation // 2 or
            (diversity < 0.2 and generations_since_recon > max(self.dim * 2, 50))
        )

        if needs_reconstruction and generations_since_recon > max(self.dim, 30):
            pop_centered = (self.population - self.mean) / self.sigma
            C_pop = np.cov(pop_centered.T) + 1e-8 * np.eye(self.dim)

            blend_weight = np.clip(0.4 + 0.6 * diversity, 0.3, 0.9)
            self.C = (1.0 - blend_weight) * self.C + blend_weight * C_pop

            noise_scale = min(0.15 * np.log10(cond + 1), 0.5) * (1.0 - diversity)
            if noise_scale > 0.01:
                noise = np.random.randn(self.dim, self.dim) * noise_scale
                noise = 0.5 * (noise + noise.T)
                np.fill_diagonal(noise, np.abs(np.diag(noise)))
                self.C += noise

            self.pc *= 0.3
            self.reconstruction_count += 1
            self.last_reconstruction_gen = self.generation
        else:
            rank_one = np.outer(self.pc, self.pc)

            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                diff = (self.population[i] - self.old_mean) / self.sigma
                rank_mu += self.weights[i] * np.outer(diff, diff)

            self.C = ((1.0 - self.ccov) * self.C + 
                      self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        self.C = self._ensure_positive_definite(self.C)
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
        """Update operator rewards using sliding window credit assignment."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity)
        reward = float(np.clip(reward, -10.0, 10.0))
        
        op = self.current_operator
        self.operator_rewards[op].append(reward)
        
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