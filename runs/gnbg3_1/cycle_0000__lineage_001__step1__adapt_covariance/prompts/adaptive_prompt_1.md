Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_adapt_covariance` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.636075e-01            5.337539e-02            6.806257e+01            -inf                    9.109618e+00            6.320314e+00            5.292891e-02            2.274841e+01            5.815463e-02            6.250180e-02            -inf                    
1      1.549302e-01            5.638518e-02            1.271924e+02            -inf                    1.345304e+01            5.088207e+00            5.088372e-02            4.353387e+01            4.428101e-02            2.858735e-01            -inf                    
2      4.121083e+00            4.958888e+00            4.891081e+02            -inf                    8.965113e+01            3.365731e+01            5.082527e+00            2.272592e+02            4.034608e+00            4.528425e-01            -inf                    
3      1.368544e+00            2.658386e+00            2.465219e+02            -inf                    5.661724e+01            2.311492e+01            1.918788e+00            1.208385e+02            2.891691e+00            2.693289e-01            -inf                    
4      9.403083e-01            1.316432e+00            5.151262e+00            -inf                    2.780968e+00            2.138145e+00            9.698604e-01            4.005622e+00            1.199919e+00            1.185821e+00            -inf                    
5      3.555988e-01            2.985930e+01            6.485374e+08            -inf                    2.167580e+06            3.102555e+04            9.255960e-01            5.637245e+07            5.212908e+01            5.396234e-02            -inf                    
6      3.781561e+02            2.365911e+02            3.168855e+03            -inf                    4.602328e+02            6.212179e+02            7.830375e+01            1.249656e+03            3.891587e+01            1.223914e+01            -inf                    
7      2.208343e+00            2.886751e+00            3.269678e+02            -inf                    4.546657e+01            1.975155e+01            2.660499e+00            1.295428e+02            3.058319e+00            1.527793e+00            -inf                    
8      1.783426e+00            1.940056e+00            4.663965e+01            -inf                    1.403115e+01            7.179565e+00            1.684457e+00            2.498076e+01            1.794772e+00            1.287129e+00            -inf                    
9      7.950909e+00            7.443909e+00            2.263137e+02            -inf                    1.684350e+01            4.423719e+01            7.591169e+00            1.041351e+02            8.943249e+00            3.094096e+00            -inf                    
10     3.579500e+01            2.064962e+01            2.455039e+02            -inf                    7.757178e+01            6.773811e+01            3.460603e+01            1.050937e+02            4.027522e+01            3.359821e+01            -inf                    
11     3.381416e+01            2.211465e+01            8.805683e+02            -inf                    2.347920e+02            1.454402e+02            3.355732e+01            3.619566e+02            1.348925e+02            7.075991e+01            -inf                    
12     2.203703e+01            1.480112e+01            1.897479e+02            -inf                    7.417043e+01            5.743889e+01            3.113302e+01            9.292683e+01            5.419929e+01            3.759564e+01            -inf                    
13     8.534861e+00            5.854455e+00            4.531026e+01            -inf                    1.950833e+01            1.982604e+01            4.833532e+00            2.728014e+01            8.748229e+00            1.019059e+01            -inf                    
14     2.664623e+00            3.296072e+00            1.448442e+01            -inf                    6.066519e+00            3.872716e+00            2.981596e+00            8.406233e+00            5.044808e+00            2.627392e+00            -inf                    
15     2.782704e+00            2.665083e+00            3.961656e+00            -inf                    3.269399e+00            3.205557e+00            2.804737e+00            3.443956e+00            3.090679e+00            2.866175e+00            -inf                    
16     3.627826e+02            4.325324e+02            1.741513e+04            -inf                    3.720896e+02            7.883180e+02            2.736884e+02            1.886027e+03            1.365525e+02            8.377296e+01            -inf                    
17     1.982770e+03            1.246179e+03            1.902278e+05            -inf                    2.451309e+04            2.006405e+04            8.579228e+02            4.936341e+04            3.623418e+03            7.951985e+03            -inf                    
18     2.423297e+01            2.091410e+01            7.729661e+01            -inf                    3.774050e+01            3.653348e+01            2.135750e+01            4.596692e+01            2.384239e+01            2.787006e+01            -inf                    
19     2.554650e+01            3.493117e+01            2.998587e+02            -inf                    8.022779e+01            5.368370e+01            1.968666e+01            1.336229e+02            6.755408e+01            1.813957e+01            -inf                    
20     2.781775e+01            2.440853e+01            5.021241e+01            -inf                    2.631625e+01            2.938122e+01            2.553852e+01            3.441832e+01            2.158385e+01            2.156542e+01            -inf                    
21     4.580732e+00            4.529195e+00            5.830578e+00            -inf                    5.238535e+00            5.050680e+00            4.514203e+00            5.440539e+00            4.946106e+00            4.608611e+00            -inf                    
22     1.159004e+01            1.066080e+01            1.827380e+01            -inf                    1.106791e+01            1.198399e+01            1.095367e+01            1.325945e+01            1.117791e+01            9.497230e+00            -inf                    
23     3.304206e+01            3.113889e+01            7.989723e+01            -inf                    4.248866e+01            3.254623e+01            2.727991e+01            5.284396e+01            2.054706e+01            1.800623e+01            -inf                    

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_06_idea_0.py  (error=5.292891e-02)
Task  1: variant_08_idea_0.py  (error=4.428101e-02)
Task  2: variant_09_idea_0.py  (error=4.528425e-01)
Task  3: variant_09_idea_0.py  (error=2.693289e-01)
Task  4: original.py  (error=9.403083e-01)
Task  5: variant_09_idea_0.py  (error=5.396234e-02)
Task  6: variant_09_idea_0.py  (error=1.223914e+01)
Task  7: variant_09_idea_0.py  (error=1.527793e+00)
Task  8: variant_09_idea_0.py  (error=1.287129e+00)
Task  9: variant_09_idea_0.py  (error=3.094096e+00)
Task 10: variant_01_idea_0.py  (error=2.064962e+01)
Task 11: variant_01_idea_0.py  (error=2.211465e+01)
Task 12: variant_01_idea_0.py  (error=1.480112e+01)
Task 13: variant_06_idea_0.py  (error=4.833532e+00)
Task 14: variant_09_idea_0.py  (error=2.627392e+00)
Task 15: variant_01_idea_0.py  (error=2.665083e+00)
Task 16: variant_09_idea_0.py  (error=8.377296e+01)
Task 17: variant_06_idea_0.py  (error=8.579228e+02)
Task 18: variant_01_idea_0.py  (error=2.091410e+01)
Task 19: variant_09_idea_0.py  (error=1.813957e+01)
Task 20: variant_09_idea_0.py  (error=2.156542e+01)
Task 21: variant_06_idea_0.py  (error=4.514203e+00)
Task 22: variant_09_idea_0.py  (error=9.497230e+00)
Task 23: variant_09_idea_0.py  (error=1.800623e+01)

WIN COUNTS:
  variant_09_idea_0.py: 13 wins
  variant_01_idea_0.py: 5 wins
  variant_06_idea_0.py: 4 wins
  variant_08_idea_0.py: 1 wins
  original.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_idea_0.py (5 wins) ---
```python
def _adapt_covariance(self):
        """
        Adapt covariance via Cholesky factor update.
        Fundamentally different from evolution-path approach - maintains C = LL^T
        structure directly without eigenvalue decomposition.
        """
        # Initialize L if needed
        if not hasattr(self, 'L') or self.L is None:
            self.L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))

        # Compute weighted mean shift in z-space (normalized coordinates)
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Rank-1 update: use the normalized mean shift directly
        z_1 = y_mean / np.maximum(np.linalg.norm(y_mean), 1e-10)

        # Rank-mu update: weighted sum of squared normalized vectors
        rank_mu = np.zeros((self.dim, self.dim))
        total_weight = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_weight += w

        # Normalize rank_mu weights
        if total_weight > 0:
            rank_mu /= total_weight

        # Effective learning rate combining rank-1 and rank-mu
        ccov_1 = 1.0 / (self.dim + 2.0)  # Rank-1 rate
        ccov_mu = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)  # Rank-mu rate

        # Update covariance directly then refactorize
        C_new = (1.0 - ccov_1 - ccov_mu) * self.C
        C_new += ccov_1 * np.outer(z_1, z_1)
        C_new += ccov_mu * rank_mu * np.sum(self.weights[:self.mu])

        # Ensure symmetry and positive definiteness
        C_new = 0.5 * (C_new + C_new.T)

        # Add minimal regularization for numerical stability
        min_eig = np.min(np.linalg.eigvalsh(C_new))
        if min_eig < 1e-10:
            C_new += (1e-8 - min_eig) * np.eye(self.dim)

        self.C = C_new

        # Update Cholesky factor using rank-1 update formula
        try:
            # Use O(d^2) rank-1 Cholesky update
            v = np.linalg.solve(self.L.T, z_1)
            alpha = ccov_1 / (1.0 + ccov_1 * np.dot(v, v))
            self.L = self.L @ (np.eye(self.dim) + alpha * np.outer(v, v))
        except np.linalg.LinAlgError:
            # Fallback: recompute Cholesky from scratch
            self.L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
```

# --- From variant_06_idea_0.py (4 wins) ---
```python
def _adapt_covariance(self):
        """
        Adapt covariance matrix with diversity-sensitive learning rate scaling.
        Detects convergence via fitness variance and eigenvalue spread, then
        aggressively escapes local optima on multi-modal tasks.
        """
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        # Diversity detection: fitness variance normalized by scale
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        # Eigenvalue spread as second convergence signal
        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)

        # Convergence detected when both signals indicate stagnation
        is_converging = (rel_var < 0.01) and (eig_spread < 1e-4)

        # Diversity-sensitive learning rate scaling
        if is_converging:
            # Up to 10x increase for escaped local optima
            scale = 10.0
        else:
            # Moderate boost proportional to diversity
            scale = 1.0 + 5.0 * min(rel_var, 0.1)

        ccov_scaled = min(self.ccov * scale, 0.5)
        cc_scaled = min(self.cc * (1.0 + scale * 0.5), 0.3)

        # Recompute pc with scaled learning rate
        self.pc = (1.0 - cc_scaled) * self.pc + np.sqrt(cc_scaled * (2.0 - cc_scaled)) * y_mean
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update with adaptive weights
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - ccov_scaled) * self.C + 
                  ccov_scaled * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_scaled * 2.0 * rank_mu))

        # Inject exploration noise when converging
        if is_converging:
            self.C += 0.05 * np.eye(self.dim)

        # Ensure positive definiteness
        self.C = 0.5 * (self.C + self.C.T)
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-10:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)
```

# --- From variant_08_idea_0.py (1 wins) ---
```python
def _adapt_covariance(self):
        """
        Adapt covariance using dual active evolution paths with exponential history weighting.
        - pc_weighted: tracks exponentially-weighted recent mean shifts (better for rugged landscapes)
        - p_cross: captures inter-step correlations (helps with non-separable rotation)
        """
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Primary evolution path (standard)
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        # Secondary weighted evolution path: exponentially weighted recent steps
        # Gives more weight to recent successful directions (alpha = 0.05)
        alpha = 0.05
        if not hasattr(self, 'pc_weighted'):
            self.pc_weighted = np.zeros(self.dim)
        self.pc_weighted = (1.0 - alpha) * self.pc_weighted + alpha * y_mean

        # Tertiary path: inter-step correlation (captures rotation in search)
        # Helps on non-separable functions where step direction changes
        beta = 0.02
        if not hasattr(self, 'p_cross'):
            self.p_cross = np.zeros(self.dim)
        if hasattr(self, 'prev_y_mean'):
            cross_contribution = np.sqrt(beta) * (y_mean - self.prev_y_mean)
            self.p_cross = (1.0 - beta) * self.p_cross + cross_contribution
        self.prev_y_mean = y_mean.copy()

        # Rank-1 updates from all three paths
        rank_one_primary = np.outer(self.pc, self.pc)
        rank_one_weighted = np.outer(self.pc_weighted, self.pc_weighted)
        rank_one_cross = np.outer(self.p_cross, self.p_cross)

        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Combine with different learning rates
        # Primary path: standard rate
        # Weighted path: 0.3 * ccov (slower, supplementary)
        # Cross path: 0.1 * ccov (minor correction for rotation)
        ccov_primary = self.ccov
        ccov_weighted = 0.3 * self.ccov
        ccov_cross = 0.1 * self.ccov

        self.C = ((1.0 - ccov_primary) * self.C + 
                  ccov_primary * rank_one_primary +
                  ccov_weighted * rank_one_weighted +
                  ccov_cross * rank_one_cross +
                  (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)

        # Ensure positive definiteness with tighter bound
        self.C = 0.5 * (self.C + self.C.T)
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-12:
            self.C += (1e-9 - min_eig) * np.eye(self.dim)

        # Clip condition number for robustness
        eigvals = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals) / np.min(eigvals)
        if cond > 1e7:
            # Dampen extreme eigenvalues
            self.C *= 0.5
```

# --- From variant_09_idea_0.py (13 wins) ---
```python
def _adapt_covariance(self):
        """
        Adapt covariance matrix with diversity-sensitive and stagnation-aware scaling.
        """
        # Track population diversity (scaled variance)
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.1, 1.0)

        # Track improvement for stagnation detection
        if not hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.improvement_ema = 1.0
        improvement = max(1e-10, self.prev_f_opt - self.f_opt)
        self.improvement_ema = 0.9 * self.improvement_ema + 0.1 * improvement
        self.prev_f_opt = self.f_opt

        stagnation = self.improvement_ema < 1e-10 * max(1.0, abs(self.f_opt))

        # Adaptive ccov: boost when diversity is low or stagnated
        base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        ccov_adaptive = base_ccov * (1.5 if stagnation else 1.0) * (1.5 if diversity < 0.3 else 1.0)
        ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 1.0)

        # Compute evolution path
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update with diversity-sensitive scaling
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Scale rank-mu: boost when diversity is low (encourage exploration)
        rank_mu_scale = 2.0 if (stagnation or diversity < 0.3) else 1.0

        # Combined update with adaptive rates
        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  ccov_adaptive * rank_mu_scale * (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)

        # Ensure positive definiteness
        self.C = 0.5 * (self.C + self.C.T)
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-10:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)
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


class CovarianceGuidedEvolutionStrategy:
    """
    A novel optimizer combining CMA-ES-style covariance adaptation
    with a particle-swarm-inspired mean tracking mechanism.
    
    Key features:
    - Simplified full covariance matrix adaptation (not diagonal-only)
    - Dual evolution paths for step-size and covariance
    - Rank-based weighted recombination
    - Automatic restart on stagnation
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        # Population size: 8 * dim (balanced for dim=30 -> NP=240)
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
        self.ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        self.damping = 1.0 + np.maximum(0.0, np.sqrt(self.mueff) - 1.0)
        
        # Restart parameters
        self.max_stagnation = 50 + self.dim * 3
        self.min_diversity = 1e-6 * (self.ub[0] - self.lb[0])
        
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
            self._adapt_covariance()
            self._check_stagnation()
            self._restart_if_needed()
        
        return self.f_opt, self.x_opt
    
    def _initialize_population(self):
        """Initialize population using Latin Hypercube sampling."""
        samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        
        # Apply LHS for better spread
        for d in range(self.dim):
            bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
            perm = np.random.permutation(self.NP)
            samples[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(self.NP)
        
        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        
        # Initialize covariance from population spread
        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        
        # Evaluate initial population
        self.fitness = self.func(self.population)
        
        # Track best solution
        best_idx = np.argmin(self.fitness)
        self.f_opt = self.fitness[best_idx]
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        
        self.stagnation_counter = 0
        self.generation = 0
    
    def _sample_trials_batch(self):
        """Sample new trial population from multivariate normal."""
        # Eigendecomposition for stable sampling
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        # Sample from N(mean, sigma^2 * C)
        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)
        self.trials = np.clip(self.trials, self.lb, self.ub)
    
    def _evaluate_batch(self):
        """Evaluate all trial candidates in single batch call."""
        self.trial_fitness = self.func(self.trials)
    
    def _update_best(self):
        """Update best solution if trial population contains improvement."""
        trial_best_idx = np.argmin(self.trial_fitness)
        if self.trial_fitness[trial_best_idx] < self.f_opt:
            self.f_opt = self.trial_fitness[trial_best_idx]
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
        """
        Adapt step-size using evolution path (CSA - Cumulation).
        """
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean
        
        self.sigma *= np.exp((np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1.0) * self.cs / self.damping)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _adapt_covariance(self):
        """
        Adapt covariance matrix using rank-1 and rank-mu updates.
        """
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
        
        # Ensure positive definiteness
        self.C = 0.5 * (self.C + self.C.T)
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-10:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)
    
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
            
            # Keep best individual, reinitialize rest
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]
            
            self._initialize_population()
            
            # Inject elite into new population
            self.population[0] = elite
            self.fitness[0] = elite_fit
            self.f_opt = elite_fit
            self.x_opt = elite.copy()
            self.f_opt_prev = elite_fit

```