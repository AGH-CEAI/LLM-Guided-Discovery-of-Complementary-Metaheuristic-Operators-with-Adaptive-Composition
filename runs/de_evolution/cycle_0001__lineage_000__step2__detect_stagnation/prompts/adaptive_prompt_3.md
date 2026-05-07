Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_detect_stagnation` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.081223e-07            1.769004e-07            1.241221e-07            8.521850e-02            1.259219e-07            1.107464e-07            4.160202e-07            5.178081e-07            4.370177e-06            1.092424e-07            2.050521e-07            
1      2.389856e-07            3.179284e-07            2.851068e-07            2.992791e+00            2.838878e-07            2.480263e-07            5.140078e-07            1.037683e-06            4.379766e-06            2.270481e-07            2.401573e-06            
2      2.355990e-05            3.701786e-05            2.660463e-05            3.530017e+00            2.721729e-05            2.331118e-05            2.651350e-05            3.567048e+02            3.317240e-05            2.362731e-05            4.325475e+02            
3      3.566622e-04            4.075177e-04            3.670340e-04            2.593454e+00            3.744210e-04            3.291411e-04            1.989581e+00            1.883065e+02            4.041251e-04            3.421770e-04            2.392057e+02            
4      2.433718e-02            2.617659e-02            2.550368e-02            1.385373e+00            2.576253e-02            2.445371e-02            2.493158e-02            3.518591e-02            3.702322e-02            2.459491e-02            2.756102e-02            
5      1.000000e-08            1.000000e-08            7.733880e+02            1.390402e+01            1.619272e+02            1.000000e-08            4.677521e-08            2.699845e+08            6.446114e+02            2.000000e-08            4.503202e+08            
6      4.043598e-07            4.337874e-07            1.535282e+02            5.195870e+02            5.736959e-07            4.080699e-07            4.589491e-06            2.234182e+03            1.070465e+03            3.789524e+02            4.964276e+03            
7      1.344123e-05            1.857318e-05            1.476088e-05            3.365288e+00            1.552235e-05            1.327038e-05            1.501513e-05            1.957698e+02            3.314659e-05            1.362083e-05            2.692902e+02            
8      1.748844e-03            1.888416e-03            1.833745e-03            1.574252e+00            1.850319e-03            1.701289e-03            1.896416e-03            3.602934e-03            2.083911e-03            1.703651e-03            2.232731e-03            
9      5.047887e-04            6.004841e-04            5.691946e-04            9.604188e+00            6.132548e+00            4.758441e-04            5.248926e+00            1.268802e+02            1.068013e-03            4.821854e-04            3.128156e+02            
10     3.361115e+01            2.521166e+01            2.169665e+01            9.274969e+01            9.333043e+00            5.821315e+00            2.482676e+00            1.515349e+02            1.025732e+02            5.897739e+00            2.537894e+02            
11     1.424043e+02            8.725613e+00            1.373032e+02            4.035545e+02            1.157063e+01            3.616556e+01            4.035466e+00            4.642662e+02            3.816763e+02            8.673190e+00            1.027160e+03            
12     4.155661e+00            3.935526e+00            1.113455e+02            1.151611e+02            2.724408e+01            1.001757e+01            6.905150e+00            1.156322e+02            1.509160e+02            3.054344e+00            1.844851e+02            
13     1.111589e+01            1.485236e+01            1.713306e+01            2.268506e+01            1.336513e+01            1.805460e+00            1.621019e+01            1.487572e+01            1.918082e+01            1.109139e+00            3.178155e+01            
14     4.256195e+00            3.702208e+00            5.921488e+00            1.076029e+01            6.679029e+00            3.890959e+00            4.837610e+00            9.418770e+00            8.227604e+00            2.469073e+00            1.591668e+01            
15     1.799734e+00            1.224650e+00            3.376066e+00            3.979596e+00            3.243766e+00            1.220384e+00            2.161833e+00            3.601175e+00            3.532686e+00            1.050369e+00            3.943063e+00            
16     1.139552e+03            1.790992e+03            7.369210e+03            2.607272e+03            1.504368e+03            2.107837e+03            1.099295e+03            6.960191e+03            8.536303e+03            5.594973e+03            1.591616e+04            
17     6.238053e+00            1.246774e+01            4.084607e+04            4.052213e+04            4.048993e+00            3.120103e+00            3.120104e+00            3.821216e+04            4.977950e+04            3.120103e+00            1.562807e+05            
18     2.714931e+01            2.865669e+01            3.242074e+01            4.389579e+01            2.874882e+01            1.543510e+01            3.368877e+01            4.870684e+01            4.939387e+01            2.674502e+00            5.671620e+01            
19     7.254457e+01            8.211408e+01            2.049983e+02            1.786649e+02            2.800483e+02            8.318907e+01            7.182677e+01            3.306650e+02            2.804783e+02            4.901195e+01            3.873000e+02            
20     2.490047e+01            2.843501e+01            2.681711e+01            3.133817e+01            3.054680e+01            1.211550e+01            2.417222e+01            3.055360e+01            3.451338e+01            6.992461e+00            4.825477e+01            
21     4.529395e+00            4.576199e+00            5.536909e+00            5.800620e+00            5.485702e+00            4.559685e+00            4.932325e+00            5.778942e+00            5.778048e+00            4.488449e+00            5.785615e+00            
22     9.369191e+00            1.016919e+01            1.390936e+01            1.596985e+01            1.548124e+01            7.397491e+00            1.284456e+01            1.717193e+01            1.800850e+01            2.273833e+00            1.889757e+01            
23     4.593962e+01            4.266850e+01            6.129001e+01            5.971966e+01            4.207201e+01            1.130870e+01            3.977845e+01            7.252710e+01            7.249519e+01            6.164246e+00            1.046457e+02            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: original.py  (error=1.081223e-07)
Task  1: variant_09_idea_0.py  (error=2.270481e-07)
Task  2: variant_05_idea_0.py  (error=2.331118e-05)
Task  3: variant_05_idea_0.py  (error=3.291411e-04)
Task  4: original.py  (error=2.433718e-02)
Task  5: original.py  (error=1.000000e-08)
Task  6: original.py  (error=4.043598e-07)
Task  7: variant_05_idea_0.py  (error=1.327038e-05)
Task  8: variant_05_idea_0.py  (error=1.701289e-03)
Task  9: variant_05_idea_0.py  (error=4.758441e-04)
Task 10: variant_06_idea_0.py  (error=2.482676e+00)
Task 11: variant_06_idea_0.py  (error=4.035466e+00)
Task 12: variant_09_idea_0.py  (error=3.054344e+00)
Task 13: variant_09_idea_0.py  (error=1.109139e+00)
Task 14: variant_09_idea_0.py  (error=2.469073e+00)
Task 15: variant_09_idea_0.py  (error=1.050369e+00)
Task 16: variant_06_idea_0.py  (error=1.099295e+03)
Task 17: variant_09_idea_0.py  (error=3.120103e+00)
Task 18: variant_09_idea_0.py  (error=2.674502e+00)
Task 19: variant_09_idea_0.py  (error=4.901195e+01)
Task 20: variant_09_idea_0.py  (error=6.992461e+00)
Task 21: variant_09_idea_0.py  (error=4.488449e+00)
Task 22: variant_09_idea_0.py  (error=2.273833e+00)
Task 23: variant_09_idea_0.py  (error=6.164246e+00)

WIN COUNTS:
  variant_09_idea_0.py: 12 wins
  variant_05_idea_0.py: 5 wins
  original.py: 4 wins
  variant_06_idea_0.py: 3 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_05_idea_0.py (5 wins) ---
```python
def _detect_stagnation(self):
        """Adaptive stagnation detection: aggressive restarts when error is large, patient when close to optimum."""

        # Track fitness history for plateau detection
        if not hasattr(self, '_fitness_history'):
            self._fitness_history = []
        self._fitness_history.append(self.best_fitness)

        # Keep history bounded
        max_hist = 200
        if len(self._fitness_history) > max_hist:
            self._fitness_history = self._fitness_history[-max_hist:]

        # Sigma collapsed — need restart
        if self.sigma < 1e-18:
            return True

        # Sigma exploded — algorithm is lost
        if self.sigma > 1e5:
            return True

        # Condition number too large — C is degenerate
        cond = np.max(self.D) / max(np.min(self.D), 1e-30)
        if cond > 1e8:
            return True

        # Two-regime stagnation limit based on current best fitness
        if self.best_fitness < 1e-4:
            # Near optimum: be very patient, allow fine-tuning
            stag_limit = 50 + int(80 * self.dim / self.pop_size)
        elif self.best_fitness < 1.0:
            # Moderate: medium patience
            stag_limit = 20 + int(40 * self.dim / self.pop_size)
        else:
            # Far from optimum: restart aggressively to explore new basins
            stag_limit = 5 + int(15 * self.dim / self.pop_size)

        if self.stagnation_counter > stag_limit:
            return True

        # Fitness plateau detection: if relative improvement over last N generations is tiny
        window = min(30, len(self._fitness_history))
        if window >= 10:
            old_fit = self._fitness_history[-window]
            new_fit = self._fitness_history[-1]
            if old_fit > 0 and np.isfinite(old_fit) and np.isfinite(new_fit):
                rel_improvement = (old_fit - new_fit) / (abs(old_fit) + 1e-30)
                # If essentially no relative improvement over the window
                if rel_improvement < 1e-12 and self.stagnation_counter > stag_limit // 2:
                    return True

        # Detect if sigma * max(D) is tiny relative to domain — search volume collapsed
        max_step = self.sigma * np.max(self.D)
        if max_step < 1e-15 * (self.ub - self.lb):
            return True

        # Detect if sigma * min(D) is tiny — one direction collapsed while others didn't
        min_step = self.sigma * np.min(self.D)
        if min_step < 1e-20 and cond > 1e4:
            return True

        return False
```

# --- From variant_06_idea_0.py (3 wins) ---
```python
def _detect_stagnation(self):
        """Adaptive stagnation detection based on fitness improvement rate."""
        # Track fitness history
        if not hasattr(self, '_fitness_window'):
            self._fitness_window = []
        self._fitness_window.append(self.best_fitness)

        # Keep window bounded
        max_window = 200
        if len(self._fitness_window) > max_window:
            self._fitness_window = self._fitness_window[-max_window:]

        # Check sigma collapse
        if self.sigma < 1e-18:
            self._fitness_window = []
            return True

        # Check condition number of C
        cond = np.max(self.D) / max(np.min(self.D), 1e-30)
        if cond > 1e8:
            self._fitness_window = []
            return True

        # Sigma explosion
        if self.sigma > 1e5:
            self._fitness_window = []
            return True

        # Adaptive window-based stagnation check
        # Use shorter window when error is large (restart faster), longer when small
        current_error = self.best_fitness
        if current_error > 1e2:
            check_window = max(8, int(5 + self.dim // 4))
        elif current_error > 1e0:
            check_window = max(12, int(10 + self.dim // 3))
        elif current_error > 1e-4:
            check_window = max(20, int(15 + self.dim // 2))
        else:
            check_window = max(30, int(20 + self.dim))

        if len(self._fitness_window) >= check_window:
            old_val = self._fitness_window[-check_window]
            new_val = self._fitness_window[-1]

            # Relative improvement over the window
            if old_val == 0 or not np.isfinite(old_val):
                rel_improvement = 0.0
            else:
                rel_improvement = (old_val - new_val) / (abs(old_val) + 1e-30)

            # Require minimum relative improvement scaled by error magnitude
            if current_error > 1e2:
                min_improvement = 1e-3  # Need at least 0.1% improvement
            elif current_error > 1e0:
                min_improvement = 1e-4
            elif current_error > 1e-4:
                min_improvement = 1e-6
            else:
                min_improvement = 1e-8

            if rel_improvement < min_improvement:
                self._fitness_window = []
                return True

        # Hard stagnation counter limit (safety net)
        hard_limit = 5 + int(15 * self.dim / self.pop_size)
        if self.stagnation_counter > hard_limit:
            self._fitness_window = []
            return True

        return False
```

# --- From variant_09_idea_0.py (12 wins) ---
```python
def _detect_stagnation(self):
        """Adaptive stagnation: fast restarts at high error, patient at low error."""
        # Base stagnation limit scales with how good current best is
        # If best_fitness is large, restart quickly to explore more
        # If best_fitness is tiny, be very patient to allow convergence

        if self.best_fitness < 1e-6:
            # Very close to solution - be extremely patient
            stag_limit = 200 + int(100 * self.dim / self.pop_size)
        elif self.best_fitness < 1e-2:
            # Good region - moderate patience
            stag_limit = 50 + int(50 * self.dim / self.pop_size)
        elif self.best_fitness < 1.0:
            # Decent but not great - shorter patience
            stag_limit = 15 + int(20 * self.dim / self.pop_size)
        elif self.best_fitness < 100.0:
            # Poor - restart quickly to try new regions
            stag_limit = 8 + int(10 * self.dim / self.pop_size)
        else:
            # Very poor - restart very aggressively
            stag_limit = 5 + int(5 * self.dim / self.pop_size)

        if self.stagnation_counter > stag_limit:
            return True

        # Sigma collapse - but only restart if error is still large
        # If error is small and sigma collapsed, that might be fine convergence
        if self.sigma < 1e-16:
            if self.best_fitness > 1e-8:
                return True
            else:
                return False  # Let it stay converged

        # Sigma explosion - lost control
        if self.sigma > 1e4:
            return True

        # Condition number check - more tolerant when error is small
        if np.min(self.D) > 0:
            cond = np.max(self.D) / np.min(self.D)
            cond_limit = 1e7 if self.best_fitness > 1e-4 else 1e10
            if cond > cond_limit:
                return True

        # If we've been running many generations with high error, force restart
        # This catches slow drift that doesn't trigger stagnation counter
        if self.generation > 0 and self.generation % (20 + int(30 * self.dim / self.pop_size)) == 0:
            if self.best_fitness > 10.0:
                # Check recent progress via fitness history
                if hasattr(self, 'best_fitness_history') and len(self.best_fitness_history) > 10:
                    recent = self.best_fitness_history[-10:]
                    if len(recent) >= 2 and (recent[0] - recent[-1]) / max(abs(recent[0]), 1e-30) < 0.01:
                        return True

        # Track fitness history for trend analysis
        if hasattr(self, 'best_fitness_history'):
            if len(self.best_fitness_history) == 0 or self.best_fitness_history[-1] != self.best_fitness:
                self.best_fitness_history.append(self.best_fitness)

        return False
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


class SimplifiedCovarianceAdaptiveSearch:
    """
    A simplified CMA-ES-inspired optimizer with:
    - Covariance matrix adaptation via rank-1 and rank-mu updates
    - Step-size adaptation via cumulative path length control
    - Elite-based mean update
    - Periodic restarts on stagnation
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.pop_size = 4 + int(3 * np.log(dim))  # ~14 for dim=30
        self.pop_size = max(self.pop_size, 8)
        # Make pop_size even
        if self.pop_size % 2 != 0:
            self.pop_size += 1
        self.mu = self.pop_size // 2  # number of parents
        self._initialize_strategy_params()

    def _initialize_strategy_params(self):
        """Set up all CMA-ES-like strategy parameters."""
        dim = self.dim
        mu = self.mu
        lam = self.pop_size

        # Recombination weights
        raw_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        # Step-size adaptation parameters
        self.c_sigma = (self.mu_eff + 2.0) / (dim + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (dim + 1.0)) - 1.0) + self.c_sigma

        # Covariance adaptation parameters
        self.c_c = (4.0 + self.mu_eff / dim) / (dim + 4.0 + 2.0 * self.mu_eff / dim)
        self.c_1 = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1.0 - self.c_1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((dim + 2.0) ** 2 + self.mu_eff))

        # Expected length of N(0,I) vector
        self.chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

    def _initialize_state(self):
        """Initialize the mean, covariance, evolution paths, and step size."""
        dim = self.dim
        self.mean = np.random.uniform(self.lb * 0.5, self.ub * 0.5, size=dim)
        self.sigma = 30.0  # initial step size
        self.C = np.eye(dim)
        self.p_sigma = np.zeros(dim)
        self.p_c = np.zeros(dim)
        self.eigen_decomp_gen = 0
        self.B = np.eye(dim)
        self.D = np.ones(dim)
        self.invsqrt_C = np.eye(dim)
        self.generation = 0
        self.best_fitness = np.inf
        self.best_x = self.mean.copy()
        self.stagnation_counter = 0
        self.best_fitness_history = []

    def _update_eigen_decomposition(self):
        """Recompute eigen decomposition of C if needed (every few gens)."""
        update_interval = max(1, int(1.0 / (self.c_1 + self.c_mu) / self.dim / 10.0))
        if self.generation - self.eigen_decomp_gen >= update_interval:
            # Enforce symmetry
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            eigenvalues, self.B = np.linalg.eigh(self.C)
            # Clamp eigenvalues for numerical stability
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            self.D = np.sqrt(eigenvalues)
            inv_D = 1.0 / self.D
            self.invsqrt_C = self.B @ np.diag(inv_D) @ self.B.T
            self.eigen_decomp_gen = self.generation

    def _sample_population_batch(self):
        """Sample a batch of lambda offspring from N(mean, sigma^2 * C)."""
        dim = self.dim
        lam = self.pop_size
        # z ~ N(0, I)
        z = np.random.randn(lam, dim)
        # y = B * D * z  (transform to N(0, C))
        y = z @ np.diag(self.D) @ self.B.T
        # x = mean + sigma * y
        population = self.mean[np.newaxis, :] + self.sigma * y
        return population, y, z

    def _clip_to_bounds_batch(self, population):
        """Clip all candidates to the search domain."""
        return np.clip(population, self.lb, self.ub)

    def _sort_by_fitness(self, population, fitness, y_vectors):
        """Sort population by fitness, return sorted arrays."""
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return population, fitness, y_vectors
        # Put invalid (NaN/inf) at the end with large fitness
        sort_fit = np.where(valid_mask, fitness, np.inf)
        order = np.argsort(sort_fit)
        return population[order], sort_fit[order], y_vectors[order]

    def _update_mean(self, sorted_pop, sorted_y):
        """Update the distribution mean using weighted recombination of the mu best."""
        old_mean = self.mean.copy()
        self.mean = self.weights @ sorted_pop[:self.mu]
        # Weighted mean of y-vectors (steps in original space)
        y_w = self.weights @ sorted_y[:self.mu]
        return old_mean, y_w

    def _update_evolution_path_sigma(self, y_w):
        """Update the conjugate evolution path for step-size control."""
        c_s = self.c_sigma
        self.p_sigma = (1.0 - c_s) * self.p_sigma + np.sqrt(c_s * (2.0 - c_s) * self.mu_eff) * (self.invsqrt_C @ y_w)

    def _update_evolution_path_c(self, y_w):
        """Update the evolution path for covariance matrix adaptation."""
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        h_sigma_threshold = (1.4 + 2.0 / (self.dim + 1.0)) * self.chi_n * np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
        h_sigma = 1.0 if p_sigma_norm < h_sigma_threshold else 0.0

        self.p_c = ((1.0 - self.c_c) * self.p_c +
                     h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * y_w)
        return h_sigma

    def _update_covariance_matrix(self, sorted_y, h_sigma):
        """Rank-1 and rank-mu update of the covariance matrix."""
        dim = self.dim
        # Rank-1 update
        rank1 = np.outer(self.p_c, self.p_c)

        # Rank-mu update
        y_sel = sorted_y[:self.mu]  # (mu, dim)
        rank_mu = np.zeros((dim, dim))
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(y_sel[i], y_sel[i])

        # Correction for h_sigma
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        self.C = ((1.0 - self.c_1 - self.c_mu + delta_h * self.c_1) * self.C +
                   self.c_1 * rank1 +
                   self.c_mu * rank_mu)

    def _update_step_size(self):
        """CSA: cumulative step-size adaptation."""
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0))
        # Clamp sigma to prevent explosion or collapse
        self.sigma = np.clip(self.sigma, 1e-20, 1e6)

    def _track_best(self, sorted_pop, sorted_fit):
        """Track the overall best solution found."""
        if len(sorted_fit) == 0:
            return
        valid = np.isfinite(sorted_fit)
        if not np.any(valid):
            return
        best_idx = np.argmin(np.where(valid, sorted_fit, np.inf))
        if sorted_fit[best_idx] < self.best_fitness:
            self.best_fitness = sorted_fit[best_idx]
            self.best_x = sorted_pop[best_idx].copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

    def _detect_stagnation(self):
        """Check if the algorithm has stagnated and needs a restart."""
        # Stagnation if no improvement for many generations
        stag_limit = 10 + int(30 * self.dim / self.pop_size)
        if self.stagnation_counter > stag_limit:
            return True
        # Check if sigma collapsed
        if self.sigma < 1e-16:
            return True
        # Check if C has degenerate eigenvalues
        if np.max(self.D) / max(np.min(self.D), 1e-30) > 1e7:
            return True
        return False

    def _restart(self):
        """IPOP-style restart: double pop size, alternate random/biased starts, large sigma."""
        saved_best_x = self.best_x.copy()
        saved_best_f = self.best_fitness

        # Track restart count
        if not hasattr(self, '_restart_count'):
            self._restart_count = 0
            self._base_pop_size = self.pop_size
        self._restart_count += 1

        # IPOP: increase population size (double each restart, cap at reasonable limit)
        new_pop_size = self._base_pop_size * (2 ** min(self._restart_count, 5))
        new_pop_size = min(new_pop_size, 512)  # cap to avoid excessive evals
        if new_pop_size % 2 != 0:
            new_pop_size += 1

        self.pop_size = new_pop_size
        self.mu = self.pop_size // 2

        # Recompute strategy parameters for new pop size
        self._initialize_strategy_params()

        # Re-initialize state
        self._initialize_state()

        # Restore best
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f

        # Alternate between strategies based on restart count
        strategy = self._restart_count % 4

        if strategy == 0:
            # Fully random restart - explore completely new region
            self.mean = np.random.uniform(self.lb, self.ub, size=self.dim)
            self.sigma = 50.0  # large sigma for broad exploration
        elif strategy == 1:
            # Start near best but with very large sigma
            perturbation = np.random.randn(self.dim) * 40.0
            self.mean = np.clip(saved_best_x + perturbation, self.lb, self.ub)
            self.sigma = 60.0
        elif strategy == 2:
            # Opposition-based restart: mirror best through center
            center = np.zeros(self.dim)  # center of domain
            opposite = 2.0 * center - saved_best_x
            noise = np.random.randn(self.dim) * 10.0
            self.mean = np.clip(opposite + noise, self.lb, self.ub)
            self.sigma = 40.0
        else:
            # Start at best with moderate sigma for local refinement
            self.mean = np.clip(saved_best_x + np.random.randn(self.dim) * 5.0, self.lb, self.ub)
            self.sigma = 15.0

        # Every 4th restart cycle, reset pop size to avoid getting stuck with huge pop
        if self._restart_count % 8 == 0:
            self._restart_count = 0

    def _handle_truncated_fitness(self, population, fitness, y_vectors):
        """Handle case where func returns fewer values than requested."""
        n_valid = len(fitness)
        if n_valid < len(population):
            population = population[:n_valid]
            y_vectors = y_vectors[:n_valid]
            fitness = fitness[:n_valid]
        return population, fitness, y_vectors

    def _inject_best_into_population(self, population, fitness, y_vectors):
        """Optionally inject the best-known solution into the population to preserve it."""
        if self.best_fitness < np.inf and len(population) > 0:
            worst_idx = np.argmax(fitness)
            if self.best_fitness < fitness[worst_idx]:
                population[worst_idx] = self.best_x.copy()
                fitness[worst_idx] = self.best_fitness
                y_vectors[worst_idx] = (self.best_x - self.mean) / max(self.sigma, 1e-30)
        return population, fitness, y_vectors

    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        self._initialize_state()

        # Initial evaluation of mean to seed best tracking
        mean_fit = func(self.mean.reshape(1, self.dim))
        if len(mean_fit) > 0 and np.isfinite(mean_fit[0]):
            self.best_fitness = mean_fit[0]
            self.best_x = self.mean.copy()

        while not stopping_condition():
            # Update eigen decomposition if needed
            self._update_eigen_decomposition()

            # Sample new population
            population, y_vectors, z_vectors = self._sample_population_batch()
            population = self._clip_to_bounds_batch(population)

            # Evaluate fitness
            fitness = func(population)
            if stopping_condition():
                # Still process results we got
                population, fitness, y_vectors = self._handle_truncated_fitness(population, fitness, y_vectors)
                if len(fitness) > 0:
                    sorted_pop, sorted_fit, sorted_y = self._sort_by_fitness(population, fitness, y_vectors)
                    self._track_best(sorted_pop, sorted_fit)
                break

            # Handle truncated results
            population, fitness, y_vectors = self._handle_truncated_fitness(population, fitness, y_vectors)
            if len(fitness) < self.mu:
                # Not enough to do a meaningful update
                self._track_best(population, fitness, y_vectors)
                break

            # Sort by fitness
            sorted_pop, sorted_fit, sorted_y = self._sort_by_fitness(population, fitness, y_vectors)

            # Track best
            self._track_best(sorted_pop, sorted_fit)

            # Update mean
            old_mean, y_w = self._update_mean(sorted_pop, sorted_y)

            # Update evolution paths
            self._update_evolution_path_sigma(y_w)
            h_sigma = self._update_evolution_path_c(y_w)

            # Update covariance matrix
            self._update_covariance_matrix(sorted_y, h_sigma)

            # Update step size
            self._update_step_size()

            self.generation += 1

            # Detect and handle stagnation
            if self._detect_stagnation():
                self._restart()

        return self.best_fitness, self.best_x

```