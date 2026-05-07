This is iteration 4 of 10. Propose ONE replacement implementation for `_update_step_size`.

Requirements:
- Keep the EXACT function signature: `def _update_step_size(self):`
- Propose a SINGLE, FUNDAMENTALLY DIFFERENT strategy vs. the variants below
- You may read any `self` attribute but do NOT modify `__init__` or other methods
- The fenced code block must contain ONLY the single replacement function
- Ensure numerical robustness (no division by zero, handle edge dims, clip to bounds)

WHY PER-TASK COVERAGE MATTERS:
After all variants are generated, an ADAPTIVE algorithm will be built that
selects the best operator FOR EACH TASK at runtime (e.g. via Thompson Sampling
or multi-armed bandit). This means:
- We do NOT need a single variant that wins everywhere.
- We DO need at least one variant that reaches error <= 1e-08 on EACH task.
- Tasks marked UNSOLVED below are critical gaps. The bigger the remaining
  error, the higher the priority — your variant should specifically target
  the WORST unsolved tasks at the top of the priority list.

TASK COVERAGE SUMMARY: 1 SOLVED (<= 1e-08), 23 UNSOLVED (of which 0 crashed) — out of 24.
Goal: drive every task to error <= 1e-08. Focus first on the WORST unsolved tasks (top of the UNSOLVED list).

UNSOLVED TASKS (worst best-error first — these are the priority targets):
  Task 16: *** UNSOLVED *** — best=9.374e+02 by variant_01_idea_0.py      (target=1e-08, ~+11.0 decades above target)
  Task 19: *** UNSOLVED *** — best=3.516e+01 by variant_01_idea_0.py      (target=1e-08, ~+9.5 decades above target)
  Task 20: *** UNSOLVED *** — best=8.427e+00 by variant_02_idea_0.py      (target=1e-08, ~+8.9 decades above target)
  Task 23: *** UNSOLVED *** — best=6.213e+00 by original.py               (target=1e-08, ~+8.8 decades above target)
  Task 11: *** UNSOLVED *** — best=4.926e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.7 decades above target)
  Task 10: *** UNSOLVED *** — best=4.722e+00 by original.py               (target=1e-08, ~+8.7 decades above target)
  Task 21: *** UNSOLVED *** — best=4.453e+00 by original.py               (target=1e-08, ~+8.6 decades above target)
  Task 17: *** UNSOLVED *** — best=3.120e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.5 decades above target)
  Task 18: *** UNSOLVED *** — best=2.674e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.4 decades above target)
  Task 14: *** UNSOLVED *** — best=2.546e+00 by variant_03_idea_0.py      (target=1e-08, ~+8.4 decades above target)
  Task 12: *** UNSOLVED *** — best=2.491e+00 by original.py               (target=1e-08, ~+8.4 decades above target)
  Task 22: *** UNSOLVED *** — best=2.324e+00 by variant_02_idea_0.py      (target=1e-08, ~+8.4 decades above target)
  Task 13: *** UNSOLVED *** — best=1.356e+00 by original.py               (target=1e-08, ~+8.1 decades above target)
  Task 15: *** UNSOLVED *** — best=1.102e+00 by variant_02_idea_0.py      (target=1e-08, ~+8.0 decades above target)
  Task  4: *** UNSOLVED *** — best=2.433e-02 by original.py               (target=1e-08, ~+6.4 decades above target)
  Task  8: *** UNSOLVED *** — best=1.588e-03 by variant_02_idea_0.py      (target=1e-08, ~+5.2 decades above target)
  Task  9: *** UNSOLVED *** — best=4.662e-04 by variant_01_idea_0.py      (target=1e-08, ~+4.7 decades above target)
  Task  3: *** UNSOLVED *** — best=2.972e-04 by variant_01_idea_0.py      (target=1e-08, ~+4.5 decades above target)
  Task  2: *** UNSOLVED *** — best=2.408e-05 by variant_01_idea_0.py      (target=1e-08, ~+3.4 decades above target)
  Task  7: *** UNSOLVED *** — best=1.350e-05 by original.py               (target=1e-08, ~+3.1 decades above target)
  Task  6: *** UNSOLVED *** — best=4.396e-07 by original.py               (target=1e-08, ~+1.6 decades above target)
  Task  1: *** UNSOLVED *** — best=2.264e-07 by original.py               (target=1e-08, ~+1.4 decades above target)
  Task  0: *** UNSOLVED *** — best=7.458e-08 by variant_02_idea_0.py      (target=1e-08, ~+0.9 decades above target)

SOLVED TASKS (already at or below target — do not regress these):
  Task  5: SOLVED  — best=1.000e-08 by original.py                   

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_idea_0.py  variant_02_idea_0.py  variant_03_idea_0.py  
---------------------------------------------------------------------------------------------
0    1.1781e-07            1.1307e-07            7.4585e-08            8.1359e-08            
1    2.2640e-07            2.3475e-07            1.2274e+02            2.4286e-07            
2    2.4435e-05            2.4083e-05            5.7213e+02            4.9149e-05            
3    3.0066e-04            2.9720e-04            2.5465e+02            3.4876e-04            
4    2.4334e-02            2.4545e-02            2.4394e-02            2.4383e-02            
5    1.0000e-08            1.0000e-08            1.5723e+09            1.9494e+04            
6    4.3956e-07            4.6647e-07            6.6246e+03            3.8153e+03            
7    1.3496e-05            1.3521e-05            3.2576e+02            2.3128e+01            
8    1.6949e-03            1.7335e-03            1.5879e-03            1.7426e-03            
9    4.6853e-04            4.6620e-04            3.2676e+02            3.3200e-03            
10   4.7223e+00            5.2169e+00            2.7579e+02            2.5155e+02            
11   5.6001e+00            4.9262e+00            1.1082e+03            9.5096e+02            
12   2.4913e+00            4.0999e+00            2.0465e+02            1.8509e+02            
13   1.3563e+00            1.5999e+00            1.6471e+00            1.5394e+00            
14   2.6240e+00            2.6955e+00            2.5663e+00            2.5458e+00            
15   1.2005e+00            1.1347e+00            1.1020e+00            1.1944e+00            
16   9.4940e+02            9.3736e+02            1.8339e+04            1.5427e+04            
17   3.1201e+00            3.1201e+00            1.5892e+05            1.4104e+05            
18   2.6744e+00            2.6744e+00            2.6745e+00            2.7846e+00            
19   3.6306e+01            3.5159e+01            3.9386e+02            3.8683e+02            
20   1.3172e+01            1.2583e+01            8.4272e+00            2.0621e+01            
21   4.4530e+00            4.5592e+00            4.5173e+00            4.5906e+00            
22   2.3248e+00            2.4808e+00            2.3237e+00            2.5258e+00            
23   6.2128e+00            6.7236e+00            1.0531e+02            1.1579e+01            

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 16: best error so far = 9.374e+02  (target = 1e-08)
  Task 19: best error so far = 3.516e+01  (target = 1e-08)
  Task 20: best error so far = 8.427e+00  (target = 1e-08)
  Task 23: best error so far = 6.213e+00  (target = 1e-08)
  Task 11: best error so far = 4.926e+00  (target = 1e-08)
  Task 10: best error so far = 4.722e+00  (target = 1e-08)
  Task 21: best error so far = 4.453e+00  (target = 1e-08)
  Task 17: best error so far = 3.120e+00  (target = 1e-08)
  ... and 15 other unsolved task(s).

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0, Idea 0

Your new proposal MUST be designed to crush the error on the WORST unsolved
tasks above. It is acceptable — even expected — for the new variant to be
worse than existing variants on already-SOLVED tasks; the adaptive selector
will handle that. Reason explicitly about the priority targets:
1. What property of those WORST unsolved tasks (multimodality, ill-conditioning,
   separability, ruggedness, noise, deceptive local optima, narrow basins,
   non-separable rotation, etc.) is preventing existing operators from reaching
   1e-08? Use the per-task error magnitudes as evidence — errors
   stuck at ~1e+1 vs ~1e-3 vs ~1e-6 imply different failure modes.
2. What specific mechanism in your proposed operator is designed to break
   through that exact obstacle and push the error several orders of magnitude
   lower?
3. Why is this approach fundamentally different from the prior variants —
   especially from whichever variant currently holds the best (but still
   insufficient) error on the priority tasks?

Current implementation:
```python
def _update_step_size(self):
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0))
        self.sigma = np.clip(self.sigma, 1e-20, 1e6)
```

Full algorithm for context:
```python
import numpy as np


class SimplifiedCovarianceAdaptiveSearch:
    """
    A simplified CMA-ES-inspired optimizer with adaptive stagnation detection.
    Uses Thompson Sampling to select among three stagnation detection strategies.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.pop_size = 4 + int(3 * np.log(dim))
        self.pop_size = max(self.pop_size, 8)
        if self.pop_size % 2 != 0:
            self.pop_size += 1
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()

        # Adaptive operator selection: Thompson Sampling for 3 stagnation strategies
        # 0 = variant_09 (patient near optimum, aggressive far)
        # 1 = variant_05 (adaptive with plateau detection)
        # 2 = variant_06 (window-based improvement rate)
        self.n_strategies = 3
        self.ts_alpha = np.ones(self.n_strategies, dtype=float)  # Beta prior successes
        self.ts_beta = np.ones(self.n_strategies, dtype=float)   # Beta prior failures
        self.current_strategy = 0
        self._strategy_fitness_start = np.inf  # fitness at start of current strategy epoch
        self._strategy_evals_start = 0
        self._strategy_epoch_length = 0

    def _initialize_strategy_params(self):
        dim = self.dim
        mu = self.mu

        raw_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        self.c_sigma = (self.mu_eff + 2.0) / (dim + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (dim + 1.0)) - 1.0) + self.c_sigma

        self.c_c = (4.0 + self.mu_eff / dim) / (dim + 4.0 + 2.0 * self.mu_eff / dim)
        self.c_1 = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1.0 - self.c_1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((dim + 2.0) ** 2 + self.mu_eff))

        self.chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

    def _initialize_state(self):
        dim = self.dim
        self.mean = np.random.uniform(self.lb * 0.5, self.ub * 0.5, size=dim)
        self.sigma = 30.0
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
        self._fitness_history = []
        self._fitness_window = []

    def _update_eigen_decomposition(self):
        update_interval = max(1, int(1.0 / (self.c_1 + self.c_mu) / self.dim / 10.0))
        if self.generation - self.eigen_decomp_gen >= update_interval:
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            eigenvalues, self.B = np.linalg.eigh(self.C)
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            self.D = np.sqrt(eigenvalues)
            inv_D = 1.0 / self.D
            self.invsqrt_C = self.B @ np.diag(inv_D) @ self.B.T
            self.eigen_decomp_gen = self.generation

    def _sample_population_batch(self):
        dim = self.dim
        lam = self.pop_size
        z = np.random.randn(lam, dim)
        y = z @ np.diag(self.D) @ self.B.T
        population = self.mean[np.newaxis, :] + self.sigma * y
        return population, y, z

    def _clip_to_bounds_batch(self, population):
        return np.clip(population, self.lb, self.ub)

    def _sort_by_fitness(self, population, fitness, y_vectors):
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return population, fitness, y_vectors
        sort_fit = np.where(valid_mask, fitness, np.inf)
        order = np.argsort(sort_fit)
        return population[order], sort_fit[order], y_vectors[order]

    def _update_mean(self, sorted_pop, sorted_y):
        old_mean = self.mean.copy()
        self.mean = self.weights @ sorted_pop[:self.mu]
        y_w = self.weights @ sorted_y[:self.mu]
        return old_mean, y_w

    def _update_evolution_path_sigma(self, y_w):
        c_s = self.c_sigma
        self.p_sigma = (1.0 - c_s) * self.p_sigma + np.sqrt(c_s * (2.0 - c_s) * self.mu_eff) * (self.invsqrt_C @ y_w)

    def _update_evolution_path_c(self, y_w):
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        h_sigma_threshold = (1.4 + 2.0 / (self.dim + 1.0)) * self.chi_n * np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
        h_sigma = 1.0 if p_sigma_norm < h_sigma_threshold else 0.0

        self.p_c = ((1.0 - self.c_c) * self.p_c +
                     h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * y_w)
        return h_sigma

    def _update_covariance_matrix(self, sorted_y, h_sigma):
        dim = self.dim
        rank1 = np.outer(self.p_c, self.p_c)

        y_sel = sorted_y[:self.mu]
        rank_mu = np.zeros((dim, dim))
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(y_sel[i], y_sel[i])

        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        self.C = ((1.0 - self.c_1 - self.c_mu + delta_h * self.c_1) * self.C +
                   self.c_1 * rank1 +
                   self.c_mu * rank_mu)

    def _update_step_size(self):
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0))
        self.sigma = np.clip(self.sigma, 1e-20, 1e6)

    def _track_best(self, sorted_pop, sorted_fit):
        if len(sorted_fit) == 0:
            return
        valid = np.isfinite(sorted_fit)
        if not np.any(valid):
            return
        best_idx = np.argmin(np.where(valid, sorted_fit, np.inf))
        if sorted_fit[best_idx] < self.best_fitness:
            self.best_fitness = float(sorted_fit[best_idx])
            self.best_x = sorted_pop[best_idx].copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

    # ---- Three stagnation detection strategies ----

    def _detect_stagnation_strategy0(self):
        """Strategy 0: variant_09 - patient near optimum, aggressive far away."""
        if self.best_fitness < 1e-6:
            stag_limit = 200 + int(100 * self.dim / self.pop_size)
        elif self.best_fitness < 1e-2:
            stag_limit = 50 + int(50 * self.dim / self.pop_size)
        elif self.best_fitness < 1.0:
            stag_limit = 15 + int(20 * self.dim / self.pop_size)
        elif self.best_fitness < 100.0:
            stag_limit = 8 + int(10 * self.dim / self.pop_size)
        else:
            stag_limit = 5 + int(5 * self.dim / self.pop_size)

        if self.stagnation_counter > stag_limit:
            return True

        if self.sigma < 1e-16:
            if self.best_fitness > 1e-8:
                return True
            else:
                return False

        if self.sigma > 1e4:
            return True

        if np.min(self.D) > 0:
            cond = np.max(self.D) / np.min(self.D)
            cond_limit = 1e7 if self.best_fitness > 1e-4 else 1e10
            if cond > cond_limit:
                return True

        if self.generation > 0 and self.generation % (20 + int(30 * self.dim / self.pop_size)) == 0:
            if self.best_fitness > 10.0:
                if hasattr(self, 'best_fitness_history') and len(self.best_fitness_history) > 10:
                    recent = self.best_fitness_history[-10:]
                    if len(recent) >= 2 and (recent[0] - recent[-1]) / max(abs(recent[0]), 1e-30) < 0.01:
                        return True

        if hasattr(self, 'best_fitness_history'):
            if len(self.best_fitness_history) == 0 or self.best_fitness_history[-1] != self.best_fitness:
                self.best_fitness_history.append(float(self.best_fitness))

        return False

    def _detect_stagnation_strategy1(self):
        """Strategy 1: variant_05 - adaptive with plateau detection."""
        if not hasattr(self, '_fitness_history'):
            self._fitness_history = []
        self._fitness_history.append(float(self.best_fitness))

        max_hist = 200
        if len(self._fitness_history) > max_hist:
            self._fitness_history = self._fitness_history[-max_hist:]

        if self.sigma < 1e-18:
            return True

        if self.sigma > 1e5:
            return True

        cond = np.max(self.D) / max(np.min(self.D), 1e-30)
        if cond > 1e8:
            return True

        if self.best_fitness < 1e-4:
            stag_limit = 50 + int(80 * self.dim / self.pop_size)
        elif self.best_fitness < 1.0:
            stag_limit = 20 + int(40 * self.dim / self.pop_size)
        else:
            stag_limit = 5 + int(15 * self.dim / self.pop_size)

        if self.stagnation_counter > stag_limit:
            return True

        window = min(30, len(self._fitness_history))
        if window >= 10:
            old_fit = self._fitness_history[-window]
            new_fit = self._fitness_history[-1]
            if old_fit > 0 and np.isfinite(old_fit) and np.isfinite(new_fit):
                rel_improvement = (old_fit - new_fit) / (abs(old_fit) + 1e-30)
                if rel_improvement < 1e-12 and self.stagnation_counter > stag_limit // 2:
                    return True

        max_step = self.sigma * np.max(self.D)
        if max_step < 1e-15 * (self.ub - self.lb):
            return True

        min_step = self.sigma * np.min(self.D)
        if min_step < 1e-20 and cond > 1e4:
            return True

        return False

    def _detect_stagnation_strategy2(self):
        """Strategy 2: variant_06 - window-based improvement rate."""
        if not hasattr(self, '_fitness_window'):
            self._fitness_window = []
        self._fitness_window.append(float(self.best_fitness))

        max_window = 200
        if len(self._fitness_window) > max_window:
            self._fitness_window = self._fitness_window[-max_window:]

        if self.sigma < 1e-18:
            self._fitness_window = []
            return True

        cond = np.max(self.D) / max(np.min(self.D), 1e-30)
        if cond > 1e8:
            self._fitness_window = []
            return True

        if self.sigma > 1e5:
            self._fitness_window = []
            return True

        current_error = float(self.best_fitness)
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

            if old_val == 0 or not np.isfinite(old_val):
                rel_improvement = 0.0
            else:
                rel_improvement = (old_val - new_val) / (abs(old_val) + 1e-30)

            if current_error > 1e2:
                min_improvement = 1e-3
            elif current_error > 1e0:
                min_improvement = 1e-4
            elif current_error > 1e-4:
                min_improvement = 1e-6
            else:
                min_improvement = 1e-8

            if rel_improvement < min_improvement:
                self._fitness_window = []
                return True

        hard_limit = 5 + int(15 * self.dim / self.pop_size)
        if self.stagnation_counter > hard_limit:
            self._fitness_window = []
            return True

        return False

    def _select_strategy(self):
        """Thompson Sampling to select a stagnation detection strategy."""
        samples = np.array([
            np.random.beta(max(self.ts_alpha[i], 0.01), max(self.ts_beta[i], 0.01))
            for i in range(self.n_strategies)
        ])
        return int(np.argmax(samples))

    def _update_strategy_reward(self, strategy_idx, fitness_before, fitness_after, n_gens):
        """Credit assignment: reward strategy based on fitness improvement rate."""
        strategy_idx = int(strategy_idx)
        if not np.isfinite(fitness_before) or not np.isfinite(fitness_after):
            return
        if n_gens <= 0:
            return

        # Compute relative improvement
        improvement = float(fitness_before) - float(fitness_after)
        rel_improvement = improvement / (abs(float(fitness_before)) + 1e-30)

        # Convert to reward: positive improvement = success
        if rel_improvement > 1e-6:
            self.ts_alpha[strategy_idx] += 1.0
        elif rel_improvement > 1e-10:
            self.ts_alpha[strategy_idx] += 0.3
            self.ts_beta[strategy_idx] += 0.3
        else:
            self.ts_beta[strategy_idx] += 1.0

        # Decay to allow adaptation over time
        decay = 0.995
        self.ts_alpha *= decay
        self.ts_beta *= decay
        # Ensure minimum values
        self.ts_alpha = np.maximum(self.ts_alpha, 0.5)
        self.ts_beta = np.maximum(self.ts_beta, 0.5)

    def _detect_stagnation(self):
        """Adaptive stagnation detection: delegates to selected strategy."""
        strategies = [
            self._detect_stagnation_strategy0,
            self._detect_stagnation_strategy1,
            self._detect_stagnation_strategy2,
        ]
        result = strategies[self.current_strategy]()
        self._strategy_epoch_length += 1
        return result

    def _restart(self):
        """IPOP-style restart with strategy reward update."""
        saved_best_x = self.best_x.copy()
        saved_best_f = float(self.best_fitness)

        # Update reward for current strategy
        self._update_strategy_reward(
            self.current_strategy,
            self._strategy_fitness_start,
            saved_best_f,
            self._strategy_epoch_length
        )

        # Select new strategy for next epoch
        self.current_strategy = self._select_strategy()
        self._strategy_fitness_start = saved_best_f
        self._strategy_epoch_length = 0

        if not hasattr(self, '_restart_count'):
            self._restart_count = 0
            self._base_pop_size = self.pop_size
        self._restart_count += 1

        new_pop_size = self._base_pop_size * (2 ** min(self._restart_count, 5))
        new_pop_size = min(new_pop_size, 512)
        if new_pop_size % 2 != 0:
            new_pop_size += 1

        self.pop_size = new_pop_size
        self.mu = self.pop_size // 2

        self._initialize_strategy_params()
        self._initialize_state()

        self.best_x = saved_best_x
        self.best_fitness = saved_best_f

        strategy = self._restart_count % 4

        if strategy == 0:
            self.mean = np.random.uniform(self.lb, self.ub, size=self.dim)
            self.sigma = 50.0
        elif strategy == 1:
            perturbation = np.random.randn(self.dim) * 40.0
            self.mean = np.clip(saved_best_x + perturbation, self.lb, self.ub)
            self.sigma = 60.0
        elif strategy == 2:
            center = np.zeros(self.dim)
            opposite = 2.0 * center - saved_best_x
            noise = np.random.randn(self.dim) * 10.0
            self.mean = np.clip(opposite + noise, self.lb, self.ub)
            self.sigma = 40.0
        else:
            self.mean = np.clip(saved_best_x + np.random.randn(self.dim) * 5.0, self.lb, self.ub)
            self.sigma = 15.0

        if self._restart_count % 8 == 0:
            self._restart_count = 0

    def _handle_truncated_fitness(self, population, fitness, y_vectors):
        n_valid = len(fitness)
        if n_valid < len(population):
            population = population[:n_valid]
            y_vectors = y_vectors[:n_valid]
            fitness = fitness[:n_valid]
        return population, fitness, y_vectors

    def _inject_best_into_population(self, population, fitness, y_vectors):
        if self.best_fitness < np.inf and len(population) > 0:
            worst_idx = np.argmax(fitness)
            if self.best_fitness < fitness[worst_idx]:
                population[worst_idx] = self.best_x.copy()
                fitness[worst_idx] = self.best_fitness
                y_vectors[worst_idx] = (self.best_x - self.mean) / max(self.sigma, 1e-30)
        return population, fitness, y_vectors

    def __call__(self, func, stopping_condition):
        self._initialize_state()

        # Initialize adaptive strategy tracking
        self.current_strategy = self._select_strategy()
        self._strategy_fitness_start = np.inf
        self._strategy_epoch_length = 0

        mean_fit = func(self.mean.reshape(1, self.dim))
        if len(mean_fit) > 0 and np.isfinite(mean_fit[0]):
            self.best_fitness = float(mean_fit[0])
            self.best_x = self.mean.copy()
            self._strategy_fitness_start = float(self.best_fitness)

        while not stopping_condition():
            self._update_eigen_decomposition()

            population, y_vectors, z_vectors = self._sample_population_batch()
            population = self._clip_to_bounds_batch(population)

            fitness = func(population)
            if stopping_condition():
                population, fitness, y_vectors = self._handle_truncated_fitness(population, fitness, y_vectors)
                if len(fitness) > 0:
                    sorted_pop, sorted_fit, sorted_y = self._sort_by_fitness(population, fitness, y_vectors)
                    self._track_best(sorted_pop, sorted_fit)
                break

            population, fitness, y_vectors = self._handle_truncated_fitness(population, fitness, y_vectors)
            if len(fitness) < self.mu:
                if len(fitness) > 0:
                    sorted_pop, sorted_fit, sorted_y = self._sort_by_fitness(population, fitness, y_vectors)
                    self._track_best(sorted_pop, sorted_fit)
                break

            sorted_pop, sorted_fit, sorted_y = self._sort_by_fitness(population, fitness, y_vectors)
            self._track_best(sorted_pop, sorted_fit)

            old_mean, y_w = self._update_mean(sorted_pop, sorted_y)
            self._update_evolution_path_sigma(y_w)
            h_sigma = self._update_evolution_path_c(y_w)
            self._update_covariance_matrix(sorted_y, h_sigma)
            self._update_step_size()

            self.generation += 1

            if self._detect_stagnation():
                self._restart()

        return self.best_fitness, self.best_x

```

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description.
```python
def _update_step_size(self, ...):
    ...
```