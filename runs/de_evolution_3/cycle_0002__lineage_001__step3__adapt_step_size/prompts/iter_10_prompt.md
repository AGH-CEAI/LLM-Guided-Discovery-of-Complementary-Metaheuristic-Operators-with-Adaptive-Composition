This is iteration 10 of 10. Propose ONE replacement implementation for `_adapt_step_size`.

Requirements:
- Keep the EXACT function signature: `def _adapt_step_size(self):`
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
  Task 21: *** UNSOLVED *** — best=4.124e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.6 decades above target)
  Task 17: *** UNSOLVED *** — best=3.120e+00 by original.py               (target=1e-08, ~+8.5 decades above target)
  Task 18: *** UNSOLVED *** — best=2.675e+00 by variant_03_idea_0.py      (target=1e-08, ~+8.4 decades above target)
  Task 19: *** UNSOLVED *** — best=2.511e+00 by variant_03_idea_0.py      (target=1e-08, ~+8.4 decades above target)
  Task 22: *** UNSOLVED *** — best=1.477e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.2 decades above target)
  Task 20: *** UNSOLVED *** — best=1.401e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.1 decades above target)
  Task 15: *** UNSOLVED *** — best=7.514e-01 by variant_01_idea_0.py      (target=1e-08, ~+7.9 decades above target)
  Task 14: *** UNSOLVED *** — best=4.530e-01 by variant_01_idea_0.py      (target=1e-08, ~+7.7 decades above target)
  Task 13: *** UNSOLVED *** — best=6.324e-02 by variant_01_idea_0.py      (target=1e-08, ~+6.8 decades above target)
  Task 23: *** UNSOLVED *** — best=6.013e-02 by variant_01_idea_0.py      (target=1e-08, ~+6.8 decades above target)
  Task 12: *** UNSOLVED *** — best=5.122e-02 by variant_01_idea_0.py      (target=1e-08, ~+6.7 decades above target)
  Task  4: *** UNSOLVED *** — best=2.376e-02 by variant_03_idea_0.py      (target=1e-08, ~+6.4 decades above target)
  Task 11: *** UNSOLVED *** — best=8.421e-03 by variant_01_idea_0.py      (target=1e-08, ~+5.9 decades above target)
  Task  8: *** UNSOLVED *** — best=1.729e-03 by variant_03_idea_0.py      (target=1e-08, ~+5.2 decades above target)
  Task 10: *** UNSOLVED *** — best=1.643e-03 by variant_01_idea_0.py      (target=1e-08, ~+5.2 decades above target)
  Task  9: *** UNSOLVED *** — best=4.805e-04 by original.py               (target=1e-08, ~+4.7 decades above target)
  Task  3: *** UNSOLVED *** — best=3.424e-04 by variant_03_idea_0.py      (target=1e-08, ~+4.5 decades above target)
  Task  2: *** UNSOLVED *** — best=2.236e-05 by original.py               (target=1e-08, ~+3.3 decades above target)
  Task  7: *** UNSOLVED *** — best=1.342e-05 by original.py               (target=1e-08, ~+3.1 decades above target)
  Task  6: *** UNSOLVED *** — best=4.101e-07 by original.py               (target=1e-08, ~+1.6 decades above target)
  Task  1: *** UNSOLVED *** — best=2.371e-07 by original.py               (target=1e-08, ~+1.4 decades above target)
  Task  0: *** UNSOLVED *** — best=8.543e-08 by variant_03_idea_0.py      (target=1e-08, ~+0.9 decades above target)
  Task 16: *** UNSOLVED *** — best=3.999e-08 by original.py               (target=1e-08, ~+0.6 decades above target)

SOLVED TASKS (already at or below target — do not regress these):
  Task  5: SOLVED  — best=1.000e-08 by original.py                   

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_02_idea_0.py  variant_03_idea_0.py  variant_04_idea_0.py  variant_05_idea_0.py  variant_06_idea_0.py  variant_07_idea_0.py  variant_08_idea_0.py  variant_09_idea_0.py  
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0    8.7621e-08            2.5188e-07            8.5425e-08            nan                   -inf                  1.0110e+02            2.4826e-07            2.0761e-07            7.7149e-04            
1    2.3715e-07            7.9888e-07            2.5194e-07            nan                   -inf                  1.9007e+02            6.7767e-07            6.5048e-07            2.1837e-05            
2    2.2358e-05            7.1519e+00            2.2452e-05            nan                   -inf                  8.2802e+02            5.2035e-05            5.0593e-05            2.9414e+00            
3    3.4716e-04            1.2558e+01            3.4239e-04            nan                   -inf                  3.4012e+02            2.3430e-03            7.9008e-04            1.0129e+00            
4    2.3894e-02            3.3346e-02            2.3764e-02            nan                   -inf                  5.9377e+00            4.2306e-02            3.1026e-02            5.3594e-01            
5    1.0000e-08            7.5722e+02            1.0000e-08            nan                   -inf                  6.6041e+09            1.0000e-08            1.0000e-08            8.5415e-03            
6    4.1014e-07            5.0469e-06            4.3272e-07            nan                   -inf                  1.0849e+04            1.5185e-06            1.5995e-06            1.5690e+00            
7    1.3419e-05            1.2594e-04            1.3698e-05            nan                   -inf                  4.7178e+02            3.8329e-05            3.7148e-05            6.6997e+00            
8    1.7538e-03            2.8956e+00            1.7294e-03            nan                   -inf                  5.5957e+01            3.1000e-03            3.4058e-03            2.5283e+00            
9    4.8054e-04            9.4967e+00            5.0025e-04            nan                   -inf                  4.4212e+02            4.0125e-03            2.0264e-02            2.7051e+00            
10   5.7341e-03            3.7348e+02            1.0590e-02            nan                   -inf                  4.0969e+02            5.8727e+00            5.1344e+00            2.0667e+02            
11   2.1010e+00            1.4053e+03            2.2378e+00            nan                   -inf                  1.6754e+03            3.7615e+00            6.9930e+00            1.5876e+01            
12   7.4971e-01            2.4542e+02            1.2401e+00            nan                   -inf                  2.7237e+02            2.4054e+00            4.6962e+00            2.3029e+02            
13   1.3930e+00            8.4036e+00            1.1932e+00            nan                   -inf                  5.7461e+01            1.4967e+00            1.2391e+00            2.3983e+01            
14   2.6419e+00            1.8324e+01            6.3082e-01            nan                   -inf                  2.0476e+01            8.4103e+00            1.0929e+01            2.5350e+00            
15   1.0318e+00            4.2360e+00            1.1042e+00            nan                   -inf                  4.3474e+00            1.5420e+00            1.9316e+00            4.3008e+00            
16   3.9992e-08            2.1600e+04            3.7590e-04            nan                   -inf                  3.1731e+04            5.7712e-06            3.2524e-02            4.9962e-01            
17   3.1201e+00            3.5033e+00            3.1201e+00            nan                   -inf                  2.7256e+05            3.1201e+00            3.1201e+00            2.7213e+01            
18   2.6757e+00            2.7219e+00            2.6748e+00            nan                   -inf                  9.1905e+01            2.6755e+00            2.6790e+00            4.6557e+01            
19   1.2326e+01            4.9242e+02            2.5105e+00            nan                   -inf                  5.4755e+02            1.0180e+02            1.3002e+02            9.2269e+01            
20   4.4910e+00            6.4658e+01            2.1044e+00            nan                   -inf                  6.9033e+01            6.2713e+00            2.3599e+01            4.9799e+01            
21   4.2319e+00            5.7462e+00            4.2512e+00            nan                   -inf                  5.9674e+00            4.2143e+00            5.2686e+00            4.1394e+00            
22   2.6187e+00            2.1725e+01            2.4530e+00            nan                   -inf                  2.2802e+01            1.0845e+01            1.4670e+01            2.1987e+01            
23   6.3878e-01            1.2510e+02            1.1240e-01            nan                   -inf                  1.4146e+02            1.8278e+01            3.2712e+01            1.6951e+00            

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 21: best error so far = 4.124e+00  (target = 1e-08)
  Task 17: best error so far = 3.120e+00  (target = 1e-08)
  Task 18: best error so far = 2.675e+00  (target = 1e-08)
  Task 19: best error so far = 2.511e+00  (target = 1e-08)
  Task 22: best error so far = 1.477e+00  (target = 1e-08)
  Task 20: best error so far = 1.401e+00  (target = 1e-08)
  Task 15: best error so far = 7.514e-01  (target = 1e-08)
  Task 14: best error so far = 4.530e-01  (target = 1e-08)
  ... and 15 other unsolved task(s).

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0

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
def _adapt_step_size(self):
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp(
            (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)
        )
        self.sigma = np.clip(self.sigma, 1e-20, 1e5)
```

Full algorithm for context:
```python
import numpy as np


class SimplifiedCovarianceAdaptiveSearch:
    """
    A simplified CMA-ES inspired optimizer with adaptive operator selection.
    Uses Thompson Sampling to select the best _sample_population_batch strategy
    during optimization from a portfolio of strategies.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.pop_size = 4 + int(3 * np.log(dim))
        self.pop_size = max(self.pop_size, 2 * dim)
        self.mu = self.pop_size // 2
        self.best_f = np.inf
        self.best_x = None
        self._initialize_strategy_params()

        # Adaptive operator selection: Thompson Sampling
        self.n_operators = 5  # number of sampling strategies
        # Beta distribution parameters for Thompson Sampling
        self.op_alpha = np.ones(self.n_operators) * 1.0
        self.op_beta = np.ones(self.n_operators) * 1.0
        self.current_operator = 0
        # Sliding window for credit assignment
        self.window_size = 20
        self.op_history = []  # list of (operator_idx, reward)
        # Track fitness improvements per operator
        self.op_total_reward = np.zeros(self.n_operators)
        self.op_count = np.zeros(self.n_operators)

        # Archive for variant_06
        self.sampling_archive = []

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
        self.c_mu_cov = min(1.0 - self.c_1,
                            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((dim + 2.0) ** 2 + self.mu_eff))

        self.chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

    def _initialize_state(self):
        dim = self.dim
        self.mean = np.random.uniform(self.lb * 0.5, self.ub * 0.5, size=dim)
        self.sigma = 30.0
        self.C = np.eye(dim)
        self.p_sigma = np.zeros(dim)
        self.p_c = np.zeros(dim)
        self.eigenvalues = np.ones(dim)
        self.eigenvectors = np.eye(dim)
        self.gen_since_eigen = 0
        self.stagnation_counter = 0
        self.prev_best_f = np.inf

    def _decompose_covariance(self):
        try:
            self.C = 0.5 * (self.C + self.C.T)
            eigenvalues, eigenvectors = np.linalg.eigh(self.C)
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
            self.gen_since_eigen = 0
        except np.linalg.LinAlgError:
            self.C = np.eye(self.dim)
            self.eigenvalues = np.ones(self.dim)
            self.eigenvectors = np.eye(self.dim)
            self.gen_since_eigen = 0

    def _select_operator(self):
        """Select operator using Thompson Sampling."""
        samples = np.array([
            np.random.beta(self.op_alpha[i], self.op_beta[i])
            for i in range(self.n_operators)
        ])
        self.current_operator = int(np.argmax(samples))
        return self.current_operator

    def _update_operator_reward(self, operator_idx, reward):
        """Update Thompson Sampling parameters based on reward."""
        reward = float(reward)
        operator_idx = int(operator_idx)

        self.op_history.append((operator_idx, reward))

        # Keep sliding window
        if len(self.op_history) > self.window_size:
            old_op, old_reward = self.op_history.pop(0)
            # Decay old contributions
            self.op_alpha[old_op] = max(1.0, self.op_alpha[old_op] - old_reward)
            self.op_beta[old_op] = max(1.0, self.op_beta[old_op] - (1.0 - old_reward))

        # Update with new reward
        self.op_alpha[operator_idx] += reward
        self.op_beta[operator_idx] += (1.0 - reward)

        # Prevent parameters from growing too large (gentle decay)
        max_param = 50.0
        for i in range(self.n_operators):
            total = self.op_alpha[i] + self.op_beta[i]
            if total > max_param:
                scale = max_param / total
                self.op_alpha[i] = max(1.0, self.op_alpha[i] * scale)
                self.op_beta[i] = max(1.0, self.op_beta[i] * scale)

    # --- Operator 0: Original sampling ---
    def _sample_original(self):
        dim = self.dim
        lam = self.pop_size
        z = np.random.randn(lam, dim)
        sqrt_eig = np.sqrt(self.eigenvalues)
        y = self.eigenvectors @ (sqrt_eig[:, None] * z.T)
        population = self.mean[None, :] + self.sigma * y.T
        return population, z

    # --- Operator 1: Antithetic variates (variant_01) ---
    def _sample_antithetic(self):
        dim = self.dim
        lam = self.pop_size

        if lam % 2 == 0:
            n_pairs = lam // 2
        else:
            n_pairs = (lam - 1) // 2

        z_pairs = np.random.randn(n_pairs, dim)
        z = np.vstack([z_pairs, -z_pairs])

        if lam % 2 == 1:
            z_extra = np.random.randn(1, dim)
            z = np.vstack([z, z_extra])

        sqrt_eig = np.sqrt(self.eigenvalues)
        y = self.eigenvectors @ (sqrt_eig[:, None] * z.T)
        population = self.mean[None, :] + self.sigma * y.T
        return population, z

    # --- Operator 2: Hybrid center (variant_02) ---
    def _sample_hybrid_center(self):
        dim = self.dim
        lam = self.pop_size
        z = np.random.randn(lam, dim)

        if self.best_x is not None and np.isfinite(self.best_f):
            diff_norm = np.linalg.norm(self.mean - self.best_x)
            drift_scale = min(diff_norm / (self.sigma + 1e-20), 1.0)
            w_best = 0.3 * drift_scale
            center = (1.0 - w_best) * self.mean + w_best * self.best_x
        else:
            center = self.mean

        sqrt_eig = np.sqrt(np.maximum(self.eigenvalues, 1e-20))
        y = self.eigenvectors @ (sqrt_eig[:, None] * z.T)
        population = center[None, :] + self.sigma * y.T
        return population, z

    # --- Operator 3: Diversity injection (variant_04) ---
    def _sample_diversity_injection(self):
        dim = self.dim
        lam = self.pop_size

        if self.stagnation_counter > 8:
            random_frac = 0.5
        elif self.stagnation_counter > 4:
            random_frac = 0.3
        elif self.stagnation_counter > 2:
            random_frac = 0.2
        else:
            random_frac = 0.1

        n_random = max(1, int(lam * random_frac))
        n_cma = lam - n_random

        z_all = np.random.randn(lam, dim)
        sqrt_eig = np.sqrt(self.eigenvalues + 1e-20)
        y = self.eigenvectors @ (sqrt_eig[:, None] * z_all.T)
        cma_samples = self.mean[None, :] + self.sigma * y.T

        random_samples = np.random.uniform(self.lb, self.ub, (n_random, dim))

        if n_random > 0 and n_cma > 0:
            population = np.vstack([cma_samples[:n_cma], random_samples])
        elif n_random > 0:
            population = random_samples
        else:
            population = cma_samples

        population = np.clip(population, self.lb, self.ub)
        return population, z_all

    # --- Operator 4: Archive-guided with heavy tails (variant_06) ---
    def _sample_archive_guided(self):
        dim = self.dim
        lam = self.pop_size

        if not hasattr(self, 'sampling_archive'):
            self.sampling_archive = []
        if len(self.sampling_archive) > 0 and len(self.sampling_archive[0]) != dim:
            self.sampling_archive = []

        is_stagnant = self.stagnation_counter > max(3, int(5 * self.dim / self.pop_size))

        n_archive = 0
        if is_stagnant and len(self.sampling_archive) >= 5:
            n_archive = int(lam * np.random.uniform(0.4, 0.7))
            n_archive = min(n_archive, len(self.sampling_archive))

        n_cmaes = lam - n_archive

        population = np.empty((lam, dim))
        z = np.empty((lam, dim))

        if n_cmaes > 0:
            z[:n_cmaes] = np.random.randn(n_cmaes, dim)
            heavy_tail_mask = np.random.rand(n_cmaes) < 0.2
            if np.any(heavy_tail_mask):
                z[:n_cmaes][heavy_tail_mask] = np.random.standard_t(2.0, size=(np.sum(heavy_tail_mask), dim))

            sqrt_eig = np.sqrt(self.eigenvalues)
            y = self.eigenvectors @ (sqrt_eig[:, None] * z[:n_cmaes].T)
            population[:n_cmaes] = self.mean[None, :] + self.sigma * y.T

        if n_archive > 0:
            archive_array = np.array(self.sampling_archive)
            archive_indices = np.random.randint(0, len(archive_array), size=n_archive)
            sampled_from_archive = archive_array[archive_indices]

            exploration_noise = np.random.randn(n_archive, dim)
            heavy_tail_pert = np.random.standard_t(2.0, size=(n_archive, dim))
            combined_noise = 0.7 * exploration_noise + 0.3 * heavy_tail_pert

            sqrt_eig = np.sqrt(self.eigenvalues)
            noise = self.eigenvectors @ (sqrt_eig[:, None] * combined_noise.T)
            population[n_cmaes:] = sampled_from_archive + self.sigma * 2.0 * noise.T

            z[n_cmaes:] = combined_noise

        return population, z

    def _sample_population_batch(self):
        """Adaptively select and use the best sampling strategy."""
        op_idx = self._select_operator()

        if op_idx == 0:
            result = self._sample_original()
        elif op_idx == 1:
            result = self._sample_antithetic()
        elif op_idx == 2:
            result = self._sample_hybrid_center()
        elif op_idx == 3:
            result = self._sample_diversity_injection()
        elif op_idx == 4:
            result = self._sample_archive_guided()
        else:
            result = self._sample_original()

        return result

    def _clip_to_bounds_batch(self, population):
        return np.clip(population, self.lb, self.ub)

    def _sort_by_fitness(self, population, fitness, z_vectors):
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return population, fitness, z_vectors

        sort_fitness = np.where(valid_mask, fitness, np.inf)
        order = np.argsort(sort_fitness)
        return population[order], sort_fitness[order], z_vectors[order]

    def _update_best(self, population, fitness):
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return
        valid_idx = np.where(valid_mask)[0]
        best_idx = valid_idx[np.argmin(fitness[valid_idx])]
        if fitness[best_idx] < self.best_f:
            self.best_f = float(fitness[best_idx])
            self.best_x = population[best_idx].copy()

    def _update_mean(self, sorted_population):
        old_mean = self.mean.copy()
        selected = sorted_population[:self.mu]
        self.mean = self.weights @ selected
        return old_mean

    def _update_evolution_path_sigma(self, old_mean):
        inv_sqrt_eig = 1.0 / np.sqrt(self.eigenvalues + 1e-20)
        inv_sqrt_C = self.eigenvectors @ np.diag(inv_sqrt_eig) @ self.eigenvectors.T

        displacement = (self.mean - old_mean) / self.sigma
        self.p_sigma = ((1.0 - self.c_sigma) * self.p_sigma +
                        np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) *
                        inv_sqrt_C @ displacement)

    def _update_evolution_path_c(self, old_mean):
        dim = self.dim
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        threshold = (1.4 + 2.0 / (dim + 1.0)) * self.chi_n
        h_sigma = 1.0 if p_sigma_norm < threshold else 0.0

        displacement = (self.mean - old_mean) / self.sigma
        self.p_c = ((1.0 - self.c_c) * self.p_c +
                     h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) *
                     displacement)
        return h_sigma

    def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        dim = self.dim
        selected = sorted_population[:self.mu]

        rank1 = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        diff = (selected - old_mean[None, :]) / self.sigma
        rank_mu = np.zeros((dim, dim))
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(diff[i], diff[i])

        self.C = ((1.0 - self.c_1 - self.c_mu_cov + delta_h * self.c_1) * self.C +
                   self.c_1 * rank1 +
                   self.c_mu_cov * rank_mu)

        self.C = 0.5 * (self.C + self.C.T)

    def _adapt_step_size(self):
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp(
            (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)
        )
        self.sigma = np.clip(self.sigma, 1e-20, 1e5)

    def _check_stagnation(self, fitness):
        valid = fitness[np.isfinite(fitness)]
        if len(valid) == 0:
            return False

        current_best = float(np.min(valid))
        improvement = self.prev_best_f - current_best

        if improvement < 1e-12 * (1.0 + abs(self.prev_best_f)):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        self.prev_best_f = min(self.prev_best_f, current_best)

        stagnation_limit = 10 + int(30 * self.dim / self.pop_size)
        return self.stagnation_counter > stagnation_limit

    def _check_condition_number(self):
        ratio = np.max(self.eigenvalues) / (np.min(self.eigenvalues) + 1e-30)
        return ratio > 1e14

    def _restart(self):
        self.pop_size = min(int(self.pop_size * 1.5), 800)
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()
        self._initialize_state()
        if self.best_x is not None and np.random.rand() < 0.3:
            self.mean = self.best_x + np.random.randn(self.dim) * 10.0
            self.mean = np.clip(self.mean, self.lb, self.ub)

    def _handle_truncated_fitness(self, population, fitness, z_vectors):
        n_valid = len(fitness)
        if n_valid < len(population):
            population = population[:n_valid]
            z_vectors = z_vectors[:n_valid]
            fitness = fitness[:n_valid]
        return population, fitness, z_vectors

    def _should_decompose(self):
        self.gen_since_eigen += 1
        decompose_freq = max(1, int(1.0 / (10.0 * self.dim * (self.c_1 + self.c_mu_cov))))
        decompose_freq = min(decompose_freq, self.pop_size)
        return self.gen_since_eigen >= decompose_freq

    def _update_archive(self, sorted_population, sorted_fitness):
        """Update sampling archive with good solutions for variant_06."""
        if not hasattr(self, 'sampling_archive'):
            self.sampling_archive = []

        # Add top mu solutions to archive
        n_add = min(self.mu, len(sorted_population))
        for i in range(n_add):
            if np.isfinite(sorted_fitness[i]):
                self.sampling_archive.append(sorted_population[i].copy())

        # Keep archive bounded
        max_archive = 200
        if len(self.sampling_archive) > max_archive:
            self.sampling_archive = self.sampling_archive[-max_archive:]

    def __call__(self, func, stopping_condition):
        self._initialize_state()
        self._decompose_covariance()
        self.best_f = np.inf
        self.best_x = np.zeros(self.dim)
        self.sampling_archive = []

        # Reset operator selection
        self.op_alpha = np.ones(self.n_operators) * 1.0
        self.op_beta = np.ones(self.n_operators) * 1.0
        self.op_history = []

        # Initial evaluation to seed the best
        initial_pop = self._clip_to_bounds_batch(
            np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        )
        if not stopping_condition():
            initial_fit = func(initial_pop)
            if stopping_condition():
                self._update_best(initial_pop[:len(initial_fit)], initial_fit)
                return float(self.best_f), self.best_x
            self._update_best(initial_pop[:len(initial_fit)], initial_fit)

            valid_mask = np.isfinite(initial_fit)
            if np.any(valid_mask):
                best_idx = np.argmin(np.where(valid_mask, initial_fit, np.inf))
                self.mean = initial_pop[best_idx].copy()

        generation = 0
        while not stopping_condition():
            # Record fitness before this generation for reward computation
            pre_gen_best = float(self.best_f)

            # Sample new population (operator is selected inside)
            population, z_vectors = self._sample_population_batch()
            used_operator = int(self.current_operator)
            population = self._clip_to_bounds_batch(population)

            # Evaluate
            fitness = func(population)

            if stopping_condition():
                population, fitness, z_vectors = self._handle_truncated_fitness(
                    population, fitness, z_vectors
                )
                self._update_best(population, fitness)
                break

            population, fitness, z_vectors = self._handle_truncated_fitness(
                population, fitness, z_vectors
            )
            self._update_best(population, fitness)

            if len(fitness) < self.mu:
                break

            # Compute reward for operator selection
            post_gen_best = float(self.best_f)
            # Reward based on relative improvement
            if pre_gen_best > 0:
                rel_improvement = (pre_gen_best - post_gen_best) / (abs(pre_gen_best) + 1e-20)
            elif pre_gen_best == 0:
                rel_improvement = 0.0
            else:
                rel_improvement = (pre_gen_best - post_gen_best) / (abs(pre_gen_best) + 1e-20)

            # Also consider the quality of the population (median fitness vs best)
            valid_fit = fitness[np.isfinite(fitness)]
            if len(valid_fit) > 0:
                median_fit = float(np.median(valid_fit))
                best_fit = float(np.min(valid_fit))
                # Reward: normalize improvement to [0, 1]
                # Use sigmoid-like mapping
                reward = float(np.clip(rel_improvement * 100.0, 0.0, 1.0))
                # Bonus if we found improvement
                if post_gen_best < pre_gen_best:
                    reward = max(reward, 0.6)
                # Small baseline reward for not crashing
                reward = max(reward, 0.1)
            else:
                reward = 0.0

            self._update_operator_reward(used_operator, reward)

            # Sort by fitness
            sorted_pop, sorted_fit, sorted_z = self._sort_by_fitness(
                population, fitness, z_vectors
            )

            # Update archive for variant_06
            self._update_archive(sorted_pop, sorted_fit)

            # Update distribution
            old_mean = self._update_mean(sorted_pop)
            self._update_evolution_path_sigma(old_mean)
            h_sigma = self._update_evolution_path_c(old_mean)
            self._update_covariance_matrix(old_mean, sorted_pop, h_sigma)
            self._adapt_step_size()

            # Eigendecomposition if needed
            if self._should_decompose():
                self._decompose_covariance()

            # Check for restart conditions
            need_restart = self._check_stagnation(sorted_fit)
            if not need_restart:
                need_restart = self._check_condition_number()

            if need_restart:
                self._restart()
                self._decompose_covariance()

            generation += 1

        return float(self.best_f), self.best_x

```

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description.
```python
def _adapt_step_size(self, ...):
    ...
```