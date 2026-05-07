This is iteration 7 of 10. Propose ONE replacement implementation for `_sample_population_batch`.

Requirements:
- Keep the EXACT function signature: `def _sample_population_batch(self):`
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
  Task 16: *** UNSOLVED *** — best=3.263e+01 by variant_02_idea_0.py      (target=1e-08, ~+9.5 decades above target)
  Task 19: *** UNSOLVED *** — best=6.190e+00 by variant_04_idea_0.py      (target=1e-08, ~+8.8 decades above target)
  Task 23: *** UNSOLVED *** — best=4.867e+00 by variant_06_idea_0.py      (target=1e-08, ~+8.7 decades above target)
  Task 21: *** UNSOLVED *** — best=4.197e+00 by variant_06_idea_0.py      (target=1e-08, ~+8.6 decades above target)
  Task 17: *** UNSOLVED *** — best=3.120e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.5 decades above target)
  Task 18: *** UNSOLVED *** — best=2.674e+00 by original.py               (target=1e-08, ~+8.4 decades above target)
  Task 22: *** UNSOLVED *** — best=2.119e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.3 decades above target)
  Task 20: *** UNSOLVED *** — best=1.950e+00 by variant_02_idea_0.py      (target=1e-08, ~+8.3 decades above target)
  Task 14: *** UNSOLVED *** — best=1.621e+00 by variant_02_idea_0.py      (target=1e-08, ~+8.2 decades above target)
  Task 13: *** UNSOLVED *** — best=8.968e-01 by variant_05_idea_0.py      (target=1e-08, ~+8.0 decades above target)
  Task 15: *** UNSOLVED *** — best=8.803e-01 by variant_01_idea_0.py      (target=1e-08, ~+7.9 decades above target)
  Task  4: *** UNSOLVED *** — best=2.321e-02 by variant_01_idea_0.py      (target=1e-08, ~+6.4 decades above target)
  Task 12: *** UNSOLVED *** — best=4.525e-03 by variant_02_idea_0.py      (target=1e-08, ~+5.7 decades above target)
  Task  8: *** UNSOLVED *** — best=1.657e-03 by variant_01_idea_0.py      (target=1e-08, ~+5.2 decades above target)
  Task 11: *** UNSOLVED *** — best=1.074e-03 by variant_02_idea_0.py      (target=1e-08, ~+5.0 decades above target)
  Task  9: *** UNSOLVED *** — best=4.552e-04 by variant_01_idea_0.py      (target=1e-08, ~+4.7 decades above target)
  Task  3: *** UNSOLVED *** — best=3.176e-04 by variant_01_idea_0.py      (target=1e-08, ~+4.5 decades above target)
  Task  2: *** UNSOLVED *** — best=1.966e-05 by variant_01_idea_0.py      (target=1e-08, ~+3.3 decades above target)
  Task  7: *** UNSOLVED *** — best=1.251e-05 by variant_01_idea_0.py      (target=1e-08, ~+3.1 decades above target)
  Task 10: *** UNSOLVED *** — best=4.128e-06 by variant_02_idea_0.py      (target=1e-08, ~+2.6 decades above target)
  Task  6: *** UNSOLVED *** — best=3.660e-07 by variant_01_idea_0.py      (target=1e-08, ~+1.6 decades above target)
  Task  1: *** UNSOLVED *** — best=2.168e-07 by variant_01_idea_0.py      (target=1e-08, ~+1.3 decades above target)
  Task  0: *** UNSOLVED *** — best=8.081e-08 by variant_01_idea_0.py      (target=1e-08, ~+0.9 decades above target)

SOLVED TASKS (already at or below target — do not regress these):
  Task  5: SOLVED  — best=1.000e-08 by original.py                   

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_idea_0.py  variant_02_idea_0.py  variant_03_idea_0.py  variant_04_idea_0.py  variant_05_idea_0.py  variant_06_idea_0.py  
---------------------------------------------------------------------------------------------------------------------------------------------------------------
0    8.1436e-08            8.0809e-08            8.7381e-08            5.2230e+01            9.7056e-08            1.1632e-07            8.5335e-08            
1    2.3315e-07            2.1678e-07            2.3082e-07            1.4023e-06            2.7028e-07            3.1511e-07            2.4591e-07            
2    2.0810e-05            1.9660e-05            2.1979e-05            8.1555e+02            2.4012e-05            2.8497e-05            2.2053e-05            
3    3.3685e-04            3.1757e-04            3.3417e-04            3.2914e+02            3.5596e-04            4.0099e-04            3.5379e-04            
4    2.3586e-02            2.3207e-02            2.3578e-02            5.8456e+00            2.4495e-02            2.5887e-02            2.4067e-02            
5    1.0000e-08            1.0000e-08            1.0000e-08            6.2384e+09            1.0000e-08            1.0000e-08            1.0000e-08            
6    4.0021e-07            3.6598e-07            3.9082e-07            1.2414e+04            4.6562e-07            5.4720e-07            4.2395e-07            
7    1.3553e-05            1.2509e-05            1.3868e-05            4.6128e+02            1.4680e-05            1.6606e-05            1.4367e-05            
8    1.7085e-03            1.6574e-03            1.7050e-03            8.3231e-03            1.8240e-03            1.9219e-03            1.7536e-03            
9    4.8725e-04            4.5519e-04            4.8861e-04            5.0149e+02            5.0897e-04            5.7208e-04            4.9782e-04            
10   3.4615e+00            4.4616e+00            4.1280e-06            4.7233e+02            2.9701e+00            4.0656e+00            2.6747e+00            
11   7.1382e-01            1.0186e+00            1.0737e-03            1.9809e+03            5.2647e-01            1.6597e+00            7.9386e-01            
12   4.3261e-01            5.3718e-01            4.5254e-03            3.3080e+02            8.8262e-01            2.0425e+00            4.9449e-01            
13   9.5956e-01            1.2512e+00            1.0872e+00            6.0149e+01            9.3923e-01            8.9684e-01            9.9094e-01            
14   2.5480e+00            3.2455e+00            1.6210e+00            2.1879e+01            2.5008e+00            2.6044e+00            2.4336e+00            
15   9.2414e-01            8.8029e-01            9.5611e-01            4.3789e+00            9.1110e-01            1.0501e+00            1.0318e+00            
16   5.0963e+01            2.9985e+02            3.2632e+01            4.3540e+04            5.8590e+01            1.6535e+02            5.6250e+01            
17   3.1201e+00            3.1201e+00            3.1201e+00            2.6685e+05            3.1201e+00            3.1201e+00            3.1201e+00            
18   2.6744e+00            2.6747e+00            2.6748e+00            9.6752e+01            2.6747e+00            2.6754e+00            2.6746e+00            
19   2.3549e+01            2.5581e+01            1.1935e+01            6.1936e+02            6.1898e+00            6.5463e+00            6.6988e+00            
20   5.6416e+00            1.7471e+01            1.9502e+00            7.5617e+01            3.8651e+00            6.4256e+00            3.2334e+00            
21   4.4063e+00            4.3448e+00            4.2277e+00            6.0641e+00            4.2041e+00            4.3810e+00            4.1969e+00            
22   2.1422e+00            2.1186e+00            2.3894e+00            2.4110e+01            2.1771e+00            2.2613e+00            2.2135e+00            
23   5.6567e+00            2.2378e+01            5.1330e+00            1.6305e+02            5.5723e+00            6.4694e+00            4.8667e+00            

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 16: best error so far = 3.263e+01  (target = 1e-08)
  Task 19: best error so far = 6.190e+00  (target = 1e-08)
  Task 23: best error so far = 4.867e+00  (target = 1e-08)
  Task 21: best error so far = 4.197e+00  (target = 1e-08)
  Task 17: best error so far = 3.120e+00  (target = 1e-08)
  Task 18: best error so far = 2.674e+00  (target = 1e-08)
  Task 22: best error so far = 2.119e+00  (target = 1e-08)
  Task 20: best error so far = 1.950e+00  (target = 1e-08)
  ... and 15 other unsolved task(s).

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0, Idea 0, Idea 0, Idea 0, Idea 0

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
def _sample_population_batch(self):
        """Sample a new population from the current distribution N(mean, sigma^2 * C)."""
        dim = self.dim
        lam = self.pop_size

        # Sample standard normal
        z = np.random.randn(lam, dim)

        # Transform: x = mean + sigma * B * D * z
        sqrt_eig = np.sqrt(self.eigenvalues)
        # y = B * D * z^T => shape (dim, lam)
        y = self.eigenvectors @ (sqrt_eig[:, None] * z.T)
        # population = mean + sigma * y^T
        population = self.mean[None, :] + self.sigma * y.T

        return population, z
```

Full algorithm for context:
```python
import numpy as np


class SimplifiedCovarianceAdaptiveSearch:
    """
    A simplified CMA-ES inspired optimizer with:
    - Covariance matrix adaptation using a rank-1 + rank-mu update
    - Step-size adaptation via cumulative path length control
    - Elite-based mean update with rank-weighted recombination
    - Periodic restarts on stagnation detection
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.pop_size = 4 + int(3 * np.log(dim))  # CMA-ES lambda
        self.pop_size = max(self.pop_size, 2 * dim)
        self.mu = self.pop_size // 2  # number of parents
        self.best_f = np.inf
        self.best_x = None
        self._initialize_strategy_params()

    def _initialize_strategy_params(self):
        """Initialize all strategy parameters for covariance adaptation."""
        dim = self.dim
        mu = self.mu
        lam = self.pop_size

        # Recombination weights (log-linear)
        raw_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        # Step-size adaptation parameters
        self.c_sigma = (self.mu_eff + 2.0) / (dim + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (dim + 1.0)) - 1.0) + self.c_sigma

        # Covariance adaptation parameters
        self.c_c = (4.0 + self.mu_eff / dim) / (dim + 4.0 + 2.0 * self.mu_eff / dim)
        self.c_1 = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu_cov = min(1.0 - self.c_1,
                            2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((dim + 2.0) ** 2 + self.mu_eff))

        # Expected norm of N(0,I)
        self.chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

    def _initialize_state(self):
        """Initialize the evolutionary state: mean, covariance, paths, step-size."""
        dim = self.dim
        self.mean = np.random.uniform(self.lb * 0.5, self.ub * 0.5, size=dim)
        self.sigma = 30.0  # initial step size
        self.C = np.eye(dim)  # covariance matrix
        self.p_sigma = np.zeros(dim)  # evolution path for sigma
        self.p_c = np.zeros(dim)  # evolution path for C
        self.eigenvalues = np.ones(dim)
        self.eigenvectors = np.eye(dim)
        self.gen_since_eigen = 0
        self.stagnation_counter = 0
        self.prev_best_f = np.inf

    def _decompose_covariance(self):
        """Eigendecompose the covariance matrix for sampling."""
        try:
            self.C = 0.5 * (self.C + self.C.T)  # enforce symmetry
            eigenvalues, eigenvectors = np.linalg.eigh(self.C)
            # Clamp eigenvalues to avoid numerical issues
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            self.eigenvalues = eigenvalues
            self.eigenvectors = eigenvectors
            self.gen_since_eigen = 0
        except np.linalg.LinAlgError:
            # Reset covariance if decomposition fails
            self.C = np.eye(self.dim)
            self.eigenvalues = np.ones(self.dim)
            self.eigenvectors = np.eye(self.dim)
            self.gen_since_eigen = 0

    def _sample_population_batch(self):
        """Sample a new population from the current distribution N(mean, sigma^2 * C)."""
        dim = self.dim
        lam = self.pop_size

        # Sample standard normal
        z = np.random.randn(lam, dim)

        # Transform: x = mean + sigma * B * D * z
        sqrt_eig = np.sqrt(self.eigenvalues)
        # y = B * D * z^T => shape (dim, lam)
        y = self.eigenvectors @ (sqrt_eig[:, None] * z.T)
        # population = mean + sigma * y^T
        population = self.mean[None, :] + self.sigma * y.T

        return population, z

    def _clip_to_bounds_batch(self, population):
        """Clip all candidates to the search bounds."""
        return np.clip(population, self.lb, self.ub)

    def _sort_by_fitness(self, population, fitness, z_vectors):
        """Sort population by fitness (ascending = better) and return sorted arrays."""
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return population, fitness, z_vectors

        # Put invalid entries at the end with inf fitness
        sort_fitness = np.where(valid_mask, fitness, np.inf)
        order = np.argsort(sort_fitness)
        return population[order], sort_fitness[order], z_vectors[order]

    def _update_best(self, population, fitness):
        """Track the global best solution found so far."""
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return
        valid_idx = np.where(valid_mask)[0]
        best_idx = valid_idx[np.argmin(fitness[valid_idx])]
        if fitness[best_idx] < self.best_f:
            self.best_f = fitness[best_idx]
            self.best_x = population[best_idx].copy()

    def _update_mean(self, sorted_population):
        """Update the distribution mean using weighted recombination of the mu best."""
        old_mean = self.mean.copy()
        selected = sorted_population[:self.mu]  # (mu, dim)
        self.mean = self.weights @ selected  # weighted sum
        return old_mean

    def _update_evolution_path_sigma(self, old_mean):
        """Update the cumulative step-size adaptation path."""
        dim = self.dim
        # Inverse square root of C: B * D^{-1} * B^T
        inv_sqrt_eig = 1.0 / np.sqrt(self.eigenvalues + 1e-20)
        inv_sqrt_C = self.eigenvectors @ np.diag(inv_sqrt_eig) @ self.eigenvectors.T

        displacement = (self.mean - old_mean) / self.sigma
        self.p_sigma = ((1.0 - self.c_sigma) * self.p_sigma +
                        np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) *
                        inv_sqrt_C @ displacement)

    def _update_evolution_path_c(self, old_mean):
        """Update the cumulative covariance adaptation path."""
        dim = self.dim
        # Check if step-size path is not too long (h_sigma)
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        threshold = (1.4 + 2.0 / (dim + 1.0)) * self.chi_n
        h_sigma = 1.0 if p_sigma_norm < threshold else 0.0

        displacement = (self.mean - old_mean) / self.sigma
        self.p_c = ((1.0 - self.c_c) * self.p_c +
                     h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) *
                     displacement)
        return h_sigma

    def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """Update the covariance matrix with rank-1 and rank-mu updates."""
        dim = self.dim
        selected = sorted_population[:self.mu]

        # Rank-1 update component
        rank1 = np.outer(self.p_c, self.p_c)

        # Correction for h_sigma
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        # Rank-mu update component
        diff = (selected - old_mean[None, :]) / self.sigma  # (mu, dim)
        rank_mu = np.zeros((dim, dim))
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(diff[i], diff[i])

        # Combined update
        self.C = ((1.0 - self.c_1 - self.c_mu_cov + delta_h * self.c_1) * self.C +
                   self.c_1 * rank1 +
                   self.c_mu_cov * rank_mu)

        # Enforce symmetry
        self.C = 0.5 * (self.C + self.C.T)

    def _adapt_step_size(self):
        """Adapt the global step size sigma using cumulative path length control."""
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp(
            (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)
        )
        # Clamp sigma to reasonable range
        self.sigma = np.clip(self.sigma, 1e-20, 1e5)

    def _check_stagnation(self, fitness):
        """Detect stagnation and trigger restart if needed."""
        valid = fitness[np.isfinite(fitness)]
        if len(valid) == 0:
            return False

        current_best = np.min(valid)
        improvement = self.prev_best_f - current_best

        if improvement < 1e-12 * (1.0 + abs(self.prev_best_f)):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        self.prev_best_f = min(self.prev_best_f, current_best)

        # Stagnation threshold: restart after many generations without improvement
        stagnation_limit = 10 + int(30 * self.dim / self.pop_size)
        return self.stagnation_counter > stagnation_limit

    def _check_condition_number(self):
        """Check if covariance matrix condition number is too high."""
        ratio = np.max(self.eigenvalues) / (np.min(self.eigenvalues) + 1e-30)
        return ratio > 1e14

    def _restart(self):
        """Perform a restart with potentially increased population size."""
        # Increase pop size for IPOP-like restarts
        self.pop_size = min(int(self.pop_size * 1.5), 800)
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()
        self._initialize_state()
        # Bias restart toward best known solution with some probability
        if self.best_x is not None and np.random.rand() < 0.3:
            self.mean = self.best_x + np.random.randn(self.dim) * 10.0
            self.mean = np.clip(self.mean, self.lb, self.ub)

    def _handle_truncated_fitness(self, population, fitness, z_vectors):
        """Handle case where func returns fewer values than expected."""
        n_valid = len(fitness)
        if n_valid < len(population):
            population = population[:n_valid]
            z_vectors = z_vectors[:n_valid]
            fitness = fitness[:n_valid]
        return population, fitness, z_vectors

    def _should_decompose(self):
        """Determine if eigendecomposition is needed this generation."""
        self.gen_since_eigen += 1
        # Decompose every few generations to save compute
        decompose_freq = max(1, int(1.0 / (10.0 * self.dim * (self.c_1 + self.c_mu_cov))))
        decompose_freq = min(decompose_freq, self.pop_size)
        return self.gen_since_eigen >= decompose_freq

    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        self._initialize_state()
        self._decompose_covariance()
        self.best_f = np.inf
        self.best_x = np.zeros(self.dim)

        # Initial evaluation to seed the best
        initial_pop = self._clip_to_bounds_batch(
            np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        )
        if not stopping_condition():
            initial_fit = func(initial_pop)
            if stopping_condition():
                self._update_best(initial_pop[:len(initial_fit)], initial_fit)
                return self.best_f, self.best_x
            self._update_best(initial_pop[:len(initial_fit)], initial_fit)

            # Set mean to best initial point
            valid_mask = np.isfinite(initial_fit)
            if np.any(valid_mask):
                best_idx = np.argmin(np.where(valid_mask, initial_fit, np.inf))
                self.mean = initial_pop[best_idx].copy()

        generation = 0
        while not stopping_condition():
            # Sample new population
            population, z_vectors = self._sample_population_batch()
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

            # Sort by fitness
            sorted_pop, sorted_fit, sorted_z = self._sort_by_fitness(
                population, fitness, z_vectors
            )

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

        return self.best_f, self.best_x

```

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description.
```python
def _sample_population_batch(self, ...):
    ...
```