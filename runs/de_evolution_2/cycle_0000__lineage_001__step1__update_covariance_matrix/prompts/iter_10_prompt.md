This is iteration 10 of 10. Propose ONE replacement implementation for `_update_covariance_matrix`.

Requirements:
- Keep the EXACT function signature: `def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):`
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
  Task 16: *** UNSOLVED *** — best=3.031e+02 by variant_07_idea_0.py      (target=1e-08, ~+10.5 decades above target)
  Task 20: *** UNSOLVED *** — best=1.821e+01 by variant_06_idea_0.py      (target=1e-08, ~+9.3 decades above target)
  Task 19: *** UNSOLVED *** — best=1.732e+01 by variant_06_idea_0.py      (target=1e-08, ~+9.2 decades above target)
  Task 23: *** UNSOLVED *** — best=6.297e+00 by variant_07_idea_0.py      (target=1e-08, ~+8.8 decades above target)
  Task 21: *** UNSOLVED *** — best=4.430e+00 by variant_09_idea_0.py      (target=1e-08, ~+8.6 decades above target)
  Task 10: *** UNSOLVED *** — best=3.413e+00 by original.py               (target=1e-08, ~+8.5 decades above target)
  Task 17: *** UNSOLVED *** — best=3.120e+00 by original.py               (target=1e-08, ~+8.5 decades above target)
  Task 18: *** UNSOLVED *** — best=2.674e+00 by variant_07_idea_0.py      (target=1e-08, ~+8.4 decades above target)
  Task 14: *** UNSOLVED *** — best=2.658e+00 by variant_06_idea_0.py      (target=1e-08, ~+8.4 decades above target)
  Task 22: *** UNSOLVED *** — best=2.240e+00 by variant_06_idea_0.py      (target=1e-08, ~+8.4 decades above target)
  Task 12: *** UNSOLVED *** — best=1.350e+00 by variant_05_idea_0.py      (target=1e-08, ~+8.1 decades above target)
  Task 15: *** UNSOLVED *** — best=1.021e+00 by variant_05_idea_0.py      (target=1e-08, ~+8.0 decades above target)
  Task 13: *** UNSOLVED *** — best=9.006e-01 by variant_01_idea_0.py      (target=1e-08, ~+8.0 decades above target)
  Task 11: *** UNSOLVED *** — best=8.944e-01 by variant_01_idea_0.py      (target=1e-08, ~+8.0 decades above target)
  Task  4: *** UNSOLVED *** — best=2.484e-02 by variant_02_idea_0.py      (target=1e-08, ~+6.4 decades above target)
  Task  8: *** UNSOLVED *** — best=1.978e-03 by variant_05_idea_0.py      (target=1e-08, ~+5.3 decades above target)
  Task  9: *** UNSOLVED *** — best=5.491e-04 by variant_02_idea_0.py      (target=1e-08, ~+4.7 decades above target)
  Task  3: *** UNSOLVED *** — best=4.120e-04 by variant_05_idea_0.py      (target=1e-08, ~+4.6 decades above target)
  Task  2: *** UNSOLVED *** — best=2.425e-05 by original.py               (target=1e-08, ~+3.4 decades above target)
  Task  7: *** UNSOLVED *** — best=1.753e-05 by original.py               (target=1e-08, ~+3.2 decades above target)
  Task  6: *** UNSOLVED *** — best=5.370e-07 by variant_02_idea_0.py      (target=1e-08, ~+1.7 decades above target)
  Task  1: *** UNSOLVED *** — best=2.574e-07 by variant_08_idea_0.py      (target=1e-08, ~+1.4 decades above target)
  Task  0: *** UNSOLVED *** — best=1.075e-07 by variant_02_idea_0.py      (target=1e-08, ~+1.0 decades above target)

SOLVED TASKS (already at or below target — do not regress these):
  Task  5: SOLVED  — best=1.000e-08 by original.py                   

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_02_idea_0.py  variant_03_idea_0.py  variant_04_idea_0.py  variant_05_idea_0.py  variant_06_idea_0.py  variant_07_idea_0.py  variant_08_idea_0.py  variant_09_idea_0.py  
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0    1.4202e-07            1.0749e-07            4.2099e+01            5.7060e+00            7.9033e-07            2.8615e-06            1.0871e-07            1.1149e-07            1.8147e-06            
1    2.9404e-07            2.8959e-07            1.0050e+02            7.1995e+00            1.3529e-06            5.3500e-06            2.9882e-07            2.5740e-07            4.1735e-06            
2    2.4248e-05            2.5608e-05            3.8757e+02            6.2181e+00            2.5577e-05            4.2327e-04            2.6936e-05            2.7298e-05            3.9338e-04            
3    4.1539e-04            4.2030e-04            1.9586e+02            2.3361e+01            4.1199e-04            3.1555e-03            4.1578e-04            4.4460e-04            3.1554e-03            
4    2.4900e-02            2.4842e-02            4.5816e+00            1.7794e+00            3.1888e-02            5.7210e-02            2.5482e-02            2.5707e-02            5.4960e-02            
5    1.0000e-08            1.0000e-08            3.9843e+08            3.8821e+02            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            
6    5.3724e-07            5.3702e-07            2.1822e+03            2.8615e+02            5.6126e-07            1.1371e-05            5.7968e-07            1.5296e+01            1.9969e-05            
7    1.7534e-05            1.7611e-05            2.4153e+02            8.7368e+00            2.3487e-05            2.3691e-04            1.8024e-05            1.8489e-05            2.1642e-04            
8    1.9971e-03            2.0204e-03            3.8514e+01            2.4945e+00            1.9785e-03            9.4766e-03            2.0410e-03            2.1102e-03            9.2395e-03            
9    5.8731e-04            5.4912e-04            9.7178e+01            7.1925e+00            3.6870e+00            2.2672e-03            4.0661e+00            6.6382e+00            1.0863e+00            
10   3.4133e+00            5.7312e+00            8.8813e+01            6.1734e+01            5.7701e+00            4.8346e+00            6.0321e+00            6.0055e+00            6.2772e+00            
11   1.7454e+00            4.1851e+00            2.7762e+02            1.4447e+02            2.9913e+00            2.8039e+00            2.2882e+00            4.3572e+00            2.3961e+00            
12   1.3811e+00            2.4607e+00            6.9247e+01            5.0461e+01            1.3504e+00            3.7405e+00            2.1954e+00            2.3634e+00            1.7071e+00            
13   1.3433e+00            1.6420e+00            2.7240e+01            1.1545e+01            1.5547e+00            1.8484e+00            1.8461e+00            1.3526e+00            1.5910e+00            
14   3.9046e+00            3.5866e+00            6.8063e+00            2.7923e+00            3.9547e+00            2.6582e+00            2.7216e+00            3.6229e+00            3.1658e+00            
15   1.0845e+00            1.2800e+00            3.4531e+00            3.2176e+00            1.0209e+00            1.4644e+00            1.2722e+00            1.0784e+00            1.0772e+00            
16   5.9131e+02            3.9262e+02            2.2865e+03            5.2859e+02            5.1650e+02            3.0994e+02            3.0314e+02            6.4237e+02            7.7204e+02            
17   3.1201e+00            3.1201e+00            3.6202e+04            1.0477e+04            3.1201e+00            3.1205e+00            3.1201e+00            3.1201e+00            3.1201e+00            
18   5.4040e+00            1.4328e+01            3.9017e+01            2.7970e+01            1.3580e+01            5.8923e+00            2.6743e+00            1.4089e+01            1.0724e+01            
19   3.1511e+01            3.2226e+01            1.5472e+02            3.1431e+01            2.9015e+01            1.7320e+01            2.2019e+01            3.1122e+01            3.1157e+01            
20   2.3426e+01            2.2249e+01            3.0987e+01            2.5611e+01            2.3634e+01            1.8208e+01            2.0197e+01            2.4103e+01            2.4770e+01            
21   4.5056e+00            4.4656e+00            5.1797e+00            4.6246e+00            4.4743e+00            4.4840e+00            4.4736e+00            4.5211e+00            4.4296e+00            
22   2.3196e+00            2.4541e+00            1.2733e+01            2.9081e+00            2.9124e+00            2.2403e+00            2.4833e+00            2.5117e+00            2.5766e+00            
23   1.9066e+01            1.9157e+01            5.1597e+01            1.9438e+01            2.0691e+01            6.6755e+00            6.2969e+00            2.2969e+01            1.9209e+01            

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 16: best error so far = 3.031e+02  (target = 1e-08)
  Task 20: best error so far = 1.821e+01  (target = 1e-08)
  Task 19: best error so far = 1.732e+01  (target = 1e-08)
  Task 23: best error so far = 6.297e+00  (target = 1e-08)
  Task 21: best error so far = 4.430e+00  (target = 1e-08)
  Task 10: best error so far = 3.413e+00  (target = 1e-08)
  Task 17: best error so far = 3.120e+00  (target = 1e-08)
  Task 18: best error so far = 2.674e+00  (target = 1e-08)
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
def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """
        Update covariance matrix using rank-1 and rank-mu updates.
        """
        # Rank-1 update component
        rank_one = np.outer(self.p_c, self.p_c)

        # Correction for h_sigma
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        # Rank-mu update component
        selected = sorted_population[:self.mu]
        diffs = (selected - old_mean[np.newaxis, :]) / self.sigma  # (mu, dim)
        weighted_diffs = self.weights[:, np.newaxis] * diffs  # (mu, dim)
        rank_mu = diffs.T @ weighted_diffs  # (dim, dim)

        # Combined update
        self.C = (1.0 - self.c_1 - self.c_mu_cov + delta_h * self.c_1) * self.C + \
                 self.c_1 * rank_one + \
                 self.c_mu_cov * rank_mu
```

Full algorithm for context:
```python
import numpy as np


class SimplifiedCovarianceAdaptiveSearch:
    """
    A simplified CMA-ES inspired optimizer with:
    - Covariance matrix adaptation using a rank-1 update
    - Step-size adaptation via cumulative path length control
    - Elite-based mean update
    - Periodic restarts when diversity collapses
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.pop_size = max(4 * dim, 20)
        self.mu = self.pop_size // 2  # number of parents
        self.sigma = 30.0  # initial step size
        self.min_sigma = 1e-12
        self.max_sigma = 100.0

        # Weights for recombination
        self.weights = self._compute_recombination_weights()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        # Learning rates
        self.c_sigma = (self.mu_eff + 2.0) / (dim + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (dim + 1.0)) - 1.0) + self.c_sigma
        self.c_c = (4.0 + self.mu_eff / dim) / (dim + 4.0 + 2.0 * self.mu_eff / dim)
        self.c_1 = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu_cov = min(1.0 - self.c_1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((dim + 2.0) ** 2 + self.mu_eff))

        self.chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

        # State
        self.mean = None
        self.C = None
        self.p_sigma = None
        self.p_c = None
        self.eigen_decomp_gen = 0
        self.B = None
        self.D = None
        self.invsqrtC = None

        self.best_x = None
        self.best_f = np.inf
        self.generation = 0
        self.stagnation_counter = 0
        self.last_best_f = np.inf

    def _compute_recombination_weights(self):
        """Compute log-based recombination weights for the top mu individuals."""
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights = raw_weights / np.sum(raw_weights)
        return weights

    def _initialize_state(self):
        """Initialize or reset the distribution parameters (mean, covariance, paths)."""
        self.mean = np.random.uniform(self.lb * 0.5, self.ub * 0.5, size=self.dim)
        self.C = np.eye(self.dim)
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)
        self.sigma = 30.0
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.invsqrtC = np.eye(self.dim)
        self.eigen_decomp_gen = 0
        self.stagnation_counter = 0
        self.last_best_f = np.inf

    def _update_eigen_decomposition(self):
        """Perform eigendecomposition of the covariance matrix C for sampling."""
        # Symmetrize C to avoid numerical issues
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        eigenvalues, self.B = np.linalg.eigh(self.C)
        # Clamp eigenvalues to avoid negative values from numerical errors
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        self.D = np.sqrt(eigenvalues)
        inv_D = 1.0 / self.D
        self.invsqrtC = self.B @ np.diag(inv_D) @ self.B.T
        self.eigen_decomp_gen = self.generation

    def _sample_population_batch(self):
        """
        Sample a batch of candidates from the multivariate normal distribution
        N(mean, sigma^2 * C). Returns array of shape (pop_size, dim).
        """
        # Sample standard normals and transform
        z = np.random.randn(self.pop_size, self.dim)
        # Transform: x = mean + sigma * B * D * z
        scaled = z * self.D[np.newaxis, :]  # (pop_size, dim)
        rotated = scaled @ self.B.T  # (pop_size, dim)
        population = self.mean[np.newaxis, :] + self.sigma * rotated
        return population, z

    def _clip_to_bounds_batch(self, population):
        """Clip all candidates to the search bounds [-100, 100]^dim."""
        return np.clip(population, self.lb, self.ub)

    def _sort_by_fitness(self, population, fitness, z_vectors):
        """
        Sort population by fitness (ascending = better) and return sorted arrays.
        Handles NaN by assigning them worst possible fitness.
        """
        safe_fitness = np.where(np.isnan(fitness), np.inf, fitness)
        sorted_indices = np.argsort(safe_fitness)
        return population[sorted_indices], safe_fitness[sorted_indices], z_vectors[sorted_indices]

    def _update_mean(self, sorted_population):
        """
        Update the distribution mean using weighted recombination of the best mu individuals.
        """
        old_mean = self.mean.copy()
        selected = sorted_population[:self.mu]  # (mu, dim)
        self.mean = self.weights @ selected  # (dim,)
        return old_mean

    def _update_evolution_paths(self, old_mean):
        """
        Update the cumulative step-size path (p_sigma) and the covariance path (p_c).
        """
        mean_shift = (self.mean - old_mean) / self.sigma
        transformed = self.invsqrtC @ mean_shift

        # Update step-size path
        c_s_complement = np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + \
                       np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * transformed

        # Heaviside function for p_c update
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        threshold = (1.4 + 2.0 / (self.dim + 1.0)) * self.chi_n * c_s_complement
        h_sigma = 1.0 if p_sigma_norm < threshold else 0.0

        # Update covariance path
        self.p_c = (1.0 - self.c_c) * self.p_c + \
                   h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * mean_shift

        return h_sigma

    def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """
        Update covariance matrix using rank-1 and rank-mu updates.
        """
        # Rank-1 update component
        rank_one = np.outer(self.p_c, self.p_c)

        # Correction for h_sigma
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        # Rank-mu update component
        selected = sorted_population[:self.mu]
        diffs = (selected - old_mean[np.newaxis, :]) / self.sigma  # (mu, dim)
        weighted_diffs = self.weights[:, np.newaxis] * diffs  # (mu, dim)
        rank_mu = diffs.T @ weighted_diffs  # (dim, dim)

        # Combined update
        self.C = (1.0 - self.c_1 - self.c_mu_cov + delta_h * self.c_1) * self.C + \
                 self.c_1 * rank_one + \
                 self.c_mu_cov * rank_mu

    def _update_step_size(self):
        """Adapt the global step size sigma using cumulative step-size adaptation (CSA)."""
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0))
        self.sigma = np.clip(self.sigma, self.min_sigma, self.max_sigma)

    def _track_best(self, population, fitness):
        """Update the global best solution found so far, ignoring NaN entries."""
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return
        valid_fitness = fitness[valid_mask]
        valid_pop = population[valid_mask]
        local_best_idx = np.argmin(valid_fitness)
        if valid_fitness[local_best_idx] < self.best_f:
            self.best_f = valid_fitness[local_best_idx]
            self.best_x = valid_pop[local_best_idx].copy()

    def _check_stagnation(self, best_gen_fitness):
        """
        Detect if the search has stagnated and trigger a restart if needed.
        Returns True if a restart was performed.
        """
        improvement = self.last_best_f - best_gen_fitness
        if improvement < 1e-12 * (1.0 + abs(self.last_best_f)):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        self.last_best_f = min(self.last_best_f, best_gen_fitness)

        # Check for various restart conditions
        should_restart = False
        if self.stagnation_counter > 20 + self.dim:
            should_restart = True
        if self.sigma < self.min_sigma * 10:
            should_restart = True
        if np.max(self.D) / np.min(self.D) > 1e7:
            should_restart = True

        if should_restart:
            self._perform_restart()
            return True
        return False

    def _perform_restart(self):
        """
        Restart the search with a new random mean, reset covariance and paths,
        but keep the global best tracked.
        """
        old_best_x = self.best_x.copy() if self.best_x is not None else None
        old_best_f = self.best_f

        self._initialize_state()

        # If we have a known best, bias the new mean slightly towards it
        if old_best_x is not None:
            blend = np.random.uniform(0.0, 0.3)
            self.mean = blend * old_best_x + (1.0 - blend) * self.mean
            self.mean = np.clip(self.mean, self.lb, self.ub)

        self.best_x = old_best_x
        self.best_f = old_best_f

    def _should_update_eigen(self):
        """Determine if eigendecomposition should be recomputed this generation."""
        update_interval = max(1, int(1.0 / (10.0 * self.dim * (self.c_1 + self.c_mu_cov))))
        return (self.generation - self.eigen_decomp_gen) >= update_interval

    def _handle_truncated_fitness(self, population, fitness, z_vectors):
        """
        If func returned fewer values than expected (budget exhaustion),
        truncate population and z_vectors to match.
        """
        n_valid = len(fitness)
        return population[:n_valid], fitness[:n_valid], z_vectors[:n_valid]

    def _inject_best_into_population(self, population, z_vectors):
        """
        Occasionally inject the global best solution into the population to
        preserve elitism across restarts.
        """
        if self.best_x is not None and self.generation % 5 == 0:
            population[-1] = self.best_x.copy()
            # Approximate z_vector (not exact but acceptable)
            z_vectors[-1] = np.zeros(self.dim)
        return population, z_vectors

    def __call__(self, func, stopping_condition):
        """
        Main optimization loop: sample, evaluate, sort, update distribution.
        """
        self._initialize_state()
        self._update_eigen_decomposition()

        # Initial evaluation of a small seed population around mean
        seed_pop = self._clip_to_bounds_batch(
            self.mean[np.newaxis, :] + self.sigma * np.random.randn(self.pop_size, self.dim)
        )
        seed_fit = func(seed_pop)
        if stopping_condition():
            self._track_best(seed_pop, seed_fit[:len(seed_pop)])
            return self.best_f, self.best_x

        self._track_best(seed_pop, seed_fit)

        while not stopping_condition():
            # Eigen decomposition update
            if self._should_update_eigen():
                try:
                    self._update_eigen_decomposition()
                except np.linalg.LinAlgError:
                    self._perform_restart()
                    self._update_eigen_decomposition()

            # Sample new population
            population, z_vectors = self._sample_population_batch()
            population = self._clip_to_bounds_batch(population)

            # Inject best occasionally
            population, z_vectors = self._inject_best_into_population(population, z_vectors)
            population = self._clip_to_bounds_batch(population)

            # Evaluate
            fitness = func(population)
            if stopping_condition():
                if len(fitness) > 0:
                    population, fitness, z_vectors = self._handle_truncated_fitness(
                        population, fitness, z_vectors
                    )
                    self._track_best(population, fitness)
                break

            # Handle truncated returns
            if len(fitness) < len(population):
                population, fitness, z_vectors = self._handle_truncated_fitness(
                    population, fitness, z_vectors
                )
                self._track_best(population, fitness)
                break

            # Track best
            self._track_best(population, fitness)

            # Sort by fitness
            sorted_pop, sorted_fit, sorted_z = self._sort_by_fitness(population, fitness, z_vectors)

            # Need at least mu valid individuals
            valid_count = np.sum(np.isfinite(sorted_fit))
            if valid_count < self.mu:
                self.generation += 1
                continue

            # Update mean
            old_mean = self._update_mean(sorted_pop)

            # Update evolution paths
            h_sigma = self._update_evolution_paths(old_mean)

            # Update covariance matrix
            self._update_covariance_matrix(old_mean, sorted_pop, h_sigma)

            # Update step size
            self._update_step_size()

            # Check stagnation
            best_gen_f = sorted_fit[0]
            if np.isfinite(best_gen_f):
                restarted = self._check_stagnation(best_gen_f)
                if restarted:
                    self._update_eigen_decomposition()

            self.generation += 1

        return self.best_f, self.best_x

```

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description.
```python
def _update_covariance_matrix(self, ...):
    ...
```