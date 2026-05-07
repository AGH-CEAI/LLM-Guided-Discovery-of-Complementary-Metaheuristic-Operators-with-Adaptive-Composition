This is iteration 3 of 10. Propose ONE replacement implementation for `_clip_to_bounds_batch`.

Requirements:
- Keep the EXACT function signature: `def _clip_to_bounds_batch(self, population):`
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
  Task 16: *** UNSOLVED *** — best=4.362e+02 by variant_01_idea_0.py      (target=1e-08, ~+10.6 decades above target)
  Task 20: *** UNSOLVED *** — best=2.272e+01 by variant_01_idea_0.py      (target=1e-08, ~+9.4 decades above target)
  Task 19: *** UNSOLVED *** — best=2.267e+01 by variant_02_idea_0.py      (target=1e-08, ~+9.4 decades above target)
  Task 23: *** UNSOLVED *** — best=1.727e+01 by variant_02_idea_0.py      (target=1e-08, ~+9.2 decades above target)
  Task 18: *** UNSOLVED *** — best=5.247e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.7 decades above target)
  Task 10: *** UNSOLVED *** — best=4.503e+00 by variant_02_idea_0.py      (target=1e-08, ~+8.7 decades above target)
  Task 21: *** UNSOLVED *** — best=4.485e+00 by variant_02_idea_0.py      (target=1e-08, ~+8.7 decades above target)
  Task 14: *** UNSOLVED *** — best=3.467e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.5 decades above target)
  Task 17: *** UNSOLVED *** — best=3.120e+00 by original.py               (target=1e-08, ~+8.5 decades above target)
  Task 11: *** UNSOLVED *** — best=2.461e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.4 decades above target)
  Task 22: *** UNSOLVED *** — best=2.252e+00 by variant_02_idea_0.py      (target=1e-08, ~+8.4 decades above target)
  Task 12: *** UNSOLVED *** — best=1.454e+00 by original.py               (target=1e-08, ~+8.2 decades above target)
  Task 13: *** UNSOLVED *** — best=1.244e+00 by original.py               (target=1e-08, ~+8.1 decades above target)
  Task 15: *** UNSOLVED *** — best=1.077e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.0 decades above target)
  Task  4: *** UNSOLVED *** — best=2.465e-02 by variant_02_idea_0.py      (target=1e-08, ~+6.4 decades above target)
  Task  8: *** UNSOLVED *** — best=1.957e-03 by variant_01_idea_0.py      (target=1e-08, ~+5.3 decades above target)
  Task  9: *** UNSOLVED *** — best=5.719e-04 by original.py               (target=1e-08, ~+4.8 decades above target)
  Task  3: *** UNSOLVED *** — best=4.032e-04 by original.py               (target=1e-08, ~+4.6 decades above target)
  Task  2: *** UNSOLVED *** — best=2.383e-05 by original.py               (target=1e-08, ~+3.4 decades above target)
  Task  7: *** UNSOLVED *** — best=1.719e-05 by original.py               (target=1e-08, ~+3.2 decades above target)
  Task  6: *** UNSOLVED *** — best=5.265e-07 by variant_01_idea_0.py      (target=1e-08, ~+1.7 decades above target)
  Task  1: *** UNSOLVED *** — best=2.835e-07 by original.py               (target=1e-08, ~+1.5 decades above target)
  Task  0: *** UNSOLVED *** — best=1.316e-07 by variant_01_idea_0.py      (target=1e-08, ~+1.1 decades above target)

SOLVED TASKS (already at or below target — do not regress these):
  Task  5: SOLVED  — best=1.000e-08 by original.py                   

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_idea_0.py  variant_02_idea_0.py  
-----------------------------------------------------------------------
0    1.3209e-07            1.3162e-07            1.3755e-07            
1    2.8355e-07            2.8720e-07            2.8685e-07            
2    2.3833e-05            2.4186e-05            2.4273e-05            
3    4.0321e-04            4.1700e-04            4.1081e-04            
4    2.4851e-02            2.4714e-02            2.4654e-02            
5    1.0000e-08            1.0000e-08            1.0000e-08            
6    5.2845e-07            5.2654e-07            5.3910e-07            
7    1.7186e-05            1.7661e-05            1.7201e-05            
8    2.0050e-03            1.9572e-03            1.9726e-03            
9    5.7186e-04            5.9539e-04            5.8700e-04            
10   6.3931e+00            5.6099e+00            4.5034e+00            
11   2.7706e+00            2.4613e+00            3.1212e+00            
12   1.4540e+00            1.8944e+00            2.0811e+00            
13   1.2442e+00            1.2627e+00            1.9721e+00            
14   3.4699e+00            3.4674e+00            3.6450e+00            
15   1.1184e+00            1.0766e+00            1.0775e+00            
16   5.0533e+02            4.3615e+02            5.5326e+02            
17   3.1201e+00            3.1201e+00            3.1201e+00            
18   5.4202e+00            5.2469e+00            5.4115e+00            
19   3.0083e+01            3.2745e+01            2.2672e+01            
20   2.4654e+01            2.2720e+01            2.3190e+01            
21   4.4960e+00            4.4985e+00            4.4850e+00            
22   2.3574e+00            2.3175e+00            2.2516e+00            
23   1.9129e+01            2.0998e+01            1.7274e+01            

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 16: best error so far = 4.362e+02  (target = 1e-08)
  Task 20: best error so far = 2.272e+01  (target = 1e-08)
  Task 19: best error so far = 2.267e+01  (target = 1e-08)
  Task 23: best error so far = 1.727e+01  (target = 1e-08)
  Task 18: best error so far = 5.247e+00  (target = 1e-08)
  Task 10: best error so far = 4.503e+00  (target = 1e-08)
  Task 21: best error so far = 4.485e+00  (target = 1e-08)
  Task 14: best error so far = 3.467e+00  (target = 1e-08)
  ... and 15 other unsolved task(s).

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0

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
def _clip_to_bounds_batch(self, population):
        """Clip all candidates to the search bounds [-100, 100]^dim."""
        return np.clip(population, self.lb, self.ub)
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
            Sample a batch of candidates with archive-based restart diversification.
            Maintains an archive of best solutions across restarts to prevent getting
            trapped in local optima by sampling from diverse historical solutions.
            """
            # Initialize archive if first call
            if not hasattr(self, 'archive') or self.archive is None:
                self.archive = []

            # Check if we should force restart diversity sampling
            force_archive_sampling = (self.stagnation_counter > self.dim)

            z = np.random.randn(self.pop_size, self.dim)

            # If using archive sampling, bias toward diverse historical solutions
            if force_archive_sampling and len(self.archive) > 0:
                archive_size = len(self.archive)
                # Sample a portion from archive (proportional to stagnation severity)
                archive_portion = min(len(self.archive), max(1, self.pop_size // 2))
                archive_indices = np.random.choice(archive_size, archive_portion, replace=False)

                for i, idx in enumerate(archive_indices[:archive_portion]):
                    if i < self.pop_size:
                        # Sample from neighborhood of archived solution
                        archive_solution = np.array(self.archive[idx])
                        # Use larger variance for exploration
                        z[i] = np.random.randn(self.dim) * 3.0

            # Ensure D and B are valid
            if self.D is None or self.B is None or np.any(self.D <= 0):
                self.D = np.ones(self.dim)
                self.B = np.eye(self.dim)

            # Transform: x = mean + sigma * B * D * z
            safe_D = np.maximum(self.D, 1e-20)
            scaled = z * safe_D[np.newaxis, :]
            rotated = scaled @ self.B.T
            population = self.mean[np.newaxis, :] + self.sigma * rotated

            # Add archived solutions directly to population
            if len(self.archive) > 0 and self.generation % 3 == 0:
                inject_count = min(len(self.archive), max(1, self.pop_size // 4))
                archive_samples = np.array(
                    list(self.archive[:inject_count]) + 
                    [np.random.randn(self.dim) for _ in range(inject_count)]
                )[:inject_count]

                if archive_samples.shape[0] >= inject_count:
                    start_idx = max(0, self.pop_size - inject_count)
                    for i in range(min(inject_count, self.pop_size - start_idx)):
                        if start_idx + i < self.pop_size:
                            population[start_idx + i] = archive_samples[i]

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
        Includes stagnation-triggered path reset to escape local optima.
        """
        # Detect stagnation and trigger evolution path reset
        if self.stagnation_counter > 20 + self.dim:
            # Reset evolution paths using current mean shift direction
            mean_shift = (self.mean - old_mean) / self.sigma
            # Preserve covariance matrix C but reset paths for fresh direction
            self.p_sigma = mean_shift * 0.1
            self.p_c = mean_shift * 0.1
            self.stagnation_counter = 0  # Reset counter after path reset

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
def _clip_to_bounds_batch(self, ...):
    ...
```