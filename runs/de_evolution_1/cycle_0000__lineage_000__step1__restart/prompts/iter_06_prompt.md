This is iteration 6 of 10. Propose ONE replacement implementation for `_restart`.

Requirements:
- Keep the EXACT function signature: `def _restart(self):`
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
  Task 17: *** UNSOLVED *** — best=1.153e+04 by original.py               (target=1e-08, ~+12.1 decades above target)
  Task 16: *** UNSOLVED *** — best=8.283e+02 by variant_04_idea_0.py      (target=1e-08, ~+10.9 decades above target)
  Task 11: *** UNSOLVED *** — best=1.138e+02 by variant_04_idea_0.py      (target=1e-08, ~+10.1 decades above target)
  Task 19: *** UNSOLVED *** — best=5.973e+01 by variant_04_idea_0.py      (target=1e-08, ~+9.8 decades above target)
  Task 23: *** UNSOLVED *** — best=4.150e+01 by variant_04_idea_0.py      (target=1e-08, ~+9.6 decades above target)
  Task 18: *** UNSOLVED *** — best=2.886e+01 by original.py               (target=1e-08, ~+9.5 decades above target)
  Task 10: *** UNSOLVED *** — best=2.801e+01 by variant_04_idea_0.py      (target=1e-08, ~+9.4 decades above target)
  Task 20: *** UNSOLVED *** — best=2.290e+01 by variant_01_idea_0.py      (target=1e-08, ~+9.4 decades above target)
  Task 12: *** UNSOLVED *** — best=1.807e+01 by variant_04_idea_0.py      (target=1e-08, ~+9.3 decades above target)
  Task 13: *** UNSOLVED *** — best=1.229e+01 by variant_02_idea_0.py      (target=1e-08, ~+9.1 decades above target)
  Task 22: *** UNSOLVED *** — best=9.414e+00 by variant_04_idea_0.py      (target=1e-08, ~+9.0 decades above target)
  Task 21: *** UNSOLVED *** — best=4.480e+00 by original.py               (target=1e-08, ~+8.7 decades above target)
  Task 14: *** UNSOLVED *** — best=3.527e+00 by original.py               (target=1e-08, ~+8.5 decades above target)
  Task 15: *** UNSOLVED *** — best=2.658e+00 by variant_04_idea_0.py      (target=1e-08, ~+8.4 decades above target)
  Task  4: *** UNSOLVED *** — best=2.434e-02 by variant_01_idea_0.py      (target=1e-08, ~+6.4 decades above target)
  Task  8: *** UNSOLVED *** — best=1.697e-03 by variant_01_idea_0.py      (target=1e-08, ~+5.2 decades above target)
  Task  9: *** UNSOLVED *** — best=4.966e-04 by variant_01_idea_0.py      (target=1e-08, ~+4.7 decades above target)
  Task  3: *** UNSOLVED *** — best=3.409e-04 by variant_05_idea_0.py      (target=1e-08, ~+4.5 decades above target)
  Task  2: *** UNSOLVED *** — best=2.373e-05 by variant_04_idea_0.py      (target=1e-08, ~+3.4 decades above target)
  Task  7: *** UNSOLVED *** — best=1.321e-05 by variant_01_idea_0.py      (target=1e-08, ~+3.1 decades above target)
  Task  6: *** UNSOLVED *** — best=4.104e-07 by variant_02_idea_0.py      (target=1e-08, ~+1.6 decades above target)
  Task  1: *** UNSOLVED *** — best=2.352e-07 by variant_03_idea_0.py      (target=1e-08, ~+1.4 decades above target)
  Task  0: *** UNSOLVED *** — best=1.105e-07 by variant_02_idea_0.py      (target=1e-08, ~+1.0 decades above target)

SOLVED TASKS (already at or below target — do not regress these):
  Task  5: SOLVED  — best=1.000e-08 by original.py                   

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_idea_0.py  variant_02_idea_0.py  variant_03_idea_0.py  variant_04_idea_0.py  variant_05_idea_0.py  
-----------------------------------------------------------------------------------------------------------------------------------------
0    1.1303e-07            1.1330e-07            1.1046e-07            1.1590e-07            1.1818e-07            1.1350e-07            
1    2.3938e-07            2.3641e-07            2.3660e-07            2.3521e-07            2.4228e-07            2.3749e-07            
2    2.4506e-05            2.4306e-05            2.3844e-05            2.4311e-05            2.3727e-05            2.4750e-05            
3    3.5433e-04            3.4696e-04            3.4339e-04            3.5420e-04            3.5473e-04            3.4085e-04            
4    2.4657e-02            2.4337e-02            2.4448e-02            2.4563e-02            2.4539e-02            2.4745e-02            
5    1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            1.0000e-08            
6    4.2932e-07            4.4707e-07            4.1037e-07            4.5602e-07            4.3585e-07            4.1521e-07            
7    1.3265e-05            1.3207e-05            1.3286e-05            1.3502e-05            1.3718e-05            1.3946e-05            
8    1.7603e-03            1.6972e-03            1.7273e-03            1.7209e-03            1.7346e-03            1.7422e-03            
9    5.0803e-04            4.9664e-04            5.1500e-04            5.0852e-04            5.0241e-04            5.0418e-04            
10   2.8402e+01            3.9212e+01            3.4370e+01            4.8395e+01            2.8008e+01            5.1482e+01            
11   1.5035e+02            1.7302e+02            1.3179e+02            1.7418e+02            1.1384e+02            1.2891e+02            
12   5.0565e+01            7.3087e+01            6.4750e+01            7.7364e+01            1.8070e+01            7.3385e+01            
13   1.2655e+01            1.5844e+01            1.2288e+01            1.7365e+01            1.5029e+01            1.4788e+01            
14   3.5267e+00            3.8915e+00            4.1145e+00            4.0121e+00            4.0358e+00            3.9581e+00            
15   3.0962e+00            2.8560e+00            3.1165e+00            3.2957e+00            2.6575e+00            3.0618e+00            
16   1.2144e+03            1.0114e+03            1.3517e+03            2.1365e+03            8.2831e+02            1.1715e+03            
17   1.1529e+04            1.3885e+04            1.2121e+04            1.5221e+04            1.2176e+04            1.2342e+04            
18   2.8857e+01            3.1950e+01            2.9838e+01            3.8901e+01            3.0559e+01            3.7662e+01            
19   6.5476e+01            6.8303e+01            7.9826e+01            7.5688e+01            5.9725e+01            8.4125e+01            
20   2.4917e+01            2.2903e+01            2.4888e+01            2.5075e+01            2.3560e+01            2.7419e+01            
21   4.4795e+00            4.5713e+00            4.5975e+00            5.1036e+00            4.5528e+00            4.6328e+00            
22   1.0381e+01            1.0261e+01            1.3015e+01            1.0087e+01            9.4144e+00            1.0580e+01            
23   4.3619e+01            4.4072e+01            4.4227e+01            4.2037e+01            4.1500e+01            4.3283e+01            

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 17: best error so far = 1.153e+04  (target = 1e-08)
  Task 16: best error so far = 8.283e+02  (target = 1e-08)
  Task 11: best error so far = 1.138e+02  (target = 1e-08)
  Task 19: best error so far = 5.973e+01  (target = 1e-08)
  Task 23: best error so far = 4.150e+01  (target = 1e-08)
  Task 18: best error so far = 2.886e+01  (target = 1e-08)
  Task 10: best error so far = 2.801e+01  (target = 1e-08)
  Task 20: best error so far = 2.290e+01  (target = 1e-08)
  ... and 15 other unsolved task(s).

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0, Idea 0, Idea 0, Idea 0

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
def _restart(self):
        """Restart the search with a new random state, keeping best solution."""
        saved_best_x = self.best_x.copy()
        saved_best_f = self.best_fitness
        self._initialize_state()
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f
        # Bias new mean towards best found so far with some probability
        if np.random.rand() < 0.3:
            perturbation = np.random.randn(self.dim) * 20.0
            self.mean = np.clip(saved_best_x + perturbation, self.lb, self.ub)
```

Full algorithm for context:
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
        """Restart the search with a new random state, keeping best solution."""
        saved_best_x = self.best_x.copy()
        saved_best_f = self.best_fitness
        self._initialize_state()
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f
        # Bias new mean towards best found so far with some probability
        if np.random.rand() < 0.3:
            perturbation = np.random.randn(self.dim) * 20.0
            self.mean = np.clip(saved_best_x + perturbation, self.lb, self.ub)

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

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description.
```python
def _restart(self, ...):
    ...
```