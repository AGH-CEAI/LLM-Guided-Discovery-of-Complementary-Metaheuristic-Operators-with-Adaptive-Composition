Below is the lineage champion metaheuristic. You will plan this lineage by
choosing EXACTLY 3 methods from the CANDIDATE list. Those 3 methods will
each be adapted in a separate cycle (in the order you give them), so your
ranking matters — pick #1 will be adapted first and should be the single
method whose adaptation is most likely to unlock the biggest benchmark gain.

Consider:
- Which operators are typically the most performance-sensitive in this family
  of algorithms? (e.g. mutation strategies in DE, recombination in CMA-ES)
- Which methods embody a choice that is known to matter (discrete strategy
  picks, per-problem-dependent heuristics) rather than mere bookkeeping?
- Methods already made adaptive in previous lineages have been REMOVED from
  the candidate list; do NOT suggest anything not listed.

CANDIDATE methods (you MUST pick exactly 3 distinct names from this list):
  - _clip_to_bounds_batch  (3 LOC)
  - _update_evolution_path_sigma  (4 LOC)
  - _update_step_size  (6 LOC)
  - _update_mean  (7 LOC)
  - _handle_truncated_fitness  (8 LOC)
  - _sort_by_fitness  (9 LOC)
  - _update_evolution_path_c  (9 LOC)
  - _inject_best_into_population  (9 LOC)
  - _sample_population_batch  (11 LOC)
  - _restart  (11 LOC)
  - _update_eigen_decomposition  (13 LOC)
  - _detect_stagnation  (13 LOC)
  - _track_best  (14 LOC)
  - _initialize_state  (17 LOC)
  - _update_covariance_matrix  (18 LOC)
  - _initialize_strategy_params  (22 LOC)

HISTORY of cycles in this lineage so far (most recent last):
(no prior cycles in this lineage)

Current champion source:
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

Reply with JSON only, listing picks in DESCENDING order of expected impact:
{"picks": ["...", "...", "..."], "reason": "..."}
