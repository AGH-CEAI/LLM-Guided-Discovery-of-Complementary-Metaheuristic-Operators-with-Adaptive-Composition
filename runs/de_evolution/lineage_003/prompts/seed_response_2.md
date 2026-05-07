

```python
import numpy as np


class AdaptiveCMAESLite:
    """
    A simplified CMA-ES inspired optimizer with:
    - Covariance adaptation via evolution path and rank-1 update
    - Step-size adaptation via cumulative step-size adaptation (CSA)
    - Restart mechanism on stagnation detection
    - Multiple restart strategies (IPOP-like increasing population)
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.base_pop_size = max(4 * dim, 20)
        self.pop_size = self.base_pop_size
        self.sigma = 30.0
        self.f_opt = np.inf
        self.x_opt = None
        self.restart_count = 0
        self.max_restarts = 20

    def _initialize_population(self, center=None):
        """Initialize population around a center or randomly."""
        if center is None:
            center = np.random.uniform(self.lb, self.ub, size=self.dim)
        self.mean = center.copy()
        self.sigma = 30.0 * (0.9 ** self.restart_count)
        self.sigma = max(self.sigma, 1.0)

        # CMA-ES parameters
        n = self.dim
        lam = self.pop_size
        self.mu = lam // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        # Learning rates
        self.c_sigma = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0) + self.c_sigma
        self.c_c = (4.0 + self.mu_eff / n) / (n + 4.0 + 2.0 * self.mu_eff / n)
        self.c_1 = 2.0 / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_mu_cov = min(1.0 - self.c_1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((n + 2.0) ** 2 + self.mu_eff))

        # Evolution paths
        self.p_sigma = np.zeros(n)
        self.p_c = np.zeros(n)

        # Use diagonal covariance for efficiency in high dim
        self.C_diag = np.ones(n)  # diagonal of covariance matrix
        self.sqrt_C_diag = np.ones(n)
        self.inv_sqrt_C_diag = np.ones(n)

        # Stagnation tracking
        self.stagnation_counter = 0
        self.best_fitness_history = []
        self.generation = 0

    def _generate_parameters_batch(self):
        """Sample a batch of candidate solutions from the current distribution."""
        n = self.dim
        lam = self.pop_size
        z = np.random.randn(lam, n)
        y = z * self.sqrt_C_diag[np.newaxis, :]
        x = self.mean[np.newaxis, :] + self.sigma * y
        return x, y, z

    def _clip_to_bounds(self, population):
        """Clip population to search bounds with reflection."""
        clipped = population.copy()
        # Reflection-based clipping
        for _ in range(2):
            below = clipped < self.lb
            above = clipped > self.ub
            clipped[below] = 2 * self.lb - clipped[below]
            clipped[above] = 2 * self.ub - clipped[above]
        # Final hard clip
        np.clip(clipped, self.lb, self.ub, out=clipped)
        return clipped

    def _evaluate_batch(self, func, population):
        """Evaluate a batch of candidates and update best tracking."""
        fitness = func(population)
        valid_count = len(fitness)
        if valid_count < len(population):
            population = population[:valid_count]
            fitness = fitness[:valid_count]
        # Update best, ignoring NaNs
        valid_mask = np.isfinite(fitness)
        if np.any(valid_mask):
            best_idx = np.argmin(np.where(valid_mask, fitness, np.inf))
            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = population[best_idx].copy()
        return population, fitness

    def _select_parents(self, population, fitness, y_batch, z_batch):
        """Select the mu best individuals and return weighted recombination info."""
        valid_mask = np.isfinite(fitness)
        fit_for_sort = np.where(valid_mask, fitness, np.inf)
        order = np.argsort(fit_for_sort)
        selected_indices = order[:self.mu]

        y_sel = y_batch[selected_indices]
        z_sel = z_batch[selected_indices]

        # Weighted mean of steps
        y_w = self.weights @ y_sel  # shape (dim,)
        z_w = self.weights @ z_sel  # shape (dim,)

        return y_w, z_w, selected_indices, fit_for_sort[order]

    def _update_evolution_paths(self, y_w, z_w):
        """Update the cumulation paths p_sigma and p_c."""
        n = self.dim
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        # CSA path
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + \
                       np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * z_w

        # h_sigma indicator
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        h_sigma_threshold = (1.4 + 2.0 / (n + 1.0)) * chi_n * \
                            np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2.0 * (self.generation + 1)))
        h_sigma = 1.0 if p_sigma_norm < h_sigma_threshold else 0.0

        # Evolution path for covariance
        self.p_c = (1.0 - self.c_c) * self.p_c + \
                   h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * y_w

        return h_sigma, chi_n

    def _update_covariance_diagonal(self, y_sel_batch, h_sigma):
        """Update the diagonal covariance matrix (sep-CMA-ES style)."""
        n = self.dim
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        # Rank-1 update
        rank1 = self.p_c ** 2

        # Rank-mu update
        rank_mu = self.weights @ (y_sel_batch ** 2)

        # Combined update
        self.C_diag = (1.0 - self.c_1 - self.c_mu_cov + self.c_1 * delta_h) * self.C_diag + \
                      self.c_1 * rank1 + self.c_mu_cov * rank_mu

        # Ensure positivity
        self.C_diag = np.maximum(self.C_diag, 1e-20)

        # Update cached quantities
        self.sqrt_C_diag = np.sqrt(self.C_diag)
        self.inv_sqrt_C_diag = 1.0 / self.sqrt_C_diag

    def _adapt_step_size(self, chi_n):
        """Adapt sigma using cumulative step-size adaptation."""
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        log_factor = (self.c_sigma / self.d_sigma) * (p_sigma_norm / chi_n - 1.0)
        # Clamp to avoid extreme changes
        log_factor = np.clip(log_factor, -0.5, 0.5)
        self.sigma *= np.exp(log_factor)
        # Keep sigma in reasonable bounds
        self.sigma = np.clip(self.sigma, 1e-20, 100.0)

    def _update_mean(self, y_w):
        """Update distribution mean."""
        self.mean = self.mean + self.sigma * y_w
        # Clip mean to bounds
        self.mean = np.clip(self.mean, self.lb, self.ub)

    def _detect_stagnation(self, sorted_fitness):
        """Detect if the algorithm is stagnating and needs a restart."""
        if len(sorted_fitness) == 0:
            return False

        current_best = sorted_fitness[0] if np.isfinite(sorted_fitness[0]) else np.inf
        self.best_fitness_history.append(current_best)

        # Check various stagnation conditions
        window = min(20 + self.dim, len(self.best_fitness_history))
        if len(self.best_fitness_history) >= window:
            recent = self.best_fitness_history[-window:]
            improvement = abs(recent[0] - recent[-1])
            scale = max(abs(recent[-1]), 1e-10)
            if improvement / scale < 1e-12:
                return True

        # Sigma too small
        if self.sigma < 1e-15:
            return True

        # Sigma too large relative to search space
        if self.sigma > 200.0:
            return True

        # Condition number of diagonal too large
        if self.C_diag.max() / (self.C_diag.min() + 1e-30) > 1e14:
            return True

        # Too many generations without global improvement
        if self.generation > 100 + 50 * self.dim:
            return True

        return False

    def _restart(self):
        """Perform a restart with potentially different strategy."""
        self.restart_count += 1

        # IPOP: increase population on restarts
        if self.restart_count % 3 == 0:
            self.pop_size = min(self.pop_size * 2, 800)
        else:
            self.pop_size = self.base_pop_size

        # Choose restart center
        if self.x_opt is not None and np.random.random() < 0.5:
            # Restart near best known with perturbation
            perturbation = np.random.randn(self.dim) * 20.0
            center = np.clip(self.x_opt + perturbation, self.lb, self.ub)
        else:
            center = None

        self._initialize_population(center=center)

    def _local_search_batch(self, func, stopping_condition):
        """Perform a small batch local search around the best solution."""
        if self.x_opt is None:
            return
        if stopping_condition():
            return

        n_local = min(self.dim * 2, 60)
        # Generate local candidates
        local_sigma = max(self.sigma * 0.1, 0.01)
        perturbations = np.random.randn(n_local, self.dim) * local_sigma
        candidates = self.x_opt[np.newaxis, :] + perturbations
        candidates = self._clip_to_bounds(candidates)

        candidates, fitness = self._evaluate_batch(func, candidates)

    def _opposition_based_candidates(self, population):
        """Generate opposition-based candidates for diversity."""
        center = (self.lb + self.ub) / 2.0
        opposition = 2.0 * center - population
        return self._clip_to_bounds(opposition)

    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        self.f_opt = np.inf
        self.x_opt = None
        self.restart_count = 0
        self.pop_size = self.base_pop_size

        # Initial population evaluation for seeding
        self._initialize_population()
        init_pop = np.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim))
        init_pop, init_fit = self._evaluate_batch(func, init_pop)
        if stopping_condition():
            return self.f_opt, self.x_opt

        # Set mean to best initial point
        if self.x_opt is not None:
            self.mean = self.x_opt.copy()

        while not stopping_condition():
            # Run one CMA-ES generation
            population, y_batch, z_batch = self._generate_parameters_batch()
            population = self._clip_to_bounds(population)

            population, fitness = self._evaluate_batch(func, population)
            if stopping_condition():
                break

            if len(fitness) < 2:
                break

            # Adjust batch sizes if truncated
            actual_size = len(fitness)
            y_batch = y_batch[:actual_size]
            z_batch = z_batch[:actual_size]

            # Selection and recombination
            effective_mu = min(self.mu, actual_size)
            if effective_mu < 1:
                break

            y_w, z_w, selected_indices, sorted_fitness = self._select_parents(
                population, fitness, y_batch, z_batch
            )

            # Get selected y for covariance update
            order = np.argsort(np.where(np.isfinite(fitness), fitness, np.inf))
            y_sel = y_batch[order[:effective_mu]]

            # Recompute weights if mu changed
            if effective_mu != self.mu:
                w = np.log(effective_mu + 0.5) - np.log(np.arange(1, effective_mu + 1))
                w = w / w.sum()
                y_w = w @ y_sel
                z_sel = z_batch[order[:effective_mu]]
                z_w = w @ z_sel

            # Update mean
            self._update_mean(y_w)

            # Update evolution paths
            h_sigma, chi_n = self._update_evolution_paths(y_w, z_w)

            # Update covariance
            self._update_covariance_diagonal(y_sel, h_sigma)

            # Adapt step size
            self._adapt_step_size(chi_n)

            self.generation += 1

            # Detect stagnation and restart
            if self._detect_stagnation(sorted_fitness):
                # Try local search before restart
                self._local_search_batch(func, stopping_condition)
                if stopping_condition():
                    break

                if self.restart_count < self.max_restarts:
                    self._restart()
                else:
                    # Reset restart count and try fresh
                    self.restart_count = 0
                    self.pop_size = self.base_pop_size
                    self._initialize_population()

            # Periodic opposition-based search for diversity
            if self.generation % 50 == 0 and not stopping_condition():
                opp_pop = self._opposition_based_candidates(
                    population[order[:min(5, actual_size)]]
                )
                self._evaluate_batch(func, opp_pop)
                if stopping_condition():
                    break

        return self.f_opt, self.x_opt
```