import numpy as np


class AdaptiveCMAESWithRestarts:
    """
    A simplified CMA-ES style optimizer with covariance adaptation,
    step-size control, and intelligent restart mechanisms.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.pop_size = max(4 + int(3 * np.log(dim)), 6 * dim)
        self.pop_size = min(self.pop_size, 300)
        self.mu = self.pop_size // 2
        self.f_opt = np.inf
        self.x_opt = None
        self._setup_weights()

    def _setup_weights(self):
        """Compute recombination weights for weighted mean update."""
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

    def _compute_learning_rates(self):
        """Compute CMA-ES learning rates based on dimension and mu_eff."""
        dim = self.dim
        n = dim
        self.c_sigma = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0) + self.c_sigma
        self.c_c = (4.0 + self.mu_eff / n) / (n + 4.0 + 2.0 * self.mu_eff / n)
        self.c_1 = 2.0 / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_mu_cov = min(1.0 - self.c_1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((n + 2.0) ** 2 + self.mu_eff))
        self.chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

    def _initialize_state(self):
        """Initialize the CMA-ES internal state variables."""
        dim = self.dim
        self.mean = np.random.uniform(self.lb * 0.5, self.ub * 0.5, size=dim)
        self.sigma = 40.0
        self.C = np.eye(dim)
        self.p_sigma = np.zeros(dim)
        self.p_c = np.zeros(dim)
        self.eigenvalues = np.ones(dim)
        self.eigenvectors = np.eye(dim)
        self.eigen_update_counter = 0
        self.stagnation_counter = 0
        self.best_fitness_history = []
        self._compute_learning_rates()

    def _decompose_covariance(self):
        """Eigendecompose the covariance matrix for sampling."""
        self.eigen_update_counter += 1
        update_freq = max(1, int(self.dim / (10.0 * (self.c_1 + self.c_mu_cov) * self.dim)))
        if self.eigen_update_counter >= update_freq:
            self.eigen_update_counter = 0
            # Ensure symmetry
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(self.C)
                # Clamp eigenvalues to avoid numerical issues
                eigenvalues = np.maximum(eigenvalues, 1e-20)
                self.eigenvalues = eigenvalues
                self.eigenvectors = eigenvectors
            except np.linalg.LinAlgError:
                self.C = np.eye(self.dim)
                self.eigenvalues = np.ones(self.dim)
                self.eigenvectors = np.eye(self.dim)

    def _sample_population_batch(self):
        """Sample a new population from the current distribution."""
        dim = self.dim
        NP = self.pop_size
        sqrt_eigenvalues = np.sqrt(self.eigenvalues)
        # Sample standard normal and transform
        z = np.random.randn(NP, dim)
        # Transform: x = mean + sigma * B * D * z
        y = z * sqrt_eigenvalues[np.newaxis, :]  # (NP, dim)
        y = y @ self.eigenvectors.T  # (NP, dim)
        population = self.mean[np.newaxis, :] + self.sigma * y
        return population, z, y

    def _clip_to_bounds(self, population):
        """Clip population to search bounds with reflection."""
        clipped = np.clip(population, self.lb, self.ub)
        return clipped

    def _evaluate_batch(self, func, population, stopping_condition):
        """Evaluate a batch of solutions, handling budget exhaustion."""
        if stopping_condition():
            return None
        fitness = func(population)
        # Handle truncated returns
        if fitness is None:
            return None
        fitness = np.asarray(fitness, dtype=np.float64)
        if len(fitness) < len(population):
            fitness = fitness[:len(fitness)]
        return fitness

    def _update_best(self, population, fitness):
        """Track the global best solution found so far."""
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return
        valid_fitness = fitness[valid_mask]
        valid_pop = population[valid_mask]
        idx = np.argmin(valid_fitness)
        if valid_fitness[idx] < self.f_opt:
            self.f_opt = valid_fitness[idx]
            self.x_opt = valid_pop[idx].copy()

    def _sort_by_fitness(self, population, fitness):
        """Sort population by fitness, return sorted arrays and indices."""
        valid_count = min(len(fitness), len(population))
        population = population[:valid_count]
        fitness = fitness[:valid_count]
        indices = np.argsort(fitness)
        return population[indices], fitness[indices], indices

    def _update_mean(self, sorted_population):
        """Update the distribution mean using weighted recombination of best mu."""
        mu = min(self.mu, len(sorted_population))
        weights = self.weights[:mu]
        weights = weights / weights.sum()
        old_mean = self.mean.copy()
        self.mean = np.dot(weights, sorted_population[:mu])
        return old_mean

    def _update_evolution_paths(self, old_mean):
        """Update the cumulation paths p_sigma and p_c."""
        dim = self.dim
        invsqrt_C = self.eigenvectors @ np.diag(1.0 / np.sqrt(self.eigenvalues)) @ self.eigenvectors.T
        mean_diff = (self.mean - old_mean) / self.sigma

        # Update p_sigma (conjugate evolution path)
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + \
                       np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * (invsqrt_C @ mean_diff)

        # Heaviside function
        norm_p_sigma = np.linalg.norm(self.p_sigma)
        h_sigma = 1.0 if norm_p_sigma / np.sqrt(
            1.0 - (1.0 - self.c_sigma) ** (2 * (self.stagnation_counter + 1))
        ) < (1.4 + 2.0 / (dim + 1.0)) * self.chi_n else 0.0

        # Update p_c (evolution path for covariance)
        self.p_c = (1.0 - self.c_c) * self.p_c + \
                   h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * mean_diff

        return mean_diff, h_sigma

    def _update_covariance(self, old_mean, sorted_population, h_sigma):
        """Update the covariance matrix using rank-one and rank-mu updates."""
        dim = self.dim
        mu = min(self.mu, len(sorted_population))
        weights = self.weights[:mu]
        weights = weights / weights.sum()

        # Rank-one update component
        rank_one = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        # Rank-mu update component
        artmp = (sorted_population[:mu] - old_mean[np.newaxis, :]) / self.sigma  # (mu, dim)
        rank_mu = np.zeros((dim, dim))
        for i in range(mu):
            rank_mu += weights[i] * np.outer(artmp[i], artmp[i])

        # Combined update
        self.C = (1.0 - self.c_1 - self.c_mu_cov + delta_h * self.c_1) * self.C + \
                 self.c_1 * rank_one + \
                 self.c_mu_cov * rank_mu

        # Fix numerical issues
        self.C = np.triu(self.C) + np.triu(self.C, 1).T

    def _update_step_size(self):
        """Adapt sigma using cumulative step-size adaptation (CSA)."""
        norm_p_sigma = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp(min(1.0, (self.c_sigma / self.d_sigma) * (norm_p_sigma / self.chi_n - 1.0)))
        # Bound sigma to prevent explosion or collapse
        self.sigma = np.clip(self.sigma, 1e-20, 1e5)

    def _detect_stagnation(self, fitness_sorted):
        """Detect if the optimization has stagnated."""
        current_best = fitness_sorted[0] if len(fitness_sorted) > 0 else np.inf
        self.best_fitness_history.append(current_best)
        
        # Check various stagnation conditions
        window = min(20 + self.dim, len(self.best_fitness_history))
        if len(self.best_fitness_history) >= window:
            recent = self.best_fitness_history[-window:]
            improvement = abs(recent[0] - recent[-1])
            scale = max(abs(recent[-1]), 1e-10)
            relative_improvement = improvement / scale
            if relative_improvement < 1e-12:
                return True

        # Condition number too large
        cond = np.max(self.eigenvalues) / max(np.min(self.eigenvalues), 1e-20)
        if cond > 1e14:
            return True

        # Sigma too small
        if self.sigma * np.max(np.sqrt(self.eigenvalues)) < 1e-15:
            return True

        # Sigma too large relative to domain
        if self.sigma * np.min(np.sqrt(self.eigenvalues)) > 200.0:
            return True

        return False

    def _restart(self):
        """Perform a restart with adapted parameters."""
        dim = self.dim
        # Increase population size slightly on restart (IPOP-like)
        self.pop_size = min(int(self.pop_size * 1.5), 500)
        self.mu = self.pop_size // 2
        self._setup_weights()
        
        # Re-initialize around best known or random
        self._initialize_state()
        
        # Sometimes restart near best known solution
        if self.x_opt is not None and np.random.rand() < 0.3:
            self.mean = self.x_opt.copy() + np.random.randn(dim) * 10.0
            self.mean = np.clip(self.mean, self.lb, self.ub)
            self.sigma = 20.0
        else:
            self.mean = np.random.uniform(self.lb * 0.8, self.ub * 0.8, size=dim)
            self.sigma = 40.0

    def _local_search_batch(self, func, stopping_condition):
        """Perform a small local search around the best solution."""
        if self.x_opt is None or stopping_condition():
            return
        dim = self.dim
        n_local = min(2 * dim, 60)
        perturbations = np.random.randn(n_local, dim) * 0.1
        candidates = self.x_opt[np.newaxis, :] + perturbations
        candidates = self._clip_to_bounds(candidates)
        fitness = self._evaluate_batch(func, candidates, stopping_condition)
        if fitness is not None and len(fitness) > 0:
            self._update_best(candidates[:len(fitness)], fitness)

    def _opposition_based_population(self, population):
        """Generate opposition-based candidates for diversity."""
        center = (self.lb + self.ub) / 2.0
        opposition = 2.0 * center - population
        opposition = self._clip_to_bounds(opposition)
        return opposition

    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        self.f_opt = np.inf
        self.x_opt = None
        self._initialize_state()

        generation = 0
        restart_count = 0

        while not stopping_condition():
            # Sample new population
            population, z, y = self._sample_population_batch()
            population = self._clip_to_bounds(population)

            # Evaluate
            fitness = self._evaluate_batch(func, population, stopping_condition)
            if fitness is None or len(fitness) == 0:
                break
            if stopping_condition():
                self._update_best(population[:len(fitness)], fitness)
                break

            # On first generation, also try opposition-based candidates
            if generation == 0 and not stopping_condition():
                opp_pop = self._opposition_based_population(population)
                opp_fitness = self._evaluate_batch(func, opp_pop, stopping_condition)
                if opp_fitness is not None and len(opp_fitness) > 0:
                    self._update_best(opp_pop[:len(opp_fitness)], opp_fitness)
                    # Merge and keep best
                    merged_pop = np.vstack([population[:len(fitness)], opp_pop[:len(opp_fitness)]])
                    merged_fit = np.concatenate([fitness[:len(fitness)], opp_fitness[:len(opp_fitness)]])
                    sort_idx = np.argsort(merged_fit)[:self.pop_size]
                    population = merged_pop[sort_idx]
                    fitness = merged_fit[sort_idx]
                if stopping_condition():
                    break

            # Update best
            self._update_best(population[:len(fitness)], fitness)

            # Sort by fitness
            sorted_pop, sorted_fit, _ = self._sort_by_fitness(population[:len(fitness)], fitness)

            # Update distribution
            old_mean = self._update_mean(sorted_pop)
            mean_diff, h_sigma = self._update_evolution_paths(old_mean)
            self._update_covariance(old_mean, sorted_pop, h_sigma)
            self._update_step_size()
            self._decompose_covariance()

            # Check for stagnation and restart if needed
            self.stagnation_counter += 1
            if self._detect_stagnation(sorted_fit):
                # Do a local search before restarting
                if not stopping_condition():
                    self._local_search_batch(func, stopping_condition)
                if stopping_condition():
                    break
                self._restart()
                restart_count += 1
                generation = 0
                continue

            generation += 1

            # Periodic local search every 50 generations
            if generation % 50 == 0 and not stopping_condition():
                self._local_search_batch(func, stopping_condition)

        # Final local search
        if not stopping_condition():
            self._local_search_batch(func, stopping_condition)

        return self.f_opt, self.x_opt
