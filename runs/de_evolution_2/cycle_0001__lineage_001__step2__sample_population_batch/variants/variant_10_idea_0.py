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
        Sample a batch combining standard CMA-ES sampling with restart-mode sampling.
        When diversity collapses (large condition number or small sigma), inject
        restart candidates sampled from a wide distribution to escape local optima.
        """
        # Standard CMA-ES sampling
        z = np.random.randn(self.pop_size, self.dim)
        scaled = z * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        cmaes_population = self.mean[np.newaxis, :] + self.sigma * rotated

        # Determine if restart-mode sampling should be triggered
        # Use condition number of covariance as diversity metric
        if self.D is not None and len(self.D) > 0:
            cond_C = np.max(self.D) / np.maximum(np.min(self.D), 1e-30)
            trigger_restart = cond_C > 1e5 or self.sigma < 0.1
        else:
            trigger_restart = False

        if trigger_restart:
            # Restart-mode: sample from a MUCH wider distribution
            # Scale is 10x larger than current sigma to escape local optima
            restart_scale = max(self.sigma * 10.0, 10.0)
            restart_z = np.random.randn(self.pop_size, self.dim)
            # Sample from uniform cube of radius restart_scale in each eigen-direction
            restart_population = self.mean[np.newaxis, :] + restart_scale * restart_z * self.D[np.newaxis, :]
            restart_population = np.clip(restart_population, self.lb, self.ub)

            # Blend: 50% CMA-ES, 50% restart-mode for diversity
            blend_mask = np.random.rand(self.pop_size) < 0.5
            population = np.where(blend_mask[:, np.newaxis], restart_population, cmaes_population)
            # Recompute z for restart candidates (for path tracking)
            z = (population - self.mean[np.newaxis, :]) / max(self.sigma, 1e-30)
            # Project back to z-space using inverse transform
            z = z @ self.B / self.D[np.newaxis, :]
        else:
            population = cmaes_population

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
