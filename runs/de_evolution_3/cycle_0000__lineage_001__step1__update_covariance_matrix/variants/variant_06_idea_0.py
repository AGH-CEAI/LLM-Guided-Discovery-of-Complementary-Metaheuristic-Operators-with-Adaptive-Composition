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
        """Update the covariance matrix using exponential fitness-weighted rank-mu update."""
        dim = self.dim
        selected = sorted_population[:self.mu]

        # Rank-1 update component (unchanged)
        rank1 = np.outer(self.p_c, self.p_c)

        # Correction for h_sigma
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        # Exponential fitness-based weights for covariance: emphasize best solutions
        # This is fundamentally different from log-linear weights used elsewhere
        inv_ranks = 1.0 / np.arange(1, self.mu + 1)
        exp_weights = np.exp(-1.5 * inv_ranks)  # exponential decay over rank
        exp_weights = exp_weights / np.sum(exp_weights)
        mu_eff_exp = 1.0 / np.sum(exp_weights ** 2)

        # Adaptive learning rate based on effective weight concentration
        c_mu_adaptive = min(0.9, self.c_mu_cov * (1.0 + 0.5 * (mu_eff_exp - 1.0)))

        # Rank-mu update with exponential weights (more aggressive selection)
        diff = (selected - old_mean[None, :]) / self.sigma  # (mu, dim)
        rank_mu = np.zeros((dim, dim))
        for i in range(self.mu):
            rank_mu += exp_weights[i] * np.outer(diff[i], diff[i])

        # Combined update with adaptive learning rate
        self.C = ((1.0 - self.c_1 - c_mu_adaptive + delta_h * self.c_1) * self.C +
                   self.c_1 * rank1 +
                   c_mu_adaptive * rank_mu)

        # Enforce symmetry and add slight diagonal push for numerical stability
        self.C = 0.5 * (self.C + self.C.T)
        self.C.flat[::dim + 1] += 1e-10 * np.ones(dim)

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
