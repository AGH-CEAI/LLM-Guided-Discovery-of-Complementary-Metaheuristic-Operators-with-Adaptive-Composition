import numpy as np


class SimplifiedCovarianceAdaptiveSearch:
    """
    A simplified CMA-ES inspired optimizer with adaptive operator selection
    for the _sample_population_batch strategy using Thompson Sampling.
    
    Strategies:
    0: Antithetic variates (variant_01 - 11 wins)
    1: Hybrid center (variant_02 - 6 wins)
    2: Heavy-tailed + archive (variant_06 - 2 wins)
    3: Original standard sampling
    4: Stagnation-driven diversity injection (variant_04 - 1 win)
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
        
        # Adaptive operator selection via Thompson Sampling
        self.n_operators = 5
        # Beta distribution parameters (alpha, beta) for each operator
        # Initialize with priors reflecting benchmark win counts
        self.ts_alpha = np.array([12.0, 7.0, 3.0, 2.0, 2.0])
        self.ts_beta = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        
        # Sliding window for credit assignment
        self.window_size = 20
        self.op_history = []  # list of (operator_idx, reward)
        self.current_operator = 0
        
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
            np.random.beta(self.ts_alpha[i], self.ts_beta[i])
            for i in range(self.n_operators)
        ])
        self.current_operator = int(np.argmax(samples))
        return self.current_operator

    def _update_operator_stats(self, operator_idx, reward):
        """Update Thompson Sampling parameters based on reward."""
        reward = float(reward)
        self.op_history.append((int(operator_idx), reward))
        
        # Trim to sliding window
        if len(self.op_history) > self.window_size:
            self.op_history = self.op_history[-self.window_size:]
        
        # Recompute alpha/beta from sliding window with priors
        # Small prior to keep exploration
        base_alpha = np.array([2.0, 1.5, 1.2, 1.0, 1.0])
        base_beta = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        self.ts_alpha = base_alpha.copy()
        self.ts_beta = base_beta.copy()
        
        for op_idx, rew in self.op_history:
            if 0 <= op_idx < self.n_operators:
                if rew > 0:
                    self.ts_alpha[op_idx] += rew
                else:
                    self.ts_beta[op_idx] += abs(rew)

    def _sample_population_batch(self):
        """Adaptively select and use one of the sampling strategies."""
        op = self._select_operator()
        
        if op == 0:
            return self._sample_antithetic()
        elif op == 1:
            return self._sample_hybrid_center()
        elif op == 2:
            return self._sample_heavy_tail_archive()
        elif op == 3:
            return self._sample_original()
        elif op == 4:
            return self._sample_diversity_injection()
        else:
            return self._sample_original()

    def _sample_original(self):
        """Original standard CMA-ES sampling."""
        dim = self.dim
        lam = self.pop_size
        z = np.random.randn(lam, dim)
        sqrt_eig = np.sqrt(self.eigenvalues)
        y = self.eigenvectors @ (sqrt_eig[:, None] * z.T)
        population = self.mean[None, :] + self.sigma * y.T
        return population, z

    def _sample_antithetic(self):
        """Antithetic variates for variance reduction (variant_01)."""
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

    def _sample_hybrid_center(self):
        """Hybrid center sampling biased toward best solution (variant_02)."""
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

    def _sample_heavy_tail_archive(self):
        """Heavy-tailed + archive-guided sampling (variant_06)."""
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

    def _sample_diversity_injection(self):
        """Stagnation-driven diversity injection (variant_04)."""
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

    def _update_archive(self, sorted_population, sorted_fitness):
        """Update the sampling archive with good solutions."""
        if not hasattr(self, 'sampling_archive'):
            self.sampling_archive = []
        
        # Add top mu solutions to archive
        n_add = min(self.mu, len(sorted_population))
        for i in range(n_add):
            if np.isfinite(sorted_fitness[i]):
                self.sampling_archive.append(sorted_population[i].copy())
        
        # Keep archive bounded
        max_archive = 10 * self.pop_size
        if len(self.sampling_archive) > max_archive:
            self.sampling_archive = self.sampling_archive[-max_archive:]

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

    def __call__(self, func, stopping_condition):
        self._initialize_state()
        self._decompose_covariance()
        self.best_f = np.inf
        self.best_x = np.zeros(self.dim)
        self.sampling_archive = []

        # Initial evaluation
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
        prev_best_for_reward = float(self.best_f)

        while not stopping_condition():
            # Select operator adaptively
            op_idx = self._select_operator()
            
            # Record best_f before this generation
            best_before = float(self.best_f)

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

            # Update archive for variant_06
            self._update_archive(sorted_pop, sorted_fit)

            # Compute reward for operator selection
            best_after = float(self.best_f)
            if best_before > -np.inf and np.isfinite(best_before) and np.isfinite(best_after):
                improvement = best_before - best_after
                if improvement > 1e-15 * (1.0 + abs(best_before)):
                    # Normalize reward: log-scale improvement
                    reward = float(min(1.0, np.log1p(improvement / (abs(best_before) + 1e-20)) + 0.5))
                else:
                    # Small negative reward for no improvement
                    reward = -0.1
            else:
                reward = 0.0
            
            self._update_operator_stats(op_idx, reward)

            # Update distribution
            old_mean = self._update_mean(sorted_pop)
            self._update_evolution_path_sigma(old_mean)
            h_sigma = self._update_evolution_path_c(old_mean)
            self._update_covariance_matrix(old_mean, sorted_pop, h_sigma)
            self._adapt_step_size()

            if self._should_decompose():
                self._decompose_covariance()

            need_restart = self._check_stagnation(sorted_fit)
            if not need_restart:
                need_restart = self._check_condition_number()

            if need_restart:
                self._restart()
                self._decompose_covariance()

            generation += 1

        return float(self.best_f), self.best_x
