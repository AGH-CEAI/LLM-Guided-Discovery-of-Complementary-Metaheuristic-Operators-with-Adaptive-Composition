import numpy as np


class SimplifiedCovarianceAdaptiveSearch:
    """
    A simplified CMA-ES-inspired optimizer with adaptive restart strategy selection.
    Uses Thompson Sampling to learn which restart strategy works best during optimization.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.pop_size = 4 + int(3 * np.log(dim))
        self.pop_size = max(self.pop_size, 8)
        if self.pop_size % 2 != 0:
            self.pop_size += 1
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()

        # Adaptive restart strategy selection via Thompson Sampling
        self.num_strategies = 7  # number of restart strategies
        # Beta distribution parameters for Thompson Sampling
        self.ts_alpha = np.ones(self.num_strategies) * 1.0
        self.ts_beta = np.ones(self.num_strategies) * 1.0
        # Track performance per strategy
        self._strategy_rewards = [[] for _ in range(self.num_strategies)]
        self._last_strategy = -1
        self._fitness_before_restart = np.inf
        
        # Restart tracking
        self._restart_count = 0
        self._base_pop_size = self.pop_size
        self._previous_means = []

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
        self.c_mu = min(1.0 - self.c_1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((dim + 2.0) ** 2 + self.mu_eff))

        self.chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

    def _initialize_state(self):
        dim = self.dim
        self.mean = np.random.uniform(self.lb * 0.5, self.ub * 0.5, size=dim)
        self.sigma = 30.0
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
        update_interval = max(1, int(1.0 / (self.c_1 + self.c_mu) / self.dim / 10.0))
        if self.generation - self.eigen_decomp_gen >= update_interval:
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            try:
                eigenvalues, self.B = np.linalg.eigh(self.C)
            except np.linalg.LinAlgError:
                self.C = np.eye(self.dim)
                eigenvalues = np.ones(self.dim)
                self.B = np.eye(self.dim)
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            self.D = np.sqrt(eigenvalues)
            inv_D = 1.0 / self.D
            self.invsqrt_C = self.B @ np.diag(inv_D) @ self.B.T
            self.eigen_decomp_gen = self.generation

    def _sample_population_batch(self):
        dim = self.dim
        lam = self.pop_size
        z = np.random.randn(lam, dim)
        y = z @ np.diag(self.D) @ self.B.T
        population = self.mean[np.newaxis, :] + self.sigma * y
        return population, y, z

    def _clip_to_bounds_batch(self, population):
        return np.clip(population, self.lb, self.ub)

    def _sort_by_fitness(self, population, fitness, y_vectors):
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return population, fitness, y_vectors
        sort_fit = np.where(valid_mask, fitness, np.inf)
        order = np.argsort(sort_fit)
        return population[order], sort_fit[order], y_vectors[order]

    def _update_mean(self, sorted_pop, sorted_y):
        old_mean = self.mean.copy()
        self.mean = self.weights @ sorted_pop[:self.mu]
        y_w = self.weights @ sorted_y[:self.mu]
        return old_mean, y_w

    def _update_evolution_path_sigma(self, y_w):
        c_s = self.c_sigma
        self.p_sigma = (1.0 - c_s) * self.p_sigma + np.sqrt(c_s * (2.0 - c_s) * self.mu_eff) * (self.invsqrt_C @ y_w)

    def _update_evolution_path_c(self, y_w):
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        h_sigma_threshold = (1.4 + 2.0 / (self.dim + 1.0)) * self.chi_n * np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
        h_sigma = 1.0 if p_sigma_norm < h_sigma_threshold else 0.0

        self.p_c = ((1.0 - self.c_c) * self.p_c +
                     h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * y_w)
        return h_sigma

    def _update_covariance_matrix(self, sorted_y, h_sigma):
        dim = self.dim
        rank1 = np.outer(self.p_c, self.p_c)
        y_sel = sorted_y[:self.mu]
        rank_mu = np.zeros((dim, dim))
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(y_sel[i], y_sel[i])

        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        self.C = ((1.0 - self.c_1 - self.c_mu + delta_h * self.c_1) * self.C +
                   self.c_1 * rank1 +
                   self.c_mu * rank_mu)

    def _update_step_size(self):
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0))
        self.sigma = np.clip(self.sigma, 1e-20, 1e6)

    def _track_best(self, sorted_pop, sorted_fit):
        if len(sorted_fit) == 0:
            return
        valid = np.isfinite(sorted_fit)
        if not np.any(valid):
            return
        best_idx = np.argmin(np.where(valid, sorted_fit, np.inf))
        if sorted_fit[best_idx] < self.best_fitness:
            self.best_fitness = float(sorted_fit[best_idx])
            self.best_x = sorted_pop[best_idx].copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

    def _detect_stagnation(self):
        stag_limit = 10 + int(30 * self.dim / self.pop_size)
        if self.stagnation_counter > stag_limit:
            return True
        if self.sigma < 1e-16:
            return True
        if np.max(self.D) / max(np.min(self.D), 1e-30) > 1e7:
            return True
        return False

    def _select_strategy(self):
        """Use Thompson Sampling to select a restart strategy."""
        samples = np.array([
            np.random.beta(self.ts_alpha[i], self.ts_beta[i])
            for i in range(self.num_strategies)
        ])
        return int(np.argmax(samples))

    def _update_strategy_reward(self, strategy_idx, fitness_before, fitness_after):
        """Update Thompson Sampling parameters based on improvement."""
        if strategy_idx < 0 or strategy_idx >= self.num_strategies:
            return
        
        fb = float(fitness_before) if np.isfinite(fitness_before) else 1e30
        fa = float(fitness_after) if np.isfinite(fitness_after) else 1e30
        
        # Reward = 1 if improved, 0 otherwise
        # Use relative improvement threshold
        if fa < fb:
            improvement = (fb - fa) / (abs(fb) + 1e-30)
            # Scale reward based on magnitude of improvement
            reward = min(1.0, improvement * 10.0)
            reward = max(0.01, reward)  # at least small reward for any improvement
        else:
            reward = 0.0
        
        self.ts_alpha[strategy_idx] += reward
        self.ts_beta[strategy_idx] += (1.0 - reward)
        
        # Decay to allow adaptation (sliding window effect)
        decay = 0.995
        self.ts_alpha = np.maximum(self.ts_alpha * decay, 1.0)
        self.ts_beta = np.maximum(self.ts_beta * decay, 1.0)

    def _restart(self):
        """Adaptive restart: use Thompson Sampling to select among strategies."""
        saved_best_x = self.best_x.copy()
        saved_best_f = float(self.best_fitness)

        # Update reward for previous strategy if applicable
        if self._last_strategy >= 0:
            self._update_strategy_reward(
                self._last_strategy, 
                self._fitness_before_restart, 
                saved_best_f
            )

        # Record fitness before this restart cycle
        self._fitness_before_restart = saved_best_f

        # Store previous mean
        self._previous_means.append(self.mean.copy())
        if len(self._previous_means) > 20:
            self._previous_means = self._previous_means[-20:]

        self._restart_count += 1

        # Select strategy via Thompson Sampling
        strategy = self._select_strategy()
        self._last_strategy = strategy

        # Compute new pop size (IPOP-style increase, common to most strategies)
        new_pop_size = self._base_pop_size * (2 ** min(self._restart_count, 5))
        new_pop_size = min(new_pop_size, 512)
        if new_pop_size % 2 != 0:
            new_pop_size += 1

        # Re-initialize state
        self._initialize_state()

        # Restore best
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f

        if strategy == 0:
            # Strategy 0: variant_09 style - fully random restart with large pop
            self.pop_size = new_pop_size
            self.mu = self.pop_size // 2
            self._initialize_strategy_params()
            self.mean = np.random.uniform(self.lb, self.ub, size=self.dim)
            self.sigma = 50.0

        elif strategy == 1:
            # Strategy 1: variant_09 style - near best with very large sigma
            self.pop_size = new_pop_size
            self.mu = self.pop_size // 2
            self._initialize_strategy_params()
            perturbation = np.random.randn(self.dim) * 40.0
            self.mean = np.clip(saved_best_x + perturbation, self.lb, self.ub)
            self.sigma = 60.0

        elif strategy == 2:
            # Strategy 2: variant_09 style - opposition-based restart
            self.pop_size = new_pop_size
            self.mu = self.pop_size // 2
            self._initialize_strategy_params()
            center = np.zeros(self.dim)
            opposite = 2.0 * center - saved_best_x
            noise = np.random.randn(self.dim) * 10.0
            self.mean = np.clip(opposite + noise, self.lb, self.ub)
            self.sigma = 40.0

        elif strategy == 3:
            # Strategy 3: variant_09 style - local refinement near best
            self.pop_size = new_pop_size
            self.mu = self.pop_size // 2
            self._initialize_strategy_params()
            self.mean = np.clip(saved_best_x + np.random.randn(self.dim) * 5.0, self.lb, self.ub)
            self.sigma = 15.0

        elif strategy == 4:
            # Strategy 4: variant_08 style - local refinement with small pop
            self.pop_size = self._base_pop_size
            if self.pop_size % 2 != 0:
                self.pop_size += 1
            self.mu = self.pop_size // 2
            self._initialize_strategy_params()
            self.mean = np.clip(saved_best_x + np.random.randn(self.dim) * 2.0, self.lb, self.ub)
            self.sigma = max(5.0, 0.1 * np.std(saved_best_x - self.mean + 1e-10))

        elif strategy == 5:
            # Strategy 5: variant_01 style - IPOP with blended mean
            old_pop_size = self.pop_size
            self.pop_size = min(new_pop_size, 512)
            if self.pop_size % 2 != 0:
                self.pop_size += 1
            self.mu = self.pop_size // 2
            self._initialize_strategy_params()
            random_point = np.random.uniform(self.lb, self.ub, size=self.dim)
            alpha = min(0.5, old_pop_size / float(self.pop_size + 1e-30))
            self.mean = np.clip(
                alpha * saved_best_x + (1.0 - alpha) * random_point,
                self.lb, self.ub
            )
            dist_to_best = np.linalg.norm(self.mean - saved_best_x)
            if dist_to_best > 1e-10:
                self.sigma = np.clip(dist_to_best / np.sqrt(self.dim), 1.0, 50.0)
            else:
                self.sigma = 30.0

        elif strategy == 6:
            # Strategy 6: variant_02 style - maximize distance from previous means
            self.pop_size = new_pop_size
            if self.pop_size % 2 != 0:
                self.pop_size += 1
            self.mu = self.pop_size // 2
            self._initialize_strategy_params()
            self.sigma = 50.0 + 20.0 * min(self._restart_count, 5)

            if len(self._previous_means) > 0 and np.random.rand() < 0.5:
                # Maximize distance from all previous means
                candidate_means = []
                for _ in range(50):
                    candidate = np.random.uniform(self.lb * 0.8, self.ub * 0.8, size=self.dim)
                    min_dist = min(np.linalg.norm(candidate - pm) for pm in self._previous_means)
                    candidate_means.append((min_dist, candidate))
                candidate_means.sort(key=lambda x: -x[0])
                self.mean = np.clip(candidate_means[0][1], self.lb, self.ub)
            else:
                # Opposition-based
                center = (self.lb + self.ub) / 2.0
                self.mean = np.clip(2.0 * center - saved_best_x, self.lb, self.ub)
                self.mean += np.random.randn(self.dim) * 10.0
                self.mean = np.clip(self.mean, self.lb, self.ub)

            # Reset covariance
            self.C = np.eye(self.dim)
            self.B = np.eye(self.dim)
            self.D = np.ones(self.dim)
            self.invsqrt_C = np.eye(self.dim)
            self.p_sigma = np.zeros(self.dim)
            self.p_c = np.zeros(self.dim)

        # Reset pop size cycle every 8 restarts to avoid permanently huge populations
        if self._restart_count % 8 == 0:
            self._restart_count = 0

        # Ensure covariance is properly reset for strategies that don't explicitly do it
        if strategy not in [6]:
            self.C = np.eye(self.dim)
            self.B = np.eye(self.dim)
            self.D = np.ones(self.dim)
            self.invsqrt_C = np.eye(self.dim)
            self.p_sigma = np.zeros(self.dim)
            self.p_c = np.zeros(self.dim)
            self.eigen_decomp_gen = 0

        self.stagnation_counter = 0

    def _handle_truncated_fitness(self, population, fitness, y_vectors):
        n_valid = len(fitness)
        if n_valid < len(population):
            population = population[:n_valid]
            y_vectors = y_vectors[:n_valid]
            fitness = fitness[:n_valid]
        return population, fitness, y_vectors

    def _inject_best_into_population(self, population, fitness, y_vectors):
        if self.best_fitness < np.inf and len(population) > 0:
            worst_idx = np.argmax(fitness)
            if self.best_fitness < fitness[worst_idx]:
                population[worst_idx] = self.best_x.copy()
                fitness[worst_idx] = self.best_fitness
                y_vectors[worst_idx] = (self.best_x - self.mean) / max(self.sigma, 1e-30)
        return population, fitness, y_vectors

    def __call__(self, func, stopping_condition):
        self._initialize_state()
        
        # Reset adaptive state
        self.ts_alpha = np.ones(self.num_strategies) * 1.0
        self.ts_beta = np.ones(self.num_strategies) * 1.0
        self._last_strategy = -1
        self._fitness_before_restart = np.inf
        self._restart_count = 0
        self._base_pop_size = self.pop_size
        self._previous_means = []

        # Initial evaluation of mean
        mean_fit = func(self.mean.reshape(1, self.dim))
        if len(mean_fit) > 0 and np.isfinite(mean_fit[0]):
            self.best_fitness = float(mean_fit[0])
            self.best_x = self.mean.copy()

        while not stopping_condition():
            self._update_eigen_decomposition()

            population, y_vectors, z_vectors = self._sample_population_batch()
            population = self._clip_to_bounds_batch(population)

            fitness = func(population)
            if stopping_condition():
                population, fitness, y_vectors = self._handle_truncated_fitness(population, fitness, y_vectors)
                if len(fitness) > 0:
                    sorted_pop, sorted_fit, sorted_y = self._sort_by_fitness(population, fitness, y_vectors)
                    self._track_best(sorted_pop, sorted_fit)
                break

            population, fitness, y_vectors = self._handle_truncated_fitness(population, fitness, y_vectors)
            if len(fitness) < self.mu:
                if len(fitness) > 0:
                    sorted_pop, sorted_fit, sorted_y = self._sort_by_fitness(population, fitness, y_vectors)
                    self._track_best(sorted_pop, sorted_fit)
                break

            sorted_pop, sorted_fit, sorted_y = self._sort_by_fitness(population, fitness, y_vectors)
            self._track_best(sorted_pop, sorted_fit)

            old_mean, y_w = self._update_mean(sorted_pop, sorted_y)
            self._update_evolution_path_sigma(y_w)
            h_sigma = self._update_evolution_path_c(y_w)
            self._update_covariance_matrix(sorted_y, h_sigma)
            self._update_step_size()

            self.generation += 1

            if self._detect_stagnation():
                self._restart()

        return self.best_fitness, self.best_x
