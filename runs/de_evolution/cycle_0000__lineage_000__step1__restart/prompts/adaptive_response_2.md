

```python
import numpy as np


class SimplifiedCovarianceAdaptiveSearch:
    """
    A simplified CMA-ES-inspired optimizer with adaptive restart strategy selection.
    Uses Thompson Sampling to choose among multiple restart strategies during optimization.
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

        # Adaptive restart operator selection (Thompson Sampling)
        self.num_restart_strategies = 7
        # Beta distribution parameters for Thompson Sampling
        self.ts_alpha = np.ones(self.num_restart_strategies) * 1.0
        self.ts_beta = np.ones(self.num_restart_strategies) * 1.0
        # Sliding window for credit assignment
        self.restart_history = []  # list of (strategy_idx, fitness_before, fitness_after)
        self.window_size = 20
        
        # Restart tracking
        self._restart_count = 0
        self._base_pop_size = self.pop_size
        self._previous_means = []
        self._current_restart_strategy = None
        self._fitness_before_restart = None

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
                eigenvalues = np.maximum(eigenvalues, 1e-20)
                self.D = np.sqrt(eigenvalues)
                inv_D = 1.0 / self.D
                self.invsqrt_C = self.B @ np.diag(inv_D) @ self.B.T
            except np.linalg.LinAlgError:
                self.C = np.eye(self.dim)
                self.B = np.eye(self.dim)
                self.D = np.ones(self.dim)
                self.invsqrt_C = np.eye(self.dim)
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

    def _select_restart_strategy(self):
        """Use Thompson Sampling to select a restart strategy."""
        samples = np.array([
            np.random.beta(self.ts_alpha[i], self.ts_beta[i])
            for i in range(self.num_restart_strategies)
        ])
        return int(np.argmax(samples))

    def _update_restart_reward(self, strategy_idx, fitness_before, fitness_after):
        """Update Thompson Sampling parameters based on restart outcome."""
        fitness_before = float(fitness_before) if np.isfinite(fitness_before) else 1e18
        fitness_after = float(fitness_after) if np.isfinite(fitness_after) else 1e18
        
        # Reward = relative improvement
        if fitness_before > 1e-30:
            improvement = (fitness_before - fitness_after) / (abs(fitness_before) + 1e-30)
        else:
            improvement = 0.0
        
        # Convert to success probability
        success = 1.0 if improvement > 0.01 else 0.0
        
        # Update Beta distribution
        self.ts_alpha[strategy_idx] += success
        self.ts_beta[strategy_idx] += (1.0 - success)
        
        # Decay to prevent lock-in (sliding window effect)
        decay = 0.995
        self.ts_alpha = 1.0 + (self.ts_alpha - 1.0) * decay
        self.ts_beta = 1.0 + (self.ts_beta - 1.0) * decay
        
        # Ensure minimum values
        self.ts_alpha = np.maximum(self.ts_alpha, 1.0)
        self.ts_beta = np.maximum(self.ts_beta, 1.0)

    def _restart_strategy_0(self, saved_best_x, saved_best_f):
        """variant_09 style: Fully random restart with large sigma, IPOP pop size."""
        new_pop_size = self._base_pop_size * (2 ** min(self._restart_count, 5))
        new_pop_size = min(new_pop_size, 512)
        if new_pop_size % 2 != 0:
            new_pop_size += 1
        self.pop_size = new_pop_size
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()
        self._initialize_state()
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f
        
        self.mean = np.random.uniform(self.lb, self.ub, size=self.dim)
        self.sigma = 50.0

    def _restart_strategy_1(self, saved_best_x, saved_best_f):
        """variant_09 style: Near best with very large sigma."""
        new_pop_size = self._base_pop_size * (2 ** min(self._restart_count, 5))
        new_pop_size = min(new_pop_size, 512)
        if new_pop_size % 2 != 0:
            new_pop_size += 1
        self.pop_size = new_pop_size
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()
        self._initialize_state()
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f
        
        perturbation = np.random.randn(self.dim) * 40.0
        self.mean = np.clip(saved_best_x + perturbation, self.lb, self.ub)
        self.sigma = 60.0

    def _restart_strategy_2(self, saved_best_x, saved_best_f):
        """variant_09 style: Opposition-based restart."""
        new_pop_size = self._base_pop_size * (2 ** min(self._restart_count, 5))
        new_pop_size = min(new_pop_size, 512)
        if new_pop_size % 2 != 0:
            new_pop_size += 1
        self.pop_size = new_pop_size
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()
        self._initialize_state()
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f
        
        center = np.zeros(self.dim)
        opposite = 2.0 * center - saved_best_x
        noise = np.random.randn(self.dim) * 10.0
        self.mean = np.clip(opposite + noise, self.lb, self.ub)
        self.sigma = 40.0

    def _restart_strategy_3(self, saved_best_x, saved_best_f):
        """variant_09 style: Local refinement near best with moderate sigma."""
        new_pop_size = self._base_pop_size * (2 ** min(self._restart_count, 5))
        new_pop_size = min(new_pop_size, 512)
        if new_pop_size % 2 != 0:
            new_pop_size += 1
        self.pop_size = new_pop_size
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()
        self._initialize_state()
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f
        
        self.mean = np.clip(saved_best_x + np.random.randn(self.dim) * 5.0, self.lb, self.ub)
        self.sigma = 15.0

    def _restart_strategy_4(self, saved_best_x, saved_best_f):
        """variant_08 style: Local refinement with small pop and small sigma."""
        self._initialize_state()
        self.pop_size = self._base_pop_size
        if self.pop_size % 2 != 0:
            self.pop_size += 1
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f
        
        self.mean = np.clip(saved_best_x + np.random.randn(self.dim) * 2.0, self.lb, self.ub)
        self.sigma = 5.0
        
        self.C = np.eye(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.invsqrt_C = np.eye(self.dim)
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)
        self.eigen_decomp_gen = 0
        self.stagnation_counter = 0

    def _restart_strategy_5(self, saved_best_x, saved_best_f):
        """variant_01 style: IPOP with blended mean initialization."""
        old_pop_size = self.pop_size
        self.pop_size = min(self.pop_size * 2, 512)
        if self.pop_size % 2 != 0:
            self.pop_size += 1
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()
        self._initialize_state()
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f
        
        random_point = np.random.uniform(self.lb, self.ub, size=self.dim)
        alpha = min(0.5, old_pop_size / float(self.pop_size))
        self.mean = np.clip(
            alpha * saved_best_x + (1.0 - alpha) * random_point,
            self.lb, self.ub
        )
        
        dist_to_best = np.linalg.norm(self.mean - saved_best_x)
        if dist_to_best > 1e-10:
            self.sigma = np.clip(dist_to_best / np.sqrt(self.dim), 1.0, 50.0)
        else:
            self.sigma = 30.0

    def _restart_strategy_6(self, saved_best_x, saved_best_f):
        """variant_02 style: IPOP with diversification avoiding previous means."""
        self._previous_means.append(self.mean.copy())
        
        old_pop_size = self.pop_size
        new_pop_size = min(old_pop_size * 2, 512)
        self._initialize_state()
        self.pop_size = new_pop_size
        if self.pop_size % 2 != 0:
            self.pop_size += 1
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()
        self.best_x = saved_best_x
        self.best_fitness = saved_best_f
        
        self.sigma = 50.0 + 20.0 * min(self._restart_count, 5)
        
        sub_strategy = self._restart_count % 3
        if sub_strategy == 1:
            center = (self.lb + self.ub) / 2.0
            self.mean = np.clip(2.0 * center - saved_best_x, self.lb, self.ub)
            self.mean += np.random.randn(self.dim) * 10.0
            self.mean = np.clip(self.mean, self.lb, self.ub)
        elif sub_strategy == 2 and len(self._previous_means) > 0:
            candidate_means = []
            for _ in range(50):
                candidate = np.random.uniform(self.lb * 0.8, self.ub * 0.8, size=self.dim)
                min_dist = min(np.linalg.norm(candidate - pm) for pm in self._previous_means)
                candidate_means.append((min_dist, candidate))
            candidate_means.sort(key=lambda x: -x[0])
            self.mean = np.clip(candidate_means[0][1], self.lb, self.ub)
        else:
            self.mean = np.random.uniform(self.lb * 0.9, self.ub * 0.9, size=self.dim)
        
        self.C = np.eye(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.invsqrt_C = np.eye(self.dim)
        self.p_sigma = np.zeros(self.dim)
        self.p_c = np.zeros(self.dim)

    def _restart(self):
        """Adaptive restart using Thompson Sampling to select among strategies."""
        saved_best_x = self.best_x.copy()
        saved_best_f = float(self.best_fitness) if np.isfinite(self.best_fitness) else 1e18
        
        # If we have a previous restart strategy, update its reward
        if self._current_restart_strategy is not None and self._fitness_before_restart is not None:
            self._update_restart_reward(
                self._current_restart_strategy,
                self._fitness_before_restart,
                saved_best_f
            )
        
        self._restart_count += 1
        
        # Select strategy via Thompson Sampling
        strategy_idx = self._select_restart_strategy()
        self._current_restart_strategy = strategy_idx
        self._fitness_before_restart = saved_best_f
        
        # Execute selected strategy
        try:
            if strategy_idx == 0:
                self._restart_strategy_0(saved_best_x, saved_best_f)
            elif strategy_idx == 1:
                self._restart_strategy_1(saved_best_x, saved_best_f)
            elif strategy_idx == 2:
                self._restart_strategy_2(saved_best_x, saved_best_f)
            elif strategy_idx == 3:
                self._restart_strategy_3(saved_best_x, saved_best_f)
            elif strategy_idx == 4:
                self._restart_strategy_4(saved_best_x, saved_best_f)
            elif strategy_idx == 5:
                self._restart_strategy_5(saved_best_x, saved_best_f)
            elif strategy_idx == 6:
                self._restart_strategy_6(saved_best_x, saved_best_f)
        except Exception:
            # Fallback: simple restart
            self._initialize_state()
            self.best_x = saved_best_x
            self.best_fitness = saved_best_f
            self.mean = np.random.uniform(self.lb * 0.8, self.ub * 0.8, size=self.dim)
            self.sigma = 50.0
        
        # Reset pop size counter periodically to avoid overly large populations
        if self._restart_count % 10 == 0:
            self._restart_count = 1

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
        self._restart_count = 0
        self._previous_means = []
        self._current_restart_strategy = None
        self._fitness_before_restart = None
        self.ts_alpha = np.ones(self.num_restart_strategies) * 1.0
        self.ts_beta = np.ones(self.num_restart_strategies) * 1.0

        # Initial evaluation of mean
        mean_fit = func(self.mean.reshape(1, self.dim))
        if len(mean_fit) > 0 and np.isfinite(mean_fit[0]):
            self.best_fitness = float(mean_fit[0])
            self.best_x = self.mean.copy()

        while not stopping_condition():
            try:
                self._update_eigen_decomposition()
            except Exception:
                self.C = np.eye(self.dim)
                self.B = np.eye(self.dim)
                self.D = np.ones(self.dim)
                self.invsqrt_C = np.eye(self.dim)

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
```