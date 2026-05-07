import numpy as np


class SimplifiedCovarianceAdaptiveSearch:
    """
    A simplified CMA-ES inspired optimizer with adaptive operator selection
    for the covariance matrix update strategy.
    
    Uses Thompson Sampling to select among multiple _update_covariance_matrix
    strategies during optimization.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.pop_size = max(4 * dim, 20)
        self.mu = self.pop_size // 2
        self.sigma = 30.0
        self.min_sigma = 1e-12
        self.max_sigma = 100.0

        self.weights = self._compute_recombination_weights()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        self.c_sigma = (self.mu_eff + 2.0) / (dim + self.mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (dim + 1.0)) - 1.0) + self.c_sigma
        self.c_c = (4.0 + self.mu_eff / dim) / (dim + 4.0 + 2.0 * self.mu_eff / dim)
        self.c_1 = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.c_mu_cov = min(1.0 - self.c_1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((dim + 2.0) ** 2 + self.mu_eff))

        self.chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

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

        # Adaptive operator selection via Thompson Sampling
        self.num_strategies = 7  # number of covariance update strategies
        self.strategy_alpha = np.ones(self.num_strategies)  # Beta dist param (successes)
        self.strategy_beta = np.ones(self.num_strategies)   # Beta dist param (failures)
        self.current_strategy = 0
        self.pre_update_best_f = np.inf
        
        # Sliding window for credit assignment
        self.window_size = 20
        self.strategy_history = []  # list of (strategy_idx, reward)

    def _compute_recombination_weights(self):
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights = raw_weights / np.sum(raw_weights)
        return weights

    def _initialize_state(self):
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
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        eigenvalues, self.B = np.linalg.eigh(self.C)
        eigenvalues = np.maximum(eigenvalues, 1e-20)
        self.D = np.sqrt(eigenvalues)
        inv_D = 1.0 / self.D
        self.invsqrtC = self.B @ np.diag(inv_D) @ self.B.T
        self.eigen_decomp_gen = self.generation

    def _sample_population_batch(self):
        z = np.random.randn(self.pop_size, self.dim)
        scaled = z * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        population = self.mean[np.newaxis, :] + self.sigma * rotated
        return population, z

    def _clip_to_bounds_batch(self, population):
        return np.clip(population, self.lb, self.ub)

    def _sort_by_fitness(self, population, fitness, z_vectors):
        safe_fitness = np.where(np.isnan(fitness), np.inf, fitness)
        sorted_indices = np.argsort(safe_fitness)
        return population[sorted_indices], safe_fitness[sorted_indices], z_vectors[sorted_indices]

    def _update_mean(self, sorted_population):
        old_mean = self.mean.copy()
        selected = sorted_population[:self.mu]
        self.mean = self.weights @ selected
        return old_mean

    def _update_evolution_paths(self, old_mean):
        mean_shift = (self.mean - old_mean) / self.sigma
        transformed = self.invsqrtC @ mean_shift

        c_s_complement = np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + \
                       np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * transformed

        p_sigma_norm = np.linalg.norm(self.p_sigma)
        threshold = (1.4 + 2.0 / (self.dim + 1.0)) * self.chi_n * c_s_complement
        h_sigma = 1.0 if p_sigma_norm < threshold else 0.0

        self.p_c = (1.0 - self.c_c) * self.p_c + \
                   h_sigma * np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * mean_shift

        return h_sigma

    def _select_strategy(self):
        """Select a covariance update strategy using Thompson Sampling."""
        samples = np.array([
            np.random.beta(self.strategy_alpha[i], self.strategy_beta[i])
            for i in range(self.num_strategies)
        ])
        self.current_strategy = int(np.argmax(samples))
        return self.current_strategy

    def _update_strategy_reward(self, pre_best_f, post_best_f):
        """Update Thompson Sampling parameters based on improvement."""
        idx = self.current_strategy
        improvement = float(pre_best_f - post_best_f)
        
        if improvement > 0:
            # Success: reward proportional to improvement magnitude
            reward = min(1.0, improvement / (abs(pre_best_f) + 1e-20))
            self.strategy_alpha[idx] += max(0.1, reward)
        else:
            # No improvement
            self.strategy_beta[idx] += 0.1
        
        # Record in sliding window
        self.strategy_history.append((idx, improvement))
        if len(self.strategy_history) > self.window_size * self.num_strategies:
            self.strategy_history = self.strategy_history[-self.window_size * self.num_strategies:]
        
        # Decay old evidence to allow adaptation
        decay = 0.995
        self.strategy_alpha *= decay
        self.strategy_beta *= decay
        # Keep minimum values
        self.strategy_alpha = np.maximum(self.strategy_alpha, 1.0)
        self.strategy_beta = np.maximum(self.strategy_beta, 1.0)

    # --- Strategy 0: Original ---
    def _update_cov_original(self, old_mean, sorted_population, h_sigma):
        rank_one = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        selected = sorted_population[:self.mu]
        diffs = (selected - old_mean[np.newaxis, :]) / self.sigma
        weighted_diffs = self.weights[:, np.newaxis] * diffs
        rank_mu = diffs.T @ weighted_diffs
        self.C = (1.0 - self.c_1 - self.c_mu_cov + delta_h * self.c_1) * self.C + \
                 self.c_1 * rank_one + \
                 self.c_mu_cov * rank_mu

    # --- Strategy 1: Active CMA-ES (variant_01) ---
    def _update_cov_active(self, old_mean, sorted_population, h_sigma):
        dim = self.dim
        rank_one = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        selected_best = sorted_population[:self.mu]
        diffs_pos = (selected_best - old_mean[np.newaxis, :]) / self.sigma
        weighted_diffs_pos = self.weights[:, np.newaxis] * diffs_pos
        rank_mu_pos = diffs_pos.T @ weighted_diffs_pos

        n_neg = min(self.mu, len(sorted_population) - self.mu)
        if n_neg > 0:
            selected_worst = sorted_population[-n_neg:]
            neg_raw_weights = np.log(n_neg + 0.5) - np.log(np.arange(1, n_neg + 1))
            neg_weights = -neg_raw_weights / np.sum(neg_raw_weights) * 0.5
            diffs_neg = (selected_worst - old_mean[np.newaxis, :]) / self.sigma
            weighted_diffs_neg = neg_weights[:, np.newaxis] * diffs_neg
            rank_mu_neg = diffs_neg.T @ weighted_diffs_neg
            rank_mu = rank_mu_pos - rank_mu_neg
        else:
            rank_mu = rank_mu_pos

        self.C = (1.0 - self.c_1 - self.c_mu_cov + delta_h * self.c_1) * self.C + \
                 self.c_1 * rank_one + \
                 self.c_mu_cov * rank_mu
        self.C = np.triu(self.C) + np.triu(self.C, 1).T

        eigenvals = np.linalg.eigvalsh(self.C)
        eigenvals = np.maximum(eigenvals, 1e-20)
        max_eigen = np.max(eigenvals)
        min_eigen = np.min(eigenvals)
        cond_limit = 1e12
        if max_eigen / min_eigen > cond_limit:
            scale_factor = np.sqrt(max_eigen / (min_eigen * cond_limit))
            self.C *= scale_factor
        min_eigen_check = np.min(np.linalg.eigvalsh(self.C))
        if min_eigen_check < 1e-15:
            self.C += np.eye(dim) * (1e-14 - min_eigen_check)

    # --- Strategy 2: Fitness-proportional weights (variant_02) ---
    def _update_cov_fitness_prop(self, old_mean, sorted_population, h_sigma):
        rank_one = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        selected = sorted_population[:self.mu]
        diffs = (selected - old_mean[np.newaxis, :]) / self.sigma

        fitness_weights = np.exp(-0.5 * np.arange(self.mu))
        fitness_weights = fitness_weights / np.sum(fitness_weights)
        weighted_diffs = fitness_weights[:, np.newaxis] * diffs
        rank_mu = diffs.T @ weighted_diffs

        adaptive_c1 = self.c_1 * (1.0 + 0.5 * np.log1p(self.sigma / 30.0))
        adaptive_c1 = np.clip(adaptive_c1, self.c_1, self.c_1 * 3.0)

        self.C = (1.0 - adaptive_c1 - self.c_mu_cov + delta_h * adaptive_c1) * self.C + \
                 adaptive_c1 * rank_one + \
                 self.c_mu_cov * rank_mu
        self.C = np.triu(self.C) + np.triu(self.C, 1).T

    # --- Strategy 3: Momentum-based (variant_05) ---
    def _update_cov_momentum(self, old_mean, sorted_population, h_sigma):
        mean_shift = self.mean - old_mean
        mean_shift_norm = np.linalg.norm(mean_shift)
        shift_ratio = mean_shift_norm / (self.sigma + 1e-20)

        rank_one = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        progress_factor = np.clip(shift_ratio / 10.0, 0.5, 2.0)
        adaptive_c_1 = self.c_1 * progress_factor
        adaptive_c_mu = self.c_mu_cov * progress_factor

        adaptive_momentum = np.clip(0.9 ** (1.0 / (1.0 + shift_ratio)), 0.85, 0.99)

        selected = sorted_population[:self.mu]
        diffs = (selected - self.mean[np.newaxis, :]) / self.sigma
        weighted_diffs = self.weights[:, np.newaxis] * diffs
        rank_mu = diffs.T @ weighted_diffs

        self.C = (adaptive_momentum * (1.0 - adaptive_c_1 - adaptive_c_mu) +
                  (1.0 - adaptive_momentum)) * self.C + \
                 adaptive_c_1 * rank_one + \
                 adaptive_c_mu * rank_mu + \
                 delta_h * self.c_1 * np.eye(self.dim)

        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        eigvals = np.linalg.eigvalsh(self.C)
        min_eig = np.min(eigvals)
        if min_eig < 1e-12:
            self.C += (1e-11 - min_eig) * np.eye(self.dim)

    # --- Strategy 4: Best-anchored (variant_06) ---
    def _update_cov_best_anchored(self, old_mean, sorted_population, h_sigma):
        dim = self.dim
        rank_one = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        best_solution = sorted_population[0]
        selected = sorted_population[:self.mu]
        diffs = (selected - best_solution[np.newaxis, :]) / max(self.sigma, 1e-30)
        weighted_diffs = self.weights[:, np.newaxis] * diffs
        rank_mu = diffs.T @ weighted_diffs

        cond = self.D[-1] / max(self.D[0], 1e-30)
        if cond > 1e7:
            damp = 0.2
        elif cond > 1e5:
            damp = 0.5
        else:
            damp = 1.0

        stagnation = (self.stagnation_counter > self.dim)
        c_mu_eff = self.c_mu_cov * damp
        if stagnation:
            c_mu_eff *= 2.5

        self.C = (1.0 - self.c_1 - c_mu_eff + delta_h * self.c_1) * self.C + \
                 self.c_1 * rank_one + \
                 c_mu_eff * rank_mu

        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        eigvals = np.linalg.eigvalsh(self.C)
        if np.min(eigvals) < 1e-15:
            self.C += np.eye(dim) * (1e-14 - np.min(eigvals))

    # --- Strategy 5: Diversity-modulated (variant_07) ---
    def _update_cov_diversity_mod(self, old_mean, sorted_population, h_sigma):
        rank_one = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        selected = sorted_population[:self.mu]
        diffs = (selected - old_mean[np.newaxis, :]) / self.sigma
        weighted_diffs = self.weights[:, np.newaxis] * diffs
        rank_mu = diffs.T @ weighted_diffs

        pop_spread = np.mean(np.std(selected, axis=0) ** 2)
        spread_ratio = pop_spread / (np.trace(self.C) / max(self.dim, 1) + 1e-20)

        path_norm = np.linalg.norm(self.p_c) + 1e-20
        path_adaptive = min(path_norm / np.sqrt(self.dim), 3.0)

        diversity_boost = 1.0 + 2.0 * np.tanh(spread_ratio - 0.5)
        condition_factor = 1.0 + path_adaptive

        c_1_adaptive = self.c_1 * diversity_boost * condition_factor
        c_mu_adaptive = self.c_mu_cov * diversity_boost * condition_factor

        c_1_adaptive = min(c_1_adaptive, 0.5)
        c_mu_adaptive = min(c_mu_adaptive, 0.9 - c_1_adaptive)

        self.C = (1.0 - c_1_adaptive - c_mu_adaptive + delta_h * c_1_adaptive) * self.C + \
                 c_1_adaptive * rank_one + \
                 c_mu_adaptive * rank_mu

        self.C = np.triu(self.C) + np.triu(self.C, 1).T

    # --- Strategy 6: Active with negative weights (variant_08) ---
    def _update_cov_active_neg(self, old_mean, sorted_population, h_sigma):
        rank_one = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        selected = sorted_population[:self.mu]
        diffs = (selected - old_mean[np.newaxis, :]) / self.sigma
        weighted_diffs = self.weights[:, np.newaxis] * diffs
        rank_mu_pos = diffs.T @ weighted_diffs

        n_neg = max(1, self.pop_size // 4)
        neg_start = max(self.mu, self.pop_size - n_neg)
        worst = sorted_population[neg_start:]
        if len(worst) >= 2:
            neg_diffs = (worst - old_mean[np.newaxis, :]) / self.sigma
            neg_raw_weights = np.log(self.pop_size + 0.5) - np.log(np.arange(neg_start + 1, self.pop_size + 1))
            neg_weights = neg_raw_weights / np.sum(np.abs(neg_raw_weights)) * (-0.5)
            neg_weighted_diffs = neg_weights[:, np.newaxis] * neg_diffs
            rank_mu_neg = neg_diffs.T @ neg_weighted_diffs
        else:
            rank_mu_neg = np.zeros((self.dim, self.dim))

        c_pos = self.c_mu_cov
        c_neg = min(0.5, c_pos * 0.5)
        self.C = (1.0 - self.c_1 - c_pos - c_neg + delta_h * self.c_1) * self.C + \
                 self.c_1 * rank_one + \
                 c_pos * rank_mu_pos + \
                 c_neg * rank_mu_neg

        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-12:
            self.C += (1e-11 - min_eig) * np.eye(self.dim)

    def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """
        Adaptive covariance matrix update using Thompson Sampling to select
        among multiple strategies.
        """
        # Save C in case strategy produces invalid result
        C_backup = self.C.copy()
        
        # Select strategy
        strategy_idx = self._select_strategy()
        
        try:
            if strategy_idx == 0:
                self._update_cov_original(old_mean, sorted_population, h_sigma)
            elif strategy_idx == 1:
                self._update_cov_active(old_mean, sorted_population, h_sigma)
            elif strategy_idx == 2:
                self._update_cov_fitness_prop(old_mean, sorted_population, h_sigma)
            elif strategy_idx == 3:
                self._update_cov_momentum(old_mean, sorted_population, h_sigma)
            elif strategy_idx == 4:
                self._update_cov_best_anchored(old_mean, sorted_population, h_sigma)
            elif strategy_idx == 5:
                self._update_cov_diversity_mod(old_mean, sorted_population, h_sigma)
            elif strategy_idx == 6:
                self._update_cov_active_neg(old_mean, sorted_population, h_sigma)
            
            # Validate result
            if not np.all(np.isfinite(self.C)):
                self.C = C_backup
                self.strategy_beta[strategy_idx] += 1.0
        except Exception:
            self.C = C_backup
            self.strategy_beta[strategy_idx] += 1.0

    def _update_step_size(self):
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0))
        self.sigma = np.clip(self.sigma, self.min_sigma, self.max_sigma)

    def _track_best(self, population, fitness):
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return
        valid_fitness = fitness[valid_mask]
        valid_pop = population[valid_mask]
        local_best_idx = np.argmin(valid_fitness)
        if valid_fitness[local_best_idx] < self.best_f:
            self.best_f = float(valid_fitness[local_best_idx])
            self.best_x = valid_pop[local_best_idx].copy()

    def _check_stagnation(self, best_gen_fitness):
        improvement = self.last_best_f - best_gen_fitness
        if improvement < 1e-12 * (1.0 + abs(self.last_best_f)):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        self.last_best_f = min(self.last_best_f, best_gen_fitness)

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
        old_best_x = self.best_x.copy() if self.best_x is not None else None
        old_best_f = self.best_f

        self._initialize_state()

        if old_best_x is not None:
            blend = np.random.uniform(0.0, 0.3)
            self.mean = blend * old_best_x + (1.0 - blend) * self.mean
            self.mean = np.clip(self.mean, self.lb, self.ub)

        self.best_x = old_best_x
        self.best_f = old_best_f

    def _should_update_eigen(self):
        update_interval = max(1, int(1.0 / (10.0 * self.dim * (self.c_1 + self.c_mu_cov))))
        return (self.generation - self.eigen_decomp_gen) >= update_interval

    def _handle_truncated_fitness(self, population, fitness, z_vectors):
        n_valid = len(fitness)
        return population[:n_valid], fitness[:n_valid], z_vectors[:n_valid]

    def _inject_best_into_population(self, population, z_vectors):
        if self.best_x is not None and self.generation % 5 == 0:
            population[-1] = self.best_x.copy()
            z_vectors[-1] = np.zeros(self.dim)
        return population, z_vectors

    def __call__(self, func, stopping_condition):
        self._initialize_state()
        self._update_eigen_decomposition()

        seed_pop = self._clip_to_bounds_batch(
            self.mean[np.newaxis, :] + self.sigma * np.random.randn(self.pop_size, self.dim)
        )
        seed_fit = func(seed_pop)
        if stopping_condition():
            self._track_best(seed_pop, seed_fit[:len(seed_pop)])
            return self.best_f, self.best_x

        self._track_best(seed_pop, seed_fit)

        while not stopping_condition():
            if self._should_update_eigen():
                try:
                    self._update_eigen_decomposition()
                except np.linalg.LinAlgError:
                    self._perform_restart()
                    self._update_eigen_decomposition()

            population, z_vectors = self._sample_population_batch()
            population = self._clip_to_bounds_batch(population)
            population, z_vectors = self._inject_best_into_population(population, z_vectors)
            population = self._clip_to_bounds_batch(population)

            fitness = func(population)
            if stopping_condition():
                if len(fitness) > 0:
                    population, fitness, z_vectors = self._handle_truncated_fitness(
                        population, fitness, z_vectors
                    )
                    self._track_best(population, fitness)
                break

            if len(fitness) < len(population):
                population, fitness, z_vectors = self._handle_truncated_fitness(
                    population, fitness, z_vectors
                )
                self._track_best(population, fitness)
                break

            self._track_best(population, fitness)

            sorted_pop, sorted_fit, sorted_z = self._sort_by_fitness(population, fitness, z_vectors)

            valid_count = np.sum(np.isfinite(sorted_fit))
            if valid_count < self.mu:
                self.generation += 1
                continue

            # Record pre-update best for credit assignment
            pre_update_best = float(self.best_f)

            old_mean = self._update_mean(sorted_pop)
            h_sigma = self._update_evolution_paths(old_mean)
            self._update_covariance_matrix(old_mean, sorted_pop, h_sigma)
            self._update_step_size()

            # Credit assignment for the selected strategy
            post_update_best = float(self.best_f)
            # Also use generation best as signal
            gen_best = float(sorted_fit[0]) if np.isfinite(sorted_fit[0]) else post_update_best
            self._update_strategy_reward(pre_update_best, post_update_best)

            best_gen_f = sorted_fit[0]
            if np.isfinite(best_gen_f):
                restarted = self._check_stagnation(float(best_gen_f))
                if restarted:
                    self._update_eigen_decomposition()

            self.generation += 1

        return self.best_f, self.best_x
