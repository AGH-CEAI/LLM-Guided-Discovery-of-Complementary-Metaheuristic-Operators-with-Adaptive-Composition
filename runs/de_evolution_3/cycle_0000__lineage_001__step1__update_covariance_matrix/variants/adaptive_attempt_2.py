import numpy as np


class SimplifiedCovarianceAdaptiveSearch:
    """
    A simplified CMA-ES inspired optimizer with adaptive operator selection
    for the covariance matrix update strategy.
    
    Uses Thompson Sampling to select among 5 covariance update strategies:
    0: original (standard rank-1 + rank-mu)
    1: variant_01 (population-wide covariance blend)
    2: variant_02 (active CMA with negative weights)
    3: variant_06 (exponential fitness-weighted)
    4: variant_07 (exponentially weighted rank-mu history)
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

        # Adaptive operator selection: Thompson Sampling
        self.num_operators = 5
        # Beta distribution parameters (alpha, beta) for each operator
        self.ts_alpha = np.ones(self.num_operators) * 1.0
        self.ts_beta = np.ones(self.num_operators) * 1.0
        
        # Sliding window for credit assignment
        self.window_size = 20
        self.op_history = []  # list of (operator_idx, reward)
        self.current_operator = 0
        
        # History for rank-mu EWMA (variant_07)
        self._rank_mu_hist = None

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
        self._rank_mu_hist = None

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

    def _sample_population_batch(self):
        dim = self.dim
        lam = self.pop_size
        z = np.random.randn(lam, dim)
        sqrt_eig = np.sqrt(self.eigenvalues)
        y = self.eigenvectors @ (sqrt_eig[:, None] * z.T)
        population = self.mean[None, :] + self.sigma * y.T
        return population, z

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

    def _select_operator(self):
        """Select operator using Thompson Sampling."""
        samples = np.array([
            np.random.beta(self.ts_alpha[i], self.ts_beta[i])
            for i in range(self.num_operators)
        ])
        self.current_operator = int(np.argmax(samples))
        return self.current_operator

    def _update_operator_stats(self, operator_idx, reward):
        """Update Thompson Sampling parameters based on reward."""
        reward = float(reward)
        # Clamp reward to [0, 1]
        reward = max(0.0, min(1.0, reward))
        
        self.op_history.append((int(operator_idx), reward))
        
        # Keep sliding window
        if len(self.op_history) > self.window_size:
            self.op_history = self.op_history[-self.window_size:]
        
        # Recompute alpha/beta from sliding window
        self.ts_alpha = np.ones(self.num_operators) * 1.0
        self.ts_beta = np.ones(self.num_operators) * 1.0
        for op_idx, rew in self.op_history:
            op_idx = int(op_idx)
            if 0 <= op_idx < self.num_operators:
                self.ts_alpha[op_idx] += float(rew)
                self.ts_beta[op_idx] += 1.0 - float(rew)

    def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        """Adaptive covariance update: select among strategies using Thompson Sampling."""
        # Save C before update for rollback if needed
        C_backup = self.C.copy()
        
        op = self._select_operator()
        
        try:
            if op == 0:
                self._cov_update_original(old_mean, sorted_population, h_sigma)
            elif op == 1:
                self._cov_update_variant01(old_mean, sorted_population, h_sigma)
            elif op == 2:
                self._cov_update_variant02(old_mean, sorted_population, h_sigma)
            elif op == 3:
                self._cov_update_variant06(old_mean, sorted_population, h_sigma)
            elif op == 4:
                self._cov_update_variant07(old_mean, sorted_population, h_sigma)
            else:
                self._cov_update_original(old_mean, sorted_population, h_sigma)
            
            # Validate the result
            if not np.all(np.isfinite(self.C)):
                self.C = C_backup.copy()
                # Penalize this operator
                self._update_operator_stats(op, 0.0)
                return
                
        except Exception:
            self.C = C_backup.copy()
            self._update_operator_stats(op, 0.0)
            return
        
        # Store operator index for deferred credit assignment
        self._pending_operator = op

    def _cov_update_original(self, old_mean, sorted_population, h_sigma):
        """Original: standard rank-1 + rank-mu update."""
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

    def _cov_update_variant01(self, old_mean, sorted_population, h_sigma):
        """Variant 01: population-wide covariance blend."""
        dim = self.dim
        lam = self.pop_size
        selected = sorted_population[:self.mu]
        rank1 = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        diff = (selected - old_mean[None, :]) / self.sigma
        rank_mu = np.zeros((dim, dim))
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(diff[i], diff[i])
        pop_diff = (sorted_population - old_mean[None, :]) / self.sigma
        pop_cov = np.cov(pop_diff.T, bias=True) if lam > 1 else np.zeros((dim, dim))
        c_pop = min(0.1, self.c_mu_cov * 0.5)
        self.C = ((1.0 - self.c_1 - self.c_mu_cov - c_pop + delta_h * self.c_1) * self.C +
                   self.c_1 * rank1 +
                   self.c_mu_cov * rank_mu +
                   c_pop * pop_cov)
        self.C = 0.5 * (self.C + self.C.T)
        diag = np.diag(self.C)
        np.fill_diagonal(self.C, np.maximum(diag, 1e-20))

    def _cov_update_variant02(self, old_mean, sorted_population, h_sigma):
        """Variant 02: active CMA with negative weights."""
        dim = self.dim
        lam = self.pop_size
        rank1 = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        diff = (sorted_population - old_mean[None, :]) / self.sigma
        mu_pos = self.mu
        raw_pos = np.log(mu_pos + 0.5) - np.log(np.arange(1, mu_pos + 1))
        pos_weights = raw_pos / np.sum(raw_pos)
        mu_neg = lam - self.mu
        if mu_neg > 0:
            raw_neg = np.log(mu_neg + 0.5) - np.log(np.arange(1, mu_neg + 1))
            neg_weights = -raw_neg / np.sum(raw_neg) * 0.4
        else:
            neg_weights = np.array([])
        all_weights = np.concatenate([pos_weights, neg_weights])
        all_weights = np.clip(all_weights, -0.5, None)
        weight_sum = np.sum(np.abs(all_weights))
        if weight_sum > 1e-30:
            all_weights = all_weights / weight_sum
        rank_mu = np.zeros((dim, dim))
        for i in range(lam):
            w = all_weights[i]
            if abs(w) > 1e-30:
                rank_mu += w * np.outer(diff[i], diff[i])
        self.C = ((1.0 - self.c_1 - self.c_mu_cov + delta_h * self.c_1) * self.C +
                   self.c_1 * rank1 +
                   self.c_mu_cov * rank_mu)
        self.C = 0.5 * (self.C + self.C.T)
        try:
            min_eig = np.min(np.linalg.eigvalsh(self.C))
            if min_eig < 1e-15:
                self.C += (1e-14 - min_eig) * np.eye(dim)
        except np.linalg.LinAlgError:
            pass

    def _cov_update_variant06(self, old_mean, sorted_population, h_sigma):
        """Variant 06: exponential fitness-weighted rank-mu update."""
        dim = self.dim
        selected = sorted_population[:self.mu]
        rank1 = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        inv_ranks = 1.0 / np.arange(1, self.mu + 1)
        exp_weights = np.exp(-1.5 * inv_ranks)
        exp_weights = exp_weights / np.sum(exp_weights)
        mu_eff_exp = 1.0 / np.sum(exp_weights ** 2)
        c_mu_adaptive = min(0.9, self.c_mu_cov * (1.0 + 0.5 * (mu_eff_exp - 1.0)))
        diff = (selected - old_mean[None, :]) / self.sigma
        rank_mu = np.zeros((dim, dim))
        for i in range(self.mu):
            rank_mu += exp_weights[i] * np.outer(diff[i], diff[i])
        self.C = ((1.0 - self.c_1 - c_mu_adaptive + delta_h * self.c_1) * self.C +
                   self.c_1 * rank1 +
                   c_mu_adaptive * rank_mu)
        self.C = 0.5 * (self.C + self.C.T)
        self.C.flat[::dim + 1] += 1e-10 * np.ones(dim)

    def _cov_update_variant07(self, old_mean, sorted_population, h_sigma):
        """Variant 07: exponentially weighted rank-mu history."""
        dim = self.dim
        selected = sorted_population[:self.mu]
        rank1 = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)
        diff = (selected - old_mean[None, :]) / self.sigma
        rank_mu = np.zeros((dim, dim))
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(diff[i], diff[i])
        if self._rank_mu_hist is None:
            self._rank_mu_hist = rank_mu.copy()
        c_mu_hist = self.c_mu_cov * 0.1
        self._rank_mu_hist = (1.0 - c_mu_hist) * self._rank_mu_hist + c_mu_hist * rank_mu
        rank_mu_blend = 0.5 * rank_mu + 0.5 * self._rank_mu_hist
        self.C = ((1.0 - self.c_1 - self.c_mu_cov + delta_h * self.c_1) * self.C +
                   self.c_1 * rank1 +
                   self.c_mu_cov * rank_mu_blend)
        self.C = 0.5 * (self.C + self.C.T)
        diag = np.diag(self.C)
        min_diag = 1e-20
        if np.any(diag < min_diag):
            np.fill_diagonal(self.C, np.maximum(diag, min_diag))

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
        self._pending_operator = 0
        self._prev_gen_best = np.inf

        # Reset adaptive operator selection
        self.ts_alpha = np.ones(self.num_operators) * 1.0
        self.ts_beta = np.ones(self.num_operators) * 1.0
        self.op_history = []

        initial_pop = self._clip_to_bounds_batch(
            np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        )
        if not stopping_condition():
            initial_fit = func(initial_pop)
            if stopping_condition():
                self._update_best(initial_pop[:len(initial_fit)], initial_fit)
                return self.best_f, self.best_x
            self._update_best(initial_pop[:len(initial_fit)], initial_fit)
            valid_mask = np.isfinite(initial_fit)
            if np.any(valid_mask):
                best_idx = np.argmin(np.where(valid_mask, initial_fit, np.inf))
                self.mean = initial_pop[best_idx].copy()
                self._prev_gen_best = float(initial_fit[best_idx])

        generation = 0
        while not stopping_condition():
            population, z_vectors = self._sample_population_batch()
            population = self._clip_to_bounds_batch(population)

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

            sorted_pop, sorted_fit, sorted_z = self._sort_by_fitness(
                population, fitness, z_vectors
            )

            # Credit assignment for previous operator
            if generation > 0:
                valid_fit = sorted_fit[np.isfinite(sorted_fit)]
                if len(valid_fit) > 0:
                    current_gen_best = float(np.min(valid_fit))
                    # Compute improvement-based reward
                    if np.isfinite(self._prev_gen_best) and self._prev_gen_best > 0:
                        relative_improvement = (self._prev_gen_best - current_gen_best) / (abs(self._prev_gen_best) + 1e-30)
                        # Map to [0, 1] using sigmoid-like transformation
                        reward = 1.0 / (1.0 + np.exp(-100.0 * relative_improvement))
                    elif np.isfinite(self._prev_gen_best):
                        improvement = self._prev_gen_best - current_gen_best
                        reward = 1.0 / (1.0 + np.exp(-improvement))
                    else:
                        reward = 0.5
                    
                    reward = float(max(0.0, min(1.0, reward)))
                    self._update_operator_stats(self._pending_operator, reward)
                    self._prev_gen_best = current_gen_best

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
                self._rank_mu_hist = None

            generation += 1

        return self.best_f, self.best_x
