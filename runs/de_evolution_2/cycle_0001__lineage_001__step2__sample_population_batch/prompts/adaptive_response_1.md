

```python
import numpy as np


class SimplifiedCovarianceAdaptiveSearch:
    """
    A simplified CMA-ES inspired optimizer with adaptive operator selection
    for the _sample_population_batch strategy. Uses a sliding window
    Thompson Sampling approach to select among multiple sampling strategies.
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

        # Adaptive operator selection state
        self.num_strategies = 7  # number of sampling strategies
        self.strategy_successes = np.ones(self.num_strategies, dtype=float)
        self.strategy_failures = np.ones(self.num_strategies, dtype=float)
        self.current_strategy = 0
        self.window_size = 50
        self.strategy_history = []  # list of (strategy_idx, reward)
        self.prev_best_f = np.inf

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

    def _select_strategy(self):
        """Use Thompson Sampling to select the next sampling strategy."""
        samples = np.array([
            np.random.beta(max(self.strategy_successes[i], 0.01),
                           max(self.strategy_failures[i], 0.01))
            for i in range(self.num_strategies)
        ])
        self.current_strategy = int(np.argmax(samples))
        return self.current_strategy

    def _update_strategy_reward(self, strategy_idx, reward):
        """Update Thompson Sampling parameters with reward signal."""
        reward = float(reward)
        strategy_idx = int(strategy_idx)
        self.strategy_history.append((strategy_idx, reward))

        # Sliding window: decay old observations
        if len(self.strategy_history) > self.window_size:
            old_idx, old_reward = self.strategy_history.pop(0)
            old_idx = int(old_idx)
            old_reward = float(old_reward)
            if old_reward > 0:
                self.strategy_successes[old_idx] = max(1.0, self.strategy_successes[old_idx] - old_reward)
            else:
                self.strategy_failures[old_idx] = max(1.0, self.strategy_failures[old_idx] + old_reward - 1.0)

        if reward > 0:
            self.strategy_successes[strategy_idx] += reward
        else:
            self.strategy_failures[strategy_idx] += 1.0

    # --- Strategy 0: Original (standard CMA-ES sampling) ---
    def _sample_strategy_original(self):
        z = np.random.randn(self.pop_size, self.dim)
        scaled = z * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        population = self.mean[np.newaxis, :] + self.sigma * rotated
        return population, z

    # --- Strategy 1: Cholesky-based (variant_01) ---
    def _sample_strategy_cholesky(self):
        z = np.random.randn(self.pop_size, self.dim)
        regularized_C = self.C + 1e-12 * np.eye(self.dim)
        try:
            L = np.linalg.cholesky(regularized_C)
        except np.linalg.LinAlgError:
            try:
                L = np.linalg.cholesky(regularized_C + 1e-8 * np.eye(self.dim))
            except np.linalg.LinAlgError:
                return self._sample_strategy_original()
        noise = z @ L.T
        population = self.mean[np.newaxis, :] + self.sigma * noise
        return population, z

    # --- Strategy 2: Stagnation-triggered diversity injection (variant_03) ---
    def _sample_strategy_diversity_injection(self):
        pop_size = self.pop_size
        dim = self.dim
        stagnation_fraction = min(self.stagnation_counter / max(1, 20 + dim), 1.0)
        injection_rate = 0.05 + 0.45 * stagnation_fraction
        n_uniform = int(np.round(pop_size * injection_rate))
        n_cmaes = pop_size - n_uniform

        z_cmaes = np.random.randn(n_cmaes, dim)
        scaled = z_cmaes * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        cmaes_pop = self.mean[np.newaxis, :] + self.sigma * rotated

        if n_uniform > 0:
            uniform_pop = np.random.uniform(self.lb, self.ub, size=(n_uniform, dim))
            if self.best_x is not None and n_uniform >= 2:
                bias_count = max(1, n_uniform // 3)
                uniform_pop[:bias_count] = self.best_x + np.random.randn(bias_count, dim) * self.sigma * 0.5
                uniform_pop[:bias_count] = np.clip(uniform_pop[:bias_count], self.lb, self.ub)
            population = np.vstack([cmaes_pop, uniform_pop])
            z_vectors = np.vstack([z_cmaes, np.zeros((n_uniform, dim))])
        else:
            population = cmaes_pop
            z_vectors = z_cmaes

        return population, z_vectors

    # --- Strategy 3: Mirrored sampling (variant_04) ---
    def _sample_strategy_mirrored(self):
        half_size = self.pop_size // 2
        z = np.random.randn(half_size, self.dim)
        z_mirrored = -z
        z_full = np.vstack([z, z_mirrored])
        # If pop_size is odd, add one more
        if z_full.shape[0] < self.pop_size:
            z_extra = np.random.randn(self.pop_size - z_full.shape[0], self.dim)
            z_full = np.vstack([z_full, z_extra])
        scaled = z_full * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        population = self.mean[np.newaxis, :] + self.sigma * rotated
        return population, z_full

    # --- Strategy 4: Bimodal sampling (variant_05) ---
    def _sample_strategy_bimodal(self):
        z_all = np.random.randn(self.pop_size, self.dim)
        cond_num = np.max(self.D) / (np.min(self.D) + 1e-20)
        exploit_ratio = np.clip(1.0 / (1.0 + np.log1p(cond_num) / 5.0), 0.1, 0.7)

        principal_dir = self.B[:, -1]
        elite_center = self.mean + 0.3 * self.sigma * principal_dir * self.D[-1] * 0.1

        n_exploit = self.pop_size // 2
        exploit_z = z_all[:n_exploit]
        exploit_scaled = exploit_z * (self.D * exploit_ratio)[np.newaxis, :]
        exploit_rotated = exploit_scaled @ self.B.T
        pop_exploit = elite_center[np.newaxis, :] + self.sigma * exploit_rotated

        n_explore = self.pop_size - n_exploit
        explore_z = z_all[n_exploit:]  # fixed: was z_all[n_explore:] which is wrong
        explore_scaled = explore_z * self.D[np.newaxis, :]
        explore_rotated = explore_scaled @ self.B.T
        pop_explore = self.mean[np.newaxis, :] + self.sigma * explore_rotated

        population = np.vstack([pop_exploit, pop_explore])
        z_vectors = z_all.copy()

        return population, z_vectors

    # --- Strategy 5: Deme-based niching (variant_06) ---
    def _sample_strategy_deme(self):
        pop_size = self.pop_size
        dim = self.dim
        n_demes = max(2, int(np.sqrt(pop_size)))
        deme_size = pop_size // n_demes

        all_candidates = []
        all_z = []

        eig_ratio = np.max(self.D) / np.min(self.D) if np.min(self.D) > 1e-20 else 1e7

        for d in range(n_demes):
            if eig_ratio > 10.0:
                axis_weight = (d - n_demes / 2) / max(n_demes / 2, 1)
                major_axis = self.B[:, -1]
                deme_center = self.mean + axis_weight * 2.0 * self.sigma * self.D[-1] * major_axis
            else:
                angle = 2.0 * np.pi * d / n_demes
                spread_radius = self.sigma * min(5.0, np.mean(self.D))
                direction = np.zeros(dim)
                direction[0] = np.cos(angle)
                if dim > 1:
                    direction[1] = np.sin(angle)
                deme_center = self.mean + spread_radius * direction

            z_deme = np.random.randn(deme_size, dim)
            scaled = z_deme * self.D[np.newaxis, :]
            rotated = scaled @ self.B.T
            deme_pop = deme_center[np.newaxis, :] + self.sigma * 0.5 * rotated
            all_candidates.append(deme_pop)
            all_z.append(z_deme)

        remainder = pop_size - n_demes * deme_size
        if remainder > 0:
            z_rem = np.random.randn(remainder, dim)
            scaled = z_rem * self.D[np.newaxis, :]
            rotated = scaled @ self.B.T
            rem_pop = self.mean[np.newaxis, :] + self.sigma * rotated
            all_candidates.append(rem_pop)
            all_z.append(z_rem)

        population = np.concatenate(all_candidates, axis=0)
        z_vectors = np.concatenate(all_z, axis=0)

        return population, z_vectors

    # --- Strategy 6: Hybrid multi-strategy (variant_09) ---
    def _sample_strategy_hybrid(self):
        pop_size = self.pop_size
        dim = self.dim

        max_D = self.D[-1] if len(self.D) > 0 else 1.0
        min_D = max(self.D[0], 1e-20) if len(self.D) > 0 else 1.0
        cond = max_D / min_D
        explore_frac = 0.35 if cond > 1e5 else 0.25
        n_explore = max(2, int(pop_size * explore_frac))
        n_boundary = max(1, int(pop_size * 0.10))
        n_cmaes = pop_size - n_explore - n_boundary

        z_main = np.random.randn(n_cmaes, dim)
        scaled = z_main * self.D[np.newaxis, :]
        rotated = scaled @ self.B.T
        pop_main = self.mean[np.newaxis, :] + self.sigma * rotated

        z_explore = np.random.randn(n_explore, dim)
        pop_explore = np.random.uniform(self.lb, self.ub, size=(n_explore, dim))

        range_arr = self.ub - self.lb
        z_boundary = np.zeros((n_boundary, dim))
        pop_boundary = np.zeros((n_boundary, dim))
        for i in range(n_boundary):
            for d in range(dim):
                if np.random.rand() < 0.3:
                    if np.random.rand() < 0.5:
                        pop_boundary[i, d] = np.random.uniform(self.lb, self.lb + 0.1 * range_arr)
                    else:
                        pop_boundary[i, d] = np.random.uniform(self.ub - 0.1 * range_arr, self.ub)
                else:
                    pop_boundary[i, d] = self.mean[d] + np.random.uniform(-1.0, 1.0) * self.sigma

        pop_explore = np.clip(pop_explore, self.lb, self.ub)
        pop_boundary = np.clip(pop_boundary, self.lb, self.ub)

        population = np.vstack([pop_main, pop_explore, pop_boundary])
        z_vectors = np.vstack([z_main, z_explore, z_boundary])

        return population, z_vectors

    def _sample_population_batch(self):
        """Adaptive sampling: select strategy via Thompson Sampling."""
        strategy = self._select_strategy()
        
        try:
            if strategy == 0:
                result = self._sample_strategy_original()
            elif strategy == 1:
                result = self._sample_strategy_cholesky()
            elif strategy == 2:
                result = self._sample_strategy_diversity_injection()
            elif strategy == 3:
                result = self._sample_strategy_mirrored()
            elif strategy == 4:
                result = self._sample_strategy_bimodal()
            elif strategy == 5:
                result = self._sample_strategy_deme()
            elif strategy == 6:
                result = self._sample_strategy_hybrid()
            else:
                result = self._sample_strategy_original()
            
            pop, z = result
            # Safety: ensure correct shapes
            if pop.shape[0] != self.pop_size:
                # Fallback to original
                result = self._sample_strategy_original()
                pop, z = result
            
            # Check for NaN/Inf
            if not np.all(np.isfinite(pop)):
                pop = np.nan_to_num(pop, nan=0.0, posinf=self.ub, neginf=self.lb)
            if not np.all(np.isfinite(z)):
                z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
                
            return pop, z
            
        except Exception:
            # Fallback to original strategy on any error
            self.current_strategy = 0
            return self._sample_strategy_original()

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
        mean_shift = (self.mean - old_mean) / max(self.sigma, 1e-30)
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

    def _update_covariance_matrix(self, old_mean, sorted_population, h_sigma):
        rank_one = np.outer(self.p_c, self.p_c)
        delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c)

        selected = sorted_population[:self.mu]
        diffs = (selected - old_mean[np.newaxis, :]) / max(self.sigma, 1e-30)
        weighted_diffs = self.weights[:, np.newaxis] * diffs
        rank_mu = diffs.T @ weighted_diffs

        self.C = (1.0 - self.c_1 - self.c_mu_cov + delta_h * self.c_1) * self.C + \
                 self.c_1 * rank_one + \
                 self.c_mu_cov * rank_mu

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
        self.prev_best_f = float(self.best_f)

        while not stopping_condition():
            if self._should_update_eigen():
                try:
                    self._update_eigen_decomposition()
                except np.linalg.LinAlgError:
                    self._perform_restart()
                    self._update_eigen_decomposition()

            # Record best before this generation
            pre_gen_best = float(self.best_f)

            # Sample new population (adaptive strategy selection happens inside)
            population, z_vectors = self._sample_population_batch()
            used_strategy = int(self.current_strategy)
            
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

            # Compute reward for the strategy used
            post_gen_best = float(self.best_f)
            # Reward = relative improvement (normalized)
            if np.isfinite(pre_gen_best) and np.isfinite(post_gen_best):
                improvement = pre_gen_best - post_gen_best
                # Also reward based on quality of best individual this gen
                valid_mask = np.isfinite(fitness)
                if np.any(valid_mask):
                    gen_best = float(np.min(fitness[valid_mask]))
                    # Normalized reward: improvement relative to scale
                    scale = max(abs(pre_gen_best), 1e-10)
                    reward = float(max(0.0, improvement / scale))
                    # Also give small reward for producing good median fitness
                    median_f = float(np.median(fitness[valid_mask]))
                    if np.isfinite(median_f) and np.isfinite(pre_gen_best):
                        median_reward = float(max(0.0, (pre_gen_best - gen_best) / max(scale, 1.0)))
                        reward = reward + 0.1 * median_reward
                else:
                    reward = 0.0
            else:
                reward = 0.0

            self._update_strategy_reward(used_strategy, reward)

            sorted_pop, sorted_fit, sorted_z = self._sort_by_fitness(population, fitness, z_vectors)

            valid_count = np.sum(np.isfinite(sorted_fit))
            if valid_count < self.mu:
                self.generation += 1
                continue

            old_mean = self._update_mean(sorted_pop)
            h_sigma = self._update_evolution_paths(old_mean)
            self._update_covariance_matrix(old_mean, sorted_pop, h_sigma)
            self._update_step_size()

            best_gen_f = sorted_fit[0]
            if np.isfinite(best_gen_f):
                restarted = self._check_stagnation(float(best_gen_f))
                if restarted:
                    self._update_eigen_decomposition()

            self.generation += 1

        return self.best_f, self.best_x
```