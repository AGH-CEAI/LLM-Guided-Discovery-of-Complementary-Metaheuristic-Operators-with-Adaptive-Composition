

```python
import numpy as np


class SimplifiedCovarianceAdaptiveSearch:
    """
    A simplified CMA-ES inspired optimizer with adaptive operator selection
    for the _sample_population_batch strategy. Uses Thompson Sampling to
    select among multiple sampling strategies during optimization.
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
        self.n_strategies = 5  # number of sampling strategies
        # Beta distribution parameters for Thompson Sampling
        self.ts_alpha = np.ones(self.n_strategies)  # successes + 1
        self.ts_beta = np.ones(self.n_strategies)   # failures + 1
        self.current_strategy = 0
        self.strategy_window_size = 20
        self.strategy_rewards = [[] for _ in range(self.n_strategies)]
        
        # Archive for variant_06 strategy
        self.sampling_archive = []

    def _initialize_strategy_params(self):
        """Initialize all strategy parameters for covariance adaptation."""
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
        """Initialize the evolutionary state."""
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
        """Eigendecompose the covariance matrix for sampling."""
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

    def _select_strategy(self):
        """Select sampling strategy using Thompson Sampling."""
        samples = np.array([
            np.random.beta(self.ts_alpha[i], self.ts_beta[i])
            for i in range(self.n_strategies)
        ])
        self.current_strategy = int(np.argmax(samples))
        return self.current_strategy

    def _update_strategy_reward(self, strategy_idx, reward):
        """Update Thompson Sampling parameters based on reward."""
        reward = float(reward)
        strategy_idx = int(strategy_idx)
        
        self.strategy_rewards[strategy_idx].append(reward)
        # Keep sliding window
        if len(self.strategy_rewards[strategy_idx]) > self.strategy_window_size:
            self.strategy_rewards[strategy_idx] = self.strategy_rewards[strategy_idx][-self.strategy_window_size:]
        
        # Convert reward to success/failure for Beta distribution
        # reward > 0 means improvement
        if reward > 0:
            self.ts_alpha[strategy_idx] += 1.0
        else:
            self.ts_beta[strategy_idx] += 1.0
        
        # Decay to prevent lock-in (slowly forget old observations)
        decay = 0.995
        self.ts_alpha = 1.0 + (self.ts_alpha - 1.0) * decay
        self.ts_beta = 1.0 + (self.ts_beta - 1.0) * decay

    def _sample_strategy_0_original(self):
        """Original CMA-ES sampling."""
        dim = self.dim
        lam = self.pop_size
        z = np.random.randn(lam, dim)
        sqrt_eig = np.sqrt(self.eigenvalues)
        y = self.eigenvectors @ (sqrt_eig[:, None] * z.T)
        population = self.mean[None, :] + self.sigma * y.T
        return population, z

    def _sample_strategy_1_antithetic(self):
        """Antithetic variates sampling (variant_01)."""
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

    def _sample_strategy_2_hybrid_center(self):
        """Hybrid center sampling (variant_02)."""
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

    def _sample_strategy_3_diversity_injection(self):
        """Restart-triggered diversity injection (variant_04)."""
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

    def _sample_strategy_4_archive_guided(self):
        """Archive-guided diversity sampling (variant_06)."""
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

    def _sample_population_batch(self):
        """Adaptively select and use the best sampling strategy."""
        strategy = self._select_strategy()
        
        if strategy == 0:
            return self._sample_strategy_0_original()
        elif strategy == 1:
            return self._sample_strategy_1_antithetic()
        elif strategy == 2:
            return self._sample_strategy_2_hybrid_center()
        elif strategy == 3:
            return self._sample_strategy_3_diversity_injection()
        elif strategy == 4:
            return self._sample_strategy_4_archive_guided()
        else:
            return self._sample_strategy_0_original()

    def _clip_to_bounds_batch(self, population):
        """Clip all candidates to the search bounds."""
        return np.clip(population, self.lb, self.ub)

    def _sort_by_fitness(self, population, fitness, z_vectors):
        """Sort population by fitness (ascending = better)."""
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return population, fitness, z_vectors

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
            self.best_f = float(fitness[best_idx])
            self.best_x = population[best_idx].copy()

    def _update_mean(self, sorted_population):
        """Update the distribution mean using weighted recombination."""
        old_mean = self.mean.copy()
        selected = sorted_population[:self.mu]
        self.mean = self.weights @ selected
        return old_mean

    def _update_evolution_path_sigma(self, old_mean):
        """Update the cumulative step-size adaptation path."""
        inv_sqrt_eig = 1.0 / np.sqrt(self.eigenvalues + 1e-20)
        inv_sqrt_C = self.eigenvectors @ np.diag(inv_sqrt_eig) @ self.eigenvectors.T

        displacement = (self.mean - old_mean) / self.sigma
        self.p_sigma = ((1.0 - self.c_sigma) * self.p_sigma +
                        np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) *
                        inv_sqrt_C @ displacement)

    def _update_evolution_path_c(self, old_mean):
        """Update the cumulative covariance adaptation path."""
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
        """Update the covariance matrix with rank-1 and rank-mu updates."""
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
        """Adapt the global step size sigma."""
        p_sigma_norm = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp(
            (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)
        )
        self.sigma = np.clip(self.sigma, 1e-20, 1e5)

    def _check_stagnation(self, fitness):
        """Detect stagnation and trigger restart if needed."""
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
        """Check if covariance matrix condition number is too high."""
        ratio = np.max(self.eigenvalues) / (np.min(self.eigenvalues) + 1e-30)
        return ratio > 1e14

    def _restart(self):
        """Perform a restart with potentially increased population size."""
        self.pop_size = min(int(self.pop_size * 1.5), 800)
        self.mu = self.pop_size // 2
        self._initialize_strategy_params()
        self._initialize_state()
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
        decompose_freq = max(1, int(1.0 / (10.0 * self.dim * (self.c_1 + self.c_mu_cov))))
        decompose_freq = min(decompose_freq, self.pop_size)
        return self.gen_since_eigen >= decompose_freq

    def _update_archive(self, sorted_population, sorted_fitness):
        """Update the sampling archive with good solutions for strategy 4."""
        n_add = min(self.mu, len(sorted_population))
        for i in range(n_add):
            if np.isfinite(sorted_fitness[i]):
                self.sampling_archive.append(sorted_population[i].copy())
        # Keep archive bounded
        max_archive = max(100, 10 * self.dim)
        if len(self.sampling_archive) > max_archive:
            self.sampling_archive = self.sampling_archive[-max_archive:]

    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        self._initialize_state()
        self._decompose_covariance()
        self.best_f = np.inf
        self.best_x = np.zeros(self.dim)
        self.sampling_archive = []

        # Reset Thompson Sampling
        self.ts_alpha = np.ones(self.n_strategies)
        self.ts_beta = np.ones(self.n_strategies)
        self.strategy_rewards = [[] for _ in range(self.n_strategies)]

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

            valid_mask = np.isfinite(initial_fit)
            if np.any(valid_mask):
                best_idx = np.argmin(np.where(valid_mask, initial_fit, np.inf))
                self.mean = initial_pop[best_idx].copy()

        generation = 0
        while not stopping_condition():
            # Record pre-generation best for reward computation
            pre_best_f = float(self.best_f) if np.isfinite(self.best_f) else 1e30

            # Sample new population using adaptive strategy selection
            population, z_vectors = self._sample_population_batch()
            used_strategy = int(self.current_strategy)
            
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

            # Update archive for strategy 4
            self._update_archive(sorted_pop, sorted_fit)

            # Compute reward for the strategy that was used
            post_best_f = float(self.best_f) if np.isfinite(self.best_f) else 1e30
            # Reward based on improvement in best fitness
            improvement = float(pre_best_f - post_best_f)
            # Also consider median fitness improvement as secondary signal
            valid_fit = sorted_fit[np.isfinite(sorted_fit)]
            if len(valid_fit) > 0:
                median_fit = float(np.median(valid_fit))
                # Normalize reward: positive if improvement happened
                reward = improvement
                # Add a small bonus for good median fitness relative to best
                if np.isfinite(median_fit) and np.isfinite(pre_best_f) and pre_best_f > 0:
                    relative_quality = float((pre_best_f - median_fit) / (abs(pre_best_f) + 1e-20))
                    reward += max(0.0, relative_quality) * 0.1
            else:
                reward = improvement
            
            self._update_strategy_reward(used_strategy, reward)

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
                self.sampling_archive = []

            generation += 1

        return self.best_f, self.best_x
```