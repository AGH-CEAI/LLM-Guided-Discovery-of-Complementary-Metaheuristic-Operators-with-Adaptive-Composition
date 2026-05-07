import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best initialization
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 8 initialization strategies (original + 7 best-performing variants)
    - Thompson Sampling for operator selection
    - Sliding window credit assignment
    - Automatic restart on stagnation
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(240, max(80, 8 * dim))
        self.bounds = (-100.0, 100.0)
        self.lb = np.full(dim, -100.0)
        self.ub = np.full(dim, 100.0)
        
        # CMA-ES inspired parameters
        self.mu = self.NP // 4
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        
        # Learning rates
        self.cs = (self.mueff + 2.0) / (self.dim + self.mueff + 5.0)
        self.cc = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        self.ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        self.damping = 1.0 + np.maximum(0.0, np.sqrt(self.mueff) - 1.0)
        
        # Restart parameters
        self.max_stagnation = 50 + self.dim * 3
        self.min_diversity = 1e-6 * (self.ub[0] - self.lb[0])
        
        # Adaptive initialization operator selection parameters
        self.num_init_operators = 8
        self.init_operator_names = [
            'original', 'variant_01', 'variant_04', 'variant_05',
            'variant_06', 'variant_07', 'variant_08', 'variant_09'
        ]
        
        # Thompson Sampling with Beta distributions for init operators
        self.init_alpha = np.ones(self.num_init_operators)
        self.init_beta = np.ones(self.num_init_operators)
        
        # Sliding window for init operator credit assignment
        self.init_reward_window_size = 8
        self.init_operator_rewards = {i: [] for i in range(self.num_init_operators)}
        self.init_operator_counts = np.zeros(self.num_init_operators)
        self.init_selection_counts = np.zeros(self.num_init_operators)
        
        # Restart archive for variant_10-style guidance
        self.restart_archive = []
        
        # Runtime state
        self.generation = 0
        self.current_init_operator = 0
        self.L = None
    
    def _clip_to_bounds(self, x):
        """Clip solution to bounds."""
        return np.clip(x, self.lb, self.ub)
    
    def _ensure_positive_definite(self, C):
        """Ensure matrix is symmetric and positive definite."""
        C = 0.5 * (C + C.T)
        min_eig = np.min(np.linalg.eigvalsh(C))
        if min_eig < 1e-10:
            C += (1e-7 - min_eig) * np.eye(self.dim)
        return C
    
    def _select_init_operator_thompson(self):
        """Select initialization operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.init_alpha, self.init_beta)
        self.current_init_operator = int(np.argmax(samples))
        self.init_operator_counts[self.current_init_operator] += 1
        return self.current_init_operator
    
    def _update_init_operator_rewards(self):
        """Credit assignment for initialization operators via exponentially-weighted cumulative improvement."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        # Initialize cumulative tracking if needed
        if not hasattr(self, 'init_operator_cumulative_reward'):
            self.init_operator_cumulative_reward = np.zeros(self.num_init_operators)
        if not hasattr(self, 'init_operator_decay_sum'):
            self.init_operator_decay_sum = np.zeros(self.num_init_operators)
        
        # Apply exponential decay to accumulated reward
        decay = 0.9
        self.init_operator_cumulative_reward *= decay
        self.init_operator_decay_sum *= decay
        
        # Log-scaled improvement reward
        if improvement > 0:
            reward = float(np.log1p(improvement * 1e10) / 10.0)
        else:
            reward = 0.0
        
        # Diversity metric
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        # Exploration bonus when stagnant and diversity is low
        is_stagnant = self.stagnation_counter > self.max_stagnation // 2
        if is_stagnant and diversity < 0.1:
            reward *= 2.5
        
        # Update cumulative weighted reward
        self.init_operator_cumulative_reward[self.current_init_operator] += reward
        self.init_operator_decay_sum[self.current_init_operator] += 1.0
        
        # Normalize by decay sum
        norm = max(self.init_operator_decay_sum[self.current_init_operator], 1.0)
        normalized_reward = self.init_operator_cumulative_reward[self.current_init_operator] / norm
        normalized_reward = float(np.clip(normalized_reward, -10.0, 10.0))
        
        op = self.current_init_operator
        self.init_operator_rewards[op].append(normalized_reward)
        
        if len(self.init_operator_rewards[op]) > self.init_reward_window_size:
            self.init_operator_rewards[op].pop(0)
        
        n = len(self.init_operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.init_operator_rewards[op]))
            var_reward = float(np.var(self.init_operator_rewards[op]))
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.init_alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.init_beta[op] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.init_operator_rewards[op]))
            if sum_reward > 0:
                self.init_alpha[op] = 1.0 + sum_reward
            else:
                self.init_beta[op] = 1.0 - sum_reward
    
    def _apply_init_strategy(self):
        """Dispatch to the selected initialization strategy."""
        if self.current_init_operator == 0:
            self._init_original()
        elif self.current_init_operator == 1:
            self._init_variant_01()
        elif self.current_init_operator == 2:
            self._init_variant_04()
        elif self.current_init_operator == 3:
            self._init_variant_05()
        elif self.current_init_operator == 4:
            self._init_variant_06()
        elif self.current_init_operator == 5:
            self._init_variant_07()
        elif self.current_init_operator == 6:
            self._init_variant_08()
        else:
            self._init_variant_09()
    
    def __call__(self, func, stopping_condition):
        self.func = func
        self._select_init_operator_thompson()
        self._initialize_population()
        
        while not stopping_condition():
            self._sample_trials_batch()
            
            if stopping_condition():
                break
                
            self._evaluate_batch()
            
            if len(self.trial_fitness) < len(self.trials):
                self.trials = self.trials[:len(self.trial_fitness)]
                
            if len(self.trial_fitness) < self.NP:
                break
                
            self._update_best()
            
            if stopping_condition():
                break
                
            self._select_survivors_batch()
            self._adapt_step_size()
            self._adapt_covariance()
            
            # Update init operator rewards based on improvement
            self._update_init_operator_rewards()
            
            self._check_stagnation()
            self._restart_if_needed()
        
        return self.f_opt, self.x_opt
    
    def _initialize_population(self):
        """Initialize population using selected strategy."""
        self._apply_init_strategy()
    
    def _init_original(self):
        """Initialize population using Sobol quasi-random sequences."""
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(self.dim, scramble=True)
            samples = sampler.random(self.NP)
            samples = qmc.scale(samples, self.lb, self.ub)
        except Exception:
            samples = np.zeros((self.NP, self.dim))
            for d in range(self.dim):
                bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
                bin_width = bins[1] - bins[0]
                samples[:, d] = bins[:-1] + np.random.random(self.NP) * bin_width
            for d in range(self.dim):
                samples[:, d] = samples[np.random.permutation(self.NP), d]

        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = self.func(self.population)

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt

        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

        self._reset_operator_state()
    
    def _init_variant_01(self):
        """Initialize population using Latin Hypercube Sampling with Gaussian perturbation."""
        lb = np.asarray(self.lb, dtype=np.float64)
        ub = np.asarray(self.ub, dtype=np.float64)

        range_vec = ub - lb
        range_vec = np.where(range_vec < 1e-15, 1.0, range_vec)

        try:
            from scipy.stats import qmc
            sampler = qmc.Halton(d=self.dim, scramble=True)
            samples = sampler.random(self.NP)

            from scipy import stats
            samples = stats.norm.ppf(np.clip(samples, 1e-10, 1 - 1e-10))

            center = (ub + lb) / 2.0
            half_range = range_vec / 2.0

            sigma_init = 0.5
            samples = center + samples * half_range * sigma_init

        except Exception:
            samples = np.zeros((self.NP, self.dim))
            for d in range(self.dim):
                bins = np.linspace(lb[d], ub[d], self.NP + 1)
                bin_width = bins[1] - bins[0]
                samples[:, d] = bins[:-1] + bin_width * (0.5 + (np.random.random(self.NP) - 0.5) * 0.8)

        samples = np.clip(samples, lb, ub)

        unique_mask = np.ones(self.NP, dtype=bool)
        for i in range(self.NP):
            if not unique_mask[i]:
                continue
            for j in range(i + 1, self.NP):
                if unique_mask[j]:
                    dist = np.linalg.norm(samples[i] - samples[j])
                    if dist < 1e-8 * np.mean(range_vec):
                        samples[j] += np.random.randn(self.dim) * range_vec * 0.01
                        samples[j] = np.clip(samples[j], lb, ub)
                        unique_mask[j] = False

        self.population = samples

        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        cov_matrix = np.cov(self.population.T)
        if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
            cov_matrix = np.eye(self.dim) * (np.mean(range_vec) / 6.0) ** 2
        min_eig = np.min(np.linalg.eigvalsh(cov_matrix))
        if min_eig < 1e-10:
            cov_matrix += (1e-8 - min_eig) * np.eye(self.dim)
        self.C = cov_matrix

        self.sigma = np.mean(np.std(self.population, axis=0))
        self.sigma = np.clip(self.sigma, 1e-10, np.mean(range_vec) / 3.0)

        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = self.func(self.population)

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt

        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

        self._reset_operator_state()
    
    def _init_variant_04(self):
        """Initialize population using LHS with elite-targeted adaptive resampling."""
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(self.dim, scramble=True)
            samples = sampler.random(self.NP)
            samples = qmc.scale(samples, self.lb, self.ub)
        except Exception:
            samples = np.zeros((self.NP, self.dim))
            for d in range(self.dim):
                bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
                bin_width = bins[1] - bins[0]
                samples[:, d] = bins[:-1] + np.random.random(self.NP) * bin_width
            for d in range(self.dim):
                samples[:, d] = samples[np.random.permutation(self.NP), d]

        self.population = samples.copy()

        init_fitness = self.func(self.population)
        init_fitness = np.asarray(init_fitness).flatten()

        if np.min(init_fitness) > 1.0:
            n_elite = max(3, self.NP // 16)
            elite_indices = np.argsort(init_fitness)[:n_elite]
            elite_samples = self.population[elite_indices]

            n_resample = self.NP // 4
            resample_per_elite = max(1, n_resample // n_elite)

            resampled = []
            for elite in elite_samples:
                spread = 0.15 * (self.ub - self.lb)
                spread = np.clip(spread, 1.0, 50.0)

                for _ in range(resample_per_elite):
                    candidate = elite + np.random.randn(self.dim) * spread
                    candidate = np.clip(candidate, self.lb, self.ub)
                    resampled.append(candidate)

            resampled = np.array(resampled[:n_resample])

            if len(resampled) > 0:
                resample_fitness = np.asarray(self.func(resampled)).flatten()

                combined_pop = np.vstack([self.population, resampled])
                combined_fit = np.concatenate([init_fitness, resample_fitness])

                keep_indices = np.argsort(combined_fit)[:self.NP]
                self.population = combined_pop[keep_indices]
                init_fitness = combined_fit[keep_indices]

        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = init_fitness[:self.NP]

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt

        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

        self._reset_operator_state()
    
    def _init_variant_05(self):
        """Initialize population using boundary-layer adaptive sampling with multi-seed starts."""
        try:
            from scipy.stats import qmc
            sampler = qmc.Halton(self.dim, scramble=True)
            base_samples = sampler.random(self.NP // 2)
            base_samples = qmc.scale(base_samples, self.lb, self.ub)
        except Exception:
            base_samples = np.random.uniform(self.lb, self.ub, (self.NP // 2, self.dim))

        boundary_ratio = 0.3
        n_boundary = max(4, int(self.NP * boundary_ratio))
        boundary_samples = np.zeros((n_boundary, self.dim))

        for i in range(n_boundary):
            boundary_layer_width = 0.15 * (self.ub - self.lb)
            for d in range(self.dim):
                side = np.random.choice([0, 1])
                if side == 0:
                    boundary_samples[i, d] = self.lb[d] + np.random.exponential(boundary_layer_width[d])
                else:
                    boundary_samples[i, d] = self.ub[d] - np.random.exponential(boundary_layer_width[d])
            boundary_samples[i] = np.clip(boundary_samples[i], self.lb, self.ub)

        n_seeds = max(2, self.dim // 4)
        seed_samples = np.zeros((n_seeds, self.dim))
        for i in range(n_seeds):
            seed_center = boundary_samples[i % len(boundary_samples)].copy()
            spread = 0.2 * (self.ub - self.lb)
            seed_samples[i] = seed_center + np.random.randn(self.dim) * spread
            seed_samples[i] = np.clip(seed_samples[i], self.lb, self.ub)

        remaining = self.NP - len(base_samples) - n_boundary - n_seeds
        if remaining > 0:
            extra_samples = np.random.uniform(self.lb, self.ub, (remaining, self.dim))
            samples = np.vstack([base_samples, boundary_samples, seed_samples, extra_samples])
        else:
            samples = np.vstack([base_samples, boundary_samples, seed_samples])

        if len(samples) > self.NP:
            samples = samples[:self.NP]
        elif len(samples) < self.NP:
            extra = np.random.uniform(self.lb, self.ub, (self.NP - len(samples), self.dim))
            samples = np.vstack([samples, extra])

        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = self.func(self.population)

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt

        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

        self._reset_operator_state()
    
    def _init_variant_06(self):
        """Initialize population using fitness-informed Latin Hypercube with adaptive clustering."""
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(self.dim, scramble=True)
            samples = sampler.random(self.NP)
            samples = qmc.scale(samples, self.lb, self.ub)
        except Exception:
            samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))

        init_fitness = self.func(samples)

        n_clusters = min(max(3, self.dim // 10), 10, self.NP // 15)

        if n_clusters >= 2 and self.NP > n_clusters * 3:
            try:
                from scipy.cluster.vq import kmeans2
                centroids, labels = kmeans2(samples, n_clusters, minit='points')

                cluster_fitness = np.full(n_clusters, np.inf)
                for c in range(n_clusters):
                    mask = labels == c
                    if np.sum(mask) > 0:
                        cluster_fitness[c] = np.mean(init_fitness[mask])

                best_clusters = np.argsort(cluster_fitness)[:max(1, n_clusters // 2)]

                n_from_best = int(0.6 * self.NP)
                n_uniform = self.NP - n_from_best

                inv_fitness = 1.0 / (cluster_fitness[best_clusters] - np.min(cluster_fitness[best_clusters]) + 1e-6)
                cluster_probs = inv_fitness / np.sum(inv_fitness)

                new_pop = []
                for _ in range(n_from_best):
                    c = np.random.choice(best_clusters, p=cluster_probs)
                    mask = labels == c
                    idx = np.random.choice(np.where(mask)[0])
                    new_pop.append(samples[idx] + np.random.randn(self.dim) * 0.1 * (self.ub - self.lb) / 100)

                indices = np.random.choice(self.NP, n_uniform, replace=False)
                for idx in indices:
                    new_pop.append(samples[idx])

                samples = np.array(new_pop)
            except Exception:
                pass

        samples = np.clip(samples, self.lb, self.ub)

        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = self.func(self.population)

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt

        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

        self._reset_operator_state()
    
    def _init_variant_07(self):
        """Initialize population using Latin Hypercube Sampling with bounded random walk exploration."""
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(self.dim, scramble=True)
            samples = sampler.random(self.NP)
            samples = qmc.scale(samples, self.lb, self.ub)
        except Exception:
            samples = np.zeros((self.NP, self.dim))
            for d in range(self.dim):
                bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
                bin_width = bins[1] - bins[0]
                samples[:, d] = bins[:-1] + np.random.random(self.NP) * bin_width
            for d in range(self.dim):
                samples[:, d] = samples[np.random.permutation(self.NP), d]

        walk_scale = 0.15 * (self.ub - self.lb)
        walk_scale = np.maximum(walk_scale, 1e-6)
        num_walk = max(1, self.NP // 5)
        walk_perturbation = np.random.randn(num_walk, self.dim) * walk_scale
        samples[:num_walk] = samples[:num_walk] + walk_perturbation

        num_extreme = max(1, self.NP // 10)
        extreme_samples = np.random.uniform(self.lb, self.ub, (num_extreme, self.dim))
        samples[num_walk:num_walk + num_extreme] = extreme_samples

        samples = np.clip(samples, self.lb, self.ub)

        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = np.cov(self.population.T)
        if self.C.size == 1:
            self.C = np.array([[max(self.C.item(), 1e-8)]])
        self.C += 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.sigma = max(self.sigma, 1e-8)
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = self.func(self.population)

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt

        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

        self._reset_operator_state()
    
    def _init_variant_08(self):
        """Initialize population using adaptive multi-method sampling with boundary focus."""
        samples_list = []

        n_sobol = int(0.6 * self.NP)
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(self.dim, scramble=True)
            sobol_samples = sampler.random(n_sobol)
            sobol_samples = qmc.scale(sobol_samples, self.lb, self.ub)
            samples_list.append(sobol_samples)
        except Exception:
            pass

        n_lhs = int(0.25 * self.NP)
        try:
            sampler = qmc.LatinHypercube(self.dim, scramble=True)
            lhs_samples = sampler.random(n_lhs)
            lhs_samples = qmc.scale(lhs_samples, self.lb, self.ub)
            samples_list.append(lhs_samples)
        except Exception:
            pass

        n_boundary = self.NP - sum(len(s) for s in samples_list)
        if n_boundary > 0:
            boundary_samples = np.zeros((n_boundary, self.dim))
            u = np.random.random((n_boundary, self.dim))
            power = 3.0
            scaled = np.power(u, power) if np.random.random() > 0.5 else 1.0 - np.power(u, power)
            dim_mask = np.random.random(self.dim) < 0.3
            boundary_samples[:, dim_mask] = scaled[:, dim_mask]
            boundary_samples[:, ~dim_mask] = u[:, ~dim_mask]
            for d in range(self.dim):
                boundary_samples[:, d] = self.lb[d] + boundary_samples[:, d] * (self.ub[d] - self.lb[d])
            samples_list.append(boundary_samples)

        if samples_list:
            samples = np.vstack(samples_list)
        else:
            samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))

        if samples.shape[0] > self.NP:
            samples = samples[:self.NP]
        elif samples.shape[0] < self.NP:
            extra = np.random.uniform(self.lb, self.ub, (self.NP - samples.shape[0], self.dim))
            samples = np.vstack([samples, extra])

        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = self.func(self.population)

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt

        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

        self._reset_operator_state()
    
    def _init_variant_09(self):
        """Initialize population using Latin Hypercube + fitness-guided niching with Cauchy perturbations."""
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(self.dim, scramble=True)
            samples = sampler.random(self.NP)
            samples = qmc.scale(samples, self.lb, self.ub)
        except Exception:
            samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))

        init_fitness = self.func(samples)

        sorted_indices = np.argsort(init_fitness)
        num_elites = max(1, self.NP // 10)
        elite_indices = sorted_indices[:num_elites]

        num_niches = min(num_elites, 4)
        samples_per_niche = self.NP // num_niches

        population = []

        for niche_id in range(num_niches):
            elite_idx = elite_indices[niche_id % len(elite_indices)]
            center = samples[elite_idx]

            population.append(center)

            remaining = samples_per_niche - 1
            for _ in range(remaining):
                scale = 0.2 * (self.ub - self.lb)
                perturbation = np.random.standard_cauchy(self.dim) * scale
                candidate = center + perturbation
                candidate = np.clip(candidate, self.lb, self.ub)
                population.append(candidate)

        remaining_slots = self.NP - len(population)
        if remaining_slots > 0:
            extra = sampler.random(remaining_slots)
            extra = qmc.scale(extra, self.lb, self.ub)
            population.extend(extra)

        self.population = np.array(population[:self.NP])
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = self.func(self.population)

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt

        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

        self._reset_operator_state()
    
    def _reset_operator_state(self):
        """Reset all operator-specific state variables."""
        self.L = None
        if hasattr(self, 'pc_weighted'):
            self.pc_weighted = np.zeros(self.dim)
        if hasattr(self, 'p_cross'):
            self.p_cross = np.zeros(self.dim)
        if hasattr(self, 'prev_y_mean'):
            delattr(self, 'prev_y_mean')
        if hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.improvement_ema = 1.0
    
    def _sample_trials_batch(self):
        """Sample new trial population using Cholesky decomposition."""
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        try:
            L = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            diag_C = np.diag(self.C)
            diag_C = np.maximum(diag_C, 1e-10)
            L = np.diag(np.sqrt(diag_C))

        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)

        self.trials = self._clip_to_bounds(self.trials)
    
    def _evaluate_batch(self):
        """Evaluate all trial candidates in single batch call."""
        self.trial_fitness = self.func(self.trials)
    
    def _update_best(self):
        """Update best solution if trial population contains improvement."""
        trial_best_idx = np.argmin(self.trial_fitness)
        trial_best_fit = float(np.asarray(self.trial_fitness[trial_best_idx]).flatten()[0])
        if trial_best_fit < self.f_opt:
            self.f_opt = trial_best_fit
            self.x_opt = self.trials[trial_best_idx].copy()
    
    def _select_survivors_batch(self):
        """Select survivors via elitist (mu, lambda)-selection."""
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])
        
        sorted_indices = np.argsort(combined_fit)
        self.population = combined_pop[sorted_indices[:self.NP]]
        self.fitness = combined_fit[sorted_indices[:self.NP]]
        
        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
    
    def _adapt_step_size(self):
        """Adapt step-size using momentum-enhanced improvement tracking with diversity-based damping."""
        recent_improved = 0
        recent_total = 0
        total_improvement = 0.0

        for i in range(self.NP):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                if self.trial_fitness[i] < self.fitness[i]:
                    recent_improved += 1
                    diff = self.fitness[i] - self.trial_fitness[i]
                    total_improvement += max(diff, 0.0)
                recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = recent_improved / recent_total
        success_rate = np.clip(success_rate, 0.0, 1.0)

        avg_improvement = total_improvement / max(recent_improved, 1)
        improvement_magnitude = np.log1p(max(avg_improvement, 1e-15))

        if not hasattr(self, 'improvement_ema'):
            self.improvement_ema = improvement_magnitude
        self.improvement_ema = 0.7 * self.improvement_ema + 0.3 * improvement_magnitude

        target_rate = 0.25
        adaptation = (success_rate - target_rate) / target_rate
        adaptation += 0.3 * np.tanh(self.improvement_ema - 1.0)
        adaptation = np.clip(adaptation, -0.8, 0.8)

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        diversity_boost = 1.0 + 2.0 * (1.0 - diversity)
        diversity_boost = np.clip(diversity_boost, 0.5, 3.0)

        damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim)) / diversity_boost
        damping_adaptive = np.clip(damping_adaptive, 0.05, 200.0)

        self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _adapt_covariance(self):
        """Original covariance adaptation (baseline)."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _compute_diversity(self):
        """Compute diversity using percentile-robust and condition-aware metrics."""
        pop_diffs = self.population - self.mean
        max_sq_dist = np.max(np.sum(pop_diffs ** 2, axis=1))
        expected_max_sq = self.dim * ((self.ub[0] - self.lb[0]) / 3.0) ** 2
        spread_norm = np.sqrt(max_sq_dist) / np.sqrt(max(expected_max_sq, 1e-10))
        spread_norm = np.clip(spread_norm, 0.0, 2.0)

        sorted_fit = np.sort(self.fitness)
        fit_range = sorted_fit[-1] - sorted_fit[0]
        fit_median = sorted_fit[len(sorted_fit) // 2]
        fit_p10 = sorted_fit[max(0, len(sorted_fit) // 10 - 1)]
        fit_percentile_range = (fit_median - fit_p10) / (abs(fit_median) + abs(fit_p10) + 1e-10)
        fit_percentile_range = np.clip(fit_percentile_range, 0.0, 1.0)

        eigvals = np.linalg.eigvalsh(self.C)
        eigvals = np.clip(eigvals, 1e-15, None)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        cond = eig_max / max(eig_min, 1e-15)
        cond_norm = 1.0 / np.log1p(cond)
        cond_norm = np.clip(cond_norm, 0.0, 1.0)

        diag = np.diag(self.C)
        diag = np.maximum(diag, 1e-15)
        max_var = np.max(diag)
        min_var = np.min(diag)
        spread_ratio = min_var / max_var
        spread_ratio = np.clip(spread_ratio, 0.0, 1.0)

        diversity = (
            0.25 * spread_norm +
            0.35 * fit_percentile_range +
            0.25 * cond_norm +
            0.15 * spread_ratio
        )

        return float(np.clip(diversity, 1e-15, None))
    
    def _check_stagnation(self):
        """Track stagnation counter with scale-adaptive threshold."""
        improvement = self.f_opt_prev - self.f_opt

        base_scale = max(abs(self.f_opt), 1.0)
        rel_threshold = 1e-6 * base_scale
        threshold = max(rel_threshold, 1e-10)

        if improvement > threshold:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        if abs(self.f_opt) > 10.0:
            self.max_stagnation = 150 + self.dim * 5
        elif abs(self.f_opt) > 1.0:
            self.max_stagnation = 100 + self.dim * 4
        else:
            self.max_stagnation = 50 + self.dim * 3

        self.f_opt_prev = self.f_opt
    
    def _restart_if_needed(self):
        """Restart population if stagnated or diversity lost."""
        diversity = self._compute_diversity()
        
        if (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            np.any(np.isnan(self.C))):
            
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]
            
            # Store in restart archive for variant_10-style guidance
            self.restart_archive.append(self.x_opt.copy())
            if len(self.restart_archive) > 10:
                self.restart_archive.pop(0)
            
            self._select_init_operator_thompson()
            self._initialize_population()
            
            self.population[0] = elite
            self.fitness[0] = elite_fit
            self.f_opt = float(np.asarray(elite_fit).flatten()[0])
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0
