import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 5 covariance adaptation strategies (original + 4 variants)
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
        
        # Adaptive operator selection parameters
        self.num_operators = 5
        self.operator_names = ['original', 'variant_01', 'variant_06', 'variant_08', 'variant_09']
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_operators)
        self.beta = np.ones(self.num_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 10
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators)
        self.selection_counts = np.zeros(self.num_operators)
        
        # Runtime state
        self.generation = 0
        self.current_operator = 0
        self.L = None
        
        # ======= ADAPTIVE INITIALIZATION SELECTION =======
        self.num_init_operators = 10
        self.init_operator_names = [
            'init_original', 'init_01', 'init_02', 'init_03', 'init_04',
            'init_05', 'init_06', 'init_07', 'init_08', 'init_09'
        ]
        
        # Thompson Sampling for initialization operators
        self.init_alpha = np.ones(self.num_init_operators)
        self.init_beta = np.ones(self.num_init_operators)
        
        # Sliding window for initialization rewards
        self.init_reward_window_size = 8
        self.init_operator_rewards = {i: [] for i in range(self.num_init_operators)}
        self.init_operator_counts = np.zeros(self.num_init_operators)
        
        # Track initialization performance
        self.current_init_operator = 0
        self.init_f_opt_at_selection = None
        
        # Restart archive for variant_10 style initialization
        self.restart_archive = []
        self.restart_archive_fitness = []
    
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
            
            # Select and apply covariance adaptation operator
            self._select_operator_thompson()
            self._adapt_covariance()
            
            # Update operator rewards based on improvement
            self._update_operator_rewards()
            
            # Update initialization operator rewards on restart
            self._update_init_operator_rewards()
            
            self._check_stagnation()
            self._restart_if_needed()
        
        return self.f_opt, self.x_opt
    
    def _select_init_operator_thompson(self):
        """Select initialization operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.init_alpha, self.init_beta)
        self.current_init_operator = int(np.argmax(samples))
        self.init_operator_counts[self.current_init_operator] += 1
        return self.current_init_operator
    
    def _update_init_operator_rewards(self):
        """Update initialization operator rewards based on improvement since selection."""
        if self.init_f_opt_at_selection is None:
            return
        
        # Compute improvement since initialization selection
        improvement = self.init_f_opt_at_selection - self.f_opt
        improvement = float(np.clip(improvement, -1e10, 1e10))
        
        # Initialize cumulative tracking if needed
        if not hasattr(self, 'init_operator_cumulative_reward'):
            self.init_operator_cumulative_reward = np.zeros(self.num_init_operators)
        if not hasattr(self, 'init_operator_decay_sum'):
            self.init_operator_decay_sum = np.zeros(self.num_init_operators)
        
        # Apply exponential decay
        decay = 0.9
        self.init_operator_cumulative_reward *= decay
        self.init_operator_decay_sum *= decay
        
        # Log-scaled reward for better discrimination
        if improvement > 1e-15:
            reward = float(np.log1p(improvement * 1e8) / 8.0)
        elif improvement < -1e-15:
            # Small penalty for degradation
            reward = float(max(improvement * 1e6, -2.0))
        else:
            reward = 0.0
        
        # Normalize by generation count for fair comparison
        gen_count = max(self.generation - self.init_generation_at_selection + 1, 1)
        normalized_reward = reward / np.sqrt(gen_count)
        normalized_reward = float(np.clip(normalized_reward, -5.0, 5.0))
        
        op = self.current_init_operator
        self.init_operator_rewards[op].append(normalized_reward)
        
        if len(self.init_operator_rewards[op]) > self.init_reward_window_size:
            self.init_operator_rewards[op].pop(0)
        
        # Update Beta distribution parameters
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
    
    def _initialize_population(self):
        """Initialize population using the selected adaptive initialization strategy."""
        # Dispatch to the selected initialization operator
        if self.current_init_operator == 0:
            self._init_original()
        elif self.current_init_operator == 1:
            self._init_variant_01()
        elif self.current_init_operator == 2:
            self._init_variant_02()
        elif self.current_init_operator == 3:
            self._init_variant_03()
        elif self.current_init_operator == 4:
            self._init_variant_04()
        elif self.current_init_operator == 5:
            self._init_variant_05()
        elif self.current_init_operator == 6:
            self._init_variant_06()
        elif self.current_init_operator == 7:
            self._init_variant_07()
        elif self.current_init_operator == 8:
            self._init_variant_08()
        else:
            self._init_variant_09()
        
        # Track for reward assignment
        self.init_f_opt_at_selection = float(self.f_opt)
        self.init_generation_at_selection = int(self.generation)
    
    # ======= INITIALIZATION STRATEGIES =======
    
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
        self._finalize_initialization()
    
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
        self._finalize_initialization()
    
    def _init_variant_02(self):
        """Initialize population using Sobol with restart-guided anti-centralization."""
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

        if hasattr(self, 'restart_archive') and len(self.restart_archive) >= 3:
            archive_array = np.array(self.restart_archive)
            archive_centroid = np.mean(archive_array, axis=0)

            distances = np.linalg.norm(archive_array - archive_centroid, axis=1)
            escape_scale = max(np.median(distances), 1.0)

            for i in range(self.NP):
                to_archive = samples[i] - archive_centroid
                dist = np.linalg.norm(to_archive)
                if dist < escape_scale * 0.8:
                    direction = to_archive / (dist + 1e-10)
                    push_dist = escape_scale * (0.5 + 0.5 * np.random.random())
                    samples[i] = samples[i] + direction * push_dist
                    samples[i] += np.random.randn(self.dim) * escape_scale * 0.3

            samples = np.clip(samples, self.lb, self.ub)

        self.population = samples
        self._finalize_initialization()
    
    def _init_variant_03(self):
        """Initialize population using Sobol with restart-guided anti-centralization (same as variant_02)."""
        self._init_variant_02()
    
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

        init_fitness = self.func(samples)
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
        
        self.fitness = init_fitness[:self.NP]
        self._finalize_initialization_with_fitness(init_fitness[:self.NP])
    
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
        self._finalize_initialization()
    
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
        self._finalize_initialization()
    
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
        self._finalize_initialization()
    
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
        self._finalize_initialization()
    
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
        self._finalize_initialization()
    
    def _finalize_initialization(self):
        """Common finalization for initialization methods that evaluate fitness."""
        self.fitness = self.func(self.population)
        self._finalize_initialization_with_fitness(self.fitness)
    
    def _finalize_initialization_with_fitness(self, fitness):
        """Finalize initialization state with provided fitness values."""
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        cov_matrix = np.cov(self.population.T)
        if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
            cov_matrix = np.eye(self.dim) * (np.mean(self.ub - self.lb) / 6.0) ** 2
        self.C = self._ensure_positive_definite(cov_matrix)
        
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.sigma = float(np.clip(self.sigma, 1e-10, np.mean(self.ub - self.lb) / 3.0))
        
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        if fitness is not None:
            self.fitness = np.asarray(fitness).flatten()
        else:
            self.fitness = self.func(self.population)
            self.fitness = np.asarray(self.fitness).flatten()

        best_idx = np.argmin(self.fitness)
        f_best = self.fitness[best_idx]
        self.f_opt = float(np.asarray(f_best).flatten()[0])
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
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _update_operator_rewards(self):
        """Credit assignment via exponentially-weighted cumulative improvement with stagnation awareness."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)

        if not hasattr(self, 'operator_cumulative_reward'):
            self.operator_cumulative_reward = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum'):
            self.operator_decay_sum = np.zeros(self.num_operators)

        decay = 0.95
        self.operator_cumulative_reward *= decay
        self.operator_decay_sum *= decay

        if improvement > 0:
            reward = float(np.log1p(improvement * 1e10) / 10.0)
        else:
            reward = 0.0

        is_stagnant = self.stagnation_counter > self.max_stagnation // 2

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        if is_stagnant and diversity < 0.1:
            reward *= 2.0

        self.operator_cumulative_reward[self.current_operator] += reward
        self.operator_decay_sum[self.current_operator] += 1.0

        norm = max(self.operator_decay_sum[self.current_operator], 1.0)
        normalized_reward = self.operator_cumulative_reward[self.current_operator] / norm
        normalized_reward = float(np.clip(normalized_reward, -10.0, 10.0))

        op = self.current_operator
        self.operator_rewards[op].append(normalized_reward)

        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)

        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            var_reward = float(np.var(self.operator_rewards[op]))
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.beta[op] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = 1.0 + sum_reward
            else:
                self.beta[op] = 1.0 - sum_reward
    
    def _adapt_covariance(self):
        """Dispatch to the selected covariance adaptation strategy."""
        if self.current_operator == 0:
            self._adapt_covariance_original()
        elif self.current_operator == 1:
            self._adapt_covariance_variant_01()
        elif self.current_operator == 2:
            self._adapt_covariance_variant_06()
        elif self.current_operator == 3:
            self._adapt_covariance_variant_08()
        else:
            self._adapt_covariance_variant_09()
    
    def _adapt_covariance_original(self):
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
    
    def _adapt_covariance_variant_01(self):
        """Archive-guided covariance perturbation for escaping deceptive local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        if not hasattr(self, 'archive_positions'):
            self.archive_positions = []
            self.archive_fitness = []
            self.archive_generations = []

        min_dist_threshold = 1e-3 * (self.ub[0] - self.lb[0])
        is_distinct = True
        for arch_pos in self.archive_positions:
            dist = np.linalg.norm(self.x_opt - arch_pos)
            if dist < min_dist_threshold:
                is_distinct = False
                break

        if is_distinct and len(self.archive_positions) < 10:
            self.archive_positions.append(self.x_opt.copy())
            self.archive_fitness.append(self.f_opt)
            self.archive_generations.append(self.generation)
        elif is_distinct and len(self.archive_positions) >= 10:
            worst_idx = np.argmax(self.archive_fitness)
            self.archive_positions[worst_idx] = self.x_opt.copy()
            self.archive_fitness[worst_idx] = self.f_opt
            self.archive_generations[worst_idx] = self.generation

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        eig_spread = eig_min / (eig_max + 1e-10)
        cond = eig_max / (max(eig_min, 1e-10))

        is_trapped = (rel_var < 1e-3) or (cond > 1e5) or (eig_spread < 1e-5)

        escape_perturb = np.zeros((self.dim, self.dim))
        if is_trapped and len(self.archive_positions) >= 3:
            centroid = np.mean(self.archive_positions, axis=0)
            toward_centroid = centroid - self.mean
            norm_toward = np.linalg.norm(toward_centroid)
            if norm_toward > 1e-10:
                toward_centroid /= norm_toward
                for arch_pos in self.archive_positions:
                    dir_to_arch = arch_pos - self.mean
                    norm_dir = np.linalg.norm(dir_to_arch)
                    if norm_dir > 1e-10:
                        dir_to_arch /= norm_dir
                        escape_perturb += 0.3 * np.outer(dir_to_arch, dir_to_arch)
            escape_perturb += 0.1 * np.eye(self.dim)

        if is_trapped:
            ccov_scale = 0.8
            cc_scale = 0.8
        else:
            ccov_scale = min(1.0, 0.2 + 2.0 * rel_var)
            cc_scale = 1.0

        ccov_1 = ccov_scale * 1.0 / (self.dim + 2.0)
        ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        cc_adapt = cc_scale * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)

        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
                  ccov_1 * rank_one +
                  ccov_mu * rank_mu)

        if is_trapped and len(self.archive_positions) >= 3:
            self.C = self.C + escape_perturb

        self.C = self._ensure_positive_definite(self.C)

        if cond > 1e8 or eig_spread < 1e-8:
            self.C = np.eye(self.dim) * np.mean(eigvals)
            self.pc = np.zeros(self.dim)
            self.L = None
    
    def _adapt_covariance_variant_06(self):
        """Diversity-sensitive learning rate scaling."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)
        
        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
        
        is_converging = (rel_var < 0.01) and (eig_spread < 1e-4)
        
        if is_converging:
            scale = 10.0
        else:
            scale = 1.0 + 5.0 * min(rel_var, 0.1)
        
        ccov_scaled = min(self.ccov * scale, 0.5)
        cc_scaled = min(self.cc * (1.0 + scale * 0.5), 0.3)
        
        self.pc = (1.0 - cc_scaled) * self.pc + np.sqrt(cc_scaled * (2.0 - cc_scaled)) * y_mean
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - ccov_scaled) * self.C + 
                  ccov_scaled * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_scaled * 2.0 * rank_mu))
        
        if is_converging:
            self.C += 0.05 * np.eye(self.dim)
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_08(self):
        """Active CMA-ES: Reduce learning rate along unfavorable eigendirections."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu_pos = np.zeros((self.dim, self.dim))
        rank_mu_neg = np.zeros((self.dim, self.dim))

        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            weight = self.weights[i]
            outer = np.outer(diff, diff)

            if weight > 0:
                rank_mu_pos += weight * outer
            else:
                rank_mu_neg += abs(weight) * outer

        pos_sum = max(np.sum(self.weights[:self.mu][self.weights[:self.mu] > 0]), 1e-10)
        neg_sum = max(np.sum(np.abs(self.weights[:self.mu][self.weights[:self.mu] < 0])), 1e-10)

        rank_mu_pos /= pos_sum
        rank_mu_neg /= neg_sum

        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * rank_one +
                  (1.0 - 1.0 / self.mueff) * self.ccov * rank_mu_pos -
                  self.ccov * 0.4 * rank_mu_neg)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_09(self):
        """Exploration temperature with fitness gradient tracking for escaping local optima."""
        if not hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.exploration_temp = 1.0

        delta_f_opt = self.prev_f_opt - self.f_opt
        self.prev_f_opt = self.f_opt

        fitness_gradient = max(abs(delta_f_opt), 1e-15)
        temp_target = np.clip(1.0 / (1.0 + np.log1p(fitness_gradient * 1e5)), 0.05, 5.0)
        self.exploration_temp = 0.95 * self.exploration_temp + 0.05 * temp_target

        base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        ccov_adaptive = base_ccov * self.exploration_temp
        ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 0.5)

        cc_adaptive = self.cc * (1.0 / max(self.exploration_temp, 0.5))
        cc_adaptive = np.clip(cc_adaptive, 0.01, 0.3)

        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - cc_adaptive) * self.pc + np.sqrt(cc_adaptive * (2.0 - cc_adaptive)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        total_w = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_w += abs(w)

        if total_w > 0:
            rank_mu /= total_w

        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)

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
            
            # Update restart archive for variant_10 style initialization
            self.restart_archive.append(elite.copy())
            self.restart_archive_fitness.append(float(elite_fit))
            if len(self.restart_archive) > 10:
                self.restart_archive = self.restart_archive[-10:]
                self.restart_archive_fitness = self.restart_archive_fitness[-10:]
            
            self._select_init_operator_thompson()
            self._initialize_population()
            
            self.population[0] = elite
            self.fitness[0] = elite_fit
            self.f_opt = float(np.asarray(elite_fit).flatten()[0])
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0
