```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best _initialize_population
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 7 covariance adaptation strategies (original + 6 variants)
    - 7 initialization strategies with adaptive selection
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
        
        # Adaptive operator selection parameters for covariance adaptation
        self.num_operators = 5
        self.operator_names = ['original', 'variant_01', 'variant_06', 'variant_08', 'variant_09']
        
        # Thompson Sampling with Beta distributions for covariance
        self.alpha = np.ones(self.num_operators)
        self.beta = np.ones(self.num_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 10
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators)
        self.selection_counts = np.zeros(self.num_operators)
        
        # Adaptive initialization selection parameters
        self.num_init_operators = 7
        self.init_operator_names = [
            'original', 'variant_01', 'variant_02', 'variant_04',
            'variant_06', 'variant_07', 'variant_09'
        ]
        
        # Thompson Sampling for initialization operators
        self.init_alpha = np.ones(self.num_init_operators)
        self.init_beta = np.ones(self.num_init_operators)
        
        # Sliding window for init operator rewards
        self.init_operator_rewards = {i: [] for i in range(self.num_init_operators)}
        self.init_operator_counts = np.zeros(self.num_init_operators)
        self.init_selection_counts = np.zeros(self.num_init_operators)
        
        # Track performance per init strategy
        self.init_best_fitness_history = {i: [] for i in range(self.num_init_operators)}
        
        # Runtime state
        self.generation = 0
        self.current_operator = 0
        self.current_init_operator = 0
        self.L = None
        self.init_generation = 0
        
        # Track restart count for adaptive selection
        self.restart_count = 0
    
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
            
            self._check_stagnation()
            self._restart_if_needed()
        
        return self.f_opt, self.x_opt
    
    def _select_init_operator_thompson(self):
        """Select initialization operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.init_alpha, self.init_beta)
        self.current_init_operator = int(np.argmax(samples))
        self.init_operator_counts[self.current_init_operator] += 1
        return self.current_init_operator
    
    def _update_init_operator_rewards(self, initial_f_opt):
        """Credit assignment for initialization operators based on achieved fitness."""
        op = self.current_init_operator
        
        # Reward is based on inverse of initial fitness (lower is better)
        if initial_f_opt > 0:
            reward = float(1.0 / (1.0 + np.log1p(initial_f_opt * 1e5)))
        else:
            reward = float(1.0 / (1.0 + np.log1p(abs(initial_f_opt) * 1e5)))
        
        # Store in sliding window
        self.init_operator_rewards[op].append(reward)
        if len(self.init_operator_rewards[op]) > self.reward_window_size:
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
        
        # Also update based on best fitness achieved during run with this init
        self.init_best_fitness_history[op].append(self.f_opt)
        if len(self.init_best_fitness_history[op]) > 5:
            self.init_best_fitness_history[op].pop(0)
    
    def _initialize_population(self):
        """Initialize population using adaptive operator selection."""
        # Select initialization strategy using Thompson Sampling
        self._select_init_operator_thompson()
        
        # Store initial fitness for reward calculation
        initial_f_opt_before = self.f_opt if hasattr(self, 'f_opt') and self.f_opt is not None else float('inf')
        
        # Dispatch to selected initialization strategy
        if self.current_init_operator == 0:
            self._initialize_population_original()
        elif self.current_init_operator == 1:
            self._initialize_population_variant_01()
        elif self.current_init_operator == 2:
            self._initialize_population_variant_02()
        elif self.current_init_operator == 3:
            self._initialize_population_variant_04()
        elif self.current_init_operator == 4:
            self._initialize_population_variant_06()
        elif self.current_init_operator == 5:
            self._initialize_population_variant_07()
        else:
            self._initialize_population_variant_09()
        
        # Update rewards based on achieved initial fitness
        self._update_init_operator_rewards(initial_f_opt_before)
        self.init_generation += 1
    
    def _initialize_population_original(self):
        """Initialize population using Latin Hypercube sampling (original)."""
        samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        
        for d in range(self.dim):
            bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
            perm = np.random.permutation(self.NP)
            samples[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(self.NP)
        
        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        
        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        
        self.fitness = self.func(self.population)
        self.fitness = np.asarray(self.fitness).flatten()
        
        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        
        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0
        
        self._reset_operator_state()
    
    def _initialize_population_variant_01(self):
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
        self.fitness = np.asarray(self.fitness).flatten()
        
        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        
        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0
        
        self._reset_operator_state()
    
    def _initialize_population_variant_02(self):
        """Initialize population using hybrid LHS + directional sampling."""
        half_lhs = self.NP // 2
        half_dir = self.NP - half_lhs
        
        samples = np.empty((self.NP, self.dim))
        
        for d in range(self.dim):
            bins = np.linspace(self.lb[d], self.ub[d], half_lhs + 1)
            perm = np.random.permutation(half_lhs)
            samples[:half_lhs, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(half_lhs)
        
        center = (self.lb + self.ub) / 2.0
        range_vec = (self.ub - self.lb) / 2.0
        
        num_dirs = max(3, half_dir // 2)
        for i in range(num_dirs):
            dir_vec = np.random.randn(self.dim)
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm < 1e-10:
                continue
            dir_vec /= dir_norm
            
            scales = np.logspace(-4, np.log10(0.5), num=2)
            for j, scale in enumerate(scales):
                idx = half_lhs + i * 2 + j
                if idx >= self.NP:
                    break
                offset = dir_vec * scale * range_vec
                offset = np.clip(offset, self.lb - center, self.ub - center)
                samples[idx] = center + offset
        
        for idx in range(half_lhs + num_dirs * 2, self.NP):
            samples[idx] = np.random.uniform(self.lb, self.ub)
        
        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        
        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        
        self.fitness = self.func(self.population)
        self.fitness = np.asarray(self.fitness).flatten()
        
        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        
        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0
        
        self._reset_operator_state()
    
    def _initialize_population_variant_04(self):
        """Initialize using adaptive basin-seeking: oversample, evaluate, cluster, reinitialize."""
        lb, ub, dim, NP = self.lb, self.ub, self.dim, self.NP
        
        oversample_factor = 3
        n_candidates = NP * oversample_factor
        
        candidates = np.zeros((n_candidates, dim))
        for d in range(dim):
            bins = np.linspace(lb[d], ub[d], n_candidates + 1)
            perm = np.random.permutation(n_candidates)
            candidates[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(n_candidates)
        
        candidate_fitness = self.func(candidates)
        candidate_fitness = np.asarray(candidate_fitness).flatten()
        
        n_top = max(10, NP // 4)
        sorted_idx = np.argsort(candidate_fitness)
        top_idx = sorted_idx[:n_top]
        top_candidates = candidates[top_idx]
        
        n_clusters = min(5, n_top)
        cluster_centers = []
        cluster_best_fitness = []
        
        if n_top >= n_clusters * 2:
            try:
                from scipy.cluster.hierarchy import fcluster, linkage
                Z = linkage(top_candidates, method='ward')
                labels = fcluster(Z, t=n_clusters, criterion='maxclust')
                for c in range(1, n_clusters + 1):
                    mask = labels == c
                    if np.sum(mask) > 0:
                        cluster_centers.append(np.mean(top_candidates[mask], axis=0))
                        cluster_best_fitness.append(np.min(candidate_fitness[top_idx[mask]]))
            except Exception:
                cluster_centers = [top_candidates[i] for i in range(min(n_top, 5))]
                cluster_best_fitness = [candidate_fitness[top_idx[i]] for i in range(min(n_top, 5))]
        else:
            cluster_centers = [top_candidates[i] for i in range(min(n_top, 5))]
            cluster_best_fitness = [candidate_fitness[top_idx[i]] for i in range(min(n_top, 5))]
        
        if not cluster_centers:
            cluster_centers = [np.mean(candidates, axis=0)]
            cluster_best_fitness = [np.min(candidate_fitness)]
        
        init_pop = np.zeros((NP, dim))
        n_basins = len(cluster_centers)
        
        inv_fitness = np.array([1.0 / (max(f, 1e-10)) for f in cluster_best_fitness])
        weights = inv_fitness / np.sum(inv_fitness)
        
        basin_alloc = np.zeros(n_basins, dtype=int)
        remainder = NP
        for b in range(n_basins):
            basin_alloc[b] = int(NP * weights[b])
            remainder -= basin_alloc[b]
        for b in range(remainder):
            basin_alloc[b % n_basins] += 1
        
        idx = 0
        spread = 0.15 * (ub[0] - lb[0])
        for b in range(n_basins):
            for _ in range(basin_alloc[b]):
                init_pop[idx] = cluster_centers[b] + np.random.randn(dim) * spread
                idx += 1
        
        while idx < NP:
            init_pop[idx] = lb + np.random.rand(dim) * (ub - lb)
            idx += 1
        
        self.population = np.clip(init_pop, lb, ub)
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        
        self.C = np.cov(self.population.T) + 1e-8 * np.eye(dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.sigma = np.clip(self.sigma, 1e-6, (ub[0] - lb[0]) / 3.0)
        self.ps = np.zeros(dim)
        self.pc = np.zeros(dim)
        
        self.fitness = self.func(self.population)
        self.fitness = np.asarray(self.fitness).flatten()
        
        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        
        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0
        
        self._reset_operator_state()
    
    def _initialize_population_variant_06(self):
        """Initialize population using adaptive multi-scale sampling with meritocratic perturbation."""
        candidates = []
        
        uniform_samples = np.random.uniform(self.lb, self.ub, (self.NP * 3, self.dim))
        candidates.append(uniform_samples)
        
        if self.dim > 1:
            for d in range(self.dim):
                if np.isfinite(self.lb[d]) and np.isfinite(self.ub[d]) and self.lb[d] > 0 and self.ub[d] > 0:
                    log_lb, log_ub = np.log(self.lb[d] + 1e-10), np.log(self.ub[d] + 1e-10)
                    log_samples = np.exp(np.random.uniform(log_lb, log_ub, (self.NP, self.dim)))
                    candidates.append(log_samples)
                elif self.lb[d] < 0 and self.ub[d] > 0:
                    for sign in [1, -1]:
                        log_mag = np.random.uniform(0, np.log(self.ub[d] + 1e-10), (self.NP // 2, self.dim))
                        samples = sign * np.exp(log_mag)
                        candidates.append(samples)
        
        sigma_init = (self.ub[0] - self.lb[0]) / (6.0 * np.sqrt(self.dim))
        sigma_init = np.clip(sigma_init, 1e-3, 10.0)
        normal_samples = np.random.randn(self.NP * 2, self.dim) * sigma_init
        normal_samples = np.clip(normal_samples, self.lb, self.ub)
        candidates.append(normal_samples)
        
        all_candidates = np.vstack(candidates)
        
        all_fitness = self.func(all_candidates)
        all_fitness = np.asarray(all_fitness).flatten()
        
        sorted_indices = np.argsort(all_fitness)
        top_indices = sorted_indices[:min(self.NP, len(all_candidates))]
        self.population = all_candidates[top_indices].copy()
        
        top_count = max(1, self.NP // 10)
        for i in range(top_count):
            if i < len(self.population):
                fit_range = max(all_fitness[top_indices[0]], 1e-10)
                adaptive_sigma = sigma_init * np.clip(1.0 / (1.0 + np.log1p(fit_range)), 0.01, 1.0)
                
                direction = np.random.randn(self.dim)
                direction /= (np.linalg.norm(direction) + 1e-10)
                magnitude = np.random.uniform(0.1, 1.0) * adaptive_sigma * (self.ub[0] - self.lb[0])
                self.population[i] += direction * magnitude
                self.population[i] = np.clip(self.population[i], self.lb, self.ub)
        
        if len(self.population) < self.NP:
            additional = np.random.uniform(self.lb, self.ub, (self.NP - len(self.population), self.dim))
            self.population = np.vstack([self.population, additional])
        
        self.fitness = self.func(self.population)
        self.fitness = np.asarray(self.fitness).flatten()
        
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        
        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.sigma = np.clip(self.sigma, 1e-6, (self.ub[0] - self.lb[0]) / 3.0)
        
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        
        best_idx = np.argmin(self.fitness)
        self.f_opt = float(self.fitness[best_idx])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        
        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0
        
        self._reset_operator_state()
    
    def _initialize_population_variant_07(self):
        """Initialize population using Sobol quasi-random sampling with fitness-based selection."""
        num_candidates = self.NP * 5
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True)
            candidates = qmc.scale(
                sampler.random(num_candidates),
                self.lb, self.ub
            )
        except Exception:
            candidates = np.random.uniform(self.lb, self.ub, (num_candidates, self.dim))
        
        candidate_fitness = self.func(candidates)
        candidate_fitness = np.asarray(candidate_fitness).flatten()
        
        sorted_indices = np.argsort(candidate_fitness)
        samples = candidates[sorted_indices[:self.NP]]
        
        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        
        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        
        self.fitness = self.func(self.population)
        self.fitness = np.asarray(self.fitness).flatten()
        
        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        
        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0
        
        self._reset_operator_state()
    
    def _initialize_population_variant_09(self):
        """Initialize population with archive-guided seeding and restart-aware exploration boost."""
        samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        for d in range(self.dim):
            bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
            perm = np.random.permutation(self.NP)
            samples[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(self.NP)
        
        if not hasattr(self, 'global_best_archive'):
            self.global_best_archive = []
        
        if hasattr(self, 'x_opt') and self.f_opt is not None:
            min_separation = 0.05 * (self.ub[0] - self.lb[0])
            is_novel = True
            for archived in self.global_best_archive:
                if np.linalg.norm(self.x_opt - archived) < min_separation:
                    is_novel = False
                    break
            if is_novel and len(self.global_best_archive) < 25:
                self.global_best_archive.append(self.x_opt.copy())
        
        n_seeds = max(1, int(0.3 * self.NP))
        if self.global_best_archive:
            spread_scale = 0.15 * (self.ub[0] - self.lb[0])
            for i in range(n_seeds):
                center = self.global_best_archive[i % len(self.global_best_archive)]
                spread = spread_scale * (0.5 + 0.5 * np.random.random())
                samples[i] = center + spread * np.random.randn(self.dim)
                samples[i] = np.clip(samples[i], self.lb, self.ub)
        
        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        
        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        
        is_restart = hasattr(self, 'generation') and self.generation > 0
        if is_restart and hasattr(self, 'f_opt') and self.f_opt is not None:
            self.sigma = min(self.sigma * 2.5, (self.ub[0] - self.lb[0]) * 0.3)
            self.C *= 2.0
        
        self.fitness = self.func(self.population)
        self.fitness = np.asarray(self.fitness).flatten()
        
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
        """Compute diversity using effective sample size of the covariance matrix."""
        diag = np.diag(self.C)
        trace = np.sum(diag)
        expected_trace = self.dim * ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        trace_norm = trace / max(expected_trace, 1e-10)

        eigvals = np.linalg.eigvalsh(self.C)
        eigvals = np.clip(eigvals, 1e-15, None)
        geo_mean = np.exp(np.mean(np.log(eigvals)))
        arith_mean = np.mean(eigvals)
        ess = geo_mean / max(arith_mean, 1e-15)

        diversity = trace_norm * np.sqrt(ess)
        return float(np.clip(diversity, 1e-15, None))
    
    def _check_stagnation(self):
        """Track stagnation counter for restart logic."""
        if self.f_opt < self.f_opt_prev - 1e-12:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
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
            
            self.restart_count += 1
            
            self._initialize_population()
            
            self.population[0] = elite
            self.fitness[0] = elite_fit
            self.f_opt = float(np.asarray(elite_fit).flatten()[0])
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0
```