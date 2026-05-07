import numpy as np


class CovarianceGuidedNichingOptimizer:
    """
    A population-based optimizer combining CMA-ES-style covariance adaptation
    with niching and subpopulation-based diversity maintenance.
    
    Key design:
    - Maintains multiple subpopulations (niches) that adapt their own distributions
    - Global covariance adaptation alongside per-niche adaptation
    - Adaptive step sizes per niche
    - Elite tracking and diversity-based restart mechanism
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population sizing
        self.np = min(max(4 * dim, 120), 10 * dim)
        self.n_niches = max(3, dim // 10)
        self.individuals_per_niche = self.np // self.n_niches
        
        # State variables
        self.population = None
        self.fitness = None
        self.best_fitness = np.inf
        self.best_solution = None
        self.generation = 0
        
        # Per-niche state
        self.niche_means = None
        self.niche_covs = None
        self.niche_steps = None
        self.niche_success_counts = None
        self.niche_trial_counts = None
        
        # Global adaptation state
        self.global_mean = None
        self.global_cov = None
        self.global_step = None
        self.global_success_count = 0
        self.global_trial_count = 0
        self.p_c = None  # Evolution path for covariance
        self.p_s = None  # Evolution path for step size
        
        # Tracking
        self.stagnation_counter = 0
        self.prev_best_fitness = np.inf
        self.elite_pool = None
        self.elite_fitness = None
        
    def _initialize_population(self):
        """Initialize population and all adaptive structures."""
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.np, self.dim)
        )
        self.fitness = self._evaluate_batch(self.population)
        
        # Initialize per-niche structures
        self.niche_means = np.zeros((self.n_niches, self.dim))
        self.niche_covs = np.array([
            np.eye(self.dim) * (20.0 ** 2) for _ in range(self.n_niches)
        ])
        self.niche_steps = np.full(self.n_niches, 10.0)
        self.niche_success_counts = np.zeros(self.n_niches, dtype=int)
        self.niche_trial_counts = np.zeros(self.n_niches, dtype=int)
        
        # Initialize global structures
        self.global_mean = np.zeros(self.dim)
        self.global_cov = np.eye(self.dim) * (50.0 ** 2)
        self.global_step = 20.0
        self.p_c = np.zeros(self.dim)
        self.p_s = np.zeros(self.dim)
        
        # Initialize niches
        self._assign_to_niches()
        
        # Initialize elite pool
        self.elite_pool = []
        self.elite_fitness = []
        
        self._update_best_tracking()
        
    def _assign_to_niches(self):
        """Assign individuals to niches based on nearest niche mean."""
        for i in range(self.n_niches):
            start_idx = i * self.individuals_per_niche
            end_idx = start_idx + self.individuals_per_niche
            if end_idx <= self.np:
                niche_pop = self.population[start_idx:end_idx]
                self.niche_means[i] = np.mean(niche_pop, axis=0)
                
    def _evaluate_batch(self, population_batch):
        """Evaluate fitness for a batch of candidates."""
        if len(population_batch) == 0:
            return np.array([])
        
        clipped = np.clip(population_batch, self.lower_bound, self.upper_bound)
        fitness = self._func(clipped)
        
        # Handle budget exhaustion
        if len(fitness) < len(population_batch):
            fitness = np.append(fitness, [np.inf] * (len(population_batch) - len(fitness)))
        
        return fitness
        
    def _func(self, population_batch):
        """Wrapper that handles various return types from func."""
        result = self._objective_func(population_batch)
        if result is None or len(result) == 0:
            return np.full(len(population_batch), np.inf)
        return np.array(result)
        
    def _update_best_tracking(self):
        """Update best solution found so far."""
        valid_mask = np.isfinite(self.fitness)
        if not np.any(valid_mask):
            return
            
        valid_fitness = self.fitness[valid_mask]
        valid_population = self.population[valid_mask]
        best_idx = np.argmin(valid_fitness)
        
        if valid_fitness[best_idx] < self.best_fitness:
            self.best_fitness = valid_fitness[best_idx]
            self.best_solution = valid_population[best_idx].copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
            
    def __call__(self, func, stopping_condition):
        self._objective_func = func
        
        self._initialize_population()
        
        while not stopping_condition():
            if self.generation % 50 == 0:
                self._manage_elite_pool()
            
            trials = self._sample_trials_batch()
            trial_fitness = self._evaluate_batch(trials)
            
            if stopping_condition():
                break
                
            if len(trial_fitness) < len(trials):
                min_len = min(len(trial_fitness), len(trials))
                trials = trials[:min_len]
                trial_fitness = trial_fitness[:min_len]
                
            self._update_trial_tracking(trials, trial_fitness)
            
            self.population, self.fitness = self._select_survivors_batch(
                self.population, self.fitness, trials, trial_fitness
            )
            
            self._update_distribution_batch()
            self._adapt_step_size()
            self._adapt_covariance_batch()
            
            self._update_best_tracking()
            
            if self._restart_if_stagnant():
                self._initialize_population()
                
            self.generation += 1
            
        return self.best_fitness, self.best_solution.copy() if self.best_solution is not None else None
        
    def _sample_trials_batch(self):
        """Generate trial population using niche-based sampling."""
        trials = np.empty((self.np, self.dim))
        
        for i in range(self.n_niches):
            start_idx = i * self.individuals_per_niche
            end_idx = min(start_idx + self.individuals_per_niche, self.np)
            count = end_idx - start_idx
            
            if count <= 0:
                continue
                
            # Sample from niche distribution
            niche_mean = self.niche_means[i]
            niche_cov = self.niche_covs[i]
            step = self.niche_steps[i]
            
            # Multi-strategy sampling
            strategy = i % 3
            if strategy == 0:
                # Standard CMA-like sampling
                samples = np.random.multivariate_normal(niche_mean, niche_cov, count)
                samples = niche_mean + (samples - niche_mean) * (step / np.sqrt(np.trace(niche_cov) / self.dim))
            elif strategy == 1:
                # Best-guided sampling
                if self.best_solution is not None:
                    direction = self.best_solution - niche_mean
                    samples = niche_mean + np.outer(
                        np.random.randn(count), direction * 0.3
                    ) + np.random.multivariate_normal(
                        np.zeros(self.dim), niche_cov * 0.5, count
                    )
                else:
                    samples = np.random.multivariate_normal(niche_mean, niche_cov, count)
            else:
                # Global-mean guided sampling
                if self.global_mean is not None:
                    direction = self.global_mean - niche_mean
                    samples = niche_mean + np.outer(
                        np.random.randn(count), direction * 0.2
                    ) + np.random.multivariate_normal(
                        niche_mean, niche_cov, count
                    )
                else:
                    samples = np.random.multivariate_normal(niche_mean, niche_cov, count)
            
            trials[start_idx:end_idx] = samples
            
        return self._clip_to_bounds(trials)
        
    def _clip_to_bounds(self, population_batch):
        """Clip population to search bounds."""
        return np.clip(population_batch, self.lower_bound, self.upper_bound)
        
    def _update_trial_tracking(self, trials, trial_fitness):
        """Track trial success for adaptation."""
        for i in range(self.n_niches):
            start_idx = i * self.individuals_per_niche
            end_idx = min(start_idx + self.individuals_per_niche, len(trial_fitness))
            
            parent_fitness = self.fitness[start_idx:end_idx]
            trial_subset = trial_fitness[start_idx:end_idx]
            
            if len(trial_subset) > 0:
                successes = np.sum(trial_subset < parent_fitness)
                self.niche_success_counts[i] += successes
                self.niche_trial_counts[i] += len(trial_subset)
                
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Select survivors using greedy selection per individual."""
        improved = trial_fitness < fitness
        
        new_population = population.copy()
        new_population[improved] = trials[improved]
        
        new_fitness = fitness.copy()
        new_fitness[improved] = trial_fitness[improved]
        
        return new_population, new_fitness
        
    def _update_distribution_batch(self):
        """Update niche means and global mean based on current population."""
        # Update per-niche means
        for i in range(self.n_niches):
            start_idx = i * self.individuals_per_niche
            end_idx = min(start_idx + self.individuals_per_niche, self.np)
            
            niche_pop = self.population[start_idx:end_idx]
            if len(niche_pop) > 0:
                new_mean = np.mean(niche_pop, axis=0)
                # Blend with previous mean for stability
                self.niche_means[i] = 0.8 * self.niche_means[i] + 0.2 * new_mean
                
        # Update global mean
        self.global_mean = np.mean(self.population, axis=0)
        
    def _adapt_step_size(self):
        """Adapt step sizes using cumulative step size adaptation."""
        target_success_rate = 0.25
        cs_factor = 1.0 / (self.dim + 5.0)
        damping = 1.0 + cs_factor
        
        # Per-niche adaptation
        for i in range(self.n_niches):
            if self.niche_trial_counts[i] >= self.individuals_per_niche:
                success_rate = self.niche_success_counts[i] / max(1, self.niche_trial_counts[i])
                
                if success_rate > target_success_rate:
                    self.niche_steps[i] *= np.exp(0.3 * (success_rate - target_success_rate) / target_success_rate)
                else:
                    self.niche_steps[i] /= damping
                    
                self.niche_steps[i] = np.clip(self.niche_steps[i], 0.1, 50.0)
                self.niche_success_counts[i] = 0
                self.niche_trial_counts[i] = 0
                
        # Global adaptation
        min_trials = self.np
        if self.global_trial_count >= min_trials:
            success_rate = self.global_success_count / max(1, self.global_trial_count)
            
            if success_rate > target_success_rate:
                self.global_step *= np.exp(0.3 * (success_rate - target_success_rate) / target_success_rate)
            else:
                self.global_step /= damping
                
            self.global_step = np.clip(self.global_step, 0.5, 80.0)
            self.global_success_count = 0
            self.global_trial_count = 0
            
    def _adapt_covariance_batch(self):
        """Apply CMA-ES style covariance adaptation."""
        # Per-niche covariance updates
        for i in range(self.n_niches):
            if self.niche_trial_counts[i] >= self.individuals_per_niche:
                self._adapt_covariance_variant_01(i)
                
        # Global covariance adaptation
        min_trials = self.np
        if self.global_trial_count >= min_trials:
            self._adapt_covariance_variant_08()
            
    def _adapt_covariance_variant_01(self, niche_idx):
        """Covariance adaptation for a specific niche using rank-1 update."""
        niche_pop = self.population[
            niche_idx * self.individuals_per_niche:
            (niche_idx + 1) * self.individuals_per_niche
        ]
        
        if len(niche_pop) < 2:
            return
            
        mean = self.niche_means[niche_idx]
        diff = niche_pop - mean
        
        # Rank-1 update
        cov_update = np.outer(diff.mean(axis=0), diff.mean(axis=0))
        
        # Rank-k update using top performers
        fitness_subset = self.fitness[
            niche_idx * self.individuals_per_niche:
            (niche_idx + 1) * self.individuals_per_niche
        ]
        
        if len(fitness_subset) > 0:
            k = max(1, len(fitness_subset) // 3)
            top_k_idx = np.argsort(fitness_subset)[:k]
            top_k_diff = niche_pop[top_k_idx] - mean
            
            if len(top_k_diff) > 0:
                rank_k_cov = np.cov(top_k_diff.T)
                if rank_k_cov.ndim == 2:
                    self.niche_covs[niche_idx] = (
                        0.7 * self.niche_covs[niche_idx] +
                        0.2 * cov_update +
                        0.1 * rank_k_cov +
                        1e-6 * np.eye(self.dim)
                    )
                    
    def _adapt_covariance_variant_08(self):
        """Global covariance adaptation using evolution path."""
        # Evolution path for covariance
        mean_diff = self.global_mean - (self.global_mean if self.generation == 0 else self.global_mean)
        
        if self.generation > 0:
            self.p_c = (1 - 1.0 / (self.dim + 1)) * self.p_c + np.sqrt(
                1.0 / (self.dim + 1)) * mean_diff / self.global_step
        else:
            self.p_c = np.zeros(self.dim)
            
        # Rank-1 update
        rank_1_term = np.outer(self.p_c, self.p_c)
        
        # Rank-k update from top performers
        k = max(1, self.np // 4)
        top_k_idx = np.argsort(self.fitness)[:k]
        top_k_pop = self.population[top_k_idx]
        top_k_diff = top_k_pop - self.global_mean
        
        rank_k_cov = np.zeros((self.dim, self.dim))
        if len(top_k_diff) > 1:
            rank_k_cov = np.cov(top_k_diff.T)
            if rank_k_cov.ndim != 2:
                rank_k_cov = np.outer(mean_diff, mean_diff)
                
        # Combine updates
        learning_rate = 1.0 / (self.dim + 5.0)
        self.global_cov = (
            (1 - learning_rate) * self.global_cov +
            learning_rate * (rank_1_term + 0.3 * rank_k_cov) +
            1e-8 * np.eye(self.dim)
        )
        
        # Ensure positive definiteness
        eigenvalues = np.linalg.eigvalsh(self.global_cov)
        if np.any(eigenvalues < 1e-10):
            self.global_cov += (1e-6 - eigenvalues.min()) * np.eye(self.dim)
            
    def _adapt_covariance_variant_09(self):
        """Alternative covariance adaptation using weighted combination of top solutions."""
        weights = np.exp(-self.fitness / (np.std(self.fitness) + 1e-10))
        weights /= weights.sum()
        
        weighted_mean = np.sum(self.population * weights[:, np.newaxis], axis=0)
        diff = self.population - weighted_mean
        
        weighted_cov = np.zeros((self.dim, self.dim))
        for i in range(len(self.population)):
            weighted_cov += weights[i] * np.outer(diff[i], diff[i])
            
        alpha = 0.1
        self.global_cov = (1 - alpha) * self.global_cov + alpha * weighted_cov
        
    def _compute_diversity(self):
        """Compute population diversity using average pairwise distance."""
        if len(self.population) < 2:
            return 0.0
            
        # Sample for efficiency
        sample_size = min(50, len(self.population))
        indices = np.random.choice(len(self.population), sample_size, replace=False)
        sample = self.population[indices]
        
        # Compute distances
        diff_matrix = sample[:, np.newaxis, :] - sample[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff_matrix ** 2, axis=2))
        
        # Exclude diagonal
        mask = ~np.eye(sample_size, dtype=bool)
        return np.mean(distances[mask])
        
    def _update_operator_rewards(self, rewards_dict):
        """Update rewards for different operators based on their success."""
        pass
        
    def _manage_elite_pool(self):
        """Maintain a pool of elite solutions for diversity."""
        max_elite_size = self.np // 2
        
        valid_mask = np.isfinite(self.fitness)
        if not np.any(valid_mask):
            return
            
        current_best_idx = np.argmin(self.fitness[valid_mask])
        current_best_fit = self.fitness[valid_mask][current_best_idx]
        current_best_sol = self.population[valid_mask][current_best_idx]
        
        # Add current best to pool
        if len(self.elite_pool) == 0 or current_best_fit < max(self.elite_fitness):
            self.elite_pool.append(current_best_sol.copy())
            self.elite_fitness.append(current_best_fit)
            
        # Keep top performers
        if len(self.elite_fitness) > max_elite_size:
            sorted_indices = np.argsort(self.elite_fitness)[:max_elite_size]
            self.elite_pool = [self.elite_pool[i] for i in sorted_indices]
            self.elite_fitness = [self.elite_fitness[i] for i in sorted_indices]
            
    def _restart_if_stagnant(self):
        """Check for stagnation and trigger restart if needed."""
        stagnation_threshold = 50 + self.dim
        diversity_threshold = 1.0
        
        should_restart = (
            self.stagnation_counter > stagnation_threshold or
            self._compute_diversity() < diversity_threshold
        )
        
        if should_restart:
            # Inject diversity from elite pool
            if len(self.elite_pool) > 0:
                n_inject = min(len(self.elite_pool), self.np // 4)
                inject_indices = np.random.choice(self.np, n_inject, replace=False)
                
                for idx, elite_idx in enumerate(inject_indices):
                    noise_scale = 5.0 * (1.0 - idx / n_inject)
                    self.population[elite_idx] = self.elite_pool[idx] + np.random.randn(self.dim) * noise_scale
                    
            self.stagnation_counter = 0
            
            # Reset adaptation state
            for i in range(self.n_niches):
                self.niche_success_counts[i] = 0
                self.niche_trial_counts[i] = 0
                
            self.global_success_count = 0
            self.global_trial_count = 0
            
            return True
            
        return False
