import numpy as np


class CovarianceGuidedEvolutionStrategy:
    """
    Covariance-Guided Evolution Strategy (CG-ES) with Adaptive Operator Selection
    
    A population-based optimizer combining CMA-ES-style covariance adaptation
    with a simplified rank-based update and adaptive population management.
    Uses Thompson Sampling to adaptively select the best _clear_archive strategy
    during optimization based on accumulated performance data.
    """
    
    def __init__(self, dim=30, pop_size=None, bounds=(-100, 100)):
        self.dim = dim
        self.bounds = bounds
        self.pop_size = pop_size if pop_size else max(50, 4 + int(3 * np.log(dim)))
        self._rng = np.random.default_rng()
        
        # Distribution parameters
        self.mean = None
        self.covariance = None
        self.step_size = 1.0
        
        # Adaptive parameters
        self.mu_eff = 1.0
        self.cc = 0.01
        self.c1 = 0.01
        self.cmu = 0.01
        self.damping = 1.0
        
        # Archive for elitism
        self.elite_archive = None
        self.archive_size = 10
        
        # Diversity and stagnation tracking
        self.best_fitness_history = []
        self.stagnation_counter = 0
        self.stagnation_threshold = 50
        
        # Adaptive operator selection for _clear_archive
        self._num_operators = 7
        self._operator_names = [
            'original', 'variant_idea_1', 'variant_idea_2', 'variant_idea_3',
            'variant_idea_4', 'variant_idea_5', 'variant_idea_6'
        ]
        # Thompson Sampling: track alpha (successes + 1) and beta (failures + 1)
        self._operator_alpha = np.ones(self._num_operators)
        self._operator_beta = np.ones(self._num_operators)
        # Sliding window for credit assignment
        self._reward_window_size = 3
        self._operator_reward_history = [[] for _ in range(self._num_operators)]
        # Track last operator used and fitness context for reward calculation
        self._last_operator_idx = -1
        self._fitness_before_restart = None
        
    def __call__(self, func, stopping_condition):
        """Run the optimizer and return (f_opt, x_opt)."""
        self._initialize_optimizer()
        
        while not stopping_condition():
            self._sample_and_evaluate(func)
            self._update_distribution_parameters()
            self._adapt_step_size()
            self._update_elite_archive()
            self._manage_stagnation()
            
            if self._should_restart():
                self._record_pre_restart_fitness()
                self._restart_optimizer()
                
        return self._get_best_result()
    
    def _initialize_optimizer(self):
        """Initialize all optimizer state variables."""
        self._initialize_distribution()
        self._initialize_archive()
        self._initialize_tracking()
        
    def _initialize_distribution(self):
        """Initialize mean vector and covariance matrix."""
        self.mean = self._rng.uniform(
            self.bounds[0], self.bounds[1], size=self.dim
        )
        self.covariance = np.eye(self.dim)
        self.step_size = 0.5 * (self.bounds[1] - self.bounds[0])
        
    def _initialize_archive(self):
        n = self.archive_size
        if n <= 0:
            self.elite_archive = np.zeros((0, self.dim))
            return
        parent1 = self._rng.uniform(self.bounds[0], self.bounds[1], size=self.dim)
        parent2 = self._rng.uniform(self.bounds[0], self.bounds[1], size=self.dim)
        alpha = 0.5
        offspring = []
        for _ in range(n):
            gene_min = np.minimum(parent1, parent2)
            gene_max = np.maximum(parent1, parent2)
            range_len = gene_max - gene_min + 1e-8
            factor = self._rng.uniform(-alpha, 1 + alpha, size=self.dim)
            child = gene_min + factor * range_len
            offspring.append(child)
        offspring = np.array(offspring)
        offspring = np.clip(offspring, self.bounds[0], self.bounds[1])
        self.elite_archive = offspring
        
    def _initialize_tracking(self):
        """Initialize stagnation and diversity tracking."""
        self.best_fitness_history = []
        self.stagnation_counter = 0
        
    def _sample_and_evaluate(self, func):
        """Sample candidates, clip bounds, and evaluate fitness."""
        candidates = self._sample_candidates()
        clipped = self._clip_to_bounds(candidates)
        self.current_fitness = func(clipped)
        self.current_population = clipped
        
        self._handle_evaluation_errors()
        
    def _sample_candidates(self):
        """Sample new candidates from the adapted distribution."""
        num_samples = self.pop_size - len(self.elite_archive)
        samples = self._sample_from_distribution(num_samples)
        return samples
        
    def _sample_from_distribution(self, num_samples):
        """Draw samples using simplified covariance sampling."""
        try:
            L = np.linalg.cholesky(self.covariance)
            noise = self._rng.standard_normal((num_samples, self.dim))
            samples = self.mean + self.step_size * (noise @ L.T)
        except np.linalg.LinAlgError:
            samples = self._fallback_sampling(num_samples)
        return samples
        
    def _fallback_sampling(self, num_samples):
        """Fallback to diagonal covariance sampling."""
        std = np.sqrt(np.diag(self.covariance)) + 1e-8
        samples = self.mean + self.step_size * self._rng.standard_normal(
            (num_samples, self.dim)
        ) * std
        return samples
        
    def _clip_to_bounds(self, candidates):
        low, high = self.bounds
        rng = high - low
        rng = rng if rng != 0 else 1e-12
        above = candidates > high
        below = candidates < low
        d_above = candidates - high
        d_below = low - candidates
        repaired = np.where(above, high - d_above * np.exp(-d_above / rng), candidates)
        repaired = np.where(below, low + d_below * np.exp(-d_below / rng), repaired)
        return np.clip(repaired, low, high)
        
    def _handle_evaluation_errors(self):
        """Replace NaN/Inf fitness values with worst possible fitness."""
        valid = np.isfinite(self.current_fitness)
        if not np.all(valid):
            worst = np.nanmax(self.current_fitness[np.isfinite(self.current_fitness)])
            worst = worst if np.isfinite(worst) else 1e10
            self.current_fitness[~valid] = worst
            
    def _update_distribution_parameters(self):
        """Update mean and covariance based on selected individuals."""
        selected_indices = self._select_indices()
        selected_pop = self._get_combined_population()[selected_indices]
        selected_fitness = self.current_fitness[selected_indices]
        
        self._update_mean(selected_pop, selected_fitness)
        self._update_covariance(selected_pop)
        
    def _select_indices(self):
        """Select top individuals based on fitness (minimization)."""
        num_selected = max(1, self.pop_size // 4)
        sorted_indices = np.argsort(self.current_fitness)
        return sorted_indices[:num_selected]
        
    def _get_combined_population(self):
        """Combine current population with elite archive."""
        if len(self.elite_archive) > 0:
            return np.vstack([self.current_population, self.elite_archive])
        return self.current_population
        
    def _update_mean(self, selected_pop, selected_fitness):
        """Update the distribution mean using weighted average."""
        weights = self._compute_selection_weights(selected_fitness)
        old_mean = self.mean.copy()
        self.mean = np.average(selected_pop, axis=0, weights=weights)
        self.mean_diff = self.mean - old_mean
        
    def _compute_selection_weights(self, fitness):
        """Compute weights for selected individuals (softer than strict selection)."""
        normalized = fitness - fitness.min() + 1e-6
        weights = 1.0 / normalized
        weights /= weights.sum()
        return weights
        
    def _update_covariance(self, selected_pop):
        """Simplified rank-based covariance update."""
        self._compute_adaptation_learning_rates()
        
        # Rank-1 update
        rank_one = np.outer(self.mean_diff, self.mean_diff)
        
        # Rank-mu update (simplified)
        rank_mu = np.zeros((self.dim, self.dim))
        for i, ind in enumerate(selected_pop):
            diff = (ind - self.mean).reshape(-1, 1)
            rank_mu += np.dot(diff, diff.T)
        rank_mu /= len(selected_pop)
        
        # Combine updates
        self.covariance = (
            (1 - self.c1 - self.cmu) * self.covariance +
            self.c1 * rank_one +
            self.cmu * rank_mu
        )
        
        # Ensure positive definiteness
        self._ensure_covariance_validity()
        
    def _compute_adaptation_learning_rates(self):
        """Compute learning rates for covariance adaptation."""
        pop_ratio = self.pop_size / self.dim
        self.c1 = 2.0 / (pop_ratio + 2.0 * self.dim)
        self.cmu = min(0.6, 2.0 / (pop_ratio + 2.0 * self.dim**0.5))
        
    def _ensure_covariance_validity(self):
        """Ensure covariance matrix is valid (symmetric, positive semi-definite)."""
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        min_eigenvalue = np.min(eigenvalues)
        
        if min_eigenvalue < 1e-10:
            self.covariance += (1e-9 - min_eigenvalue) * np.eye(self.dim)
            
    def _adapt_step_size(self):
        """Adapt step size based on successful mutations."""
        current_best = np.min(self.current_fitness)
        self._update_stagnation_tracking(current_best)
        
        # Simple step size adaptation
        if len(self.best_fitness_history) >= 5:
            improvement_rate = self._compute_improvement_rate()
            self._adjust_step_size(improvement_rate)
            
    def _update_stagnation_tracking(self, current_best):
        """Track fitness history for stagnation detection."""
        self.best_fitness_history.append(current_best)
        if len(self.best_fitness_history) > 20:
            self.best_fitness_history.pop(0)
            
        if len(self.best_fitness_history) >= 2:
            if self.best_fitness_history[-1] >= self.best_fitness_history[-2]:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                
    def _compute_improvement_rate(self):
        """Compute recent improvement rate."""
        history = np.array(self.best_fitness_history[-5:])
        if len(history) < 2:
            return 0.0
        return np.mean(np.diff(history))
        
    def _adjust_step_size(self, improvement_rate):
        """Adjust step size based on improvement rate."""
        if improvement_rate >= 0:
            self.step_size *= 1.1
        else:
            self.step_size *= 0.9
        self.step_size = np.clip(self.step_size, 1e-10, 
                                  (self.bounds[1] - self.bounds[0]) * 0.5)
        
    def _update_elite_archive(self):
        """Update the elite archive with best individuals."""
        num_elites = min(self.archive_size, self.pop_size // 5)
        sorted_indices = np.argsort(self.current_fitness)
        new_elites = self.current_population[sorted_indices[:num_elites]]
        
        self._add_to_archive(new_elites)
        self._prune_archive()
        
    def _add_to_archive(self, new_elites):
        """Add new elite individuals to archive."""
        self.elite_archive = np.vstack([self.elite_archive, new_elites])
        
    def _prune_archive(self):
        """Maintain archive size by keeping only best individuals."""
        if len(self.elite_archive) > self.archive_size:
            fitness_values = self._evaluate_archive_members()
            best_indices = np.argsort(fitness_values)[:self.archive_size]
            self.elite_archive = self.elite_archive[best_indices]
            
    def _evaluate_archive_members(self):
        """Evaluate archive members (for pruning)."""
        fitness_values = np.array([
            self.current_fitness[np.argmin(
                np.sum((self.current_population - elite)**2, axis=1)
            )]
            for elite in self.elite_archive
        ])
        return fitness_values
        
    def _manage_stagnation(self):
        """Check and respond to stagnation conditions."""
        diversity = self._compute_diversity()
        self.diversity_threshold = 1e-6 * (self.bounds[1] - self.bounds[0])
        
        if diversity < self.diversity_threshold:
            self._boost_diversity()
            
    def _compute_diversity(self):
        """Compute population diversity as average pairwise distance."""
        population = self._get_combined_population()
        if len(population) < 2:
            return 1.0
            
        centroid = np.mean(population, axis=0)
        distances = np.sqrt(np.sum((population - centroid)**2, axis=1))
        return np.mean(distances)
        
    def _boost_diversity(self):
        """Boost diversity by perturbing the mean and increasing step size."""
        perturbation = self._rng.standard_normal(self.dim)
        self.mean += 0.3 * self.step_size * perturbation
        self.step_size *= 1.5
        self.step_size = np.clip(self.step_size, 1e-10,
                                  (self.bounds[1] - self.bounds[0]) * 0.5)
        
    def _should_restart(self):
        """Determine if optimizer should restart."""
        return (
            self.stagnation_counter > self.stagnation_threshold or
            self.step_size < 1e-10 or
            np.any(np.isnan(self.mean)) or
            np.any(np.isinf(self.mean))
        )
        
    def _restart_optimizer(self):
        """Perform strategic restart of the optimizer."""
        self._reinitialize_distribution()
        self._clear_archive()
        self._reset_tracking()
        self._update_operator_reward()
        
    def _reinitialize_distribution(self):
        """Reinitialize distribution around current best."""
        best_idx = np.argmin(self.current_fitness)
        best_position = self.current_population[best_idx]
        
        self.mean = best_position.copy()
        self.covariance = np.eye(self.dim) * (self.step_size**2)
        self.step_size = 0.5 * (self.bounds[1] - self.bounds[0])
        
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling."""
        samples = np.array([
            self._rng.beta(
                float(self._operator_alpha[i]),
                float(self._operator_beta[i])
            )
            for i in range(self._num_operators)
        ])
        return int(np.argmax(samples))
        
    def _compute_reward(self, fitness_before, fitness_after):
        """Compute reward for operator based on fitness improvement."""
        fitness_before = float(fitness_before)
        fitness_after = float(fitness_after)
        
        if not np.isfinite(fitness_before) or not np.isfinite(fitness_after):
            return 0.0
            
        improvement = fitness_before - fitness_after
        
        if improvement > 0:
            scale = max(1.0, abs(fitness_before))
            reward = improvement / scale
            return float(np.clip(reward, -5.0, 5.0))
        else:
            return float(np.clip(improvement * 0.1, -2.0, 0.1))
        
    def _record_pre_restart_fitness(self):
        """Record fitness before restart for reward calculation."""
        current_best = np.min(self.current_fitness)
        self._fitness_before_restart = float(current_best) if np.isfinite(current_best) else 1e10
        
    def _update_operator_reward(self):
        """Update operator reward based on post-restart performance."""
        if self._last_operator_idx < 0 or self._fitness_before_restart is None:
            return
            
        current_best = np.min(self.current_fitness)
        fitness_after = float(current_best) if np.isfinite(current_best) else 1e10
        
        reward = self._compute_reward(self._fitness_before_restart, fitness_after)
        reward = float(reward)
        
        history = self._operator_reward_history[self._last_operator_idx]
        history.append(reward)
        if len(history) > self._reward_window_size:
            history.pop(0)
        
        avg_reward = float(np.mean(history)) if history else 0.0
        
        if avg_reward > 0.05:
            self._operator_alpha[self._last_operator_idx] += 1.0
        elif avg_reward < -0.05:
            self._operator_beta[self._last_operator_idx] += 1.0
        else:
            self._operator_alpha[self._last_operator_idx] += 0.3
            self._operator_beta[self._last_operator_idx] += 0.3
            
        self._operator_alpha = np.clip(self._operator_alpha, 0.1, 1e6)
        self._operator_beta = np.clip(self._operator_beta, 0.1, 1e6)
        self._fitness_before_restart = None
        
    def _clear_archive_original(self):
        """Original: Clear the elite archive."""
        self.elite_archive = np.zeros((0, self.dim))
        
    def _clear_archive_variant_1(self):
        """Exponential crossover operator applied to archive members before discarding."""
        if self.elite_archive.shape[0] >= 2:
            idx = self._rng.choice(self.elite_archive.shape[0], 2, replace=False)
            p1 = self.elite_archive[idx[0]]
            p2 = self.elite_archive[idx[1]]
            lam = max(1, self.dim // 2)
            probs = np.exp(-np.arange(self.dim) / lam) / self.dim
            mask = self._rng.random(self.dim) < probs
            offspring = np.where(mask, p1, p2)
            low, high = self.bounds
            offspring = np.clip(offspring, low, high)
        self.elite_archive = np.zeros((0, self.dim))
        
    def _clear_archive_variant_2(self):
        """Arithmetic blend (BLX-α) crossover between archive centroid and current mean before clearing."""
        if self.elite_archive.shape[0] > 0:
            alpha = 0.5
            parent1 = np.mean(self.elite_archive, axis=0)
            parent2 = self.mean
            gene_min = np.minimum(parent1, parent2)
            gene_max = np.maximum(parent1, parent2)
            range_len = gene_max - gene_min
            range_len = np.where(range_len == 0, 1e-8, range_len)
            factor = self._rng.uniform(-alpha, 1 + alpha, size=self.dim)
            blended = gene_min + factor * range_len
            low, high = self.bounds
            blended = np.clip(blended, low, high)
        self.elite_archive = np.zeros((0, self.dim))
        
    def _clear_archive_variant_3(self):
        """Segment-based crossover: swap a random contiguous segment between two archive members."""
        if self.elite_archive.shape[0] >= 2:
            idx = self._rng.choice(self.elite_archive.shape[0], 2, replace=False)
            p1 = self.elite_archive[idx[0]]
            p2 = self.elite_archive[idx[1]]
            seg_start = self._rng.integers(0, self.dim)
            max_len = max(1, self.dim - seg_start)
            seg_len = self._rng.integers(1, max_len + 1)
            offspring = p1.copy()
            offspring[seg_start:seg_start + seg_len] = p2[seg_start:seg_start + seg_len]
            low, high = self.bounds
            offspring = np.clip(offspring, low, high)
        self.elite_archive = np.zeros((0, self.dim))
        
    def _clear_archive_variant_4(self):
        """Rotate archive using eigenvectors of the current covariance matrix before clearing."""
        if self.elite_archive.shape[0] > 0:
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
                centered = self.elite_archive - self.mean
                rotated = centered @ eigenvectors
                low, high = self.bounds
                rotated = np.clip(rotated, low, high)
            except np.linalg.LinAlgError:
                pass
        self.elite_archive = np.zeros((0, self.dim))
        
    def _clear_archive_variant_5(self):
        """Deterministic crowding: replace the worst archive member with the best before clearing."""
        if self.elite_archive.shape[0] > 0:
            distances = np.linalg.norm(self.elite_archive - self.mean, axis=1)
            best_idx = np.argmin(distances)
            worst_idx = np.argmax(distances)
            self.elite_archive[worst_idx] = self.elite_archive[best_idx]
            low, high = self.bounds
            self.elite_archive = np.clip(self.elite_archive, low, high)
        self.elite_archive = np.zeros((0, self.dim))
        
    def _clear_archive_variant_6(self):
        """Clustering-based compression: run simple k-means on archive before clearing."""
        if self.elite_archive.shape[0] > 1:
            k = max(2, self.elite_archive.shape[0] // 2)
            centroids = self.elite_archive[self._rng.choice(self.elite_archive.shape[0], k, replace=False)]
            for _ in range(5):
                dists = np.sum((self.elite_archive[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
                assignments = np.argmin(dists, axis=1)
                for c in range(k):
                    mask = assignments == c
                    if np.any(mask):
                        centroids[c] = np.mean(self.elite_archive[mask], axis=0)
            low, high = self.bounds
            centroids = np.clip(centroids, low, high)
        self.elite_archive = np.zeros((0, self.dim))
        
    def _clear_archive(self):
        """Select and apply _clear_archive strategy using Thompson Sampling."""
        self._last_operator_idx = self._select_operator_thompson()
        
        if self._last_operator_idx == 0:
            self._clear_archive_original()
        elif self._last_operator_idx == 1:
            self._clear_archive_variant_1()
        elif self._last_operator_idx == 2:
            self._clear_archive_variant_2()
        elif self._last_operator_idx == 3:
            self._clear_archive_variant_3()
        elif self._last_operator_idx == 4:
            self._clear_archive_variant_4()
        elif self._last_operator_idx == 5:
            self._clear_archive_variant_5()
        elif self._last_operator_idx == 6:
            self._clear_archive_variant_6()
        
    def _reset_tracking(self):
        """Reset tracking variables after restart."""
        self.stagnation_counter = 0
        self.best_fitness_history = []
        
    def _get_best_result(self):
        """Extract best solution found from current population."""
        best_idx = np.argmin(self.current_fitness)
        best_position = self.current_population[best_idx]
        best_fitness = self.current_fitness[best_idx]
        
        return best_fitness, best_position.copy()
