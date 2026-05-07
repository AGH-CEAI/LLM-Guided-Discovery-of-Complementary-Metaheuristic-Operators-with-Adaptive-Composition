import numpy as np


class CovarianceGuidedEvolutionStrategy:
    """
    Covariance-Guided Evolution Strategy (CG-ES) with Adaptive Clip-to-Bounds
    
    A population-based optimizer combining CMA-ES-style covariance adaptation
    with an adaptive operator selection mechanism that dynamically chooses
    the best clipping strategy using Thompson Sampling with sliding window
    credit assignment.
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
        
        # ===== ADAPTIVE OPERATOR SELECTION =====
        # 8 clipping strategies indexed 0-7
        self._num_operators = 8
        self._operator_names = [
            'clip_v1_exp_decay',   # 0: variant_idea_1 (7 wins)
            'clip_v2_rand_weight', # 1: variant_idea_2 (3 wins)
            'clip_v3_reflect',     # 2: variant_idea_3 (1 win)
            'clip_v4_cov_eigen',   # 3: variant_idea_4 (1 win)
            'clip_v5_iterative',   # 4: variant_idea_5 (1 win)
            'clip_v6_stoch',       # 5: variant_idea_6 (5 wins)
            'clip_v7_tanh',        # 6: variant_idea_7 (1 win)
            'clip_v8_shrink',      # 7: variant_idea_8 (4 wins)
        ]
        
        # Thompson Sampling: Beta distribution parameters (alpha, beta)
        self._alpha = np.ones(self._num_operators, dtype=np.float64)
        self._beta = np.ones(self._num_operators, dtype=np.float64)
        
        # Sliding window credit assignment
        self._window_size = 10
        self._reward_history = [[] for _ in range(self._num_operators)]
        self._operator_usage = np.zeros(self._num_operators, dtype=np.int64)
        
        # Exploration bonus
        self._exploration_bonus = 2.0
        self._min_pulls_for_exploitation = 3
        
        # Current selected operator
        self._current_operator = 0
        self._last_fitness = np.inf
        
        # Periodic full evaluation for exploration
        self._eval_interval = 20
        self._iteration_count = 0
        
    def __call__(self, func, stopping_condition):
        """Run the optimizer and return (f_opt, x_opt)."""
        self._initialize_optimizer()
        
        while not stopping_condition():
            self._sample_and_evaluate(func)
            self._update_distribution_parameters()
            self._adapt_step_size()
            self._update_elite_archive()
            self._manage_stagnation()
            self._update_operator_rewards()
            
            if self._should_restart():
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
        self._last_fitness = np.inf
        self._iteration_count = 0
        
    def _sample_and_evaluate(self, func):
        """Sample candidates, clip bounds, and evaluate fitness."""
        candidates = self._sample_candidates()
        clipped = self._clip_to_bounds(candidates)
        self.current_fitness = func(clipped)
        self.current_population = clipped
        self._iteration_count += 1
        
        self._handle_evaluation_errors()
        
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling with UCB fallback."""
        # Ensure minimum exploration
        if np.any(self._operator_usage < self._min_pulls_for_exploitation):
            # Force exploration of under-sampled operators
            candidates = np.where(self._operator_usage < self._min_pulls_for_exploitation)[0]
            return int(self._rng.choice(candidates))
        
        # Thompson Sampling: sample from Beta distribution for each operator
        samples = np.array([
            self._rng.beta(float(self._alpha[i]), float(self._beta[i]))
            for i in range(self._num_operators)
        ])
        
        # Add UCB-style exploration bonus
        ucb_bonus = np.sqrt(
            2.0 * np.log(np.sum(self._operator_usage) + 1) / (self._operator_usage + 1)
        )
        scores = samples + self._exploration_bonus * ucb_bonus
        
        return int(np.argmax(scores))
    
    def _clip_to_bounds(self, candidates):
        """Adaptive clip to bounds using selected operator."""
        # Select operator using Thompson Sampling
        self._current_operator = self._select_operator_thompson()
        
        # Apply the selected clipping strategy
        if self._current_operator == 0:
            return self._clip_variant_1(candidates)
        elif self._current_operator == 1:
            return self._clip_variant_2(candidates)
        elif self._current_operator == 2:
            return self._clip_variant_3(candidates)
        elif self._current_operator == 3:
            return self._clip_variant_4(candidates)
        elif self._current_operator == 4:
            return self._clip_variant_5(candidates)
        elif self._current_operator == 5:
            return self._clip_variant_6(candidates)
        elif self._current_operator == 6:
            return self._clip_variant_7(candidates)
        else:
            return self._clip_variant_8(candidates)
    
    # ===== CLIPPING STRATEGY VARIANTS =====
    
    def _clip_variant_1(self, candidates):
        """Variant 1: Exponential decay clipping (7 wins)."""
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
    
    def _clip_variant_2(self, candidates):
        """Variant 2: Random weight clipping (3 wins)."""
        low, high = self.bounds
        above = candidates > high
        below = candidates < low
        w = self._rng.uniform(0.5, 1.5, size=candidates.shape)
        repaired = np.where(above, high - w * (candidates - high), candidates)
        repaired = np.where(below, low + w * (low - candidates), repaired)
        return np.clip(repaired, low, high)
    
    def _clip_variant_3(self, candidates):
        """Variant 3: Reflection/toroidal clipping (1 win)."""
        low, high = self.bounds
        segment = high - low
        if segment == 0:
            return np.full_like(candidates, low)
        period = 2 * segment
        shifted = candidates - low
        t = shifted % period
        reflected = np.where(t <= segment, low + t, high - (t - segment))
        return np.clip(reflected, low, high)
    
    def _clip_variant_4(self, candidates):
        """Variant 4: Covariance eigenvalue clipping (1 win)."""
        low, high = self.bounds
        if self.covariance is not None:
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
                eigenvalues = np.maximum(eigenvalues, 1e-12)
                V = eigenvectors
                centered = candidates - self.mean
                y = centered @ V
                sigma = np.sqrt(eigenvalues)
                y_clipped = np.clip(y, -sigma, sigma)
                repaired = y_clipped @ V.T + self.mean
                return np.clip(repaired, low, high)
            except np.linalg.LinAlgError:
                pass
        return np.clip(candidates, low, high)
    
    def _clip_variant_5(self, candidates):
        """Variant 5: Iterative clipping (1 win)."""
        low, high = self.bounds
        repaired = candidates.copy()
        max_iter = 10
        for _ in range(max_iter):
            above = repaired > high
            below = repaired < low
            if not (np.any(above) or np.any(below)):
                break
            diff = repaired[above] - high
            repaired[above] = high - diff
            diff = low - repaired[below]
            repaired[below] = low + diff
        return np.clip(repaired, low, high)
    
    def _clip_variant_6(self, candidates):
        """Variant 6: Stochastic clipping (5 wins)."""
        low, high = self.bounds
        repaired = candidates.copy()
        if self.covariance is not None:
            std = np.sqrt(np.diag(self.covariance))
        else:
            std = np.full(self.dim, self.step_size)
        std = np.maximum(std, 1e-12)
        noise = self._rng.normal(0, std, size=candidates.shape)
        above = repaired > high
        below = repaired < low
        if np.any(above):
            repaired[above] = np.clip(high + noise[above], low, high)
        if np.any(below):
            repaired[below] = np.clip(low - noise[below], low, high)
        return repaired
    
    def _clip_variant_7(self, candidates):
        """Variant 7: Tanh compression clipping (1 win)."""
        low, high = self.bounds
        midpoint = (high + low) / 2.0
        rng = high - low
        if rng == 0:
            return np.full_like(candidates, low)
        z = (candidates - midpoint) / (rng / 2.0)
        compressed = np.tanh(z)
        repaired = low + (compressed + 1.0) * 0.5 * rng
        return np.clip(repaired, low, high)
    
    def _clip_variant_8(self, candidates):
        """Variant 8: Shrink factor clipping (4 wins)."""
        low, high = self.bounds
        rng = high - low
        if rng == 0:
            return np.full_like(candidates, low)
        above = candidates > high
        below = candidates < low
        d_above = candidates - high
        d_below = low - candidates
        shrink_above = 1.0 / (1.0 + d_above / rng)
        shrink_below = 1.0 / (1.0 + d_below / rng)
        repaired = np.where(above, high - d_above * shrink_above, candidates)
        repaired = np.where(below, low + d_below * shrink_below, repaired)
        return np.clip(repaired, low, high)
    
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
    
    def _update_operator_rewards(self):
        """Update operator rewards using sliding window credit assignment."""
        current_best = float(np.min(self.current_fitness))
        
        # Compute reward as relative improvement
        if np.isfinite(self._last_fitness) and self._last_fitness > 0:
            reward = (self._last_fitness - current_best) / (abs(self._last_fitness) + 1e-10)
        else:
            reward = 0.0
        
        # Clamp reward to reasonable range
        reward = float(np.clip(reward, -5.0, 5.0))
        
        # Store reward in sliding window
        op_idx = int(self._current_operator)
        self._reward_history[op_idx].append(reward)
        
        # Maintain window size
        if len(self._reward_history[op_idx]) > self._window_size:
            self._reward_history[op_idx].pop(0)
        
        # Update Thompson Sampling parameters
        if reward > 0:
            self._alpha[op_idx] += reward
        else:
            self._beta[op_idx] += abs(reward)
        
        # Ensure minimum parameters for numerical stability
        self._alpha[op_idx] = max(self._alpha[op_idx], 1.0)
        self._beta[op_idx] = max(self._beta[op_idx], 1.0)
        
        # Increment usage counter
        self._operator_usage[op_idx] += 1
        
        # Update last fitness
        self._last_fitness = current_best
    
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
        
    def _reinitialize_distribution(self):
        """Reinitialize distribution around current best."""
        best_idx = np.argmin(self.current_fitness)
        best_position = self.current_population[best_idx]
        
        self.mean = best_position.copy()
        self.covariance = np.eye(self.dim) * (self.step_size**2)
        self.step_size = 0.5 * (self.bounds[1] - self.bounds[0])
        
    def _clear_archive(self):
        """Clear the elite archive."""
        self.elite_archive = np.zeros((0, self.dim))
        
    def _reset_tracking(self):
        """Reset tracking variables after restart."""
        self.stagnation_counter = 0
        self.best_fitness_history = []
        self._last_fitness = np.inf
        
    def _get_best_result(self):
        """Extract best solution found from current population."""
        best_idx = np.argmin(self.current_fitness)
        best_position = self.current_population[best_idx]
        best_fitness = self.current_fitness[best_idx]
        
        return best_fitness, best_position.copy()
