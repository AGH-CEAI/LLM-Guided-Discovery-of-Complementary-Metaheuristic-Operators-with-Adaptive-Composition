```python
import numpy as np


class CovarianceGuidedEvolutionStrategy:
    """
    Adaptive Covariance-Guided Evolution Strategy (CG-ES)
    
    A population-based optimizer combining CMA-ES-style covariance adaptation
    with adaptive operator selection for the elite archive update strategy.
    Uses Thompson Sampling with sliding-window credit assignment to dynamically
    select the best _add_to_archive operator during optimization.
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
        
        # Adaptive operator selection using Thompson Sampling with sliding window
        self._num_operators = 7
        self._operator_successes = np.ones(self._num_operators, dtype=np.float64)
        self._operator_failures = np.ones(self._num_operators, dtype=np.float64)
        self._current_operator = 0
        self._operator_usage_counts = np.zeros(self._num_operators, dtype=np.int64)
        self._operator_rewards = np.zeros(self._num_operators, dtype=np.float64)
        self._total_improvement = 0.0
        self._iteration_counter = 0
        
        # Sliding window for credit assignment
        self._window_size = 30
        self._reward_history = [np.zeros(self._window_size) for _ in range(self._num_operators)]
        self._reward_indices = np.zeros(self._num_operators, dtype=np.int64)
        
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
                self._restart_optimizer()
                
        return self._get_best_result()
    
    def _initialize_optimizer(self):
        """Initialize all optimizer state variables."""
        self._initialize_distribution()
        self._initialize_archive()
        self._initialize_tracking()
        self._select_operator()
        
    def _select_operator(self):
        """Select operator using Thompson Sampling with Beta distribution."""
        samples = np.empty(self._num_operators, dtype=np.float64)
        for i in range(self._num_operators):
            alpha = max(1.0, float(self._operator_successes[i]))
            beta = max(1.0, float(self._operator_failures[i]))
            samples[i] = self._rng.beta(alpha, beta)
        
        self._current_operator = int(np.argmax(samples))
        
    def _update_operator_reward(self, reward):
        """Update operator statistics using sliding window credit assignment."""
        op = self._current_operator
        
        # Store reward in sliding window
        idx = int(self._reward_indices[op])
        self._reward_history[op][idx] = float(reward)
        self._reward_indices[op] = (idx + 1) % self._window_size
        
        # Compute average reward from sliding window
        window_mean = float(np.mean(self._reward_history[op]))
        
        # Update cumulative Thompson Sampling parameters
        if reward > 0:
            self._operator_successes[op] += 0.1 + window_mean * 0.2
        else:
            self._operator_failures[op] += 0.1 + (1.0 - window_mean) * 0.2
        
        # Apply soft decay to prevent stale statistics
        decay = 0.9995
        self._operator_successes[op] *= decay
        self._operator_failures[op] *= decay
        
        # Ensure minimum priors
        self._operator_successes[op] = max(1.0, self._operator_successes[op])
        self._operator_failures[op] = max(1.0, self._operator_failures[op])
        
        # Periodically re-select operator based on recent performance
        if self._iteration_counter % 5 == 0:
            self._select_operator()
        
        self._iteration_counter += 1
        
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
        self._iteration_counter = 0
        self._total_improvement = 0.0
        
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
        self.best_fitness_history.append(float(current_best))
        if len(self.best_fitness_history) > 20:
            self.best_fitness_history.pop(0)
            
        if len(self.best_fitness_history) >= 2:
            if self.best_fitness_history[-1] >= self.best_fitness_history[-2]:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                
    def _compute_improvement_rate(self):
        """Compute recent improvement rate."""
        history = np.array(self.best_fitness_history[-5:], dtype=np.float64)
        if len(history) < 2:
            return 0.0
        return float(np.mean(np.diff(history)))
        
    def _adjust_step_size(self, improvement_rate):
        """Adjust step size based on improvement rate."""
        if improvement_rate >= 0:
            self.step_size *= 1.1
        else:
            self.step_size *= 0.9
        self.step_size = np.clip(self.step_size, 1e-10, 
                                  (self.bounds[1] - self.bounds[0]) * 0.5)
        
    def _update_elite_archive(self):
        """Update the elite archive with best individuals using adaptive operator selection."""
        num_elites = min(self.archive_size, self.pop_size // 5)
        sorted_indices = np.argsort(self.current_fitness)
        new_elites = self.current_population[sorted_indices[:num_elites]]
        
        # Store previous best fitness for reward calculation
        prev_best = float(np.min(self.current_fitness)) if len(self.current_fitness) > 0 else float('inf')
        
        # Use adaptive operator selection
        self._add_to_archive_adaptive(new_elites)
        self._prune_archive()
        
        # Calculate reward based on fitness improvement
        current_best = float(np.min(self.current_fitness))
        improvement = prev_best - current_best
        
        # Normalize reward to [0, 1] range with clipping
        if improvement > 0:
            normalized_reward = min(1.0, improvement / (abs(prev_best) + 1e-6))
            self._total_improvement += improvement
        else:
            normalized_reward = 0.0
        
        # Update operator reward
        self._update_operator_reward(normalized_reward)
    
    def _add_to_archive_adaptive(self, new_elites):
        """Add new elite individuals using the selected adaptive operator."""
        op = int(self._current_operator)
        self._operator_usage_counts[op] += 1
        
        if op == 0:
            self._add_to_archive_v1(new_elites)
        elif op == 1:
            self._add_to_archive_v2(new_elites)
        elif op == 2:
            self._add_to_archive_v3(new_elites)
        elif op == 3:
            self._add_to_archive_v4(new_elites)
        elif op == 4:
            self._add_to_archive_v5(new_elites)
        elif op == 5:
            self._add_to_archive_v6(new_elites)
        else:
            self._add_to_archive_v7(new_elites)
    
    def _add_to_archive_v1(self, new_elites):
        """Add new elite individuals using exponential crossover (variant_idea_1)."""
        if new_elites is None or len(new_elites) == 0:
            return
        if self.elite_archive is None or len(self.elite_archive) == 0:
            self.elite_archive = np.atleast_2d(new_elites)
            return
        
        low, high = self.bounds
        n_new = len(new_elites)
        n_archive = len(self.elite_archive)
        indices = self._rng.choice(n_archive, size=n_new, replace=True)
        parents = self.elite_archive[indices]
        offspring = np.empty_like(new_elites)
        
        for i in range(n_new):
            child = new_elites[i].copy()
            k = self._rng.integers(0, self.dim)
            for j in range(self.dim):
                idx = (k + j) % self.dim
                if self._rng.random() < 0.5:
                    child[idx] = parents[i, idx]
            offspring[i] = np.clip(child, low, high)
        
        self.elite_archive = np.vstack([self.elite_archive, offspring])
    
    def _add_to_archive_v2(self, new_elites):
        """Add new elite individuals using BLX-α arithmetic blend crossover (variant_idea_2)."""
        if new_elites is None or len(new_elites) == 0:
            return
        if self.elite_archive is None or len(self.elite_archive) == 0:
            self.elite_archive = np.atleast_2d(new_elites)
            return
        
        low, high = self.bounds
        n_new = len(new_elites)
        alpha = 0.5
        indices = self._rng.choice(len(self.elite_archive), size=n_new, replace=True)
        parents = self.elite_archive[indices]
        offspring = np.empty_like(new_elites)
        
        for i in range(n_new):
            p_min = np.minimum(new_elites[i], parents[i])
            p_max = np.maximum(new_elites[i], parents[i])
            rng = p_max - p_min + 1e-12
            gamma = 1 + 2 * alpha
            child = p_min - alpha * rng + self._rng.uniform(0, gamma, size=self.dim) * (p_max - p_min + alpha * rng)
            offspring[i] = np.clip(child, low, high)
        
        self.elite_archive = np.vstack([self.elite_archive, offspring])
    
    def _add_to_archive_v3(self, new_elites):
        """Add new elite individuals using segment-based crossover (variant_idea_3)."""
        if new_elites is None or len(new_elites) == 0:
            return
        if self.elite_archive is None or len(self.elite_archive) == 0:
            self.elite_archive = np.atleast_2d(new_elites)
            return
        
        low, high = self.bounds
        n_new = len(new_elites)
        num_segments = max(2, self.dim // 5)
        indices = self._rng.choice(len(self.elite_archive), size=n_new, replace=True)
        parents = self.elite_archive[indices]
        offspring = np.empty_like(new_elites)
        
        for i in range(n_new):
            child = new_elites[i].copy()
            boundaries = self._rng.choice(self.dim - 1, size=num_segments - 1, replace=False)
            boundaries = np.sort(boundaries)
            starts = np.concatenate([[0], boundaries + 1])
            ends = np.concatenate([boundaries, [self.dim]])
            for s, e in zip(starts, ends):
                if self._rng.random() < 0.5:
                    child[s:e] = parents[i, s:e]
            offspring[i] = np.clip(child, low, high)
        
        self.elite_archive = np.vstack([self.elite_archive, offspring])
    
    def _add_to_archive_v4(self, new_elites):
        """Add new elite individuals using eigenvector-rotated crossover (variant_idea_4)."""
        if new_elites is None or len(new_elites) == 0:
            return
        if self.elite_archive is None or len(self.elite_archive) == 0:
            self.elite_archive = np.atleast_2d(new_elites)
            return
        
        low, high = self.bounds
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            sqrt_eigenvalues = np.sqrt(eigenvalues)
            T = eigenvectors * sqrt_eigenvalues
            T_inv = eigenvectors.T / sqrt_eigenvalues
            new_rotated = new_elites @ T
            indices = self._rng.choice(len(self.elite_archive), size=len(new_elites), replace=True)
            parents_rotated = self.elite_archive[indices] @ T
            mask = self._rng.random((len(new_elites), self.dim)) < 0.5
            offspring_rotated = np.where(mask, new_rotated, parents_rotated)
            offspring = offspring_rotated @ T_inv
        except np.linalg.LinAlgError:
            indices = self._rng.choice(len(self.elite_archive), size=len(new_elites), replace=True)
            offspring = np.where(
                self._rng.random((len(new_elites), self.dim)) < 0.5,
                new_elites, self.elite_archive[indices]
            )
        
        self.elite_archive = np.vstack([self.elite_archive, np.clip(offspring, low, high)])
    
    def _add_to_archive_v5(self, new_elites):
        """Add new elite individuals using fitness-weighted probabilistic merge (variant_idea_5)."""
        if new_elites is None or len(new_elites) == 0:
            return
        if self.elite_archive is None or len(self.elite_archive) == 0:
            self.elite_archive = np.atleast_2d(new_elites)
            return
        
        low, high = self.bounds
        fitnesses = self._evaluate_archive_members()
        valid_fitness = np.where(np.isfinite(fitnesses), fitnesses, np.max(fitnesses) + 1e6)
        inv_fitness = 1.0 / (valid_fitness - np.min(valid_fitness) + 1e-6)
        probs = inv_fitness / np.sum(inv_fitness)
        indices = self._rng.choice(len(self.elite_archive), size=len(new_elites), replace=True, p=probs)
        parents = self.elite_archive[indices]
        weights = self._rng.uniform(0.3, 0.7, size=(len(new_elites), 1))
        offspring = weights * new_elites + (1 - weights) * parents
        self.elite_archive = np.vstack([self.elite_archive, np.clip(offspring, low, high)])
    
    def _add_to_archive_v6(self, new_elites):
        """Add new elite individuals using crowding distance for diversity (variant_idea_6)."""
        if new_elites is None or len(new_elites) == 0:
            return
        if self.elite_archive is None or len(self.elite_archive) == 0:
            self.elite_archive = np.atleast_2d(new_elites)
            return
        
        low, high = self.bounds
        all_points = np.vstack([self.elite_archive, new_elites])
        n = len(all_points)
        distances = np.zeros(n)
        
        for d in range(self.dim):
            sorted_idx = np.argsort(all_points[:, d])
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            span = all_points[sorted_idx[-1], d] - all_points[sorted_idx[0], d] + 1e-12
            for i in range(1, n - 1):
                distances[sorted_idx[i]] += (all_points[sorted_idx[i + 1], d] - all_points[sorted_idx[i - 1], d]) / span
        
        archive_len = len(self.elite_archive)
        candidates = list(range(archive_len, n))
        sorted_candidates = sorted(candidates, key=lambda i: distances[i], reverse=True)
        
        if np.any(distances < np.inf):
            threshold = np.percentile(distances[distances < np.inf], 50)
        else:
            threshold = 0.0
        
        selected = [c for c in sorted_candidates if distances[c] > threshold]
        if not selected:
            selected = sorted_candidates[:max(1, len(sorted_candidates) // 2)]
        
        selected_indices = [c - archive_len for c in selected]
        self.elite_archive = np.vstack([self.elite_archive, np.clip(new_elites[selected_indices], low, high)])
    
    def _add_to_archive_v7(self, new_elites):
        """Add new elite individuals using simple stacking (original)."""
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
        return float(np.mean(distances))
        
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
        
    def _get_best_result(self):
        """Extract best solution found from current population."""
        best_idx = np.argmin(self.current_fitness)
        best_position = self.current_population[best_idx]
        best_fitness = self.current_fitness[best_idx]
        
        return float(best_fitness), best_position.copy()
```