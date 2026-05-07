import numpy as np


class CovarianceGuidedEvolutionStrategy:
    """
    Adaptive Covariance-Guided Evolution Strategy (CG-ES)
    
    A population-based optimizer combining CMA-ES-style covariance adaptation
    with a simplified rank-based update and adaptive population management.
    Uses Thompson Sampling to adaptively select the best _initialize_archive
    strategy during optimization based on benchmark results.
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
        
        # Adaptive operator selection for _initialize_archive using Thompson Sampling
        self._num_init_strategies = 8
        # Beta distribution parameters (alpha, beta) for each strategy
        self._strategy_alpha = np.ones(self._num_init_strategies)
        self._strategy_beta = np.ones(self._num_init_strategies)
        self._current_init_strategy = 0
        self._init_reward_window = 5
        self._strategy_rewards = [[] for _ in range(self._num_init_strategies)]
        self._last_init_fitness = float('inf')
        self._total_init_calls = 0
        
    def __call__(self, func, stopping_condition):
        """Run the optimizer and return (f_opt, x_opt)."""
        self._initialize_optimizer()
        
        while not stopping_condition():
            self._sample_and_evaluate(func)
            self._update_distribution_parameters()
            self._adapt_step_size()
            self._update_elite_archive()
            self._update_init_strategy_reward()
            self._manage_stagnation()
            
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
        """Initialize elite archive using adaptively selected strategy."""
        self._current_init_strategy = self._select_init_strategy_thompson()
        self._last_init_fitness = float('inf')
        self._total_init_calls += 1
        
        if self._current_init_strategy == 0:
            self._initialize_archive_v1()
        elif self._current_init_strategy == 1:
            self._initialize_archive_v2()
        elif self._current_init_strategy == 2:
            self._initialize_archive_v3()
        elif self._current_init_strategy == 3:
            self._initialize_archive_v4()
        elif self._current_init_strategy == 4:
            self._initialize_archive_v5()
        elif self._current_init_strategy == 5:
            self._initialize_archive_v6()
        elif self._current_init_strategy == 6:
            self._initialize_archive_v7()
        else:
            self._initialize_archive_v8()
    
    def _select_init_strategy_thompson(self):
        """Select initialization strategy using Thompson Sampling."""
        samples = np.array([
            self._rng.beta(max(0.1, float(self._strategy_alpha[i])), 
                         max(0.1, float(self._strategy_beta[i])))
            for i in range(self._num_init_strategies)
        ])
        return int(np.argmax(samples))
    
    def _update_init_strategy_reward(self):
        """Update Thompson Sampling rewards based on initialization effectiveness."""
        current_best = np.min(self.current_fitness)
        
        if np.isfinite(self._last_init_fitness) and self._last_init_fitness > 0:
            improvement = (self._last_init_fitness - current_best) / (abs(self._last_init_fitness) + 1e-10)
        else:
            improvement = 0.0
        
        reward = float(np.clip(improvement, -10.0, 10.0))
        self._strategy_rewards[self._current_init_strategy].append(reward)
        
        if len(self._strategy_rewards[self._current_init_strategy]) > self._init_reward_window:
            self._strategy_rewards[self._current_init_strategy].pop(0)
        
        alpha_sum = 1.0
        beta_sum = 1.0
        for r in self._strategy_rewards[self._current_init_strategy]:
            if r > 0:
                alpha_sum += r * 0.5
            else:
                beta_sum += abs(r) * 0.5
        
        self._strategy_alpha[self._current_init_strategy] = max(0.1, alpha_sum)
        self._strategy_beta[self._current_init_strategy] = max(0.1, beta_sum)
    
    # Strategy 1: LHS-based initialization (2 wins)
    def _initialize_archive_v1(self):
        n = self.archive_size
        if n <= 0:
            self.elite_archive = np.zeros((0, self.dim))
            return
        intervals = np.linspace(0, 1, n + 1)
        points = np.zeros((n, self.dim))
        for d in range(self.dim):
            perm = self._rng.permutation(n)
            randoms = self._rng.random(n)
            points[:, d] = intervals[perm] + randoms[perm] * (intervals[1] - intervals[0])
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        self.elite_archive = np.clip(points, self.bounds[0], self.bounds[1])
    
    # Strategy 2: Segment-based crossover initialization (4 wins)
    def _initialize_archive_v2(self):
        n = self.archive_size
        if n <= 0:
            self.elite_archive = np.zeros((0, self.dim))
            return
        parent1 = self.mean.copy()
        parent2 = self._rng.uniform(self.bounds[0], self.bounds[1], size=self.dim)
        offspring = []
        for _ in range(n):
            child = parent2.copy()
            start = self._rng.integers(0, self.dim)
            length = self._rng.integers(0, self.dim + 1)
            indices = np.arange(start, start + length) % self.dim
            child[indices] = parent1[indices]
            offspring.append(child)
        offspring = np.array(offspring)
        self.elite_archive = np.clip(offspring, self.bounds[0], self.bounds[1])
    
    # Strategy 3: BLX-alpha based initialization (4 wins)
    def _initialize_archive_v3(self):
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
        self.elite_archive = np.clip(offspring, self.bounds[0], self.bounds[1])
    
    # Strategy 4: Half-space center initialization (1 win)
    def _initialize_archive_v4(self):
        n = self.archive_size
        if n <= 0:
            self.elite_archive = np.zeros((0, self.dim))
            return
        centers = []
        mid = (self.bounds[0] + self.bounds[1]) / 2.0
        lower_center = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * 0.25
        upper_center = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * 0.75
        for d in range(self.dim):
            point = np.full(self.dim, mid)
            point[d] = lower_center
            centers.append(point)
            point = np.full(self.dim, mid)
            point[d] = upper_center
            centers.append(point)
        centers = np.array(centers)
        if n >= len(centers):
            selected = centers
        else:
            selected = self._rng.choice(centers, size=n, replace=False)
        self.elite_archive = np.clip(selected, self.bounds[0], self.bounds[1])
    
    # Strategy 5: Eigenvector-based initialization (1 win)
    def _initialize_archive_v5(self):
        n = self.archive_size
        if n <= 0:
            self.elite_archive = np.zeros((0, self.dim))
            return
        try:
            _, eigenvectors = np.linalg.eigh(self.covariance)
        except np.linalg.LinAlgError:
            eigenvectors = np.eye(self.dim)
        points = []
        step = self.step_size * 0.5
        for i in range(n):
            if i < self.dim:
                point = self.mean.copy() + step * eigenvectors[:, i]
            else:
                point = self.mean.copy() + step * self._rng.standard_normal(self.dim)
            points.append(point)
        points = np.array(points)
        self.elite_archive = np.clip(points, self.bounds[0], self.bounds[1])
    
    # Strategy 6: Halton sequence initialization (4 wins)
    def _initialize_archive_v6(self):
        n = self.archive_size
        if n <= 0:
            self.elite_archive = np.zeros((0, self.dim))
            return
        def halton_sequence(index, base):
            result = 0.0
            f = 1.0
            while index > 0:
                f /= base
                result += f * (index % base)
                index //= base
            return result
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        points = np.zeros((n, self.dim))
        for d in range(self.dim):
            base = bases[d % len(bases)]
            for i in range(n):
                points[i, d] = halton_sequence(i + 1, base)
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        self.elite_archive = np.clip(points, self.bounds[0], self.bounds[1])
    
    # Strategy 7: Orthogonal direction initialization (4 wins)
    def _initialize_archive_v7(self):
        n = self.archive_size
        if n <= 0:
            self.elite_archive = np.zeros((0, self.dim))
            return
        random_vectors = self._rng.standard_normal((n, self.dim))
        orthogonal_vectors = []
        for i in range(n):
            v = random_vectors[i].copy()
            for j in range(len(orthogonal_vectors)):
                v = v - np.dot(orthogonal_vectors[j], v) * orthogonal_vectors[j]
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                v = v / norm
                orthogonal_vectors.append(v)
            else:
                v = self._rng.standard_normal(self.dim)
                for j in range(len(orthogonal_vectors)):
                    v = v - np.dot(orthogonal_vectors[j], v) * orthogonal_vectors[j]
                norm = np.linalg.norm(v)
                if norm > 1e-10:
                    v = v / norm
                    orthogonal_vectors.append(v)
        orthogonal_vectors = np.array(orthogonal_vectors)
        points = []
        step = self.step_size * 0.5
        for i in range(min(len(orthogonal_vectors), n)):
            points.append(self.mean.copy() + step * orthogonal_vectors[i])
        while len(points) < n:
            points.append(self.mean.copy() + step * self._rng.standard_normal(self.dim))
        points = np.array(points)
        self.elite_archive = np.clip(points, self.bounds[0], self.bounds[1])
    
    # Strategy 8: Corner and edge initialization (1 win)
    def _initialize_archive_v8(self):
        n = self.archive_size
        if n <= 0:
            self.elite_archive = np.zeros((0, self.dim))
            return
        points = []
        center = (self.bounds[0] + self.bounds[1]) / 2.0
        points.append(np.full(self.dim, center))
        for d in range(self.dim):
            point = np.full(self.dim, self.bounds[0])
            point[d] = center
            points.append(point)
            point = np.full(self.dim, self.bounds[1])
            point[d] = center
            points.append(point)
        for bits in range(1 << self.dim):
            if len(points) >= n:
                break
            corner = np.array([self.bounds[1] if (bits >> d) & 1 else self.bounds[0] for d in range(self.dim)])
            points.append(corner)
        points = np.array(points)
        if len(points) > n:
            points = points[:n]
        self.elite_archive = np.clip(points, self.bounds[0], self.bounds[1])
        
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
        """Clip all candidates to the search bounds."""
        return np.clip(candidates, self.bounds[0], self.bounds[1])
        
    def _handle_evaluation_errors(self):
        """Replace NaN/Inf fitness values with worst possible fitness."""
        valid = np.isfinite(self.current_fitness)
        if not np.all(valid):
            finite_fitness = self.current_fitness[np.isfinite(self.current_fitness)]
            worst = np.nanmax(finite_fitness) if len(finite_fitness) > 0 else 1e10
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
        
        rank_one = np.outer(self.mean_diff, self.mean_diff)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i, ind in enumerate(selected_pop):
            diff = (ind - self.mean).reshape(-1, 1)
            rank_mu += np.dot(diff, diff.T)
        rank_mu /= len(selected_pop)
        
        self.covariance = (
            (1 - self.c1 - self.cmu) * self.covariance +
            self.c1 * rank_one +
            self.cmu * rank_mu
        )
        
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
        history = np.array(self.best_fitness_history[-5:], dtype=float)
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
