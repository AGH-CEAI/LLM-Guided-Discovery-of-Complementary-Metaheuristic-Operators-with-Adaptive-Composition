import numpy as np


class CovarianceGuidedEvolutionStrategy:
    """
    Covariance-Guided Evolution Strategy (CG-ES) with Adaptive Archive Initialization
    
    An adaptive population-based optimizer that learns the best _initialize_archive
    strategy during optimization using Thompson Sampling with sliding window credit
    assignment.
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
        
        # Adaptive initialization tracking
        self._selected_strategy = 0
        self._sliding_window_size = 15
        self._strategy_rewards = [[] for _ in range(8)]
        self._strategy_counts = np.zeros(8, dtype=float)
        self._strategy_cumulative_reward = np.zeros(8, dtype=float)
        self._exploration_bonus = 2.0
        
        # Map strategies to their implementations
        self._strategy_methods = [
            self._initialize_archive_variant1,
            self._initialize_archive_variant2,
            self._initialize_archive_variant3,
            self._initialize_archive_variant4,
            self._initialize_archive_variant5,
            self._initialize_archive_variant6,
            self._initialize_archive_variant7,
            self._initialize_archive_variant8,
        ]
        
    def _initialize_archive_variant1(self):
        """LHS-based initialization (variant_idea_1.py)."""
        n = self.archive_size
        intervals = np.linspace(0, 1, n + 1)
        points = np.zeros((n, self.dim))
        for d in range(self.dim):
            perm = self._rng.permutation(n)
            randoms = self._rng.random(n)
            points[:, d] = intervals[perm] + randoms[perm] * (intervals[1] - intervals[0])
        points = self.bounds[0] + points * (self.bounds[1] - self.bounds[0])
        points = np.clip(points, self.bounds[0], self.bounds[1])
        self.elite_archive = points
        
    def _initialize_archive_variant2(self):
        """Crossover-based initialization (variant_idea_2.py)."""
        n = self.archive_size
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
        offspring = np.clip(offspring, self.bounds[0], self.bounds[1])
        self.elite_archive = offspring
        
    def _initialize_archive_variant3(self):
        """SBX-inspired initialization (variant_idea_3.py)."""
        n = self.archive_size
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
        
    def _initialize_archive_variant4(self):
        """Edge-centered initialization (variant_idea_4.py)."""
        n = self.archive_size
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
        selected = np.clip(selected, self.bounds[0], self.bounds[1])
        self.elite_archive = selected
        
    def _initialize_archive_variant5(self):
        """Eigenvector-based initialization (variant_idea_5.py)."""
        n = self.archive_size
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
        points = np.clip(points, self.bounds[0], self.bounds[1])
        self.elite_archive = points
        
    def _initialize_archive_variant6(self):
        """Halton sequence-based initialization (variant_idea_6.py)."""
        n = self.archive_size
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
        points = np.clip(points, self.bounds[0], self.bounds[1])
        self.elite_archive = points
        
    def _initialize_archive_variant7(self):
        """Orthogonal direction-based initialization (variant_idea_7.py)."""
        n = self.archive_size
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
        points = np.clip(points, self.bounds[0], self.bounds[1])
        self.elite_archive = points
        
    def _initialize_archive_variant8(self):
        """Corner and edge-based initialization (variant_idea_8.py)."""
        n = self.archive_size
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
        points = np.clip(points, self.bounds[0], self.bounds[1])
        self.elite_archive = points
        
    def _select_init_strategy_ucb(self):
        """Select initialization strategy using Upper Confidence Bound (UCB)."""
        total_counts = float(np.sum(self._strategy_counts))
        
        if total_counts < 1e-10:
            return int(self._rng.integers(0, 8))
        
        ucb_values = np.zeros(8)
        for i in range(8):
            if self._strategy_counts[i] > 0:
                avg_reward = float(self._strategy_cumulative_reward[i]) / float(self._strategy_counts[i])
            else:
                avg_reward = 0.0
            exploration = self._exploration_bonus * np.sqrt(np.log(total_counts + 1) / (self._strategy_counts[i] + 1e-10))
            ucb_values[i] = avg_reward + exploration
        
        max_ucb = float(np.max(ucb_values))
        ties = np.where(np.abs(ucb_values - max_ucb) < 1e-12)[0]
        return int(self._rng.choice(ties))
        
    def _update_strategy_reward(self, strategy_idx, fitness):
        """Update reward for a strategy using sliding window credit assignment."""
        reward = float(-fitness)
        
        self._strategy_rewards[strategy_idx].append(reward)
        if len(self._strategy_rewards[strategy_idx]) > self._sliding_window_size:
            self._strategy_rewards[strategy_idx].pop(0)
        
        self._strategy_counts[strategy_idx] += 1.0
        self._strategy_cumulative_reward[strategy_idx] += reward
        
        if self._strategy_counts[strategy_idx] > 1e6:
            self._strategy_counts[strategy_idx] *= 0.999
            self._strategy_cumulative_reward[strategy_idx] *= 0.999
            
    def _initialize_archive(self):
        """Initialize archive using adaptive strategy selection."""
        n = self.archive_size
        if n <= 0:
            self.elite_archive = np.zeros((0, self.dim))
            return
        
        strategy_idx = self._select_init_strategy_ucb()
        self._strategy_methods[strategy_idx]()
        self._selected_strategy = strategy_idx
        
    def __call__(self, func, stopping_condition):
        """Run the optimizer and return (f_opt, x_opt)."""
        self._initialize_optimizer()
        
        while not stopping_condition():
            self._sample_and_evaluate(func)
            self._update_distribution_parameters()
            self._adapt_step_size()
            self._update_elite_archive()
            self._manage_stagnation()
            
            current_best = float(np.min(self.current_fitness))
            self._update_strategy_reward(self._selected_strategy, current_best)
            
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
            if len(finite_fitness) > 0:
                worst = float(np.nanmax(finite_fitness))
            else:
                worst = 1e10
            if not np.isfinite(worst):
                worst = 1e10
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
        normalized = fitness - float(fitness.min()) + 1e-6
        weights = 1.0 / normalized
        weights /= weights.sum()
        return weights
        
    def _update_covariance(self, selected_pop):
        """Simplified rank-based covariance update."""
        self._compute_adaptation_learning_rates()
        
        rank_one = np.outer(self.mean_diff, self.mean_diff)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for ind in selected_pop:
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
        min_eigenvalue = float(np.min(eigenvalues))
        
        if min_eigenvalue < 1e-10:
            self.covariance += (1e-9 - min_eigenvalue) * np.eye(self.dim)
            
    def _adapt_step_size(self):
        """Adapt step size based on successful mutations."""
        current_best = float(np.min(self.current_fitness))
        self._update_stagnation_tracking(current_best)
        
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
        best_fitness = float(self.current_fitness[best_idx])
        
        return best_fitness, best_position.copy()
