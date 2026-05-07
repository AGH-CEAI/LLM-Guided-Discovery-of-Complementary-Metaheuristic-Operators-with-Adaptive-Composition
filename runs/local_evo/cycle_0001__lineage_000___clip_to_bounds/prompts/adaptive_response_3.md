```python
import numpy as np


class CovarianceGuidedEvolutionStrategy:
    """
    Covariance-Guided Evolution Strategy (CG-ES) with Adaptive Clipping
    
    A population-based optimizer combining CMA-ES-style covariance adaptation
    with a simplified rank-based update and adaptive population management.
    Uses a small elite archive and restart mechanism for robustness.
    
    Adaptive clipping uses Thompson Sampling to dynamically select the best
    boundary handling strategy based on observed performance.
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
        
        # Adaptive clipping strategy selection (Thompson Sampling)
        self._init_adaptive_clipping()
        
    def _init_adaptive_clipping(self):
        """Initialize adaptive clipping mechanism with Thompson Sampling."""
        self.num_clip_strategies = 9
        # Beta distribution parameters for Thompson Sampling
        self.clip_alpha = np.ones(self.num_clip_strategies, dtype=np.float64)
        self.clip_beta = np.ones(self.num_clip_strategies, dtype=np.float64)
        # Track rewards for each strategy
        self.clip_rewards = [[] for _ in range(self.num_clip_strategies)]
        self.max_reward_history = 50
        # Track which strategy is currently selected
        self.current_clip_strategy = 0
        self.clip_strategy_attempts = np.zeros(self.num_clip_strategies, dtype=int)
        # Previous best fitness for reward calculation
        self._prev_best_fitness = None
        
    def __call__(self, func, stopping_condition):
        """Run the optimizer and return (f_opt, x_opt)."""
        self._initialize_optimizer()
        
        while not stopping_condition():
            # Select clipping strategy before sampling (Thompson Sampling)
            self._select_clip_strategy()
            
            self._sample_and_evaluate(func)
            
            # Update adaptive mechanism after evaluation
            self._update_clip_strategy_reward()
            
            self._update_distribution_parameters()
            self._adapt_step_size()
            self._update_elite_archive()
            self._manage_stagnation()
            
            if self._should_restart():
                self._restart_optimizer()
                
        return self._get_best_result()
    
    def _select_clip_strategy(self):
        """Select clipping strategy using Thompson Sampling."""
        if np.any(self.clip_strategy_attempts > 0):
            samples = np.array([
                self._sample_beta(self.clip_alpha[i], self.clip_beta[i])
                for i in range(self.num_clip_strategies)
            ])
            # Ensure we have valid samples
            valid_samples = np.where(np.isfinite(samples), samples, 0.0)
            self.current_clip_strategy = int(np.argmax(valid_samples))
        else:
            # Uniform random selection for first few iterations
            self.current_clip_strategy = self._rng.integers(0, self.num_clip_strategies)
        
        self.clip_strategy_attempts[self.current_clip_strategy] += 1
    
    def _sample_beta(self, alpha, beta):
        """Sample from Beta distribution using gamma sampling."""
        alpha = max(float(alpha), 1e-10)
        beta = max(float(beta), 1e-10)
        try:
            x = self._rng.gamma(alpha, 1.0)
            y = self._rng.gamma(beta, 1.0)
            if x + y > 0:
                return float(x) / float(x + y)
        except (ValueError, FloatingPointError):
            pass
        return 0.5
    
    def _update_clip_strategy_reward(self):
        """Update the reward for the selected clipping strategy."""
        if self._prev_best_fitness is None:
            self._prev_best_fitness = float(np.min(self.current_fitness))
            return
        
        current_best = float(np.min(self.current_fitness))
        improvement = self._prev_best_fitness - current_best
        
        # Store reward as float scalar
        reward = float(improvement)
        
        # Clamp extreme rewards to prevent numerical issues
        reward = float(np.clip(reward, -1e10, 1e10))
        
        if np.isfinite(reward):
            strategy = self.current_clip_strategy
            self.clip_rewards[strategy].append(reward)
            
            # Maintain sliding window
            if len(self.clip_rewards[strategy]) > self.max_reward_history:
                self.clip_rewards[strategy].pop(0)
            
            # Update Beta distribution parameters
            pos_sum = sum(max(0, r) for r in self.clip_rewards[strategy])
            neg_sum = sum(max(0, -r) for r in self.clip_rewards[strategy])
            
            self.clip_alpha[strategy] = 1.0 + pos_sum
            self.clip_beta[strategy] = 1.0 + neg_sum
        
        self._prev_best_fitness = current_best
    
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
        self._prev_best_fitness = None
        
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
        """Apply the selected adaptive clipping strategy."""
        strategy = self.current_clip_strategy
        
        if strategy == 0:
            return self._clip_strategy_original(candidates)
        elif strategy == 1:
            return self._clip_strategy_v1(candidates)
        elif strategy == 2:
            return self._clip_strategy_v2(candidates)
        elif strategy == 3:
            return self._clip_strategy_v3(candidates)
        elif strategy == 4:
            return self._clip_strategy_v4(candidates)
        elif strategy == 5:
            return self._clip_strategy_v5(candidates)
        elif strategy == 6:
            return self._clip_strategy_v6(candidates)
        elif strategy == 7:
            return self._clip_strategy_v7(candidates)
        elif strategy == 8:
            return self._clip_strategy_v8(candidates)
        else:
            return self._clip_strategy_original(candidates)
    
    def _clip_strategy_original(self, candidates):
        """Original simple clipping."""
        return np.clip(candidates, self.bounds[0], self.bounds[1])
    
    def _clip_strategy_v1(self, candidates):
        """Exponential decay clipping (7 wins - variant_idea_1)."""
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
    
    def _clip_strategy_v2(self, candidates):
        """Random weight reflection (3 wins - variant_idea_2)."""
        low, high = self.bounds
        above = candidates > high
        below = candidates < low
        w = self._rng.uniform(0.5, 1.5, size=candidates.shape)
        repaired = np.where(above, high - w * (candidates - high), candidates)
        repaired = np.where(below, low + w * (low - candidates), repaired)
        return np.clip(repaired, low, high)
    
    def _clip_strategy_v3(self, candidates):
        """Periodic reflection (1 win - variant_idea_3)."""
        low, high = self.bounds
        segment = high - low
        if segment == 0:
            return np.full_like(candidates, low)
        period = 2 * segment
        shifted = candidates - low
        t = shifted % period
        reflected = np.where(t <= segment, low + t, high - (t - segment))
        return np.clip(reflected, low, high)
    
    def _clip_strategy_v4(self, candidates):
        """Eigendecomposition-based clipping (1 win - variant_idea_4)."""
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
    
    def _clip_strategy_v5(self, candidates):
        """Iterative shrinking (1 win - variant_idea_5)."""
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
    
    def _clip_strategy_v6(self, candidates):
        """Gaussian noise clipping (5 wins - variant_idea_6)."""
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
    
    def _clip_strategy_v7(self, candidates):
        """Tanh compression (1 win - variant_idea_7)."""
        low, high = self.bounds
        midpoint = (high + low) / 2.0
        rng = high - low
        if rng == 0:
            return np.full_like(candidates, low)
        z = (candidates - midpoint) / (rng / 2.0)
        compressed = np.tanh(z)
        repaired = low + (compressed + 1.0) * 0.5 * rng
        return np.clip(repaired, low, high)
    
    def _clip_strategy_v8(self, candidates):
        """Hyperbolic shrink (4 wins - variant_idea_8)."""
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
        self._prev_best_fitness = None
        
    def _get_best_result(self):
        """Extract best solution found from current population."""
        best_idx = np.argmin(self.current_fitness)
        best_position = self.current_population[best_idx]
        best_fitness = self.current_fitness[best_idx]
        
        return float(best_fitness), best_position.copy()
```