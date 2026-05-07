import numpy as np


class CovarianceRingAdaptationOptimizer:
    """
    A population-based optimizer combining CMA-ES-inspired covariance adaptation
    with a ring-based population topology. Each individual samples from an
    adaptive multivariate normal distribution centered on its ring neighborhood mean.
    """
    
    def __init__(self, dim, pop_size=None, F=0.5, CR=0.9, target_success_rate=0.2):
        self.dim = dim
        self.pop_size = pop_size if pop_size else max(4 * dim, 40)
        self.F = F
        self.CR = CR
        self.target_success_rate = target_success_rate
        
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        self.population = None
        self.fitness = None
        self.best_idx = None
        self.best_fitness = None
        self.best_position = None
        
        self.covariance = None
        self.step_size = None
        self.mean = None
        
        self.success_count = 0
        self.failure_count = 0
        self.adaptation_counter = 0
        self.past_best_fitness = None
        self.stagnation_counter = 0
        
        self._max_generations = 10000
    
    def _initialize_population(self):
        """Initialize population uniformly in bounds."""
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound,
            size=(self.pop_size, self.dim)
        )
        self._clip_to_bounds_batch()
    
    def _initialize_adaptation_params(self):
        """Initialize CMA-ES-inspired covariance and step size."""
        self.covariance = np.eye(self.dim)
        self.step_size = 0.3 * (self.upper_bound - self.lower_bound)
        self.mean = np.mean(self.population, axis=0)
    
    def _clip_to_bounds_batch(self):
        """Clip all individuals to search bounds."""
        np.clip(self.population, self.lower_bound, self.upper_bound, out=self.population)
    
    def _compute_ring_neighborhood_means(self):
        """
        Compute ring topology neighborhood means.
        Each individual interacts with itself and its two neighbors.
        Returns array of shape (NP, dim).
        """
        left = np.roll(self.population, 1, axis=0)
        right = np.roll(self.population, -1, axis=0)
        center = self.population
        neighborhood_means = (left + center + right) / 3.0
        return neighborhood_means
    
    def _update_mean_and_covariance(self):
        """Update distribution mean and covariance based on successful mutations."""
        self.mean = np.mean(self.population, axis=0)
        
        centered = self.population - self.mean
        cov_estimate = np.dot(centered.T, centered) / self.pop_size
        
        eigenvalue_eps = 1e-6 * np.eye(self.dim)
        cov_estimate = cov_estimate + eigenvalue_eps
        
        mixing_factor = 0.01
        self.covariance = (1 - mixing_factor) * self.covariance + mixing_factor * cov_estimate
    
    def _sample_from_distribution(self, mean, n_samples):
        """Sample n_samples individuals from multivariate normal."""
        try:
            L = np.linalg.cholesky(self.covariance)
            samples = mean + self.step_size * np.dot(
                np.random.randn(n_samples, self.dim), L.T
            )
        except np.linalg.LinAlgError:
            samples = mean + self.step_size * np.random.randn(n_samples, self.dim)
        return samples
    
    def _generate_covariance_mutations(self, neighborhood_means):
        """
        Generate mutations using CMA-ES-style sampling around neighborhood means.
        Returns array of shape (NP, dim).
        """
        mutations = self._sample_from_distribution(
            neighborhood_means[0], self.pop_size
        )
        return mutations
    
    def _generate_ring_differential_mutations(self, neighborhood_means):
        """
        Generate DE-style mutations using ring topology.
        For each target, pick two different neighbors for differential variation.
        """
        left = np.roll(self.population, 1, axis=0)
        right = np.roll(self.population, -1, axis=0)
        
        r1_indices = np.random.randint(0, self.pop_size, size=self.pop_size)
        r2_indices = (r1_indices + np.random.randint(1, self.pop_size, size=self.pop_size)) % self.pop_size
        
        r1 = self.population[r1_indices]
        r2 = self.population[r2_indices]
        
        base_indices = np.random.randint(0, 3, size=self.pop_size)
        base = np.where(base_indices == 0, self.population,
               np.where(base_indices == 1, left, right))
        
        F_scaled = self.F * (1.0 + 0.1 * np.random.randn(self.pop_size))
        F_scaled = np.clip(F_scaled, 0.1, 2.0)
        
        mutations = base + F_scaled[:, np.newaxis] * (r1 - r2)
        return mutations
    
    def _apply_crossover_batch(self, targets, mutations):
        """
        Apply binomial crossover between targets and mutations.
        Returns array of shape (NP, dim).
        """
        coin_flips = np.random.rand(self.pop_size, self.dim) < self.CR
        
        j_rand = np.random.randint(0, self.dim, size=self.pop_size)
        mask = np.zeros_like(coin_flips)
        mask[np.arange(self.pop_size), j_rand] = True
        
        trials = np.where(coin_flips | mask, mutations, targets)
        return trials
    
    def _clip_to_bounds_batch(self):
        """Clip trial candidates to search bounds."""
        np.clip(self.population, self.lower_bound, self.upper_bound, out=self.population)
    
    def _eval_wrapper(self, candidates):
        """
        Evaluate candidates with budget safety check.
        Returns fitness array and whether budget is exhausted.
        """
        fitness = func(candidates)
        
        if len(fitness) < len(candidates):
            actual_len = len(fitness)
            candidates = candidates[:actual_len]
            fitness = fitness[:actual_len]
            budget_exhausted = True
        else:
            budget_exhausted = False
        
        return fitness, budget_exhausted
    
    def _select_survivors_batch(self, targets, target_fitness, trials, trial_fitness):
        """
        Greedy selection between targets and trials.
        Returns updated population and fitness.
        """
        improvements = trial_fitness < target_fitness
        
        new_population = targets.copy()
        new_population[improvements] = trials[improvements]
        
        new_fitness = target_fitness.copy()
        new_fitness[improvements] = trial_fitness[improvements]
        
        return new_population, new_fitness, improvements
    
    def _adapt_step_size(self, n_success, n_total):
        """Adapt step size based on cumulative step-size adaptation (CMA-ES)."""
        if n_total == 0:
            return
        
        success_rate = n_success / n_total
        
        if success_rate > self.target_success_rate:
            self.step_size *= 1.2
        elif success_rate < self.target_success_rate:
            self.step_size /= 1.2
        
        self.step_size = np.clip(self.step_size, 1e-10, 1e4)
    
    def _update_velocity_batch(self, old_pop, new_pop):
        """
        Track implicit velocity for adaptive purposes.
        Returns velocity array of shape (NP, dim).
        """
        velocity = new_pop - old_pop
        return velocity
    
    def _check_stagnation(self):
        """Check if algorithm has stagnated."""
        if self.past_best_fitness is None:
            self.past_best_fitness = self.best_fitness
            return False
        
        improvement_threshold = 1e-12 * max(1.0, abs(self.best_fitness))
        
        if self.best_fitness < self.past_best_fitness - improvement_threshold:
            self.past_best_fitness = self.best_fitness
            self.stagnation_counter = 0
            return False
        else:
            self.stagnation_counter += 1
            return self.stagnation_counter > 50
    
    def _restart_if_stagnant(self):
        """Reinitialize population while preserving best solution."""
        if not self._check_stagnation():
            return
        
        best = self.best_position.copy() if self.best_position is not None else np.zeros(self.dim)
        
        self._initialize_population()
        self._initialize_adaptation_params()
        
        self.population[0] = best
        self._clip_to_bounds_batch()
        
        self.success_count = 0
        self.failure_count = 0
        self.adaptation_counter = 0
        self.past_best_fitness = self.best_fitness
        self.stagnation_counter = 0
    
    def _compute_diversity(self):
        """
        Compute population diversity as mean distance to centroid.
        """
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _update_tracking(self, fitness):
        """Update best solution tracking."""
        current_best_idx = np.argmin(fitness)
        current_best_fitness = fitness[current_best_idx]
        
        if self.best_fitness is None or current_best_fitness < self.best_fitness:
            self.best_idx = current_best_idx
            self.best_fitness = current_best_fitness
            self.best_position = self.population[current_best_idx].copy()
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer.
        
        Args:
            func: Fitness function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        self._initialize_population()
        self._initialize_adaptation_params()
        
        fitness, budget_exhausted = self._eval_wrapper(self.population)
        if budget_exhausted or stopping_condition():
            self._update_tracking(fitness)
            return self.best_fitness, self.best_position
        
        self._update_tracking(fitness)
        
        neighborhood_means = self._compute_ring_neighborhood_means()
        
        for gen in range(self._max_generations):
            if stopping_condition():
                break
            
            cov_mutations = self._generate_covariance_mutations(neighborhood_means)
            de_mutations = self._generate_ring_differential_mutations(neighborhood_means)
            
            mutation_choice = np.random.rand(self.pop_size) < 0.5
            mutations = np.where(mutation_choice[:, np.newaxis], cov_mutations, de_mutations)
            
            trials = self._apply_crossover_batch(self.population, mutations)
            np.clip(trials, self.lower_bound, self.upper_bound, out=trials)
            
            trial_fitness, budget_exhausted = self._eval_wrapper(trials)
            
            if stopping_condition() or budget_exhausted:
                break
            
            old_pop = self.population.copy()
            self.population, fitness, improvements = self._select_survivors_batch(
                self.population, fitness, trials, trial_fitness
            )
            
            velocity = self._update_velocity_batch(old_pop, self.population)
            
            self.success_count += np.sum(improvements)
            self.failure_count += self.pop_size - np.sum(improvements)
            
            self.adaptation_counter += 1
            if self.adaptation_counter >= self.pop_size:
                self._adapt_step_size(self.success_count, self.success_count + self.failure_count)
                self.success_count = 0
                self.failure_count = 0
                self.adaptation_counter = 0
            
            self._update_tracking(fitness)
            
            self._update_mean_and_covariance()
            neighborhood_means = self._compute_ring_neighborhood_means()
            
            self._restart_if_stagnant()
            
            if budget_exhausted:
                break
        
        return self.best_fitness, self.best_position
