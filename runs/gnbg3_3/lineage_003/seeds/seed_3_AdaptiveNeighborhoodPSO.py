import numpy as np


class AdaptiveNeighborhoodPSO:
    """
    Particle Swarm Optimizer with adaptive neighborhood topology,
    multiple velocity strategies, and diversity-driven adaptation.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.pop_size = min(kwargs.get('pop_size', 5 * dim), 300)
        self.lower = -100.0
        self.upper = 100.0
        self.bound_range = self.upper - self.lower
        
        self.inertia = kwargs.get('inertia', 0.729)
        self.cognitive = kwargs.get('cognitive', 1.496)
        self.social = kwargs.get('social', 1.496)
        self.max_velocity = 0.15 * self.bound_range
        
        self.neighborhood_k = kwargs.get('neighborhood_k', 4)
        self.diversity_threshold = kwargs.get('diversity_threshold', 1e-4)
        self.stagnation_limit = kwargs.get('stagnation_limit', 30)
        
        self._population = None
        self._velocity = None
        self._fitness = None
        self._personal_best = None
        self._personal_best_fitness = None
        self._global_best = None
        self._global_best_fitness = None
        self._neighborhood_best = None
        self._neighborhood_best_fitness = None
        self._topology = None
        self._fitness_history = []
        self._stagnation_counter = 0
        self._generation = 0
        
    def __call__(self, func, stopping_condition):
        self._initialize_population()
        self._fitness = func(self._population)
        self._handle_budget_exhaustion(self._fitness, self._population)
        self._update_personal_best_batch()
        self._update_global_best()
        self._build_topology_batch()
        self._compute_neighborhood_best_batch()
        
        while not stopping_condition():
            self._generation += 1
            self._velocity = self._compute_velocity_batch()
            self._update_position_batch()
            self._clip_to_bounds_batch()
            self._fitness = func(self._population)
            if self._handle_budget_exhaustion(self._fitness, self._population):
                break
            self._update_personal_best_batch()
            self._update_global_best()
            self._adapt_inertia_weight()
            self._adapt_social_coefficient()
            self._maybe_evolve_topology()
            self._compute_neighborhood_best_batch()
            self._apply_diversity_control()
            self._fitness_history.append(self._global_best_fitness)
            if self._check_stagnation():
                self._restart_diversified()
            if stopping_condition():
                break
        return self._global_best_fitness, self._global_best.copy()
    
    def _initialize_population(self):
        self._population = np.random.uniform(
            self.lower, self.upper, (self.pop_size, self.dim)
        )
        self._velocity = np.random.uniform(
            -self.max_velocity, self.max_velocity, (self.pop_size, self.dim)
        )
        self._personal_best = self._population.copy()
        self._personal_best_fitness = np.full(self.pop_size, np.inf)
        self._global_best = np.zeros(self.dim)
        self._global_best_fitness = np.inf
        self._generation = 0
        self._stagnation_counter = 0
        self._fitness_history = []
    
    def _handle_budget_exhaustion(self, fitness, population):
        if len(fitness) < len(population):
            actual_len = len(fitness)
            self._population = population[:actual_len]
            self._velocity = self._velocity[:actual_len]
            self._personal_best = self._personal_best[:actual_len]
            self._personal_best_fitness = self._personal_best_fitness[:actual_len]
            if self._neighborhood_best is not None:
                self._neighborhood_best = self._neighborhood_best[:actual_len]
                self._neighborhood_best_fitness = self._neighborhood_best_fitness[:actual_len]
            self.pop_size = actual_len
            return True
        return False
    
    def _clip_to_bounds_batch(self):
        np.clip(self._population, self.lower, self.upper, out=self._population)
    
    def _compute_velocity_batch(self):
        r1 = np.random.rand(self.pop_size, self.dim)
        r2 = np.random.rand(self.pop_size, self.dim)
        cognitive_component = self.cognitive * r1 * (
            self._personal_best - self._population
        )
        social_component = self.social * r2 * (
            self._neighborhood_best - self._population
        )
        velocity = self.inertia * self._velocity + cognitive_component + social_component
        velocity = np.clip(velocity, -self.max_velocity, self.max_velocity)
        return velocity
    
    def _update_position_batch(self):
        self._population += self._velocity
    
    def _update_personal_best_batch(self):
        improved = self._fitness < self._personal_best_fitness
        self._personal_best_fitness[improved] = self._fitness[improved]
        self._personal_best[improved] = self._population[improved].copy()
    
    def _update_global_best(self):
        best_idx = np.argmin(self._personal_best_fitness)
        if self._personal_best_fitness[best_idx] < self._global_best_fitness:
            self._global_best = self._personal_best[best_idx].copy()
            self._global_best_fitness = self._personal_best_fitness[best_idx]
            self._stagnation_counter = 0
        else:
            self._stagnation_counter += 1
    
    def _build_topology_batch(self):
        k = min(self.neighborhood_k, self.pop_size - 1)
        self._topology = np.zeros((self.pop_size, k + 1), dtype=int)
        for i in range(self.pop_size):
            indices = [(i + j) % self.pop_size for j in range(k + 1)]
            self._topology[i] = indices
    
    def _compute_neighborhood_best_batch(self):
        self._neighborhood_best = np.empty((self.pop_size, self.dim))
        self._neighborhood_best_fitness = np.empty(self.pop_size)
        for i in range(self.pop_size):
            neighbor_indices = self._topology[i]
            neighbor_fitness = self._personal_best_fitness[neighbor_indices]
            best_neighbor = neighbor_indices[np.argmin(neighbor_fitness)]
            self._neighborhood_best[i] = self._personal_best[best_neighbor]
            self._neighborhood_best_fitness[i] = self._personal_best_fitness[best_neighbor]
    
    def _adapt_inertia_weight(self):
        if len(self._fitness_history) < 5:
            return
        recent_improvement = self._fitness_history[-5] - self._fitness_history[-1]
        if recent_improvement > 0:
            self.inertia *= 0.98
        else:
            self.inertia *= 1.02
        self.inertia = np.clip(self.inertia, 0.1, 0.99)
    
    def _adapt_social_coefficient(self):
        if self._stagnation_counter > 10:
            self.social = min(2.0, self.social * 1.05)
        else:
            self.social = max(1.2, self.social * 0.99)
    
    def _maybe_evolve_topology(self):
        if self._stagnation_counter > 15 and self._stagnation_counter % 5 == 0:
            k_new = min(self.neighborhood_k + 1, self.pop_size - 1)
            self.neighborhood_k = k_new
            self._build_topology_batch()
        elif self._stagnation_counter == 0 and self.neighborhood_k > 2:
            self.neighborhood_k = max(2, self.neighborhood_k - 1)
            self._build_topology_batch()
    
    def _compute_diversity(self):
        centroid = np.mean(self._population, axis=0)
        distances = np.linalg.norm(self._population - centroid, axis=1)
        return np.mean(distances) / (self.bound_range * 0.5)
    
    def _apply_diversity_control(self):
        diversity = self._compute_diversity()
        if diversity < self.diversity_threshold:
            perturbation = np.random.randn(self.pop_size, self.dim)
            perturbation = perturbation / np.linalg.norm(
                perturbation, axis=1, keepdims=True
            )
            random_factors = np.random.uniform(0.1, 0.3, (self.pop_size, 1))
            self._velocity = perturbation * self.max_velocity * random_factors
            self.inertia = min(0.9, self.inertia * 1.1)
    
    def _check_stagnation(self):
        if len(self._fitness_history) < self.stagnation_limit:
            return False
        recent_best = min(self._fitness_history[-self.stagnation_limit:])
        current_best = self._global_best_fitness
        return (current_best >= recent_best and 
                self._stagnation_counter >= self.stagnation_limit)
    
    def _restart_diversified(self):
        elite_count = max(1, self.pop_size // 5)
        elite_indices = np.argsort(self._personal_best_fitness)[:elite_count]
        elites = self._personal_best[elite_indices].copy()
        
        self._population = np.empty((self.pop_size, self.dim))
        self._velocity = np.empty((self.pop_size, self.dim))
        
        for i in range(elite_count):
            self._population[i] = elites[i] + np.random.randn(self.dim) * (
                self.bound_range * 0.05
            )
        remaining = self.pop_size - elite_count
        if remaining > 0:
            self._population[elite_count:] = np.random.uniform(
                self.lower, self.upper, (remaining, self.dim)
            )
        self._velocity = np.random.uniform(
            -self.max_velocity, self.max_velocity, (self.pop_size, self.dim)
        )
        self.inertia = 0.729
        self.social = 1.496
        self._stagnation_counter = 0
