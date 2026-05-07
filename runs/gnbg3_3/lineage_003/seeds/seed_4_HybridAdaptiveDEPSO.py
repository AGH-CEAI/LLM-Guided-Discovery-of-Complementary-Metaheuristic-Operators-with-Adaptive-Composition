import numpy as np

class HybridAdaptiveDEPSO:
    """
    Hybrid Differential Evolution with PSO-inspired velocity guidance.
    Combines DE mutation/crossover with PSO velocity components for
    enhanced exploration-exploitation balance.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(max(5 * dim, 50), 300)
        
    def __call__(self, func, stopping_condition):
        self.func = func
        self.gen = 0
        self.last_improvement_gen = 0
        self.stagnation_threshold = 30
        self.local_search_step = 0.1 * (200.0 / self.dim)
        self.velocity_blend = 0.1
        self.inertia = 0.7
        self.success_F = []
        self.success_CR = []
        self.F_archive = np.full(self.NP, 0.5)
        self.CR_archive = np.full(self.NP, 0.9)
        
        self._initialize_population()
        self.pbest_positions = self.population.copy()
        self.pbest_fitness = self.fitness.copy()
        self.gbest_position = self.best_position.copy()
        
        while not stopping_condition():
            self.gen += 1
            
            mutant_batch = self._mutate_batch()
            velocity_batch = self._velocity_update_batch()
            mutated_velocity_batch = self._apply_velocity_to_mutation(mutant_batch, velocity_batch)
            trial_batch = self._crossover_batch(self.population, mutated_velocity_batch)
            trial_batch = self._clip_to_bounds_batch(trial_batch)
            
            trial_fitness = func(trial_batch)
            if len(trial_fitness) < len(trial_batch):
                break
            
            improved_mask = trial_fitness < self.fitness
            self.trial_population = trial_batch
            self._select_survivors_batch(trial_batch, trial_fitness)
            self._update_personal_best_batch()
            self._update_global_best()
            self._adapt_parameters(improved_mask, trial_fitness)
            self._adapt_inertia_weight()
            self._adapt_velocity_blend()
            
            if self.gen % 5 == 0:
                self._apply_local_search_batch()
            
            if stopping_condition():
                break
            
            self._restart_if_stagnant()
        
        return self.best_fitness, self.best_position
    
    def _initialize_population(self):
        """Initialize population with random values within bounds."""
        self.population = np.random.uniform(-100, 100, (self.NP, self.dim))
        self.fitness = self.func(self.population)
        if len(self.fitness) < self.NP:
            self.fitness = np.full(self.NP, np.inf)
        best_idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[best_idx]
        self.best_position = self.population[best_idx].copy()
    
    def _select_three_random_indices(self):
        """Select three distinct random indices for mutation."""
        indices = np.arange(self.NP)
        np.random.shuffle(indices)
        return indices[0], indices[1], indices[2]
    
    def _get_best_vector(self):
        """Get best individual tiled for batch operations."""
        return np.tile(self.best_position, (self.NP, 1))
    
    def _get_gbest_vector(self):
        """Get global best position tiled for batch operations."""
        return np.tile(self.gbest_position, (self.NP, 1))
    
    def _mutate_batch(self):
        """Generate mutant vectors using multiple DE strategies."""
        r1, r2, r3 = self._select_three_random_indices()
        best_vec = self._get_best_vector()
        
        mutant_rand = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
        mutant_best = best_vec + self.F * (self.population[r1] - self.population[r2])
        mutant_ttb = self.population + self.F * (best_vec - self.population) + \
                     self.F * (self.population[r1] - self.population[r2])
        
        weights = np.random.dirichlet(np.ones(3))
        mutant_batch = weights[0] * mutant_rand + weights[1] * mutant_best + weights[2] * mutant_ttb
        return mutant_batch
    
    def _velocity_update_batch(self):
        """Compute PSO-inspired velocity for cognitive and social guidance."""
        cognitive_component = self.inertia * np.random.random((self.NP, self.dim)) * \
                              (self.pbest_positions - self.population)
        gbest_vec = self._get_gbest_vector()
        social_component = self.inertia * np.random.random((self.NP, self.dim)) * \
                           (gbest_vec - self.population)
        return cognitive_component + social_component
    
    def _apply_velocity_to_mutation(self, mutant_batch, velocity_batch):
        """Blend DE mutation with PSO velocity contribution."""
        return mutant_batch + self.velocity_blend * velocity_batch
    
    def _crossover_batch(self, target_batch, mutant_batch):
        """Perform binomial crossover between targets and mutants."""
        cr_mask = np.random.random((self.NP, self.dim)) < self.CR
        j_rand = np.random.randint(self.dim, size=self.NP)
        forced_cross = np.zeros((self.NP, self.dim), dtype=bool)
        forced_cross[np.arange(self.NP), j_rand] = True
        cross_mask = cr_mask | forced_cross
        return np.where(cross_mask, mutant_batch, target_batch)
    
    def _clip_to_bounds_batch(self, batch):
        """Clip all candidates to the search bounds."""
        return np.clip(batch, -100, 100)
    
    def _select_survivors_batch(self, trial_batch, trial_fitness):
        """Greedy selection between target and trial vectors."""
        improved = trial_fitness < self.fitness
        self.population = np.where(improved[:, np.newaxis], trial_batch, self.population)
        self.fitness = np.where(improved, trial_fitness, self.fitness)
        if improved.any():
            self.last_improvement_gen = self.gen
    
    def _update_personal_best_batch(self):
        """Update personal best positions based on current fitness."""
        improved = self.fitness < self.pbest_fitness
        self.pbest_fitness = np.where(improved, self.fitness, self.pbest_fitness)
        self.pbest_positions = np.where(improved[:, np.newaxis], self.population, self.pbest_positions)
    
    def _update_global_best(self):
        """Update global best position if current best is better."""
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_position = self.population[best_idx].copy()
            self.gbest_position = self.best_position.copy()
    
    def _adapt_parameters(self, improved_mask, trial_fitness):
        """Adapt F and CR based on historical success rates."""
        if improved_mask.any():
            successful_F = self.F_archive[improved_mask]
            successful_CR = self.CR_archive[improved_mask]
            self.success_F.extend(successful_F.tolist())
            self.success_CR.extend(successful_CR.tolist())
            if len(self.success_F) > 50:
                self.success_F = self.success_F[-50:]
                self.success_CR = self.success_CR[-50:]
    
    def _adapt_inertia_weight(self):
        """Adapt inertia weight based on recent F parameter success."""
        if len(self.success_F) >= 10:
            recent_F_mean = np.mean(self.success_F[-10:])
            self.inertia = 0.4 + 0.3 * (1 - recent_F_mean)
            self.inertia = np.clip(self.inertia, 0.2, 0.9)
    
    def _compute_diversity(self):
        """Compute population diversity as normalized average distance to centroid."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances) / (self.dim * 50.0)
    
    def _adapt_velocity_blend(self):
        """Adapt velocity blend factor to maintain target diversity."""
        diversity = self._compute_diversity()
        target_diversity = 0.15
        error = target_diversity - diversity
        self.velocity_blend = self.velocity_blend + 0.05 * error
        self.velocity_blend = np.clip(self.velocity_blend, 0.0, 0.3)
    
    def _apply_local_search_batch(self):
        """Apply coordinate-wise local search to top performers."""
        n_ls = max(2, self.NP // 5)
        top_indices = np.argsort(self.fitness)[:n_ls]
        
        for idx in top_indices:
            x = self.population[idx].copy()
            step = self.local_search_step
            directions = np.vstack([np.eye(self.dim) * step, -np.eye(self.dim) * step])
            neighbors = x + directions
            neighbors = np.clip(neighbors, -100, 100)
            neighbor_fitness = self.func(neighbors)
            
            if len(neighbor_fitness) < len(neighbors):
                break
            
            best_neighbor_idx = np.argmin(neighbor_fitness)
            if neighbor_fitness[best_neighbor_idx] < self.fitness[idx]:
                self.population[idx] = neighbors[best_neighbor_idx]
                self.fitness[idx] = neighbor_fitness[best_neighbor_idx]
                self.local_search_step *= 1.2
            else:
                self.local_search_step *= 0.5
    
    def _restart_if_stagnant(self):
        """Replace worst individuals if no improvement for extended period."""
        stagnation = self.gen - self.last_improvement_gen
        if stagnation > self.stagnation_threshold:
            n_replace = max(5, self.NP // 4)
            replace_indices = np.argsort(self.fitness)[-n_replace:]
            random_replacements = np.random.uniform(-100, 100, (n_replace, self.dim))
            self.population[replace_indices] = random_replacements
            self.pbest_positions[replace_indices] = random_replacements.copy()
            self.last_improvement_gen = self.gen
            self.stagnation_threshold = min(100, self.stagnation_threshold + 20)
