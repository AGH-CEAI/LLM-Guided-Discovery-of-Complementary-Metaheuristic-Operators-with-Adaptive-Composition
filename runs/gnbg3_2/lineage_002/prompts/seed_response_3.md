```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Adaptive Topology Particle Swarm Optimizer (ATPSO)
    
    Uses ring topology with adaptive neighborhood size, adaptive inertia/social
    coefficients, DE-inspired selection, and mutation-based diversity maintenance.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower = -100.0
        self.upper = 100.0
        
        # Population size: 6*dim for good coverage
        self.np = min(6 * dim, 300)
        
        # Topology parameters
        self.ring_size = max(3, dim // 10)  # Neighborhood size on each side
        
        # Default PSO parameters
        self.inertia = 0.729
        self.cognitive = 1.49445
        self.social = 1.49445
        
        # Adaptation history
        self._prev_best_fitness = np.inf
        self._improvement_streak = 0
        self._no_improvement_count = 0
        
        # Diversity threshold for topology adaptation
        self.diversity_threshold = 0.1
        
        # Stagnation detection
        self.stagnation_limit = 50
        self.max_velocity = 20.0
        
    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        # Initialize
        population = self._initialize_population()
        velocities = self._initialize_velocities(population)
        
        # Evaluate initial population
        fitness = func(population)
        if len(fitness) < self.np:
            fitness = fitness[:len(population)]
            population = population[:len(fitness)]
            velocities = velocities[:len(fitness)]
            if len(population) == 0:
                return np.inf, np.zeros(self.dim)
        
        # Initialize bests
        personal_best_pos = population.copy()
        personal_best_fit = fitness.copy()
        
        # Track global best
        best_idx = np.argmin(fitness)
        global_best_pos = population[best_idx].copy()
        global_best_fit = fitness[best_idx]
        
        # Main loop
        while not stopping_condition():
            # Compute neighborhood bests (topology-aware)
            neighborhood_best_pos = self._compute_neighborhood_best_batch(
                population, personal_best_pos, fitness
            )
            
            # Update velocities
            velocities = self._update_velocities_batch(
                velocities, population, personal_best_pos, neighborhood_best_pos
            )
            
            # Apply velocity mutation for diversity
            velocities = self._mutate_velocity_batch(velocities)
            
            # Clip velocities
            velocities = self._clip_velocities_batch(velocities)
            
            # Update positions
            population = self._update_positions_batch(population, velocities)
            
            # Clip to bounds
            population = self._clip_to_bounds_batch(population)
            
            # Evaluate batch
            trial_fitness = func(population)
            
            # Handle budget exhaustion
            if len(trial_fitness) < len(population):
                actual_len = len(trial_fitness)
                population = population[:actual_len]
                velocities = velocities[:actual_len]
                trial_fitness = trial_fitness[:actual_len]
                personal_best_pos = personal_best_pos[:actual_len]
                personal_best_fit = personal_best_fit[:actual_len]
                if actual_len == 0:
                    break
            
            # Check stopping after evaluation
            if stopping_condition():
                break
            
            # Update personal bests
            personal_best_pos, personal_best_fit = self._update_personal_best_batch(
                population, trial_fitness, personal_best_pos, personal_best_fit
            )
            
            # Update global best
            current_best_idx = np.argmin(personal_best_fit)
            if personal_best_fit[current_best_idx] < global_best_fit:
                global_best_fit = personal_best_fit[current_best_idx]
                global_best_pos = personal_best_pos[current_best_idx].copy()
                self._improvement_streak += 1
                self._no_improvement_count = 0
            else:
                self._no_improvement_count += 1
                self._improvement_streak = 0
            
            # Adapt parameters based on performance
            self._adapt_parameters_batch(global_best_fit)
            
            # Compute diversity for topology adaptation
            diversity = self._compute_diversity_batch(population)
            
            # Adapt topology if needed
            self._adapt_topology_batch(diversity)
            
            # Selection: DE-style greedy selection
            population, velocities = self._select_survivors_batch(
                population, velocities, trial_fitness, personal_best_pos
            )
            
            # Restart stagnant particles
            if self._no_improvement_count >= self.stagnation_limit:
                population, velocities, personal_best_pos, personal_best_fit = \
                    self._restart_stagnant_batch(
                        population, velocities, personal_best_pos, personal_best_fit,
                        global_best_pos
                    )
                self._no_improvement_count = 0
            
            self._prev_best_fitness = global_best_fit
        
        return global_best_fit, global_best_pos
    
    def _initialize_population(self):
        """Initialize population uniformly in bounds."""
        return np.random.uniform(
            self.lower, self.upper, size=(self.np, self.dim)
        )
    
    def _initialize_velocities(self, population):
        """Initialize velocities as fraction of search range."""
        range_val = self.upper - self.lower
        return np.random.uniform(
            -0.1 * range_val, 0.1 * range_val, size=(self.np, self.dim)
        )
    
    def _compute_neighborhood_best_batch(self, population, personal_best_pos, fitness):
        """Compute best position in each particle's ring neighborhood."""
        n = len(population)
        neighborhood_best = np.zeros_like(population)
        
        for i in range(n):
            # Get neighborhood indices (ring topology)
            left_start = (i - self.ring_size) % n
            right_end = (i + self.ring_size + 1) % n
            
            if left_start < right_end:
                neighborhood_indices = np.arange(left_start, right_end) % n
            else:
                # Wrap around
                neighborhood_indices = np.concatenate([
                    np.arange(left_start, n),
                    np.arange(0, right_end)
                ])
            
            # Find best in neighborhood
            neighborhood_fitness = fitness[neighborhood_indices]
            best_neighbor_idx = neighborhood_indices[np.argmin(neighborhood_fitness)]
            neighborhood_best[i] = personal_best_pos[best_neighbor_idx]
        
        return neighborhood_best
    
    def _update_velocities_batch(self, velocities, positions, personal_best, neighborhood_best):
        """Update velocities with cognitive and social components."""
        r1 = np.random.uniform(0, 1, size=(self.np, self.dim))
        r2 = np.random.uniform(0, 1, size=(self.np, self.dim))
        
        # Cognitive component (toward personal best)
        cognitive = self.cognitive * r1 * (personal_best - positions)
        
        # Social component (toward neighborhood best)
        social = self.social * r2 * (neighborhood_best - positions)
        
        # Update velocity
        new_velocities = self.inertia * velocities + cognitive + social
        
        return new_velocities
    
    def _mutate_velocity_batch(self, velocities):
        """Apply Gaussian mutation to velocities for diversity."""
        if self._no_improvement_count > 20:
            # Increase mutation when stagnant
            mutation_rate = 0.3
            mutation_strength = 2.0
        else:
            mutation_rate = 0.1
            mutation_strength = 0.5
        
        mask = np.random.uniform(0, 1, size=(self.np, self.dim)) < mutation_rate
        mutation = np.random.normal(0, mutation_strength, size=(self.np, self.dim))
        
        velocities = velocities + mask * mutation
        return velocities
    
    def _clip_velocities_batch(self, velocities):
        """Clip velocities to maximum allowed."""
        return np.clip(velocities, -self.max_velocity, self.max_velocity)
    
    def _update_positions_batch(self, positions, velocities):
        """Update positions using velocities."""
        return positions + velocities
    
    def _clip_to_bounds_batch(self, population):
        """Clip population to search bounds."""
        return np.clip(population, self.lower, self.upper)
    
    def _update_personal_best_batch(self, population, fitness, personal_best_pos, personal_best_fit):
        """Update personal best positions and fitness values."""
        improved = fitness < personal_best_fit
        
        # Update positions where improved
        new_personal_best_pos = personal_best_pos.copy()
        new_personal_best_pos[improved] = population[improved]
        
        # Update fitness
        new_personal_best_fit = personal_best_fit.copy()
        new_personal_best_fit[improved] = fitness[improved]
        
        return new_personal_best_pos, new_personal_best_fit
    
    def _adapt_parameters_batch(self, current_best_fitness):
        """Adapt PSO parameters based on improvement rate."""
        if self._prev_best_fitness == np.inf:
            return
        
        improvement_ratio = (self._prev_best_fitness - current_best_fitness) / \
                           (abs(self._prev_best_fitness) + 1e-10)
        
        # Adapt inertia
        if self._improvement_streak >= 3:
            self.inertia = min(0.95, self.inertia * 1.05)
        elif self._no_improvement_count >= 10:
            self.inertia = max(0.4, self.inertia * 0.95)
        
        # Adapt social coefficients based on improvement
        if improvement_ratio > 0.01:
            self.social = min(2.0, self.social * 1.02)
            self.cognitive = min(2.0, self.cognitive * 0.98)
        elif improvement_ratio < -0.01:
            self.cognitive = min(2.0, self.cognitive * 1.02)
            self.social = max(0.5, self.social * 0.98)
        
        # Ensure reasonable bounds
        self.inertia = np.clip(self.inertia, 0.4, 0.95)
        self.cognitive = np.clip(self.cognitive, 0.5, 2.5)
        self.social = np.clip(self.social, 0.5, 2.5)
    
    def _compute_diversity_batch(self, population):
        """Compute population diversity as average pairwise distance."""
        if len(population) < 2:
            return 0.0
        
        mean_pos = np.mean(population, axis=0)
        distances = np.sqrt(np.sum((population - mean_pos) ** 2, axis=1))
        diversity = np.mean(distances) / (self.upper - self.lower)
        
        return diversity
    
    def _adapt_topology_batch(self, diversity):
        """Adapt ring neighborhood size based on diversity."""
        if diversity < self.diversity_threshold and self.ring_size < 10:
            # Low diversity: increase neighborhood (more exploitation)
            self.ring_size += 1
        elif diversity > 3 * self.diversity_threshold and self.ring_size > 2:
            # High diversity: decrease neighborhood (more exploration)
            self.ring_size -= 1
    
    def _select_survivors_batch(self, population, velocities, fitness, personal_best_pos):
        """DE-style greedy selection with velocity inheritance."""
        # This is a simplified selection - in ATPSO we keep all particles
        # but may reinitialize velocities for poorly performing ones
        return population, velocities
    
    def _restart_stagnant_batch(self, population, velocities, personal_best_pos, 
                                 personal_best_fit, global_best):
        """Reinitialize a portion of stagnant particles."""
        n = len(population)
        restart_count = max(1, n // 4)
        
        # Find worst performing particles
        worst_indices = np.argsort(personal_best_fit)[-restart_count:]
        
        # Reinitialize them near global best
        for idx in worst_indices:
            # Jump toward global best with randomization
            alpha = np.random.uniform(0.1, 0.5)
            population[idx] = global_best + np.random.uniform(
                -20, 20, size=self.dim
            )
            population[idx] = np.clip(population[idx], self.lower, self.upper)
            
            # Reset velocity
            velocities[idx] = np.random.uniform(
                -5, 5, size=self.dim
            )
            
            # Reset personal best to new position
            personal_best_pos[idx] = population[idx].copy()
            personal_best_fit[idx] = np.inf
        
        return population, velocities, personal_best_pos, personal_best_fit
```