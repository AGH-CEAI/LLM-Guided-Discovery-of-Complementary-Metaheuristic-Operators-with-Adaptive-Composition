```python
import numpy as np

class AdaptiveTopologySwarmOptimizer:
    """
    Particle Swarm Optimizer with adaptive topology, inertia weight,
    acceleration coefficients, and diversity-triggered mutation/restart.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(5 * dim, 30), 8 * dim)
        self.lb = -100.0
        self.ub = 100.0
        
        # Swarm state
        self.population = None
        self.velocity = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.neighborhood_best = None
        self.global_best_fitness = np.inf
        self.global_best_position = None
        
        # Adaptive parameters
        self.inertia = 0.729
        self.cognitive = 1.494
        self.social = 1.494
        self.adaptive_rate = 0.05
        
        # Topology management
        self.topology = 'ring'
        self.topology_list = ['ring', 'star', 'von_neumann', 'random']
        self.neighbors = None
        
        # Diversity and stagnation tracking
        self.diversity_history = []
        self.stagnation_counter = 0
        self.stagnation_limit = 20
        self.improvement_history = []
        self.topology_stagnation = 0
        
    def _initialize_population(self):
        """Initialize swarm with random positions and velocities."""
        self.population = np.random.uniform(self.lb, self.ub, (self.np, self.dim))
        self.velocity = np.random.uniform(-1.0, 1.0, (self.np, self.dim))
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.np, np.inf)
        self.neighborhood_best = np.zeros_like(self.population)
        self._update_topology_batch()
        
    def _update_topology_batch(self):
        """Update neighborhood indices based on current topology setting."""
        indices = np.arange(self.np)
        
        if self.topology == 'ring':
            left = np.roll(indices, 1)
            right = np.roll(indices, -1)
            self.neighbors = np.column_stack([left, right])
            
        elif self.topology == 'star':
            gbest_idx = np.argmin(self.personal_best_fitness) if np.any(np.isfinite(self.personal_best_fitness)) else 0
            self.neighbors = np.full((self.np, 2), gbest_idx, dtype=int)
            
        elif self.topology == 'von_neumann':
            rows = int(np.ceil(np.sqrt(self.np)))
            cols = int(np.ceil(self.np / rows))
            grid = np.full((rows, cols), -1, dtype=int)
            for i, idx in enumerate(indices):
                r, c = i // cols, i % cols
                if r < rows and c < cols:
                    grid[r, c] = idx
            neighbors_list = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r, c] >= 0:
                        nr, nc = (r - 1) % rows, c
                        sr, sc = (r + 1) % rows, c
                        er, ec = r, (c - 1) % cols
                        wr, wc = r, (c + 1) % cols
                        neighbors_list.append([grid[nr, nc], grid[sr, sc], grid[er, ec], grid[wr, wc]])
            self.neighbors = np.array(neighbors_list, dtype=int)[:self.np]
            
        elif self.topology == 'random':
            k = 3
            neighbors_list = []
            for i in range(self.np):
                others = np.concatenate([indices[:i], indices[i+1:]])
                selected = np.random.choice(others, min(k, len(others)), replace=False)
                neighbors_list.append(selected)
            self.neighbors = np.array(neighbors_list, dtype=int)
            
    def _evaluate_batch(self, func):
        """Evaluate fitness for all particles in a single batch call."""
        clipped = np.clip(self.population, self.lb, self.ub)
        fitness = func(clipped)
        
        if len(fitness) < len(self.population):
            return None
        return fitness
    
    def _update_personal_best_batch(self, fitness):
        """Update personal bests for all particles where improvement occurred."""
        improved = fitness < self.personal_best_fitness
        self.personal_best_fitness[improved] = fitness[improved]
        self.personal_best[improved] = self.population[improved]
        
    def _update_neighborhood_best_batch(self):
        """Update neighborhood bests based on current topology."""
        neighbor_best_idx = np.array([np.argmin(self.personal_best_fitness[self.neighbors[i]]) 
                                      for i in range(self.np)])
        flat_indices = self.neighbors[np.arange(self.np), neighbor_best_idx]
        self.neighborhood_best = self.personal_best[flat_indices]
        
    def _update_global_best_batch(self):
        """Update global best across entire swarm."""
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            self.global_best_position = self.personal_best[best_idx].copy()
            
    def _adapt_parameters(self):
        """Adapt inertia and acceleration coefficients based on improvement rate."""
        recent = self.improvement_history[-10:] if self.improvement_history else [True]
        improvement_rate = sum(recent) / len(recent)
        
        if improvement_rate > 0.5:
            self.inertia = min(0.95, self.inertia * (1 + self.adaptive_rate))
        elif improvement_rate < 0.2:
            self.inertia = max(0.3, self.inertia * (1 - self.adaptive_rate))
            
        cognitive_factor = 0.5 + np.random.uniform(0, 0.5)
        social_factor = 0.5 + np.random.uniform(0, 0.5)
        self.cognitive = 2.05 * cognitive_factor * self.inertia
        self.social = 2.05 * social_factor * self.inertia
        
    def _update_velocity_batch(self):
        """Update velocities for all particles using adaptive parameters."""
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = self.cognitive * r1 * (self.personal_best - self.population)
        social_component = self.social * r2 * (self.neighborhood_best - self.population)
        
        self.velocity = self.inertia * self.velocity + cognitive_component + social_component
        
        max_velocity = 0.2 * (self.ub - self.lb)
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)
        
    def _update_position_batch(self):
        """Update positions for all particles."""
        self.population = self.population + self.velocity
        self.population = np.clip(self.population, self.lb, self.ub)
        
    def _compute_diversity(self):
        """Compute swarm diversity as mean distance to centroid."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _apply_mutation_batch(self):
        """Apply mutation to diverse subset of particles to escape local optima."""
        diversity = self._compute_diversity()
        range_scale = (self.ub - self.lb)
        low_diversity = diversity < 0.05 * range_scale * np.sqrt(self.dim)
        
        if low_diversity or self.stagnation_counter > 5:
            mutation_count = max(2, self.np // 4)
            worst_indices = np.argsort(self.personal_best_fitness)[-mutation_count:]
            
            for idx in worst_indices:
                mutant = self.global_best_position + np.random.uniform(-0.5, 0.5, self.dim) * range_scale
                self.population[idx] = np.clip(mutant, self.lb, self.ub)
                self.velocity[idx] = np.random.uniform(-0.1, 0.1, self.dim) * range_scale
                
    def _switch_topology(self):
        """Switch to a different topology when stagnation is detected."""
        current_idx = self.topology_list.index(self.topology)
        new_idx = (current_idx + 1) % len(self.topology_list)
        self.topology = self.topology_list[new_idx]
        self._update_topology_batch()
        self.topology_stagnation = 0
        
    def _restart_if_needed(self):
        """Check stagnation and restart population if necessary."""
        if self.stagnation_counter >= self.stagnation_limit:
            self._restart_population()
            return True
        return False
    
    def _restart_population(self):
        """Reinitialize population while preserving global best."""
        self.population = np.random.uniform(self.lb, self.ub, (self.np, self.dim))
        self.velocity = np.random.uniform(-1.0, 1.0, (self.np, self.dim))
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.np, np.inf)
        self.stagnation_counter = 0
        self.improvement_history = []
        
    def _track_improvement(self, fitness):
        """Track whether global best improved in this iteration."""
        best_current = np.min(fitness)
        improved = best_current < self.global_best_fitness - 1e-12
        self.improvement_history.append(improved)
        if len(self.improvement_history) > 20:
            self.improvement_history.pop(0)
        return improved
    
    def _update_stagnation(self, improved):
        """Update stagnation counter based on improvement status."""
        if improved:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
            self.topology_stagnation += 1
            
    def _select_survivors_batch(self, fitness):
        """Select surviving particles based on fitness (for budget management)."""
        sorted_indices = np.argsort(fitness)
        selected = sorted_indices[:self.np]
        self.population = self.population[selected]
        self.velocity = self.velocity[selected]
        self.personal_best = self.personal_best[selected]
        self.personal_best_fitness = self.personal_best_fitness[selected]
        
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
        
        fitness = self._evaluate_batch(func)
        if fitness is None:
            return self.global_best_fitness, self.global_best_position
            
        self._update_personal_best_batch(fitness)
        self._update_neighborhood_best_batch()
        self._update_global_best_batch()
        
        while not stopping_condition():
            self._adapt_parameters()
            
            self._update_personal_best_batch(fitness)
            self._update_neighborhood_best_batch()
            self._update_global_best_batch()
            
            self._update_velocity_batch()
            self._update_position_batch()
            
            fitness = self._evaluate_batch(func)
            if fitness is None:
                break
                
            if stopping_condition():
                break
                
            improved = self._track_improvement(fitness)
            self._update_stagnation(improved)
            
            if self.topology_stagnation > 15:
                self._switch_topology()
                
            self._apply_mutation_batch()
            
            if self._restart_if_needed():
                fitness = self._evaluate_batch(func)
                if fitness is None:
                    break
                    
        return self.global_best_fitness, self.global_best_position
```