```python
import numpy as np


class TopologyGuidedSwarmOptimizer:
    """
    Particle Swarm Optimizer with adaptive ring topology and hybrid mutation.
    Uses batched operations throughout for efficiency.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5*dim is a good balance for dim=30
        self.np = min(max(5 * dim, 50), 300)
        
        # PSO parameters with adaptive components
        self.inertia = 0.729  # Base inertia weight (convergent value)
        self.inertia_min = 0.3
        self.inertia_max = 0.9
        self.cognitive = 1.494  # Cognitive coefficient
        self.social = 1.494    # Social coefficient
        
        # Topology parameters
        self.neighbor_size = 3  # Ring neighborhood size
        self.diversity_threshold = 0.15  # Trigger topology adaptation
        
        # Mutation parameters (hybrid component)
        self.mutation_prob = 0.05
        self.mutation_strength = 0.1
        
        # Restart parameters
        self.stagnation_limit = 50
        self.max_restarts = 3
        
        # State tracking
        self.population = None
        self.velocities = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.local_best = None
        self.local_best_fitness = None
        self.global_best = None
        self.global_best_fitness = np.inf
        
        self._rng = np.random.default_rng()
    
    def _initialize_population(self):
        """Initialize population using opposition-based learning for better coverage."""
        # Generate initial population
        uniform_pop = self._rng.uniform(
            self.lower_bound, self.upper_bound, 
            size=(self.np, self.dim)
        )
        
        # Generate opposition population
        opp_pop = self.lower_bound + self.upper_bound - uniform_pop
        
        # Combine and evaluate both
        combined = np.vstack([uniform_pop, opp_pop])
        fitness = self._evaluate_batch(combined)
        
        # Select best NP individuals from combined pool
        indices = np.argsort(fitness)[:self.np]
        self.population = combined[indices]
        self.personal_best_fitness = fitness[indices].copy()
        self.personal_best = self.population.copy()
        
        # Initialize velocities as fraction of search space
        range_val = self.upper_bound - self.lower_bound
        self.velocities = self._rng.uniform(
            -0.1 * range_val, 0.1 * range_val,
            size=(self.np, self.dim)
        )
        
        # Find initial global best
        best_idx = np.argmin(self.personal_best_fitness)
        self.global_best = self.personal_best[best_idx].copy()
        self.global_best_fitness = self.personal_best_fitness[best_idx]
        
        # Initialize local bests (ring topology)
        self._initialize_ring_topology()
    
    def _initialize_ring_topology(self):
        """Initialize local bests for ring topology."""
        self.local_best = self.personal_best.copy()
        self.local_best_fitness = self.personal_best_fitness.copy()
    
    def _evaluate_batch(self, population):
        """Evaluate fitness for entire population batch."""
        # Clip to bounds before evaluation
        clipped = np.clip(population, self.lower_bound, self.upper_bound)
        fitness = self._func(clipped)
        
        # Handle budget exhaustion gracefully
        if len(fitness) < len(population):
            # Truncate population to evaluated portion
            self.population = clipped[:len(fitness)]
            self.np = len(fitness)
            self.velocities = self.velocities[:len(fitness)]
            self.personal_best = self.personal_best[:len(fitness)]
            self.personal_best_fitness = self.personal_best_fitness[:len(fitness)]
        
        return fitness
    
    def _update_personal_best_batch(self, fitness):
        """Update personal bests for all individuals in batch."""
        improved = fitness < self.personal_best_fitness
        self.personal_best[improved] = self.population[improved]
        self.personal_best_fitness[improved] = fitness[improved]
    
    def _get_ring_neighbors(self, indices):
        """Get neighbor indices for ring topology (vectorized)."""
        n = self.np
        half_k = self.neighbor_size // 2
        
        # Create neighbor index arrays
        neighbors = np.zeros((len(indices), self.neighbor_size), dtype=int)
        for i, idx in enumerate(indices):
            neighbor_indices = [(idx - j) % n for j in range(half_k, 0, -1)]
            neighbor_indices += [(idx + j) % n for j in range(1, half_k + 1)]
            neighbors[i] = neighbor_indices
        
        return neighbors
    
    def _update_local_best_batch(self):
        """Update local bests based on ring neighborhood (batched)."""
        indices = np.arange(self.np)
        neighbors = self._get_ring_neighbors(indices)
        
        # Gather neighbor best fitness values
        neighbor_best_fit = self.personal_best_fitness[neighbors]  # (NP, neighbor_size)
        neighbor_best_pos = self.personal_best[neighbors]  # (NP, neighbor_size, dim)
        
        # Find best neighbor for each individual
        best_neighbor_idx = np.argmin(neighbor_best_fit, axis=1)  # (NP,)
        
        # Get best neighbor positions and fitness
        batch_indices = np.arange(self.np)
        new_local_best_fit = neighbor_best_fit[batch_indices, best_neighbor_idx]
        new_local_best_pos = neighbor_best_pos[batch_indices, best_neighbor_idx]
        
        # Update local bests where improvement found
        improved = new_local_best_fit < self.local_best_fitness
        self.local_best_fitness[improved] = new_local_best_fit[improved]
        self.local_best[improved] = new_local_best_pos[improved]
    
    def _update_global_best_batch(self):
        """Update global best from personal bests."""
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best = self.personal_best[best_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[best_idx]
    
    def _compute_diversity(self):
        """Compute swarm diversity as average distance to centroid."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances) / (self.upper_bound - self.lower_bound)
    
    def _adapt_inertia(self, diversity):
        """Adapt inertia weight based on swarm diversity."""
        # High diversity -> higher inertia for exploration
        # Low diversity -> lower inertia for exploitation
        target_inertia = self.inertia_min + (self.inertia_max - self.inertia_min) * diversity
        # Smooth adaptation
        self.inertia = 0.9 * self.inertia + 0.1 * target_inertia
    
    def _adapt_topology(self, diversity):
        """Adapt neighborhood size based on diversity."""
        if diversity < self.diversity_threshold:
            # Low diversity: increase neighborhood for stronger social influence
            self.neighbor_size = min(self.neighbor_size + 2, self.np // 2)
        else:
            # High diversity: decrease neighborhood for more individual exploration
            self.neighbor_size = max(self.neighbor_size - 1, 3)
    
    def _update_velocities_batch(self):
        """Update velocities for all individuals using PSO formula (batched)."""
        # Generate random coefficients
        r1 = self._rng.uniform(0, 1, size=(self.np, self.dim))
        r2 = self._rng.uniform(0, 1, size=(self.np, self.dim))
        
        # Cognitive component: attraction to personal best
        cognitive = self.cognitive * r1 * (self.personal_best - self.population)
        
        # Social component: attraction to local best (ring topology)
        social = self.social * r2 * (self.local_best - self.population)
        
        # Update velocity
        self.velocities = self.inertia * self.velocities + cognitive + social
        
        # Velocity clamping to prevent explosion
        max_velocity = 0.2 * (self.upper_bound - self.lower_bound)
        self.velocities = np.clip(self.velocities, -max_velocity, max_velocity)
    
    def _apply_mutation_batch(self):
        """Apply Gaussian mutation to positions (hybrid component)."""
        mutation_mask = self._rng.uniform(0, 1, size=(self.np, self.dim)) < self.mutation_prob
        
        # Mutation toward global best with Gaussian perturbation
        mutation_direction = self.global_best - self.population
        mutation = mutation_mask * self._rng.normal(
            0, self.mutation_strength * (self.upper_bound - self.lower_bound),
            size=(self.np, self.dim)
        )
        
        self.velocities += mutation
    
    def _update_positions_batch(self):
        """Update positions based on velocities."""
        self.population = self.population + self.velocities
    
    def _clip_positions_batch(self):
        """Clip all positions to search bounds."""
        self.population = np.clip(
            self.population, 
            self.lower_bound, 
            self.upper_bound
        )
    
    def _restart_stagnant_portion(self, stagnation_counter):
        """Replace stagnant portion of population with new random individuals."""
        if stagnation_counter >= self.stagnation_limit and self.max_restarts > 0:
            # Replace 30% of worst individuals
            replace_count = int(0.3 * self.np)
            worst_indices = np.argsort(self.personal_best_fitness)[-replace_count:]
            
            # Generate new individuals near global best with perturbation
            new_range = (self.upper_bound - self.lower_bound) * 0.3
            new_individuals = self.global_best + self._rng.normal(
                0, new_range, size=(replace_count, self.dim)
            )
            new_individuals = np.clip(
                new_individuals, self.lower_bound, self.upper_bound
            )
            
            self.population[worst_indices] = new_individuals
            
            # Reset corresponding velocities
            self.velocities[worst_indices] = self._rng.uniform(
                -new_range, new_range, size=(replace_count, self.dim)
            )
            
            # Re-evaluate replaced individuals
            new_fitness = self._evaluate_batch(new_individuals)
            self.personal_best_fitness[worst_indices] = new_fitness
            self.personal_best[worst_indices] = new_individuals
            
            self.max_restarts -= 1
            return True
        return False
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer.
        
        Args:
            func: Fitness function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        self._func = func
        
        # Initialize
        self._initialize_population()
        
        # Main optimization loop
        stagnation_counter = 0
        prev_best = np.inf
        generation = 0
        
        while not stopping_condition():
            # Evaluate current population
            fitness = self._evaluate_batch(self.population)
            
            # Check for budget exhaustion
            if len(fitness) < len(self.population):
                break
            
            # Check stopping condition after evaluation
            if stopping_condition():
                break
            
            # Update personal bests
            self._update_personal_best_batch(fitness)
            
            # Update local bests (ring topology)
            self._update_local_best_batch()
            
            # Update global best
            self._update_global_best_batch()
            
            # Check for stagnation
            if self.global_best_fitness < prev_best - 1e-10:
                stagnation_counter = 0
                prev_best = self.global_best_fitness
            else:
                stagnation_counter += 1
            
            # Adapt inertia based on diversity
            diversity = self._compute_diversity()
            self._adapt_inertia(diversity)
            
            # Adapt topology based on diversity
            self._adapt_topology(diversity)
            
            # Restart if stagnant
            if self._restart_stagnant_portion(stagnation_counter):
                stagnation_counter = 0
                continue
            
            # Update velocities
            self._update_velocities_batch()
            
            # Apply mutation (hybrid component)
            self._apply_mutation_batch()
            
            # Update positions
            self._update_positions_batch()
            
            # Clip to bounds
            self._clip_positions_batch()
            
            generation += 1
        
        return self.global_best_fitness, self.global_best
```