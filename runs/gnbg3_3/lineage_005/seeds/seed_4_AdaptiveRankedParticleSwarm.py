import numpy as np

class AdaptiveRankedParticleSwarm:
    """
    A population-based optimizer combining:
    - Rank-based attraction (particles attracted to top-ranked individuals, not just the best)
    - Adaptive PSO dynamics with cognitive/social components
    - Diversity-triggered repulsion from centroid
    - Covariance-guided perturbation for exploration
    - Gradient-free local search on elite subset
    - Adaptive parameter control (inertia, neighborhood, step size)
    - Stagnation-triggered restart with elite preservation
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower = -100.0
        self.upper = 100.0
        
        # Population size: modest for dim=30
        self.np = min(150, max(4 * dim, 8 * dim))
        
        # Adaptive PSO parameters
        self.inertia = 0.729
        self.cognitive = 1.494
        self.social = 1.494
        self.repulsion_strength = 0.2
        self.temporal_drift = 0.15
        
        # Adaptive neighborhood
        self.neighborhood_size = max(3, dim // 5)
        self.max_neighborhood_size = self.np // 2
        
        # Step size for local search and perturbation
        self.step_size = 1.0
        self.min_step_size = 1e-4
        
        # Tracking
        self.f_opt = np.inf
        self.x_opt = None
        self.stagnation_counter = 0
        self.max_stagnation = 50
        self.diversity_threshold = 1e-5
        
        # Population storage
        self.population = None
        self.fitness = None
        self.velocities = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.attraction_pool = None
        self.attraction_pool_fitness = None
        
    def _initialize_population(self, func):
        """Initialize population uniformly within bounds and evaluate batch."""
        self.population = np.random.uniform(
            self.lower, self.upper, size=(self.np, self.dim)
        )
        self.velocities = np.random.uniform(
            -self.step_size, self.step_size, size=(self.np, self.dim)
        )
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.np, np.inf)
        
        # Evaluate initial population in batch
        self.fitness = func(self.population)
        
        # Handle budget exhaustion on init
        if len(self.fitness) < self.np:
            self.fitness = np.pad(
                self.fitness, (0, self.np - len(self.fitness)), constant_values=np.inf
            )
        
        # Update global best
        idx = np.argmin(self.fitness)
        if self.fitness[idx] < self.f_opt:
            self.f_opt = self.fitness[idx]
            self.x_opt = self.population[idx].copy()
        
        self._update_attraction_pool()
        return self.fitness
    
    def _update_attraction_pool(self):
        """Create rank-based attraction pool from top-ranked individuals."""
        valid_fitness = np.where(np.isfinite(self.fitness), self.fitness, np.inf)
        ranks = np.argsort(valid_fitness)
        pool_size = max(2, self.np // 5)
        self.attraction_pool = self.population[ranks[:pool_size]]
        self.attraction_pool_fitness = valid_fitness[ranks[:pool_size]]
    
    def _compute_diversity(self):
        """Compute population diversity using sampling-based pairwise distance."""
        if self.population is None:
            return 0.0
        
        sample_size = min(50, self.np)
        indices = np.random.choice(self.np, sample_size, replace=False)
        sample = self.population[indices]
        
        # Mean pairwise Euclidean distance, normalized
        dists = np.linalg.norm(sample[:, None] - sample[None, :], axis=2)
        return np.mean(dists) / (self.upper - self.lower)
    
    def _adapt_inertia_weight(self):
        """Adapt inertia weight based on diversity and stagnation status."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold:
            # Low diversity: increase inertia for more exploration
            self.inertia = min(0.95, self.inertia * 1.05)
        else:
            # Adequate diversity: decrease inertia for more exploitation
            self.inertia = max(0.3, self.inertia * 0.97)
        
        # Increase inertia during stagnation to escape local optima
        if self.stagnation_counter > 10:
            self.inertia = min(0.95, self.inertia * 1.02)
    
    def _adapt_neighborhood_size(self):
        """Adapt neighborhood size based on stagnation to balance exploitation/exploration."""
        if self.stagnation_counter > 5:
            self.neighborhood_size = min(
                self.max_neighborhood_size,
                int(self.neighborhood_size * 1.15)
            )
        else:
            self.neighborhood_size = max(
                3,
                int(self.neighborhood_size * 0.97)
            )
    
    def _adapt_step_size(self):
        """Adapt step size for local search and perturbation."""
        if self.stagnation_counter > 5:
            self.step_size = min(self.step_size * 1.2, 3.0)
        else:
            self.step_size = max(self.step_size * 0.98, self.min_step_size)
    
    def _select_attractor_batch(self):
        """Select attractor for each particle from rank-based pool (vectorized)."""
        n_attractors = len(self.attraction_pool)
        attractor_indices = np.random.randint(0, n_attractors, size=self.np)
        return self.attraction_pool[attractor_indices]
    
    def _velocity_update_batch(self, attractors):
        """Update velocities using cognitive, social, and diversity components."""
        # Cognitive: pull toward personal best
        cognitive = self.cognitive * np.random.uniform(0, 1, size=(self.np, self.dim)) * \
                    (self.personal_best - self.population)
        
        # Social: pull toward selected attractor
        social = self.social * np.random.uniform(0, 1, size=(self.np, self.dim)) * \
                 (attractors - self.population)
        
        # Diversity: repulsion from population centroid
        centroid = np.mean(self.population, axis=0)
        repulsion = self.repulsion_strength * np.random.uniform(0, 1, size=(self.np, self.dim)) * \
                    (self.population - centroid)
        
        # Combine with inertia
        self.velocities = self.inertia * self.velocities + cognitive + social + repulsion
        
        # Limit velocity magnitude to prevent explosion
        max_vel = 0.25 * (self.upper - self.lower)
        norms = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        self.velocities = np.where(
            norms > max_vel,
            self.velocities * max_vel / np.maximum(norms, 1e-10),
            self.velocities
        )
    
    def _position_update_batch(self):
        """Update positions using velocities and temporal drift."""
        self.population += self.velocities
        
        # Temporal drift for exploration
        drift = self.temporal_drift * np.random.randn(self.np, self.dim)
        self.population += drift
        
        # Clip to bounds
        self.population = np.clip(self.population, self.lower, self.upper)
    
    def _position_update_fitness_rank(self):
        """Apply rank-based position adjustment: worse particles move toward better ones."""
        valid_fitness = np.where(np.isfinite(self.fitness), self.fitness, np.inf)
        ranks = np.argsort(valid_fitness)
        ranks_normalized = ranks / max(self.np - 1, 1)
        
        # Identify worse half of population
        worse_mask = valid_fitness > np.median(valid_fitness)
        n_worse = np.sum(worse_mask)
        
        if n_worse > 0:
            # For each worse particle, move toward a randomly chosen better particle
            worse_indices = np.where(worse_mask)[0]
            better_indices = np.where(~worse_mask)[0]
            
            target_indices = np.random.choice(better_indices, size=n_worse)
            targets = self.population[target_indices]
            
            # Adjustment proportional to rank (worse = more movement)
            adjustments = self.step_size * ranks_normalized[worse_indices, None] * \
                         (targets - self.population[worse_indices])
            self.population[worse_indices] += adjustments
        
        self.population = np.clip(self.population, self.lower, self.upper)
    
    def _apply_local_search_batch(self, func):
        """Apply batched gradient-free local search to elite subset."""
        elite_count = max(3, self.np // 10)
        elite_indices = np.argsort(self.fitness)[:elite_count]
        
        # Create batch of trial points using random direction perturbation
        n_directions = min(8, self.dim)
        elite_batch = self.population[elite_indices]
        
        # Generate random orthonormal directions for each elite
        directions = np.random.randn(elite_count, n_directions, self.dim)
        directions = directions / (np.linalg.norm(directions, axis=2, keepdims=True) + 1e-10)
        
        # Create trial points along positive and negative directions
        trial_points = []
        for i in range(elite_count):
            for j in range(n_directions):
                trial_points.append(elite_batch[i] + self.step_size * 0.5 * directions[i, j])
                trial_points.append(elite_batch[i] - self.step_size * 0.5 * directions[i, j])
        
        trial_batch = np.array(trial_points)
        trial_batch = np.clip(trial_batch, self.lower, self.upper)
        
        # Evaluate all trials in one batch
        trial_fitness = func(trial_batch)
        
        # Handle budget exhaustion
        if len(trial_fitness) < len(trial_batch):
            trial_batch = trial_batch[:len(trial_fitness)]
            trial_fitness = trial_fitness[:len(trial_fitness)]
        
        # Update each elite if improved found
        for i, elite_idx in enumerate(elite_indices):
            start = i * 2 * n_directions
            end = start + 2 * n_directions
            local_trials = trial_batch[start:end]
            local_fitness = trial_fitness[start:end]
            
            best_local_idx = np.argmin(local_fitness)
            if local_fitness[best_local_idx] < self.fitness[elite_idx]:
                self.population[elite_idx] = local_trials[best_local_idx]
                self.fitness[elite_idx] = local_fitness[best_local_idx]
                
                if self.fitness[elite_idx] < self.f_opt:
                    self.f_opt = self.fitness[elite_idx]
                    self.x_opt = self.population[elite_idx].copy()
    
    def _restart_if_stagnant(self, func):
        """Restart population if stagnating or diversity is too low."""
        should_restart = (
            self.stagnation_counter >= self.max_stagnation or
            self._compute_diversity() < self.diversity_threshold or
            self.step_size < self.min_step_size
        )
        
        if should_restart:
            # Preserve best solution
            best_idx = np.argmin(self.fitness)
            best_x = self.population[best_idx].copy()
            best_f = self.fitness[best_idx]
            
            # Reinitialize population
            self.population = np.random.uniform(
                self.lower, self.upper, size=(self.np, self.dim)
            )
            self.population[0] = best_x
            
            # Evaluate new population
            self.fitness = func(self.population)
            
            # Handle budget exhaustion
            if len(self.fitness) < self.np:
                self.fitness = np.pad(
                    self.fitness, (0, self.np - len(self.fitness)), constant_values=np.inf
                )
            self.fitness[0] = best_f
            
            # Reset velocities and tracking
            self.velocities = np.random.uniform(
                -self.step_size, self.step_size, size=(self.np, self.dim)
            )
            self.personal_best = self.population.copy()
            self.personal_best_fitness = self.fitness.copy()
            
            # Reset adaptive parameters
            self.inertia = 0.729
            self.step_size = max(self.step_size, 0.5)
            self.stagnation_counter = 0
            
            self._update_attraction_pool()
    
    def __call__(self, func, stopping_condition):
        """
        Main optimization loop.
        
        Args:
            func: Objective function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        # Initialize population and evaluate
        self.fitness = self._initialize_population(func)
        self.stagnation_counter = 0
        
        # Main optimization loop
        while not stopping_condition():
            # Adapt parameters based on current state
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            self._adapt_step_size()
            
            # Select attractors from rank-based pool
            attractors = self._select_attractor_batch()
            
            # Update velocities with cognitive, social, and diversity components
            self._velocity_update_batch(attractors)
            
            # Update positions using velocities and temporal drift
            self._position_update_batch()
            
            # Apply rank-based position adjustment for worse particles
            self._position_update_fitness_rank()
            
            # Evaluate entire population in single batch
            fitness_new = func(self.population)
            
            # Handle budget exhaustion
            if len(fitness_new) < self.np:
                valid_mask = np.isfinite(fitness_new)
                self.fitness[valid_mask] = fitness_new[valid_mask]
                self.fitness[~valid_mask] = np.inf
                break
            
            self.fitness = fitness_new
            
            # Update personal bests
            improved = self.fitness < self.personal_best_fitness
            self.personal_best[improved] = self.population[improved]
            self.personal_best_fitness[improved] = self.fitness[improved]
            
            # Update global best and stagnation counter
            idx = np.argmin(self.fitness)
            if self.fitness[idx] < self.f_opt:
                self.f_opt = self.fitness[idx]
                self.x_opt = self.population[idx].copy()
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            # Check stopping condition after population evaluation
            if stopping_condition():
                break
            
            # Apply batched local search to elite subset
            self._apply_local_search_batch(func)
            
            # Update attraction pool for next iteration
            self._update_attraction_pool()
            
            # Check for stagnation and restart if needed
            self._restart_if_stagnant(func)
        
        return self.f_opt, self.x_opt
