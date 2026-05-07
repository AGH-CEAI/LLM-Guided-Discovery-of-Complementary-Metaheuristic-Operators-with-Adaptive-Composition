import numpy as np


class TopologyAdaptiveSwarm:
    """
    Topology-Adaptive Particle Swarm with Covariance-Guided Perturbations (TAPC)
    
    A PSO variant featuring:
    - Dynamic ring topology with adaptive neighborhood radius
    - Covariance-guided mutation for exploration (CMA-ES inspired)
    - Adaptive inertia weight based on success history
    - Diversity-triggered re-initialization
    - Fully vectorized batch operations
    
    Optimizer contract:
    - func(population) -> fitness array of shape (N,)
    - stopping_condition() -> bool (True when to stop)
    - Returns (f_opt, x_opt)
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = kwargs.get('lower_bound', -100.0)
        self.upper_bound = kwargs.get('upper_bound', 100.0)
        
        # Population size: 6*dim with bounds [120, 300]
        self.np = min(max(6 * dim, 120), 300)
        
        # Adaptive inertia weight (standard PSO recommended value)
        self.inertia = 0.729
        self.inertia_min = 0.3
        self.inertia_max = 0.9
        
        # Acceleration coefficients
        self.cognitive = 1.496
        self.social = 1.496
        
        # Topology
        self.neighborhood_size = max(3, dim // 4)
        
        # Covariance adaptation (CMA-ES inspired)
        self.cov_decay = 0.9
        self.cov_learning_rate = 0.02
        self.cov_matrix = np.eye(dim)
        self.mean_shift = np.zeros(dim)
        
        # Diversity monitoring
        self.diversity_threshold = 0.1 * (self.upper_bound - self.lower_bound)
        self.stagnation_counter = 0
        self.stagnation_limit = 50
        
        # Internal state
        self.population = None
        self.velocities = None
        self.fitness = None
        self.personal_best = None
        self.personal_best_fit = None
        self.neighborhood_best = None
        self.neighborhood_best_fit = None
        
        # Success history for inertia adaptation
        self.recent_improvements = np.zeros(5)
        
    # =========================================================================
    # INITIALIZATION METHODS
    # =========================================================================
    
    def _initialize_population(self):
        """Initialize particle positions, velocities, and tracking arrays."""
        range_width = self.upper_bound - self.lower_bound
        
        # Random uniform initialization
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.np, self.dim)
        )
        
        # Velocity initialization (bounded)
        self.velocities = np.random.uniform(
            -0.5 * range_width, 0.5 * range_width, (self.np, self.dim)
        )
        
        # Personal best = initial positions
        self.personal_best = self.population.copy()
        self.personal_best_fit = np.full(self.np, np.inf)
        
        # Neighborhood best tracking (will be computed after first eval)
        self.neighborhood_best = np.zeros((self.np, self.dim))
        self.neighborhood_best_fit = np.full(self.np, np.inf)
        
        # Reset covariance
        self.cov_matrix = np.eye(self.dim)
        self.mean_shift = np.zeros(self.dim)
        
        # Reset stagnation
        self.stagnation_counter = 0
        self.recent_improvements.fill(0)
        
    def _initialize_fitness(self, func):
        """Evaluate initial population fitness."""
        self.fitness = func(self.population)
        if len(self.fitness) < self.np:
            self.fitness = np.pad(self.fitness, (0, self.np - len(self.fitness)), 
                                   constant_values=np.nan)
        
        # Initialize personal bests
        valid_mask = ~np.isnan(self.fitness)
        self.personal_best_fit[valid_mask] = self.fitness[valid_mask]
        self.personal_best[valid_mask] = self.population[valid_mask]
        
        # Initialize neighborhood bests
        self._update_neighborhood_bests()
        
    # =========================================================================
    # TOPOLOGY METHODS
    # =========================================================================
    
    def _compute_ring_topology(self):
        """
        Compute ring neighborhood indices for each particle.
        Returns array of shape (NP, max_neighbors).
        """
        neighbors = np.zeros((self.np, 2 * self.neighborhood_size + 1), dtype=int)
        
        for i in range(self.np):
            indices = np.arange(i - self.neighborhood_size, i + self.neighborhood_size + 1)
            indices = ((indices % self.np) + self.np) % self.np
            neighbors[i] = indices
            
        return neighbors
    
    def _update_neighborhood_bests(self):
        """Update best position in each particle's neighborhood (vectorized)."""
        neighbors = self._compute_ring_topology()
        
        # For each particle, find the best in its neighborhood
        for i in range(self.np):
            neighbor_fits = self.personal_best_fit[neighbors[i]]
            best_neighbor_idx = neighbors[i][np.nanargmin(neighbor_fits)]
            
            self.neighborhood_best_fit[i] = self.personal_best_fit[best_neighbor_idx]
            self.neighborhood_best[i] = self.personal_best[best_neighbor_idx]
    
    # =========================================================================
    # MOVEMENT METHODS (BATCH)
    # =========================================================================
    
    def _update_velocities_batch(self):
        """
        Update all particle velocities using PSO velocity formula.
        v = w*v + c1*r1*(pbest - x) + c2*r2*(nbest - x)
        """
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = self.cognitive * r1 * (self.personal_best - self.population)
        social_component = self.social * r2 * (self.neighborhood_best - self.population)
        
        self.velocities = self.inertia * self.velocities + cognitive_component + social_component
        
        # Velocity clamping to prevent explosion
        v_max = 0.5 * (self.upper_bound - self.lower_bound)
        self.velocities = np.clip(self.velocities, -v_max, v_max)
        
    def _update_positions_batch(self):
        """Update all particle positions."""
        self.population = self.population + self.velocities
        
        # Hard bounds clipping
        self.population = np.clip(self.population, self.lower_bound, self.upper_bound)
        
    # =========================================================================
    # COVARIANCE-GUIDED MUTATION (CMA-ES INSPIRED)
    # =========================================================================
    
    def _generate_covariance_mutations(self):
        """
        Generate mutation perturbations using adapted covariance matrix.
        Only applied to a fraction of the population for diversity.
        """
        mutation_mask = np.random.uniform(0, 1, self.np) < 0.15
        num_mutations = np.sum(mutation_mask)
        
        if num_mutations == 0:
            return np.zeros((self.np, self.dim))
        
        # Generate samples from multivariate normal
        try:
            L = np.linalg.cholesky(self.cov_matrix + 1e-8 * np.eye(self.dim))
            noise = L @ np.random.randn(self.dim, num_mutations)
            noise = noise.T
        except np.linalg.LinAlgError:
            noise = np.random.randn(num_mutations, self.dim)
        
        # Scale mutation strength
        mutation_strength = 0.1 * (self.upper_bound - self.lower_bound)
        perturbations = mutation_strength * noise
        
        # Apply only to selected particles
        result = np.zeros((self.np, self.dim))
        result[mutation_mask] = perturbations
        
        return result
    
    def _adapt_covariance_matrix(self):
        """
        Adapt covariance matrix based on recent successful samples.
        Simplified CMA-ES rank-1 update.
        """
        # Weighted mean of recent improvements
        if np.any(self.recent_improvements > 0):
            weight = 0.1
        else:
            weight = 0.01
            
        # Rank-1 update: cov = (1-w)*cov + w*outer(diff)
        self.cov_matrix = (1 - weight) * self.cov_matrix + weight * np.outer(
            self.mean_shift, self.mean_shift
        )
        
        # Ensure positive definiteness
        self.cov_matrix = 0.5 * (self.cov_matrix + self.cov_matrix.T)
        eigenvalues = np.linalg.eigvalsh(self.cov_matrix)
        if eigenvalues.min() < 1e-10:
            self.cov_matrix += (1e-9 - eigenvalues.min()) * np.eye(self.dim)
            
    # =========================================================================
    # SELECTION AND ADAPTATION
    # =========================================================================
    
    def _select_survivors_batch(self, trial_population, trial_fitness):
        """
        Greedy selection: keep better of trial vs current for each particle.
        Updates personal bests and tracks improvements.
        """
        # Handle truncated returns
        if len(trial_fitness) < self.np:
            trial_population = trial_population[:len(trial_fitness)]
            trial_fitness = np.pad(trial_fitness, (0, self.np - len(trial_fitness)), 
                                    constant_values=np.inf)
        
        # Determine winners
        valid_trial = ~np.isnan(trial_fitness)
        valid_current = ~np.isnan(self.fitness)
        valid_both = valid_trial & valid_current
        
        # Trial wins
        trial_wins = valid_both & (trial_fitness < self.fitness)
        current_wins = valid_both & (trial_fitness >= self.fitness)
        
        # Update population with winners
        self.population[trial_wins] = trial_population[trial_wins]
        self.fitness[trial_wins] = trial_fitness[trial_wins]
        
        # Update personal bests
        improved = trial_wins & (trial_fitness < self.personal_best_fit)
        not_improved = trial_wins & ~improved
        
        self.personal_best_fit[improved] = trial_fitness[improved]
        self.personal_best[improved] = trial_population[improved]
        
        # Track improvement for covariance adaptation
        if np.any(improved):
            best_improved_idx = np.argmin(trial_fitness[improved])
            self.mean_shift = trial_population[improved][best_improved_idx] - np.mean(
                self.population, axis=0
            )
        
        # Update neighborhood bests
        self._update_neighborhood_bests()
        
        # Return improvement count for adaptation
        return np.sum(trial_wins)
    
    def _adapt_inertia_weight(self, improvements):
        """Adapt inertia weight based on recent success rate."""
        # Shift history
        self.recent_improvements = np.roll(self.recent_improvements, -1)
        self.recent_improvements[-1] = improvements
        
        success_rate = np.mean(self.recent_improvements > 0)
        
        if success_rate > 0.6:
            # Good progress: increase inertia for exploration
            self.inertia = min(self.inertia_max, self.inertia * 1.05)
        elif success_rate < 0.2:
            # Poor progress: decrease inertia for exploitation
            self.inertia = max(self.inertia_min, self.inertia * 0.95)
            
    # =========================================================================
    # DIVERSITY AND RESTART
    # =========================================================================
    
    def _compute_diversity(self):
        """Compute population diversity as average Euclidean distance to centroid."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _check_stagnation(self, best_fitness, gbest_fit):
        """Check if algorithm is stagnating."""
        if best_fitness < gbest_fit - 1e-10:
            self.stagnation_counter = 0
            return False
        else:
            self.stagnation_counter += 1
            return self.stagnation_counter > self.stagnation_limit
        
    def _should_restart(self):
        """Determine if population should be re-initialized."""
        diversity = self._compute_diversity()
        return (diversity < self.diversity_threshold or 
                self.stagnation_counter > self.stagnation_limit)
    
    def _restart_population(self):
        """Re-initialize population while preserving best solution."""
        # Save global best
        gbest_idx = np.nanargmin(self.fitness)
        gbest_pos = self.population[gbest_idx].copy()
        gbest_fit = self.fitness[gbest_idx]
        
        # Re-initialize
        self._initialize_population()
        
        # Inject best solution into population
        replace_idx = np.random.randint(self.np)
        self.population[replace_idx] = gbest_pos
        self.fitness[replace_idx] = gbest_fit
        self.personal_best[replace_idx] = gbest_pos
        self.personal_best_fit[replace_idx] = gbest_fit
        
        # Re-evaluate to update neighborhood
        self._update_neighborhood_bests()
        
    # =========================================================================
    # MAIN OPTIMIZATION LOOP
    # =========================================================================
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimization.
        
        Args:
            func: Fitness function f(population) -> fitness array
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        # Initialize
        self._initialize_population()
        self._initialize_fitness(func)
        
        # Track global best
        gbest_idx = np.nanargmin(self.fitness)
        gbest_fit = self.fitness[gbest_idx]
        gbest_pos = self.population[gbest_idx].copy()
        
        # Main loop
        generation = 0
        while not stopping_condition():
            # Generate trial positions
            self._update_velocities_batch()
            self._update_positions_batch()
            
            # Apply covariance-guided mutation
            perturbations = self._generate_covariance_mutations()
            self.population = self.population + perturbations
            self.population = np.clip(self.population, self.lower_bound, self.upper_bound)
            
            # Evaluate trials
            trial_fitness = func(self.population)
            
            # Check for budget exhaustion
            if len(trial_fitness) < self.np:
                break
                
            # Selection
            improvements = self._select_survivors_batch(self.population, trial_fitness)
            
            # Update global best
            current_best_idx = np.nanargmin(self.fitness)
            current_best_fit = self.fitness[current_best_idx]
            
            if current_best_fit < gbest_fit:
                gbest_fit = current_best_fit
                gbest_pos = self.population[current_best_idx].copy()
            
            # Adapt parameters
            self._adapt_inertia_weight(improvements)
            self._adapt_covariance_matrix()
            
            # Check stagnation
            if self._check_stagnation(current_best_fit, gbest_fit):
                if self._should_restart():
                    self._restart_population()
            
            generation += 1
            
        return gbest_fit, gbest_pos
