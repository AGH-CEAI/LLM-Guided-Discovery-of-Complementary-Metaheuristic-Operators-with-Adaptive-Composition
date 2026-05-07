import numpy as np


class AdaptiveTopologyDE:
    """
    Differential Evolution with Ring Topology, Covariance Adaptation,
    and Composite Mutation Strategies for Robust Black-Box Optimization.
    
    Key features:
    - Ring topology with local best information sharing
    - CMA-ES inspired covariance adaptation
    - Composite mutation: current-to-pbest + ring-guided + historical memory
    - Adaptive F and Cr parameters per dimension
    - Diversity-guided restart mechanism
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(150, max(50, 5 * dim))  # Population size
        self.bounds = np.array([-100.0, 100.0])
        self.lb = self.bounds[0]
        self.ub = self.bounds[1]
        
        # Control parameters
        self.F = 0.5 * np.ones(dim)  # Scaling factor per dimension
        self.Cr = 0.9 * np.ones(dim)  # Crossover rate per dimension
        self.p_best = 0.1  # Top p% for pbest mutation
        self.cov_scale = 0.1  # Initial covariance scale
        
        # Ring topology neighbors
        self.ring_neighbors = 3  # Each individual considers this many neighbors
        
        # Adaptation parameters
        self.success_history_F = np.ones((20, dim))
        self.success_history_Cr = np.ones((20, dim))
        self.success_counter = 0
        self.adapt_interval = 10
        
        # Memory for historical best solutions
        self.archive_size = 50
        self.archive = None
        self.archive_fitness = None
        
        # Diversity and stagnation
        self.min_diversity = 1e-6
        self.stagnation_threshold = 50
        self.generations_without_improvement = 0
        
        # Internal state
        self.population = None
        self.fitness = None
        self.f_best = np.inf
        self.x_best = None
        self.gen = 0
        
    def _initialize_population(self):
        """Initialize population using Latin Hypercube Sampling for better coverage."""
        sampler = np.linspace(0, 1, self.np)
        population = np.zeros((self.np, self.dim))
        
        for d in range(self.dim):
            perm = np.random.permutation(self.np)
            population[:, d] = self.lb + (self.ub - self.lb) * (
                sampler[perm] + np.random.uniform(0, 1/self.np, self.np)
            )
        
        # Ensure bounds
        population = np.clip(population, self.lb, self.ub)
        
        # Initialize archive with random solutions
        self.archive = population[:self.archive_size].copy()
        self.archive_fitness = np.full(self.archive_size, np.inf)
        
        return population
    
    def _evaluate_batch(self, population):
        """Evaluate fitness for entire population in one batch call."""
        return self._func(population)
    
    def _clip_to_bounds(self, population):
        """Clip population to search bounds."""
        return np.clip(population, self.lb, self.ub)
    
    def _compute_pbest_indices(self, fitness):
        """Compute indices of top p% individuals for pbest mutation."""
        p_count = max(1, int(self.np * self.p_best))
        sorted_indices = np.argsort(fitness)
        return sorted_indices[:p_count]
    
    def _get_ring_neighbors(self, idx):
        """Get neighbor indices in ring topology."""
        neighbors = []
        for offset in range(1, self.ring_neighbors + 1):
            neighbors.append((idx - offset) % self.np)
            neighbors.append((idx + offset) % self.np)
        return neighbors
    
    def _mutate_ring_topology_batch(self, population, fitness):
        """
        Generate ring-topology guided mutation vectors.
        Uses local best from ring neighborhood plus archive memory.
        """
        trials = np.zeros_like(population)
        pbest_indices = self._compute_pbest_indices(fitness)
        pbest_idx = pbest_indices[0]  # Best individual
        pbest = population[pbest_idx]
        
        for i in range(self.np):
            neighbors = self._get_ring_neighbors(i)
            
            # Get best neighbor fitness
            neighbor_fitness = fitness[neighbors]
            best_neighbor_idx = neighbors[np.argmin(neighbor_fitness)]
            best_neighbor = population[best_neighbor_idx]
            
            # Random archive member
            if self.archive_fitness[0] < np.inf:
                archive_idx = np.random.randint(len(self.archive))
                archive_member = self.archive[archive_idx]
            else:
                archive_member = best_neighbor
            
            # Select 2 random distinct individuals from population
            candidates = list(range(self.np))
            candidates.remove(i)
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            
            # Composite mutation: blend of strategies
            strategy_weight = np.random.rand()
            
            if strategy_weight < 0.4:
                # Current-to-pbest/1
                base = population[i]
                diff = pbest - population[i]
                donor = base + self.F * diff + self.F * (population[r1] - population[r2])
            elif strategy_weight < 0.7:
                # Ring-guided mutation
                donor = best_neighbor + self.F * (archive_member - population[i]) + \
                       self.F * (population[r1] - population[r2])
            else:
                # Archive enhanced mutation
                donor = population[r1] + self.F * (best_neighbor - archive_member) + \
                       self.F * (population[r2] - population[i])
            
            trials[i] = donor
        
        return trials
    
    def _apply_covariance_perturbation(self, trials, population):
        """
        Apply CMA-ES inspired covariance perturbation to enhance exploration.
        Uses historical success to adapt perturbation strength.
        """
        # Compute covariance of successful mutations from history
        if self.success_counter > 5:
            # Use recent success history to estimate good directions
            cov_strength = self.cov_scale * np.mean(self.success_history_F[:self.success_counter])
        else:
            cov_strength = self.cov_scale
        
        # Generate perturbation matrix
        noise = np.random.randn(self.np, self.dim)
        
        # Apply dimension-wise scaling based on F
        noise = noise * (self.F * cov_strength)
        
        # Add perturbation to trials
        trials = trials + noise
        
        return trials
    
    def _crossover_batch(self, population, trials):
        """
        Binomial crossover with dimension-wise Cr adaptation.
        """
        mask = np.random.rand(self.np, self.dim) < self.Cr
        
        # Ensure at least one dimension is different
        rand_dim = np.random.randint(0, self.dim, size=self.np)
        mask[np.arange(self.np), rand_dim] = True
        
        # Create trial population
        trial_population = np.where(mask, trials, population)
        
        return trial_population
    
    def _select_survivors_batch(self, population, fitness, trial_population, trial_fitness):
        """
        Greedy selection between parent and trial vectors.
        """
        better_mask = trial_fitness < fitness
        
        # Update population
        new_population = np.where(better_mask[:, np.newaxis], trial_population, population)
        new_fitness = np.where(better_mask, trial_fitness, fitness)
        
        # Track improvements for adaptation
        improvements = trial_fitness < fitness
        
        return new_population, new_fitness, improvements
    
    def _update_success_history(self, improvements, trial_population, population, trial_fitness, fitness):
        """Update success history for F and Cr adaptation."""
        if not np.any(improvements):
            return
        
        improved_mask = improvements
        n_improved = np.sum(improved_mask)
        
        if n_improved > 0:
            # Compute F used for successful mutations (approximation)
            diff_norm = np.linalg.norm(
                trial_population[improved_mask] - population[improved_mask],
                axis=1
            )
            
            # Update success history (shift and add new)
            self.success_history_F = np.roll(self.success_history_F, -1, axis=0)
            self.success_history_Cr = np.roll(self.success_history_Cr, -1, axis=0)
            
            # Store success ratio for F adaptation
            self.success_history_F[-1] = np.clip(
                diff_norm / (np.linalg.norm(population[improved_mask], axis=1) + 1e-10),
                0.1, 3.0
            )
            
            # For Cr, use whether crossover helped
            self.success_history_Cr[-1] = np.where(
                trial_fitness[improved_mask] < fitness[improved_mask],
                0.9, 0.5
            )
            
            self.success_counter = min(self.success_counter + 1, 20)
    
    def _adapt_parameters(self):
        """
        Adapt F and Cr parameters based on success history.
        Uses weighted median for robust adaptation.
        """
        if self.success_counter < 3:
            return
        
        # Adapt F per dimension using weighted median of successful F values
        weights = np.arange(1, self.success_counter + 1)
        weighted_F = np.average(
            self.success_history_F[:self.success_counter],
            axis=0,
            weights=weights
        )
        
        # Smooth adaptation
        self.F = 0.5 * self.F + 0.5 * np.clip(weighted_F, 0.1, 2.0)
        
        # Adapt Cr per dimension
        weighted_Cr = np.average(
            self.success_history_Cr[:self.success_counter],
            axis=0,
            weights=weights
        )
        self.Cr = 0.7 * self.Cr + 0.3 * np.clip(weighted_Cr, 0.1, 0.99)
    
    def _update_archive(self, population, fitness):
        """Update archive with best solutions found."""
        # Combine current population with archive
        combined = np.vstack([self.archive, population])
        combined_fitness = np.concatenate([self.archive_fitness, fitness])
        
        # Select best archive_size solutions
        best_indices = np.argsort(combined_fitness)[:self.archive_size]
        self.archive = combined[best_indices]
        self.archive_fitness = combined_fitness[best_indices]
    
    def _compute_diversity(self, population):
        """Compute population diversity using average pairwise distance."""
        if self.population is None:
            return 1.0
        
        # Sample-based diversity estimation
        sample_size = min(50, self.np)
        indices = np.random.choice(self.np, sample_size, replace=False)
        sample = population[indices]
        
        # Compute distances to centroid
        centroid = np.mean(sample, axis=0)
        distances = np.linalg.norm(sample - centroid, axis=1)
        
        return np.mean(distances) / (self.ub - self.lb)
    
    def _check_stagnation(self, fitness):
        """Check if optimization has stagnated."""
        current_best = np.min(fitness)
        
        if current_best < self.f_best - 1e-10:
            self.f_best = current_best
            self.generations_without_improvement = 0
            best_idx = np.argmin(fitness)
            self.x_best = self.population[best_idx].copy()
            return False
        else:
            self.generations_without_improvement += 1
            return self.generations_without_improvement > self.stagnation_threshold
    
    def _restart_population(self):
        """Reinitialize population with diversity injection."""
        # Keep best solution
        best_idx = np.argmin(self.fitness)
        best_solution = self.population[best_idx].copy()
        best_fitness = self.fitness[best_idx]
        
        # Reinitialize with LHS + keep best
        self.population = self._initialize_population()
        self.population[0] = best_solution
        self.fitness = np.full(self.np, np.inf)
        self.fitness[0] = best_fitness
        
        # Reset adaptation parameters
        self.F = 0.5 * np.ones(self.dim)
        self.Cr = 0.9 * np.ones(self.dim)
        self.generations_without_improvement = 0
        
        # Reduce covariance scale to encourage exploration
        self.cov_scale *= 0.5
    
    def _adapt_covariance(self):
        """
        Adapt covariance scale based on recent performance.
        Inspired by successful _adapt_covariance method.
        """
        if self.success_counter > 0:
            # Increase covariance when struggling
            recent_success_rate = np.mean(
                self.success_history_Cr[:self.success_counter] > 0.7
            )
            
            if recent_success_rate < 0.3:
                self.cov_scale *= 1.2
            elif recent_success_rate > 0.6:
                self.cov_scale *= 0.9
            
            # Keep scale bounded
            self.cov_scale = np.clip(self.cov_scale, 0.01, 1.0)
    
    def _sample_trials_batch(self, population, fitness):
        """
        Sample trial vectors using composite mutation strategy.
        Wrapper method for batch trial generation.
        """
        # Generate base mutations using ring topology
        base_trials = self._mutate_ring_topology_batch(population, fitness)
        
        # Apply covariance perturbation
        trials = self._apply_covariance_perturbation(base_trials, population)
        
        # Clip to bounds
        trials = self._clip_to_bounds(trials)
        
        return trials
    
    def __call__(self, func, stopping_condition):
        """
        Main optimization loop.
        
        Args:
            func: Fitness function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and solution found
        """
        self._func = func
        
        # Initialize
        self.population = self._initialize_population()
        self.fitness = self._evaluate_batch(self.population)
        self.fitness = np.clip(self.fitness, -1e100, 1e100)
        
        # Handle budget exhaustion at initialization
        if len(self.fitness) < self.np:
            valid_len = len(self.fitness)
            self.fitness = np.pad(self.fitness, (0, self.np - valid_len), constant_values=np.inf)
        
        self.f_best = np.min(self.fitness)
        best_idx = np.argmin(self.fitness)
        self.x_best = self.population[best_idx].copy()
        
        self.gen = 0
        
        # Main loop
        while not stopping_condition():
            self.gen += 1
            
            # Generate trials using batch method
            trials = self._sample_trials_batch(self.population, self.fitness)
            
            # Crossover
            trial_population = self._crossover_batch(self.population, trials)
            trial_population = self._clip_to_bounds(trial_population)
            
            # Evaluate trials
            trial_fitness = self._evaluate_batch(trial_population)
            trial_fitness = np.clip(trial_fitness, -1e100, 1e100)
            
            # Handle budget exhaustion
            if len(trial_fitness) < self.np:
                actual_len = len(trial_fitness)
                # Truncate population to match
                self.population = self.population[:actual_len]
                self.fitness = self.fitness[:actual_len]
                trial_population = trial_population[:actual_len]
                trial_fitness = np.pad(trial_fitness, (0, actual_len - len(trial_fitness)), constant_values=np.inf)
                self.np = actual_len
            
            # Check stopping condition after evaluation
            if stopping_condition():
                break
            
            # Selection
            self.population, self.fitness, improvements = self._select_survivors_batch(
                self.population, self.fitness, trial_population, trial_fitness
            )
            
            # Update success history
            self._update_success_history(
                improvements, trial_population, self.population, trial_fitness, self.fitness
            )
            
            # Periodic adaptation
            if self.gen % self.adapt_interval == 0:
                self._adapt_parameters()
                self._adapt_covariance()
            
            # Update archive
            self._update_archive(self.population, self.fitness)
            
            # Check stagnation
            if self._check_stagnation(self.fitness):
                diversity = self._compute_diversity(self.population)
                if diversity < self.min_diversity:
                    self._restart_population()
                    # Re-evaluate after restart
                    self.fitness = self._evaluate_batch(self.population)
                    self.fitness = np.clip(self.fitness, -1e100, 1e100)
                    continue
            
            # Check stopping condition after each generation
            if stopping_condition():
                break
        
        return self.f_best, self.x_best
