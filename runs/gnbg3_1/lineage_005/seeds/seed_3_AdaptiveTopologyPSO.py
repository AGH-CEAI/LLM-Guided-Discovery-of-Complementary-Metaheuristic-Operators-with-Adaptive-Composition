import numpy as np


class AdaptiveTopologyPSO:
    """
    Particle Swarm Optimizer with adaptive neighborhood topologies,
    covariance-guided acceleration, and DE-style selection.
    
    Key features:
    - Multiple neighborhood topologies (ring, Von Neumann, star, random)
    - Adaptive inertia and acceleration coefficients
    - Covariance-guided mutation direction
    - DE-style crossover and selection for exploration
    - Diversity-based topology switching
    - Restart mechanism for stagnation
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 120), 300)  # Population size
        self.lower = -100.0
        self.upper = 100.0
        
        # Default parameters
        self.chi = 0.7298  # Constriction coefficient
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_max = 2.5
        self.c1_min = 1.5
        self.c2_max = 2.5
        self.c2_min = 1.5
        self.c3 = 0.3  # Covariance influence weight
        
        # Topology management
        self.topologies = ['ring', 'von_neumann', 'star', 'random']
        self.current_topology = 'ring'
        self.topology_switch_interval = 50
        self.diversity_threshold_high = 0.8
        self.diversity_threshold_low = 0.2
        
        # Restart parameters
        self.max_stagnation = 100
        self.restart_ratio = 0.5
        
        # Internal state
        self.history = {'best_fitness': [], 'diversity': [], 'topology': []}
        
    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        # Initialize
        population = self._initialize_population()
        velocities = self._initialize_velocities(population)
        personal_best_pos = population.copy()
        
        # Initial evaluation
        fitness = self._evaluate_batch(func, population)
        personal_best_fitness = fitness.copy()
        
        # Initialize neighborhood bests
        neighborhood_best_pos, neighborhood_best_fitness = self._init_neighborhood_best(
            population, fitness
        )
        
        # Global best
        gbest_idx = np.argmin(fitness)
        gbest_pos = population[gbest_idx].copy()
        gbest_fitness = fitness[gbest_idx]
        
        stagnation_counter = 0
        generation = 0
        
        while not stopping_condition():
            # Adapt parameters
            self._adapt_parameters(generation, fitness, gbest_fitness)
            
            # Update neighborhood bests
            neighborhood_best_pos, neighborhood_best_fitness = self._update_neighborhood_best_batch(
                population, fitness, neighborhood_best_pos, neighborhood_best_fitness
            )
            
            # Compute covariance-guided direction
            cov_direction = self._compute_covariance_direction(population, gbest_pos)
            
            # Generate trial positions via PSO update
            trials = self._generate_pso_trials(
                population, velocities, personal_best_pos,
                neighborhood_best_pos, cov_direction
            )
            
            # DE-style crossover for exploration
            trials = self._crossover_batch(population, trials)
            
            # Clip to bounds
            trials = self._clip_to_bounds_batch(trials)
            
            # Evaluate trials
            trial_fitness = self._evaluate_batch(func, trials)
            
            # Check budget exhaustion
            if len(trial_fitness) < len(trials):
                valid_mask = np.arange(len(trials)) < len(trial_fitness)
                trials = trials[valid_mask]
                trial_fitness = trial_fitness[:np.sum(valid_mask)]
            
            if stopping_condition():
                break
            
            # Selection: keep better of trial vs original
            improvement_mask = trial_fitness < fitness
            population = np.where(improvement_mask[:, np.newaxis], trials, population)
            fitness = np.where(improvement_mask, trial_fitness, fitness)
            velocities = np.where(improvement_mask[:, np.newaxis], 
                                  trials - population + velocities * 0.5, velocities)
            
            # Update personal bests
            better_than_personal = fitness < personal_best_fitness
            personal_best_pos = np.where(better_than_personal[:, np.newaxis], 
                                         population, personal_best_pos)
            personal_best_fitness = np.where(better_than_personal, 
                                              fitness, personal_best_fitness)
            
            # Update global best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < gbest_fitness:
                gbest_pos = population[current_best_idx].copy()
                gbest_fitness = fitness[current_best_idx]
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Track history
            diversity = self._compute_diversity(population)
            self.history['best_fitness'].append(gbest_fitness)
            self.history['diversity'].append(diversity)
            self.history['topology'].append(self.current_topology)
            
            # Check for restart
            if stagnation_counter >= self.max_stagnation:
                population, velocities = self._restart_if_stagnant(
                    population, fitness, personal_best_pos
                )
                personal_best_fitness = fitness.copy()
                stagnation_counter = 0
                self._adapt_topology(diversity)
            
            # Periodic topology adaptation
            if generation % self.topology_switch_interval == 0:
                self._adapt_topology(diversity)
            
            generation += 1
        
        return gbest_fitness, gbest_pos
    
    def _initialize_population(self):
        """Initialize population uniformly in bounds."""
        return np.random.uniform(
            self.lower, self.upper, size=(self.np, self.dim)
        )
    
    def _initialize_velocities(self, population):
        """Initialize velocities based on range."""
        v_range = (self.upper - self.lower) * 0.1
        return np.random.uniform(-v_range, v_range, size=(self.np, self.dim))
    
    def _evaluate_batch(self, func, population):
        """Evaluate fitness for entire population batch."""
        return func(population)
    
    def _clip_to_bounds_batch(self, population):
        """Clip population to search bounds."""
        return np.clip(population, self.lower, self.upper)
    
    def _init_neighborhood_best(self, population, fitness):
        """Initialize neighborhood best positions and fitness."""
        neighborhood_best_pos = population.copy()
        neighborhood_best_fitness = fitness.copy()
        return neighborhood_best_pos, neighborhood_best_fitness
    
    def _get_neighborhood_indices(self):
        """Get neighbor indices based on current topology."""
        n = self.np
        
        if self.current_topology == 'ring':
            # Each particle has 2 neighbors in a ring
            left = np.roll(np.arange(n), 1)
            right = np.roll(np.arange(n), -1)
            return left, right
        
        elif self.current_topology == 'von_neumann':
            # Von Neumann neighborhood (4 neighbors in grid)
            row = np.arange(n) // int(np.sqrt(n))
            col = np.arange(n) % int(np.sqrt(n))
            grid_size = int(np.sqrt(n))
            
            up = row * grid_size + col
            down = row * grid_size + col
            left = row * grid_size + col
            right = row * grid_size + col
            
            up = np.where(row > 0, (row - 1) * grid_size + col, (grid_size - 1) * grid_size + col)
            down = np.where(row < grid_size - 1, (row + 1) * grid_size + col, col)
            left = np.where(col > 0, row * grid_size + (col - 1), row * grid_size + (grid_size - 1))
            right = np.where(col < grid_size - 1, row * grid_size + (col + 1), row * grid_size)
            
            return up, down, left, right
        
        elif self.current_topology == 'star':
            # Star: all neighbors are the global best
            return None
        
        else:  # random
            # Random topology: each particle connects to ~sqrt(n) random others
            np.random.seed()
            connections = int(np.sqrt(n))
            neighbors = np.array([
                np.random.choice(np.concatenate([np.arange(i), np.arange(i+1, n)]), 
                               size=connections, replace=False)
                for i in range(n)
            ])
            return neighbors
    
    def _update_neighborhood_best_batch(self, population, fitness, 
                                        neighborhood_best_pos, neighborhood_best_fitness):
        """Update neighborhood best positions using current topology."""
        n = self.np
        new_nb_pos = neighborhood_best_pos.copy()
        new_nb_fit = neighborhood_best_fitness.copy()
        
        if self.current_topology == 'ring':
            left, right = self._get_neighborhood_indices()
            for i in range(n):
                for j in [left[i], right[i]]:
                    if fitness[j] < new_nb_fit[i]:
                        new_nb_fit[i] = fitness[j]
                        new_nb_pos[i] = population[j]
                        
        elif self.current_topology == 'von_neumann':
            up, down, left, right = self._get_neighborhood_indices()
            for i in range(n):
                for j in [up[i], down[i], left[i], right[i]]:
                    if fitness[j] < new_nb_fit[i]:
                        new_nb_fit[i] = fitness[j]
                        new_nb_pos[i] = population[j]
                        
        elif self.current_topology == 'star':
            # Star topology: neighborhood is global best
            gbest_idx = np.argmin(fitness)
            new_nb_fit[:] = fitness[gbest_idx]
            new_nb_pos[:] = population[gbest_idx]
            
        else:  # random
            neighbors = self._get_neighborhood_indices()
            for i in range(n):
                for j in neighbors[i]:
                    if fitness[j] < new_nb_fit[i]:
                        new_nb_fit[i] = fitness[j]
                        new_nb_pos[i] = population[j]
        
        return new_nb_pos, new_nb_fit
    
    def _compute_covariance_direction(self, population, gbest_pos):
        """Compute covariance-guided direction for acceleration."""
        # Compute mean-centered population
        mean = np.mean(population, axis=0)
        centered = population - mean
        
        # Compute covariance matrix with regularization
        cov = np.cov(centered.T)
        cov += np.eye(self.dim) * 1e-6  # Regularization
        
        try:
            # Compute eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            
            # Principal direction (largest eigenvector)
            principal_dir = eigenvectors[:, -1]
            
            # Direction to global best from mean
            to_gbest = gbest_pos - mean
            to_gbest_norm = np.linalg.norm(to_gbest)
            if to_gbest_norm > 1e-10:
                to_gbest = to_gbest / to_gbest_norm
            
            # Combine principal component with direction to best
            cov_direction = 0.5 * principal_dir + 0.5 * to_gbest
            cov_direction = cov_direction / (np.linalg.norm(cov_direction) + 1e-10)
            
        except np.linalg.LinAlgError:
            cov_direction = np.zeros(self.dim)
        
        return cov_direction
    
    def _generate_pso_trials(self, population, velocities, personal_best_pos,
                            neighborhood_best_pos, cov_direction):
        """Generate trial positions using PSO update rule."""
        # Compute adaptive inertia
        progress = len(self.history['best_fitness']) / max(len(self.history['best_fitness']), 1)
        w = self.w_max - (self.w_max - self.w_min) * progress
        
        # Compute adaptive acceleration coefficients
        c1 = self.c1_max - (self.c1_max - self.c1_min) * progress
        c2 = self.c2_max - (self.c2_max - self.c2_min) * progress
        
        # Random coefficients
        r1 = np.random.uniform(0, 1, size=(self.np, self.dim))
        r2 = np.random.uniform(0, 1, size=(self.np, self.dim))
        r3 = np.random.uniform(0, 1, size=(self.np, self.dim))
        
        # PSO velocity update
        cognitive = c1 * r1 * (personal_best_pos - population)
        social = c2 * r2 * (neighborhood_best_pos - population)
        covariance = self.c3 * r3 * cov_direction
        
        velocities = w * velocities + cognitive + social + covariance
        
        # Apply constriction coefficient
        velocities = self.chi * velocities
        
        # Limit velocity
        v_max = (self.upper - self.lower) * 0.2
        velocities = np.clip(velocities, -v_max, v_max)
        
        # Position update
        trials = population + velocities
        
        return trials
    
    def _crossover_batch(self, population, trials):
        """DE-style binomial crossover for exploration."""
        cr = 0.7  # Crossover rate
        n = self.np
        
        # Random crossover mask
        mask = np.random.random(size=(n, self.dim)) < cr
        
        # Ensure at least one dimension is different
        enforce_mask = np.zeros((n, self.dim), dtype=bool)
        enforce_mask[np.arange(n), np.random.randint(0, self.dim, size=n)] = True
        
        mask = mask | enforce_mask
        
        # Crossover
        trials = np.where(mask, trials, population)
        
        return trials
    
    def _compute_diversity(self, population):
        """Compute population diversity (average distance to mean)."""
        mean = np.mean(population, axis=0)
        distances = np.linalg.norm(population - mean, axis=1)
        return np.mean(distances) / (self.upper - self.lower)
    
    def _adapt_parameters(self, generation, fitness, gbest_fitness):
        """Adapt algorithm parameters based on progress."""
        # Adaptation logic handled in _generate_pso_trials via progress
        pass
    
    def _adapt_topology(self, diversity):
        """Adapt topology based on diversity level."""
        if diversity > self.diversity_threshold_high:
            # High diversity: switch to star for exploitation
            self.current_topology = 'star'
        elif diversity < self.diversity_threshold_low:
            # Low diversity: switch to ring for exploration
            self.current_topology = 'ring'
        else:
            # Medium diversity: use Von Neumann
            self.current_topology = 'von_neumann'
    
    def _restart_if_stagnant(self, population, fitness, personal_best_pos):
        """Partially restart population when stagnated."""
        n = self.np
        n_restart = int(n * self.restart_ratio)
        
        # Keep best half
        sorted_indices = np.argsort(fitness)
        keep_indices = sorted_indices[:n // 2]
        
        new_population = population.copy()
        new_velocities = np.zeros_like(population)
        
        # Generate new individuals for worst half
        for i in sorted_indices[n // 2:]:
            # Mix from good solutions
            parent1_idx = np.random.choice(keep_indices)
            parent2_idx = np.random.choice(keep_indices)
            
            # Blend crossover
            alpha = np.random.uniform(-0.5, 1.5)
            new_population[i] = population[parent1_idx] + alpha * (
                population[parent2_idx] - population[parent1_idx]
            )
            
            # Add perturbation
            new_population[i] += np.random.normal(0, 10, size=self.dim)
        
        # Clip to bounds
        new_population = self._clip_to_bounds_batch(new_population)
        
        return new_population, new_velocities
