import numpy as np


class TopologicalSwarmOptimizer:
    """
    Particle Swarm Optimizer with adaptive ring topology and covariance-guided perturbation.
    Uses batched operations throughout for maximum efficiency.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.bounds = (-100.0, 100.0)
        self.lower = np.full(dim, self.bounds[0])
        self.upper = np.full(dim, self.bounds[1])
        
        # Population size: 6 * dim for good coverage
        self.np = min(6 * dim, 300)
        
        # PSO parameters
        self.w_inertia = 0.729
        self.c1_personal = 1.494
        self.c2_social = 1.494
        self.chi = 0.729  # constriction factor
        
        # Adaptation state
        self.step_size = 0.1 * (self.bounds[1] - self.bounds[0])
        self.covariance_scale = 1.0
        self.adapt_success_count = 0
        self.adapt_failure_count = 0
        
        # Ring topology: connect to k neighbors on each side
        self.topology_k = max(2, dim // 10)
        
        # Tracking
        self.f_opt = np.inf
        self.x_opt = None
        self.gen_count = 0
        self.stagnation_counter = 0
        self.max_stagnation = 50
        
    def __call__(self, func, stopping_condition):
        """Main entry point for optimization."""
        population = self._initialize_population()
        fitness = func(population)
        self._handle_budget_exhaustion(fitness, population, "init")
        
        personal_best_pos, personal_best_fit = self._initialize_personal_best(population, fitness)
        neighborhood_best_pos, neighborhood_best_fit = self._compute_neighborhood_best_batch(
            population, fitness
        )
        
        self._update_global_best(fitness, population)
        
        while not stopping_condition():
            self.gen_count += 1
            
            # Build trial batch using PSO + covariance perturbation
            trials = self._generate_trials_batch(
                population, personal_best_pos, neighborhood_best_pos
            )
            
            # Evaluate trials
            trial_fitness = func(trials)
            if self._handle_budget_exhaustion(trial_fitness, trials, "trials"):
                break
            
            # Selection and update
            population, fitness, personal_best_pos, personal_best_fit = \
                self._select_and_update_batch(
                    population, fitness, trials, trial_fitness,
                    personal_best_pos, personal_best_fit
                )
            
            # Recompute neighborhood bests
            neighborhood_best_pos, neighborhood_best_fit = \
                self._compute_neighborhood_best_batch(population, fitness)
            
            # Update global best
            improved = self._update_global_best(fitness, population)
            
            # Adaptation methods
            self._adapt_step_size(improved)
            self._adapt_topology_connectivity()
            self._adapt_covariance_scale(trial_fitness, fitness)
            
            # Restart if stagnant
            if self._check_stagnation(improved):
                population, fitness = self._restart_population(population, fitness)
                personal_best_pos, personal_best_fit = \
                    self._initialize_personal_best(population, fitness)
                neighborhood_best_pos, neighborhood_best_fit = \
                    self._compute_neighborhood_best_batch(population, fitness)
            
            if stopping_condition():
                break
        
        return self.f_opt, self.x_opt
    
    def _initialize_population(self):
        """Initialize random population within bounds."""
        return np.random.uniform(
            self.lower, self.upper, size=(self.np, self.dim)
        ).astype(np.float64)
    
    def _initialize_personal_best(self, population, fitness):
        """Store initial personal bests."""
        return population.copy(), fitness.copy()
    
    def _generate_trials_batch(self, population, pbest, nbest):
        """Generate trial positions using PSO velocity + covariance perturbation."""
        # PSO velocity update
        r1 = np.random.uniform(0, 1, size=(self.np, self.dim))
        r2 = np.random.uniform(0, 1, size=(self.np, self.dim))
        
        velocity = self._update_velocity_batch(population, pbest, nbest, r1, r2)
        
        # Update positions
        trials = population + velocity
        
        # Apply covariance-guided perturbation
        trials = self._apply_covariance_perturbation_batch(trials, population)
        
        # Clip to bounds
        trials = np.clip(trials, self.lower, self.upper)
        
        return trials
    
    def _update_velocity_batch(self, population, pbest, nbest, r1, r2):
        """Compute PSO velocity for entire population."""
        cognitive = self.c1_personal * r1 * (pbest - population)
        social = self.c2_social * r2 * (nbest - population)
        velocity = self.chi * (self.w_inertia * velocity + cognitive + social)
        
        # Limit velocity magnitude
        max_vel = 0.2 * (self.bounds[1] - self.bounds[0])
        velocity_mag = np.linalg.norm(velocity, axis=1, keepdims=True)
        velocity = velocity * np.minimum(1.0, max_vel / (velocity_mag + 1e-10))
        
        return velocity
    
    def _apply_covariance_perturbation_batch(self, trials, population):
        """Apply adaptive covariance-based perturbation to trials."""
        # Sample perturbation from anisotropic distribution
        scale = self.step_size * self.covariance_scale
        perturbation = np.random.normal(0, scale, size=(self.np, self.dim))
        
        # Anisotropic scaling based on fitness gradient
        fitness_normalized = (population.mean(axis=1, keepdims=True) - 
                             population.min(axis=1, keepdims=True) + 1e-10)
        anisotropic_weight = 1.0 + 0.5 * np.random.uniform(0, 1, size=(self.np, 1))
        perturbation = perturbation * anisotropic_weight
        
        return trials + perturbation
    
    def _compute_neighborhood_best_batch(self, population, fitness):
        """Compute best position in each particle's ring neighborhood."""
        # Sort by fitness to find neighborhoods
        sorted_indices = np.argsort(fitness)
        sorted_pop = population[sorted_indices]
        sorted_fit = fitness[sorted_indices]
        
        # For each particle, find best in its neighborhood
        nbest_pos = np.empty_like(population)
        nbest_fit = np.empty_like(fitness)
        
        for i in range(self.np):
            # Ring neighborhood indices
            idx = np.where(sorted_indices == i)[0][0]
            neighbors = []
            for k in range(1, self.topology_k + 1):
                neighbors.append((idx - k) % self.np)
                neighbors.append((idx + k) % self.np)
            neighbors = list(set(neighbors))
            
            # Find best among neighbors
            neighbor_fits = sorted_fit[neighbors]
            best_neighbor_idx = neighbors[np.argmin(neighbor_fits)]
            nbest_pos[i] = sorted_pop[best_neighbor_idx]
            nbest_fit[i] = sorted_fit[best_neighbor_idx]
        
        return nbest_pos, nbest_fit
    
    def _select_and_update_batch(self, population, fitness, trials, trial_fitness,
                                 pbest_pos, pbest_fit):
        """Perform batch selection and update personal bests."""
        # Selection: keep better of trial vs current
        better_mask = trial_fitness < fitness
        better_mask = np.broadcast_to(better_mask[:, np.newaxis], (self.np, self.dim))
        
        new_population = np.where(better_mask, trials, population)
        new_fitness = np.where(trial_fitness < fitness, trial_fitness, fitness)
        
        # Update personal bests
        improved_mask = trial_fitness < pbest_fit
        new_pbest_pos = pbest_pos.copy()
        new_pbest_pos[improved_mask] = trials[improved_mask]
        new_pbest_fit = np.where(improved_mask, trial_fitness, pbest_fit)
        
        return new_population, new_fitness, new_pbest_pos, new_pbest_fit
    
    def _update_global_best(self, fitness, population):
        """Update global best solution."""
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < self.f_opt:
            self.f_opt = fitness[best_idx]
            self.x_opt = population[best_idx].copy()
            self.stagnation_counter = 0
            return True
        else:
            self.stagnation_counter += 1
            return False
    
    def _adapt_step_size(self, improved):
        """Adapt step size based on improvement rate."""
        if improved:
            self.adapt_success_count += 1
            self.adapt_failure_count = 0
            if self.adapt_success_count >= 3:
                self.step_size *= 1.1
                self.adapt_success_count = 0
        else:
            self.adapt_failure_count += 1
            self.adapt_success_count = 0
            if self.adapt_failure_count >= 5:
                self.step_size *= 0.9
                self.adapt_failure_count = 0
    
    def _adapt_topology_connectivity(self):
        """Adapt neighborhood connectivity based on progress."""
        if self.stagnation_counter > 20:
            # Increase connectivity during stagnation
            self.topology_k = min(self.topology_k + 1, self.np // 4)
        elif self.stagnation_counter == 0 and self.gen_count > 10:
            # Decrease connectivity when doing well
            self.topology_k = max(2, self.topology_k - 1)
    
    def _adapt_covariance_scale(self, trial_fitness, fitness):
        """Adapt covariance scale based on trial acceptance rate."""
        accepted = np.sum(trial_fitness < fitness)
        acceptance_rate = accepted / self.np
        
        if acceptance_rate > 0.3:
            self.covariance_scale *= 1.05
        elif acceptance_rate < 0.1:
            self.covariance_scale *= 0.95
        
        self.covariance_scale = np.clip(self.covariance_scale, 0.1, 10.0)
    
    def _check_stagnation(self, improved):
        """Check if optimization has stagnated."""
        return self.stagnation_counter >= self.max_stagnation
    
    def _restart_population(self, population, fitness):
        """Reinitialize part of the population to escape local optima."""
        # Keep best 20% of population
        n_keep = max(1, self.np // 5)
        keep_indices = np.argsort(fitness)[:n_keep]
        
        new_population = population.copy()
        new_population[keep_indices] = population[keep_indices]
        
        # Reinitialize remaining with focus on promising regions
        reinit_mask = np.ones(self.np, dtype=bool)
        reinit_mask[keep_indices] = False
        reinit_indices = np.where(reinit_mask)[0]
        
        # Sample from distribution centered on best solutions
        for idx in reinit_indices:
            center_idx = keep_indices[np.random.randint(len(keep_indices))]
            new_population[idx] = np.random.normal(
                population[center_idx], 
                self.step_size * 2
            )
        
        new_population = np.clip(new_population, self.lower, self.upper)
        new_fitness = fitness.copy()
        new_fitness[reinit_indices] = np.inf
        
        self.stagnation_counter = 0
        self.step_size *= 1.5
        
        return new_population, new_fitness
    
    def _handle_budget_exhaustion(self, fitness, population, phase):
        """Handle budget exhaustion gracefully."""
        if len(fitness) < len(population):
            # Budget exhausted - fill remaining with current best
            if phase == "init":
                return True
            deficit = len(population) - len(fitness)
            fitness = np.append(fitness, [np.nan] * deficit)
        return False
    
    def _compute_diversity(self, population):
        """Compute population diversity metric."""
        centroid = population.mean(axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return distances.mean()
