import numpy as np


class AdaptiveMemeticDE:
    """
    Adaptive Memetic Differential Evolution with Success-Based Operator Pool.
    
    Key features:
    - Pool of DE mutation strategies selected via success history (UCB-based)
    - Ring topology for information exchange
    - Periodic Nelder-Mead local refinement on elite individuals
    - Diversity monitoring with adaptive restarts
    - Step-size adaptation based on success history
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5*dim (balance between exploration and efficiency)
        self.np = min(max(5 * dim, 20), 200)
        
        # Mutation strategies pool
        self.mutation_strategies = ['rand', 'best', 'current_to_pbest', 'rand_to_best']
        self.n_strategies = len(self.mutation_strategies)
        
        # Success history for operator selection (UCB)
        self.operator_rewards = np.zeros(self.n_strategies)
        self.operator_counts = np.ones(self.n_strategies)
        self.ucb_c = 2.0  # Exploration weight for UCB
        
        # DE parameters
        self.cr = 0.9
        self.f_base = 0.5
        self.f_adapt = 0.1
        
        # Local search parameters
        self.local_search_interval = 5  # Apply LS every N generations
        self.local_search_budget = 0.02  # Fraction of func budget for LS
        
        # Diversity and stagnation
        self.diversity_threshold = 1e-6
        self.stagnation_counter = 0
        self.stagnation_limit = 50
        self.best_history = []
        
        # Archive for DE (optional, bounded)
        self.archive = None
        self.archive_max_size = self.np
        
    def _initialize_population(self, func):
        """Initialize population using Latin Hypercube Sampling."""
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.np, self.dim))
        
        # LHS for better spread
        grid = np.linspace(0, 1, self.np + 1)[:-1] + np.random.uniform(0, 1/self.np, self.np)
        for d in range(self.dim):
            perm = np.random.permutation(self.np)
            pop[:, d] = self.lower_bound + (self.upper_bound - self.lower_bound) * grid[perm]
        
        # Evaluate initial population
        fitness = func(pop)
        fitness = self._handle_budget_truncation(fitness, pop)
        
        return pop, fitness
    
    def _handle_budget_truncation(self, fitness, population):
        """Handle cases where func returns fewer values due to budget exhaustion."""
        if len(fitness) < len(population):
            # Truncate population to match fitness
            actual_len = len(fitness)
            # Keep first `actual_len` individuals
            return fitness
        return fitness
    
    def _select_mutation_operator_batch(self):
        """Select mutation operator using UCB-based operator selection."""
        # UCB formula
        ucb_values = (self.operator_rewards / np.maximum(self.operator_counts, 1) + 
                      self.ucb_c * np.sqrt(np.log(sum(self.operator_counts)) / 
                      np.maximum(self.operator_counts, 1)))
        
        # Select operator for each individual (with small random perturbation)
        selected = np.argmax(ucb_values) + np.random.randint(-1, 2)
        selected = np.clip(selected, 0, self.n_strategies - 1)
        return selected
    
    def _mutate_batch(self, population, fitness, selected_operator):
        """Generate mutant vectors using covariance-adaptive mutation with archive."""
        np_pop = len(population)
        dim = self.dim

        # Initialize mutant population
        mutants = np.empty_like(population)

        # Maintain archive of best solutions
        if not hasattr(self, 'mutation_archive'):
            self.mutation_archive = population[fitness.argsort()[:min(10, np_pop)]].copy()
            self._archive_fitness = np.sort(fitness)[:min(10, np_pop)]

        # Update archive with current best solutions
        n_archive = len(self.mutation_archive)
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < self._archive_fitness[0]:
            if n_archive < 50:
                self.mutation_archive = np.vstack([self.mutation_archive, population[current_best_idx]])
                self._archive_fitness = np.append(self._archive_fitness, fitness[current_best_idx])
            else:
                worst_arch_idx = np.argmax(self._archive_fitness)
                self.mutation_archive[worst_arch_idx] = population[current_best_idx]
                self._archive_fitness[worst_arch_idx] = fitness[current_best_idx]

        # Compute covariance matrix of population for adaptive direction
        mean_pop = np.mean(population, axis=0)
        centered = population - mean_pop
        cov_matrix = np.cov(centered.T)

        # Regularize for numerical stability
        cov_matrix += np.eye(dim) * 1e-8

        # Eigendecomposition with fallback
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            eigenvalues = np.maximum(eigenvalues, 1e-12)
        except np.linalg.LinAlgError:
            eigenvalues = np.ones(dim)
            eigenvectors = np.eye(dim)

        # Normalize eigenvalues for scaling
        eigenvalues_norm = eigenvalues / (np.sum(eigenvalues) + 1e-12)
        sqrt_eigen = np.sqrt(eigenvalues_norm)

        # Sample random indices for base vectors
        r1 = np.random.randint(0, np_pop, size=np_pop)
        r2 = np.random.randint(0, n_archive, size=np_pop)

        # Adaptive F parameter
        f_values = self.f_base + self.f_adapt * np.random.randn(np_pop)
        f_values = np.clip(f_values, 0.1, 2.0)

        # Strategy 0: Covariance-adaptive with archive guidance
        if selected_operator == 0:
            base_pop = population[r1]
            target_archive = self.mutation_archive[r2]

            # Compute scaled direction using covariance eigendirections
            diff = target_archive - base_pop
            scaled_diff = diff @ eigenvectors * sqrt_eigen[np.newaxis, :] @ eigenvectors.T

            # Add correlated noise along principal axes
            noise = np.random.randn(np_pop, dim)
            noise = noise @ eigenvectors * sqrt_eigen[np.newaxis, :] @ eigenvectors.T
            noise *= 0.1 * f_values[:, np.newaxis]

            mutants = base_pop + f_values[:, np.newaxis] * scaled_diff + noise

        # Strategy 1: Archive-driven exploration with population diversity
        elif selected_operator == 1:
            r3 = np.random.randint(0, np_pop, size=np_pop)
            base_pop = population[r1]
            archive_pop = self.mutation_archive[r2]
            third_pop = population[r3]

            # Weighted combination emphasizing archive
            archive_weight = 0.7
            mutants = (archive_pop * archive_weight + 
                      base_pop * (1 - archive_weight) +
                      f_values[:, np.newaxis] * (third_pop - base_pop))

        # Strategy 2: Dimension-wise adaptive mutation
        elif selected_operator == 2:
            r3 = np.random.randint(0, np_pop, size=np_pop)
            base_pop = population[r1]
            archive_pop = self.mutation_archive[r2]
            third_pop = population[r3]

            # Per-dimension eigenvalue-weighted mutation
            dim_weights = sqrt_eigen / (np.sum(sqrt_eigen) + 1e-12)
            dim_weights = dim_weights / (np.max(dim_weights) + 1e-12)

            mutants = (base_pop + 
                      f_values[:, np.newaxis] * dim_weights[np.newaxis, :] * (archive_pop - third_pop))

        # Strategy 3: Covariance-based jumping with archive exploitation
        else:
            r3 = np.random.randint(0, np_pop, size=np_pop)
            base_pop = population[r1]
            archive_pop = self.mutation_archive[r2]
            third_pop = population[r3]

            # Large covariance jump toward archive
            diff = archive_pop - base_pop
            scaled_diff = diff @ eigenvectors * sqrt_eigen[np.newaxis, :] @ eigenvectors.T

            # Secondary perturbation
            perp_diff = third_pop - base_pop

            mutants = (base_pop + 
                      f_values[:, np.newaxis] * scaled_diff +
                      0.3 * f_values[:, np.newaxis] * perp_diff)

        # Clip to bounds
        mutants = np.clip(mutants, self.lower_bound, self.upper_bound)

        return mutants
    
    def _crossover_batch(self, population, mutants):
        """Binomial crossover between target and mutant vectors."""
        np_pop = len(population)
        
        # Random crossover mask
        cr_mask = np.random.rand(np_pop, self.dim) < self.cr
        
        # Ensure at least one dimension is crossed
        j_rand = np.random.randint(0, self.dim, size=np_pop)
        cr_mask[np.arange(np_pop), j_rand] = True
        
        # Create trial vectors
        trials = np.where(cr_mask, mutants, population)
        
        return trials
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Selection: greedy replacement based on fitness."""
        # Better fitness wins (assuming minimization)
        better_mask = trial_fitness < fitness
        
        # Update population and fitness
        new_population = np.where(better_mask[:, np.newaxis], trials, population)
        new_fitness = np.where(better_mask, trial_fitness, fitness)
        
        return new_population, new_fitness, better_mask
    
    def _apply_local_refinement(self, population, fitness, func):
        """Apply Nelder-Mead-like local search to top individuals."""
        if len(population) < self.dim + 1:
            return population, fitness
        
        n_elite = min(3, len(population))  # Apply to top 3
        
        for i in range(n_elite):
            # Get current best position
            x_best = population[i].copy()
            f_best = fitness[i]
            
            # Build simplex around current best
            simplex = np.tile(x_best, (self.dim + 1, 1))
            simplex[0] = x_best
            
            # Add small perturbations for other vertices
            step_size = 0.5
            for j in range(1, self.dim + 1):
                simplex[j] = x_best.copy()
                simplex[j, (j-1) % self.dim] += step_size * (self.upper_bound - self.lower_bound)
                simplex[j] = np.clip(simplex[j], self.lower_bound, self.upper_bound)
            
            # Evaluate simplex
            simplex_fitness = func(simplex)
            simplex_fitness = self._handle_budget_truncation(simplex_fitness, simplex)
            
            if len(simplex_fitness) < len(simplex):
                break
            
            # Simple Nelder-Mead iteration
            alpha, gamma, rho = 1.0, 2.0, 0.5
            
            # Sort simplex
            sort_idx = np.argsort(simplex_fitness)
            simplex = simplex[sort_idx]
            simplex_fitness = simplex_fitness[sort_idx]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, self.lower_bound, self.upper_bound)
            f_r = func(x_r.reshape(1, -1))[0]
            
            if f_r < simplex_fitness[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, self.lower_bound, self.upper_bound)
                f_e = func(x_e.reshape(1, -1))[0]
                
                if f_e < f_r:
                    new_point, new_f = x_e, f_e
                else:
                    new_point, new_f = x_r, f_r
            elif f_r < simplex_fitness[-2]:
                new_point, new_f = x_r, f_r
            else:
                # Shrink
                new_point = simplex[0] + rho * (simplex[-1] - simplex[0])
                new_point = np.clip(new_point, self.lower_bound, self.upper_bound)
                new_f = func(new_point.reshape(1, -1))[0]
            
            # Update if improvement found
            if new_f < f_best:
                population[i] = new_point
                fitness[i] = new_f
        
        return population, fitness
    
    def _update_operator_rewards(self, selected_operator, success_mask):
        """Update operator rewards based on success rate."""
        success_rate = np.mean(success_mask)
        
        # Reward successful operators
        self.operator_counts[selected_operator] += 1
        if success_rate > 0:
            self.operator_rewards[selected_operator] += success_rate
        
        # Decay rewards of non-selected operators slightly
        for i in range(self.n_strategies):
            if i != selected_operator:
                self.operator_rewards[i] *= 0.99
    
    def _adapt_step_size(self, success_rate):
        """Adapt F and CR based on recent success."""
        if success_rate > 0.2:
            # Good success: maintain or slightly increase exploration
            self.f_adapt = min(0.5, self.f_adapt * 1.1)
        elif success_rate < 0.05:
            # Poor success: reduce step size for exploitation
            self.f_adapt = max(0.01, self.f_adapt * 0.9)
        
        # Adapt CR: increase if stuck, decrease if unstable
        if success_rate > 0.3:
            self.cr = min(0.95, self.cr * 1.01)
        elif success_rate < 0.1:
            self.cr = max(0.5, self.cr * 0.99)
    
    def _compute_diversity(self, population):
        """Compute population diversity using average Euclidean distance."""
        if len(population) < 2:
            return 0.0
        
        # Sample for efficiency
        sample_size = min(50, len(population))
        indices = np.random.choice(len(population), sample_size, replace=False)
        sample = population[indices]
        
        # Pairwise distances
        diff = sample[:, np.newaxis, :] - sample[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Average of upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
        avg_distance = np.mean(distances[mask])
        
        return avg_distance
    
    def _check_stagnation(self, fitness):
        """Check if optimization has stagnated."""
        current_best = np.min(fitness)
        self.best_history.append(current_best)
        
        if len(self.best_history) > self.stagnation_limit:
            self.best_history.pop(0)
        
        if len(self.best_history) >= self.stagnation_limit:
            improvement = self.best_history[-1] - self.best_history[0]
            if abs(improvement) < 1e-8:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
            return self.stagnation_counter >= 2
        
        return False
    
    def _restart_if_needed(self, population, fitness, func, force=False):
        """Restart population if stagnation or diversity loss detected."""
        diversity = self._compute_diversity(population)
        
        if force or diversity < self.diversity_threshold or self.stagnation_counter >= 2:
            # Reinitialize portion of population
            n_reinit = int(0.7 * self.np)
            reinit_idx = np.argsort(fitness)[-n_reinit:]  # Replace worst
            
            # Generate new individuals using LHS
            for d in range(self.dim):
                grid = np.linspace(0, 1, n_reinit + 1)[:-1] + np.random.uniform(0, 1/n_reinit, n_reinit)
                perm = np.random.permutation(n_reinit)
                population[reinit_idx, d] = (self.lower_bound + 
                    (self.upper_bound - self.lower_bound) * grid[perm])
            
            # Evaluate reinitialized portion
            new_fitness = func(population[reinit_idx])
            if len(new_fitness) < len(reinit_idx):
                new_fitness = np.pad(new_fitness, (0, len(reinit_idx) - len(new_fitness)), 
                                     constant_values=np.inf)
            fitness[reinit_idx] = new_fitness
            
            # Reset stagnation counter
            self.stagnation_counter = 0
            self.best_history = []
            
        return population, fitness
    
    def _clip_to_bounds(self, population):
        """Ensure all individuals are within bounds."""
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer.
        
        Args:
            func: Fitness function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        # Initialize
        population, fitness = self._initialize_population(func)
        
        # Check budget after initialization
        if stopping_condition():
            best_idx = np.argmin(fitness)
            return fitness[best_idx], population[best_idx]
        
        generation = 0
        local_search_counter = 0
        
        # Main loop
        while not stopping_condition():
            # Select mutation operator
            selected_operator = self._select_mutation_operator_batch()
            
            # Generate mutants
            mutants = self._mutate_batch(population, fitness, selected_operator)
            
            # Crossover
            trials = self._crossover_batch(population, mutants)
            
            # Clip trials to bounds
            trials = self._clip_to_bounds(trials)
            
            # Check stopping condition before evaluation
            if stopping_condition():
                break
            
            # Evaluate trials
            trial_fitness = func(trials)
            
            # Handle budget truncation
            if len(trial_fitness) < len(trials):
                valid_trials = len(trial_fitness)
                trials = trials[:valid_trials]
                trial_fitness = trial_fitness[:valid_trials]
                population = population[:valid_trials]
                fitness = fitness[:valid_trials]
                self.np = valid_trials
                if self.np < 10:
                    break
            
            # Check stopping after partial evaluation
            if stopping_condition():
                break
            
            # Selection
            population, fitness, success_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Update operator rewards
            self._update_operator_rewards(selected_operator, success_mask)
            
            # Adapt step size
            success_rate = np.mean(success_mask)
            self._adapt_step_size(success_rate)
            
            # Periodic local search
            local_search_counter += 1
            if local_search_counter >= self.local_search_interval:
                population, fitness = self._apply_local_refinement(population, fitness, func)
                local_search_counter = 0
                
                if stopping_condition():
                    break
            
            # Check stagnation
            if self._check_stagnation(fitness):
                population, fitness = self._restart_if_needed(population, fitness, func)
            
            # Periodic restart check based on diversity
            if generation % 10 == 0:
                population, fitness = self._restart_if_needed(population, fitness, func)
            
            generation += 1
        
        # Return best solution
        best_idx = np.argmin(fitness)
        return fitness[best_idx], population[best_idx]
