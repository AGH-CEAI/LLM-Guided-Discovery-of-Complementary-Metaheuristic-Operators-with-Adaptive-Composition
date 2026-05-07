import numpy as np


class SwarmGradientEvolutionStrategy:
    """
    Swarm Gradient Evolution Strategy (SGES)
    
    A hybrid metaheuristic combining:
    - Ring topology with gradient approximation from neighborhood best
    - Adaptive memory archive of elite solutions
    - Nelder-Mead simplex local search on top performers
    - Success-based adaptation of control parameters
    
    Key differences from prior approaches:
    - Uses ring topology (not fully connected) for information flow
    - Approximates gradient from population structure (not covariance)
    - Hybrid local search (not embedded in mutation)
    - Memory-guided exploration with adaptive recall
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        # Population size: 5*dim, bounded for efficiency
        self.NP = min(max(5 * dim, 40), 300)
        self.f_bounds = [-100, 100]
        
        # DE control parameters
        self.cr = 0.7
        self.f_curr = 0.5
        self.f_max = 1.5
        self.f_min = 0.2
        
        # Local search config
        self.ls_fraction = 0.15  # Top 15% get local search
        self.ls_alpha = 1.0      # Nelder-Mead reflection coefficient
        
        # Memory archive
        self.memory_size = 15
        self.memory_fitness = np.full(self.memory_size, np.inf)
        
        # Ring topology (built in _initialize)
        self.neighbor_idx = np.zeros((self.NP, 2), dtype=int)
        
        # Adaptation state
        self.success_buffer = []
        self.cr_buffer = []
        self.buffer_size = 7
        self.improvement_count = 0
        
        # Stagnation detection
        self.generations_without_improvement = 0
        
    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        self.func = func
        
        # Initialize all data structures
        population = self._initialize_population()
        fitness = self._evaluate_batch(population, name="initial")
        self._update_memory(population, fitness)
        
        best_idx = np.argmin(fitness)
        best_fitness_history = [fitness[best_idx]]
        
        # Main evolution loop
        while not stopping_condition():
            # Build ring topology based on current fitness ranking
            self._update_topology(fitness)
            
            # Generate trial batch
            trials = self._generate_trial_batch(population, fitness)
            
            # Evaluate trials in batch
            trial_fitness = self._evaluate_batch(trials, name="trials")
            
            # Check for budget exhaustion
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
            
            # Stopping check after evaluation
            if stopping_condition():
                break
            
            # Selection: keep better of target/trial
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Update memory archive
            self._update_memory(population, fitness)
            
            # Adapt control parameters based on success
            self._adapt_parameters(improved_mask)
            
            # Apply local search to top performers
            population, fitness = self._apply_local_search_batch(
                population, fitness, stopping_condition
            )
            
            # Check for stagnation
            current_best = np.min(fitness)
            if current_best < best_fitness_history[-1] - 1e-12:
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1
            
            best_fitness_history.append(np.min(fitness))
            
            # Diversity-based restart if needed
            if self._should_restart():
                population, fitness = self._diversity_restart(population, fitness)
            
            # Stopping check after local search
            if stopping_condition():
                break
        
        # Return best found
        best_idx = np.argmin(fitness)
        return fitness[best_idx], population[best_idx].copy()
    
    def _initialize_population(self):
        """Create initial population within bounds."""
        pop = np.random.uniform(
            self.f_bounds[0], self.f_bounds[1],
            size=(self.NP, self.dim)
        )
        # Ensure even spread via Latin hypercube sampling approximation
        for d in range(self.dim):
            order = np.random.permutation(self.NP)
            pop[:, d] = np.linspace(self.f_bounds[0], self.f_bounds[1], self.NP)[order]
            pop[:, d] += np.random.uniform(-1, 1, self.NP) * (
                (self.f_bounds[1] - self.f_bounds[0]) / (2 * self.NP)
            )
        pop = np.clip(pop, self.f_bounds[0], self.f_bounds[1])
        return pop
    
    def _evaluate_batch(self, population, name=""):
        """Evaluate entire population in one func call."""
        fitness = self.func(population)
        # Handle NaN/inf in fitness
        if isinstance(fitness, list):
            fitness = np.array(fitness)
        fitness = np.asarray(fitness, dtype=np.float64)
        return fitness
    
    def _update_topology(self, fitness):
        """Update ring topology neighbors based on fitness ranking."""
        # Sort by fitness, best individuals get neighbors who are also good
        rank = np.argsort(fitness)
        # Each agent's neighbors: one better, one worse in ranking
        for i in range(self.NP):
            rank_pos = np.where(rank == i)[0][0]
            left = rank[(rank_pos - 1) % self.NP]
            right = rank[(rank_pos + 1) % self.NP]
            self.neighbor_idx[i] = [left, right]
    
    def _generate_trial_batch(self, population, fitness):
        """Generate trial population using DE mutation + gradient approximation."""
        trials = np.empty((self.NP, self.dim))
        
        # Compute neighborhood bests
        nb_best = np.array([
            population[self.neighbor_idx[i]].min(axis=0)
            for i in range(self.NP)
        ])
        
        # Compute gradient approximation: direction toward neighborhood best
        gradient = nb_best - population  # Shape: (NP, dim)
        
        # Compute adaptive f based on diversity
        f_adapted = self._compute_adaptive_f(population, fitness)
        
        for i in range(self.NP):
            # Target vector
            target = population[i]
            
            # Select random neighbor for direction
            nb = self.neighbor_idx[i, np.random.randint(2)]
            
            # Mutation: target + F * (neighbor_best - target) + F * gradient
            mutant = target + f_adapted[i] * gradient[i] + f_adapted[i] * 0.5 * (population[nb] - target)
            
            # Crossover: binomial
            j_rand = np.random.randint(self.dim)
            mask = np.random.random(self.dim) < self.cr
            mask[j_rand] = True
            trials[i] = np.where(mask, mutant, target)
        
        # Clip to bounds
        trials = np.clip(trials, self.f_bounds[0], self.f_bounds[1])
        return trials
    
    def _compute_adaptive_f(self, population, fitness):
        """Compute per-individual scaling factor based on diversity and position."""
        f_base = self.f_curr * np.ones(self.NP)
        
        # Compute population spread
        pop_range = population.max(axis=0) - population.min(axis=0)
        spread = np.linalg.norm(pop_range) / (self.dim * 10)
        
        # Scale F based on distance to best
        best_idx = np.argmin(fitness)
        dist_to_best = np.linalg.norm(population - population[best_idx], axis=1)
        dist_normalized = dist_to_best / (dist_to_best.max() + 1e-10)
        
        # Individuals far from best get larger F (more exploration)
        f_adapted = f_base * (1.0 + 0.3 * dist_normalized)
        f_adapted = np.clip(f_adapted, self.f_min, self.f_max)
        
        return f_adapted
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Select better individuals between target and trial populations."""
        improved = trial_fitness < fitness
        
        # Create new population: keep original or take trial
        new_pop = np.where(improved[:, np.newaxis], trials, population)
        new_fit = np.where(improved, trial_fitness, fitness)
        
        return new_pop, new_fit, improved
    
    def _update_memory(self, population, fitness):
        """Update memory archive with best solutions found."""
        combined_pop = np.vstack([population, self._get_memory_array()])
        combined_fit = np.concatenate([fitness, self.memory_fitness])
        
        # Get indices of best memory_size solutions
        if len(combined_fit) > self.memory_size:
            best_mem_idx = np.argpartition(combined_fit, self.memory_size)[:self.memory_size]
        else:
            best_mem_idx = np.arange(len(combined_fit))
        
        # Update memory
        self.memory_fitness = combined_fit[best_mem_idx]
        self.memory = combined_pop[best_mem_idx]
        
        # Sort memory by fitness
        sort_order = np.argsort(self.memory_fitness)
        self.memory_fitness = self.memory_fitness[sort_order]
        self.memory = self.memory[sort_order]
    
    def _get_memory_array(self):
        """Get memory as array for concatenation."""
        if self.memory is None:
            return np.empty((0, self.dim))
        return self.memory
    
    def _adapt_parameters(self, improved_mask):
        """Adapt F and CR based on recent success rates."""
        self.success_buffer.append(improved_mask.mean())
        self.cr_buffer.append(self.cr)
        
        if len(self.success_buffer) > self.buffer_size:
            self.success_buffer.pop(0)
            self.cr_buffer.pop(0)
        
        success_rate = np.mean(self.success_buffer)
        
        # Adapt f_curr: decrease on success, increase on failure
        if success_rate > 0.2:
            self.f_curr *= 0.95
        elif success_rate < 0.1:
            self.f_curr *= 1.05
        self.f_curr = np.clip(self.f_curr, self.f_min, self.f_max)
        
        # Adapt cr: increase when improving, decrease when stagnating
        if success_rate > 0.25:
            self.cr = min(0.95, self.cr * 1.02)
        else:
            self.cr = max(0.3, self.cr * 0.98)
    
    def _apply_local_search_batch(self, population, fitness, stopping_condition):
        """Apply Nelder-Mead simplex search to top performers."""
        n_ls = max(1, int(self.NP * self.ls_fraction))
        top_indices = np.argpartition(fitness, n_ls)[:n_ls]
        
        for idx in top_indices:
            if stopping_condition():
                break
            
            # Run simplified Nelder-Mead
            simplex = self._init_simplex(population[idx])
            simplex_fit = self._evaluate_batch(simplex, name="simplex")
            
            # Check budget
            if len(simplex_fit) < len(simplex):
                break
            
            # Nelder-Mead iterations
            for _ in range(self.ls_iter):
                if stopping_condition():
                    break
                    
                # Sort simplex by fitness
                sort_idx = np.argsort(simplex_fit)
                simplex = simplex[sort_idx]
                simplex_fit = simplex_fit[sort_idx]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                worst = simplex[-1]
                reflected = centroid + self.ls_alpha * (centroid - worst)
                reflected = np.clip(reflected, self.f_bounds[0], self.f_bounds[1])
                reflected_fit = self._evaluate_batch(reflected.reshape(1, -1), name="reflect")
                
                if len(reflected_fit) < 1:
                    break
                    
                if reflected_fit[0] < simplex_fit[0]:
                    # Expansion
                    expanded = centroid + 2.0 * (reflected - centroid)
                    expanded = np.clip(expanded, self.f_bounds[0], self.f_bounds[1])
                    expanded_fit = self._evaluate_batch(expanded.reshape(1, -1), name="expand")
                    
                    if len(expanded_fit) >= 1 and expanded_fit[0] < reflected_fit[0]:
                        simplex[-1] = expanded
                        simplex_fit[-1] = expanded_fit[0]
                    else:
                        simplex[-1] = reflected
                        simplex_fit[-1] = reflected_fit[0]
                elif reflected_fit[0] < simplex_fit[-2]:
                    # Accept reflection
                    simplex[-1] = reflected
                    simplex_fit[-1] = reflected_fit[0]
                else:
                    # Contraction
                    contracted = centroid + 0.5 * (worst - centroid)
                    contracted = np.clip(contracted, self.f_bounds[0], self.f_bounds[1])
                    contracted_fit = self._evaluate_batch(contracted.reshape(1, -1), name="contract")
                    
                    if len(contracted_fit) >= 1 and contracted_fit[0] < simplex_fit[-1]:
                        simplex[-1] = contracted
                        simplex_fit[-1] = contracted_fit[0]
                    else:
                        # Shrink toward best
                        best = simplex[0]
                        for j in range(1, len(simplex)):
                            simplex[j] = best + 0.5 * (simplex[j] - best)
                            simplex[j] = np.clip(simplex[j], self.f_bounds[0], self.f_bounds[1])
                        simplex_fit[1:] = self._evaluate_batch(simplex[1:], name="shrink")
                        if len(simplex_fit) < len(simplex) - 1:
                            break
            
            # Update population if improved
            if simplex_fit[0] < fitness[idx]:
                population[idx] = simplex[0]
                fitness[idx] = simplex_fit[0]
        
        return population, fitness
    
    def _init_simplex(self, x_base):
        """Initialize Nelder-Mead simplex around base point."""
        simplex = np.tile(x_base, (self.dim + 1, 1))
        step = 1.0  # Initial step size
        
        for j in range(self.dim):
            simplex[j + 1, j] += step
            simplex[j + 1] = np.clip(simplex[j + 1], self.f_bounds[0], self.f_bounds[1])
        
        return simplex
    
    def _should_restart(self):
        """Determine if population should be restarted for diversity."""
        stagnation_threshold = 50 + self.dim
        
        if self.generations_without_improvement > stagnation_threshold:
            return True
        
        # Also restart if diversity is extremely low
        if self.improvement_count < 5 and self.generations_without_improvement > 20:
            return True
            
        return False
    
    def _diversity_restart(self, population, fitness):
        """Reinitialize population while preserving best solutions."""
        n_elite = max(2, self.NP // 10)
        n_from_memory = max(2, self.NP // 5)
        
        # Keep top performers
        elite_indices = np.argpartition(fitness, n_elite)[:n_elite]
        new_pop = population[elite_indices].copy()
        new_fit = fitness[elite_indices].copy()
        
        # Add from memory with perturbation
        for _ in range(n_from_memory):
            if len(self.memory) > 0:
                mem_idx = np.random.randint(min(5, len(self.memory)))
                candidate = self.memory[mem_idx] + np.random.uniform(-10, 10, self.dim)
                candidate = np.clip(candidate, self.f_bounds[0], self.f_bounds[1])
                new_pop = np.vstack([new_pop, candidate])
                new_fit = np.append(new_fit, np.inf)
        
        # Fill rest randomly
        while len(new_pop) < self.NP:
            random_candidate = np.random.uniform(
                self.f_bounds[0], self.f_bounds[1], self.dim
            )
            new_pop = np.vstack([new_pop, random_candidate])
            new_fit = np.append(new_fit, np.inf)
        
        # Evaluate new candidates
        new_fit = self._evaluate_batch(new_pop, name="restart")
        
        # Reset counters
        self.generations_without_improvement = 0
        self.improvement_count = 0
        
        return new_pop[:self.NP], new_fit[:self.NP]
