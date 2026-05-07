```python
import numpy as np


class ArchipelagoMemeticDE:
    """
    Island-model Memetic Differential Evolution with ring topology migration,
    adaptive scaling factors, and Nelder-Mead local search on archipelago bests.
    
    Key design choices (different from CulturalDE, CMA-PSO, AdaptiveTopologyPSO):
    - Ring topology migration instead of centralized archive
    - Island-based subpopulations with periodic best-migration
    - Nelder-Mead simplex refinement on archipelago global bests (probabilistic)
    - Success-history adaptation with linear population size reduction
    - Opposition-based initialization for faster convergence
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower = -100.0
        self.upper = 100.0
        
        # Population sizing: 6*dim islands, 5 individuals per island
        self.islands = 6 * max(1, dim // 10)
        self.island_size = 5
        self.NP = self.islands * self.island_size
        
        # Migration ring: each island has a neighbor
        self.migration_interval = kwargs.get('migration_interval', 7)
        self.migration_rate = kwargs.get('migration_rate', 0.2)
        
        # Local search parameters
        self.ls_prob = kwargs.get('ls_prob', 0.08)
        self.ls_max_evals_ratio = kwargs.get('ls_max_evals_ratio', 0.03)
        
        # Adaptation
        self.f_min = 0.1
        self.f_max = 1.0
        self.cr_min = 0.1
        self.cr_max = 0.9
        
        # Restart
        self.stagnation_threshold = kwargs.get('stagnation_threshold', 15)
        self.diversity_threshold = kwargs.get('diversity_threshold', 0.05)
        
        # Internal state
        self.population = None
        self.fitness = None
        self.best_fitness = None
        self.best_position = None
        self.f_values = None
        self.cr_values = None
        self.success_history_f = None
        self.success_history_cr = None
        self.archive = None
        self.island_best_fitness = None
        self.island_best_positions = None
        self.stagnation_counter = 0
        self.generation = 0
        self.total_evals = 0
        
    def _initialize_population_batch(self):
        """Initialize with opposition-based learning across islands."""
        pop = np.random.uniform(self.lower, self.upper, (self.NP, self.dim))
        
        # Generate opposites
        opposites = self.lower + self.upper - pop
        
        # Batch evaluate both
        fit_original = self._eval_wrapper(pop)
        fit_opposite = self._eval_wrapper(opposites)
        
        # Select best half from union
        all_pop = np.vstack([pop, opposites])
        all_fit = np.concatenate([fit_original, fit_opposite])
        
        # Sort and keep top NP
        indices = np.argsort(all_fit)[:self.NP]
        self.population = all_pop[indices]
        self.fitness = all_fit[indices]
        
        # Initialize island structure
        self._init_island_structure()
        
        # Initialize adaptive parameters
        self.f_values = np.random.uniform(self.f_min, self.f_max, self.NP)
        self.cr_values = np.random.uniform(self.cr_min, self.cr_max, self.NP)
        self.success_history_f = []
        self.success_history_cr = []
        
        # Archive for p-best mutation
        self.archive = self.population.copy()
        
        # Track best
        best_idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[best_idx]
        self.best_position = self.population[best_idx].copy()
        
    def _init_island_structure(self):
        """Set up island topology and island best trackers."""
        self.island_best_fitness = np.full(self.islands, np.inf)
        self.island_best_positions = np.zeros((self.islands, self.dim))
        
        for i in range(self.islands):
            start = i * self.island_size
            end = start + self.island_size
            island_pop = self.population[start:end]
            island_fit = self.fitness[start:end]
            best_local = np.argmin(island_fit)
            self.island_best_fitness[i] = island_fit[best_local]
            self.island_best_positions[i] = island_pop[best_local]
            
    def _eval_wrapper(self, pop):
        """Evaluate population batch with budget awareness."""
        if len(pop) == 0:
            return np.array([])
        
 fitness = self._func(pop)
        
        # Handle truncated/NaN returns near budget exhaustion
        if len(fitness) < len(pop):
            actual = len(fitness)
            fitness = np.resize(fitness, actual)
            if np.any(np.isnan(fitness)):
                nan_mask = np.isnan(fitness)
                fitness[nan_mask] = np.inf
            return fitness
        
        if np.any(np.isnan(fitness)):
            nan_mask = np.isnan(fitness)
            fitness[nan_mask] = np.inf
            
        self.total_evals += len(pop)
        return fitness
        
    def _mutate_current_to_pbest_batch(self):
        """Current-to-pbest/1/bin mutation with ring-influenced targets."""
        trials = np.empty((self.NP, self.dim))
        
        for i in range(self.NP):
            island_idx = i // self.island_size
            island_start = island_idx * self.island_size
            island_end = island_start + self.island_size
            
            # Get island members (for p-best selection)
            island_members = self.population[island_start:island_end]
            island_fits = self.fitness[island_start:island_end]
            
            # Also consider neighbor island's best
            neighbor_idx = (island_idx + 1) % self.islands
            neighbor_best = self.island_best_positions[neighbor_idx]
            
            # Combine candidates for p-best
            combined_pop = np.vstack([island_members, neighbor_best])
            combined_fits = np.concatenate([island_fits, [self.island_best_fitness[neighbor_idx]]])
            
            # Select p-best (top 20%)
            p_size = max(1, len(combined_fits) // 5)
            p_indices = np.argsort(combined_fits)[:p_size]
            pbest = combined_pop[p_indices[np.random.randint(p_size)]]
            
            # Select r1, r2 from archive + population (excluding current)
            candidates = np.vstack([self.population, self.archive])
            mask = np.ones(len(candidates), dtype=bool)
            mask[i] = False
            candidates = candidates[mask]
            
            if len(candidates) < 2:
                candidates = self.population
                mask = np.ones(len(candidates), dtype=bool)
                mask[i] = False
                candidates = candidates[mask]
            
            indices = np.random.choice(len(candidates), size=2, replace=False)
            r1, r2 = candidates[indices]
            
            # Current-to-pbest mutation
            f = self.f_values[i]
            trials[i] = self.population[i] + f * (pbest - self.population[i]) + f * (r1 - r2)
            
        return trials
        
    def _crossover_batch(self, targets, mutants):
        """Binomial crossover with dimension-wise probability."""
        trials = np.empty((self.NP, self.dim))
        
        for i in range(self.NP):
            cr = self.cr_values[i]
            j_rand = np.random.randint(self.dim)
            
            mask = np.random.rand(self.dim) < cr
            mask[j_rand] = True
            trials[i] = np.where(mask, mutants[i], targets[i])
            
        return trials
        
    def _clip_to_bounds_batch(self, pop):
        """Hard bounds enforcement."""
        return np.clip(pop, self.lower, self.upper)
        
    def _select_survivors_batch(self, targets, target_fitness, trials, trial_fitness):
        """Greedy selection, update archive and island bests."""
        improved_mask = trial_fitness < target_fitness
        
        # Update population
        self.population = np.where(improved_mask[:, np.newaxis], trials, targets)
        self.fitness = np.where(improved_mask, trial_fitness, target_fitness)
        
        # Update archive with replaced individuals (worse ones)
        worse_mask = ~improved_mask
        if np.any(worse_mask):
            worse_individuals = targets[worse_mask]
            self.archive = np.vstack([self.archive, worse_individuals])
            
            # Limit archive size
            max_archive = self.NP * 5
            if len(self.archive) > max_archive:
                remove_n = len(self.archive) - max_archive
                remove_idx = np.random.choice(len(self.archive), remove_n, replace=False)
                self.archive = np.delete(self.archive, remove_idx, axis=0)
        
        # Update island bests
        for i in range(self.islands):
            start = i * self.island_size
            end = start + self.island_size
            island_fit = self.fitness[start:end]
            best_local = np.argmin(island_fit)
            
            if island_fit[best_local] < self.island_best_fitness[i]:
                self.island_best_fitness[i] = island_fit[best_local]
                self.island_best_positions[i] = self.population[start + best_local].copy()
        
        # Update global best
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_position = self.population[best_idx].copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
            
        return improved_mask
        
    def _adapt_parameters_batch(self, improved_mask):
        """Success-history adaptation of F and CR."""
        if not np.any(improved_mask):
            return
            
        # Collect successful F and CR values
        f_successful = self.f_values[improved_mask]
        cr_successful = self.cr_values[improved_mask]
        
        self.success_history_f.extend(f_successful.tolist())
        self.success_history_cr.extend(cr_successful.tolist())
        
        # Keep history bounded
        max_history = self.NP * 5
        if len(self.success_history_f) > max_history:
            self.success_history_f = self.success_history_f[-max_history:]
            self.success_history_cr = self.success_history_cr[-max_history:]
        
        # Reinitialize for next generation
        for i in range(self.NP):
            if len(self.success_history_f) > 0:
                # Weighted random from history
                weights = np.exp(-0.05 * np.arange(len(self.success_history_f)))
                weights /= weights.sum()
                
                self.f_values[i] = np.random.choice(self.success_history_f, p=weights)
                self.cr_values[i] = np.random.choice(self.success_history_cr, p=weights)
            else:
                self.f_values[i] = np.random.uniform(self.f_min, self.f_max)
                self.cr_values[i] = np.random.uniform(self.cr_min, self.cr_max)
                
            # Add jitter
            self.f_values[i] += np.random.uniform(-0.1, 0.1)
            self.f_values[i] = np.clip(self.f_values[i], 0.0, 2.0)
            
    def _migrate_between_islands(self):
        """Ring topology migration: each island sends best to next."""
        for i in range(self.islands):
            donor_island = i
            receiver_island = (i + 1) % self.islands
            
            # Send best from donor to random position in receiver
            start = donor_island * self.island_size
            end = start + self.island_size
            donor_pop = self.population[start:end]
            donor_fit = self.fitness[start:end]
            best_donor = np.argmin(donor_fit)
            
            # Receive into random position
            recv_start = receiver_island * self.island_size
            recv_idx = recv_start + np.random.randint(self.island_size)
            
            # Only migrate if receiver is worse
            if donor_fit[best_donor] < self.fitness[recv_idx]:
                self.population[recv_idx] = donor_pop[best_donor].copy()
                self.fitness[recv_idx] = donor_fit[best_donor]
                
    def _apply_nelder_mead_batch(self):
        """Apply Nelder-Mead simplex refinement to top archipelago bests."""
        # Select islands for local search (probabilistic, top performers)
        n_ls_islands = max(1, self.islands // 4)
        
        # Sort by fitness, pick top n
        sorted_islands = np.argsort(self.island_best_fitness)[:n_ls_islands]
        
        for island_idx in sorted_islands:
            if np.random.rand() > self.ls_prob:
                continue
                
            # Get island best as simplex centroid
            x_best = self.island_best_positions[island_idx].copy()
            f_best = self.island_best_fitness[island_idx]
            
            # Build simplex around x_best
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Generate simplex vertices
            simplex = np.empty((self.dim + 1, self.dim))
            simplex[0] = x_best
            
            # Perturb each dimension
            for j in range(self.dim):
                vertex = x_best.copy()
                step = 0.5 * (self.upper - self.lower) / 100.0
                vertex[j] += step if x_best[j] < 0 else -step
                vertex = np.clip(vertex, self.lower, self.upper)
                simplex[j + 1] = vertex
            
            # Evaluate simplex
            simplex_fit = self._eval_wrapper(simplex)
            if len(simplex_fit) < len(simplex):
                break
                
            # Check stopping condition after LM evaluation
            if self._stopping_condition():
                return
                
            # Sort simplex
            order = np.argsort(simplex_fit)
            simplex = simplex[order]
            simplex_fit = simplex_fit[order]
            
            # Nelder-Mead iterations
            for _ in range(max(3, self.dim // 2)):
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                worst = simplex[-1]
                reflected = centroid + alpha * (centroid - worst)
                reflected = np.clip(reflected, self.lower, self.upper)
                reflected_fit = self._eval_wrapper(reflected.reshape(1, -1))[0]
                
                if self._stopping_condition():
                    return
                    
                if reflected_fit >= simplex_fit[0] and reflected_fit < simplex_fit[-2]:
                    simplex[-1] = reflected
                    simplex_fit[-1] = reflected_fit
                elif reflected_fit < simplex_fit[0]:
                    # Expansion
                    expanded = centroid + gamma * (reflected - centroid)
                    expanded = np.clip(expanded, self.lower, self.upper)
                    expanded_fit = self._eval_wrapper(expanded.reshape(1, -1))[0]
                    
                    if self._stopping_condition():
                        return
                        
                    if expanded_fit < reflected_fit:
                        simplex[-1] = expanded
                        simplex_fit[-1] = expanded_fit
                    else:
                        simplex[-1] = reflected
                        simplex_fit[-1] = reflected_fit
                else:
                    # Contraction
                    contracted = centroid + rho * (worst - centroid)
                    contracted = np.clip(contracted, self.lower, self.upper)
                    contracted_fit = self._eval_wrapper(contracted.reshape(1, -1))[0]
                    
                    if self._stopping_condition():
                        return
                        
                    if contracted_fit < simplex_fit[-1]:
                        simplex[-1] = contracted
                        simplex_fit[-1] = contracted_fit
                    else:
                        # Shrink
                        for k in range(1, len(simplex)):
                            simplex[k] = simplex[0] + sigma * (simplex[k] - simplex[0])
                            simplex[k] = np.clip(simplex[k], self.lower, self.upper)
                        simplex_fit[1:] = self._eval_wrapper(simplex[1:])[0]
                        
                        if self._stopping_condition():
                            return
                            
                # Sort
                order = np.argsort(simplex_fit)
                simplex = simplex[order]
                simplex_fit = simplex_fit[order]
                
                if simplex_fit[0] < f_best:
                    f_best = simplex_fit[0]
                    x_best = simplex[0].copy()
                    
                    # Update island best if improved
                    if f_best < self.island_best_fitness[island_idx]:
                        self.island_best_fitness[island_idx] = f_best
                        self.island_best_positions[island_idx] = x_best.copy()
                        
                        # Update global best
                        if f_best < self.best_fitness:
                            self.best_fitness = f_best
                            self.best_position = x_best.copy()
                            self.stagnation_counter = 0
                            
            # Update population with improved individual
            start = island_idx * self.island_size
            end = start + self.island_size
            worst_in_island = np.argmax(self.fitness[start:end])
            global_worst_idx = start + worst_in_island
            
            if f_best < self.fitness[global_worst_idx]:
                self.population[global_worst_idx] = x_best.copy()
                self.fitness[global_worst_idx] = f_best
                
    def _compute_diversity_batch(self):
        """Compute fitness diversity (coefficient of variation)."""
        if len(self.fitness) < 2:
            return 0.0
        return np.std(self.fitness) / (np.mean(self.fitness) + 1e-10)
        
    def _should_restart(self):
        """Check if restart conditions are met."""
        diversity = self._compute_diversity_batch()
        return (self.stagnation_counter >= self.stagnation_threshold or 
                diversity < self.diversity_threshold)
        
    def _restart_population(self):
        """Reinitialize with diverse population when stagnated."""
        # Keep current best
        best = self.best_position.copy()
        best_fit = self.best_fitness
        
        # Generate new random population
        new_pop = np.random.uniform(self.lower, self.upper, (self.NP, self.dim))
        
        # Inject best into random positions
        n_best_copies = max(1, self.NP // 10)
        inject_indices = np.random.choice(self.NP, n_best_copies, replace=False)
        for idx in inject_indices:
            new_pop[idx] = best.copy()
            
        # Evaluate new population
        new_fitness = self._eval_wrapper(new_pop)
        if len(new_fitness) < self.NP:
            new_fitness = np.resize(new_fitness, self.NP)
            new_fitness[len(new_fitness):] = np.inf
            
        self.population = new_pop
        self.fitness = new_fitness
        
        # Reinitialize island structure
        self._init_island_structure()
        
        # Reset archive
        self.archive = self.population.copy()
        
        # Reset adaptation
        self.f_values = np.random.uniform(self.f_min, self.f_max, self.NP)
        self.cr_values = np.random.uniform(self.cr_min, self.cr_max, self.NP)
        self.success_history_f = []
        self.success_history_cr = []
        
        self.stagnation_counter = 0
        
    def _stopping_condition(self):
        """Placeholder for external stopping condition check."""
        return False
        
    def __call__(self, func, stopping_condition):
        """
        Run optimization.
        
        Args:
            func: Fitness function f(x) where x is (N, dim) array
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        self._func = func
        self._stopping_condition = stopping_condition
        self.total_evals = 0
        self.generation = 0
        
        # Initialize
        self._initialize_population_batch()
        
        # Main loop
        while not stopping_condition():
            self.generation += 1
            
            # Build trial population
            mutants = self._mutate_current_to_pbest_batch()
            trials = self._crossover_batch(self.population, mutants)
            trials = self._clip_to_bounds_batch(trials)
            
            # Evaluate trials
            trial_fitness = self._eval_wrapper(trials)
            if len(trial_fitness) < len(trials):
                # Budget exhausted
                break
                
            if stopping_condition():
                break
                
            # Selection
            improved = self._select_survivors_batch(
                self.population, self.fitness, trials, trial_fitness
            )
            
            # Adaptation
            self._adapt_parameters_batch(improved)
            
            # Migration
            if self.generation % self.migration_interval == 0:
                self._migrate_between_islands()
                
            # Local search on archipelago bests
            self._apply_nelder_mead_batch()
            
            # Restart check
            if self._should_restart():
                self._restart_population()
                
        return self.best_fitness, self.best_position
```