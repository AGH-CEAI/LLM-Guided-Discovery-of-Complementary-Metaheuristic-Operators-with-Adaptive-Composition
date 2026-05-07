import numpy as np


class HybridMemeticDE:
    """
    Hybrid Memetic Optimizer combining:
    - DE/current-to-pbest/1 with archive for global exploration
    - Nelder-Mead local search on elite individuals
    - Adaptive parameter control based on success history
    - Diversity-triggered restart mechanism
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(150, max(4 * dim, 60))
        self.lb = -100.0
        self.ub = 100.0
        
        # DE parameters
        self.F_init = 0.5
        self.CR_init = 0.5
        self.p_best = 0.15
        self.archive_rate = 0.3
        
        # Local search parameters
        self.local_search_interval = 8
        self.n_elite_local = 3
        self.nm_max_iter = 15
        self.nm_alpha = 1.0
        self.nm_gamma = 2.0
        self.nm_rho = 0.5
        self.nm_sigma = 0.5
        
        # Adaptation parameters
        self.success_history = []
        self.history_size = 7
        self.F = self.F_init
        self.CR = self.CR_init
        self.F_adaptive = 0.8
        self.CR_adaptive = 0.9
        
        # Restart parameters
        self.stagnation_limit = 25
        self.max_restarts = 3
        self.min_diversity = 1e-6
        
        # Internal state
        self.population = None
        self.fitness = None
        self.archive = None
        self.best_fitness = np.inf
        self.best_solution = None
        self.generation = 0
        self.stagnation_counter = 0
        self.restart_count = 0
        self.elite_indices = None
        self.func = None
        
    def __call__(self, func, stopping_condition):
        self.func = func
        self._initialize_optimization()
        
        while not stopping_condition():
            self.generation += 1
            
            trials = self._generate_trials_batched()
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
            
            if stopping_condition():
                break
            
            self._update_success_history(self.fitness, trial_fitness)
            self._select_and_update_batched(trial_fitness)
            
            if stopping_condition():
                break
            
            self._adapt_parameters()
            self._update_elite_indices()
            self._apply_local_search_to_elite()
            
            self._check_stagnation()
            
            if self.stagnation_counter >= self.stagnation_limit:
                self._restart_if_needed()
        
        return self.best_fitness, self.best_solution.copy()
    
    def _initialize_optimization(self):
        """Initialize population, archive, and tracking variables."""
        self.population = self._initialize_population()
        self.archive = self._initialize_archive()
        self.fitness = self.func(self.population)
        
        self.best_fitness = np.min(self.fitness)
        self.best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[self.best_idx].copy()
        
        self.generation = 0
        self.stagnation_counter = 0
        self.restart_count = 0
        self.F = self.F_init
        self.CR = self.CR_init
        self.success_history = []
        
        self._update_elite_indices()
    
    def _initialize_population(self):
        """Generate initial population using opposition-based sampling."""
        uniform = np.random.uniform(self.lb, self.ub, (self.np, self.dim))
        opposition = self.lb + self.ub - uniform
        combined = np.vstack([uniform, opposition])
        
        if len(combined) > self.np:
            fitness_init = self.func(combined[:self.np])
            idx = np.argsort(fitness_init)[:self.np]
            return combined[idx]
        return combined
    
    def _initialize_archive(self):
        """Initialize empty archive for non-dominated solutions."""
        return np.zeros((0, self.dim))
    
    def _generate_trials_batched(self):
        """Generate trial vectors using DE/current-to-pbest/1 with archive."""
        mutants = self._mutate_batched()
        trials = self._crossover_batched(mutants)
        trials = np.clip(trials, self.lb, self.ub)
        return trials
    
    def _mutate_batched(self):
        """
        DE/current-to-pbest/1 mutation with archive enhancement.
        Generates (NP, dim) mutant matrix.
        """
        pbest_indices = self._select_pbest_indices()
        pbest = self.population[pbest_indices]
        
        if len(self.archive) >= 2 and np.random.random() < self.archive_rate:
            archive_idx = np.random.choice(len(self.archive), self.np, replace=True)
            r1_archive = self.archive[archive_idx]
            r2_indices = np.random.choice(self.np, self.np, replace=False)
            r2 = self.population[r2_indices]
        else:
            r1_indices = np.random.choice(self.np, self.np, replace=False)
            r2_indices = np.random.choice(self.np, self.np, replace=False)
            r1 = self.population[r1_indices]
            r2 = self.population[r2_indices]
        
        base = self.population
        F_scale = np.random.uniform(0.5 * self.F, 1.2 * self.F, (self.np, 1))
        mutants = base + F_scale * (pbest - base) + self.F * (r1 - r2)
        
        mutants = np.clip(mutants, self.lb, self.ub)
        return mutants
    
    def _select_pbest_indices(self):
        """Select indices for p-best individuals."""
        n_select = max(1, int(self.p_best * self.np))
        sorted_idx = np.argsort(self.fitness)
        pbest_pool = sorted_idx[:n_select]
        return np.random.choice(pbest_pool, self.np, replace=True)
    
    def _crossover_batched(self, mutants):
        """
        Binomial crossover with dimension preservation.
        Returns (NP, dim) trial matrix.
        """
        cr_pattern = np.random.random((self.np, self.dim)) < self.CR
        
        rnd_dims = np.random.randint(0, self.dim, self.np)
        cr_pattern[np.arange(self.np), rnd_dims] = True
        
        trials = np.where(cr_pattern, mutants, self.population)
        return trials
    
    def _select_and_update_batched(self, trial_fitness):
        """
        Greedy selection and archive update.
        Operates on full population at once.
        """
        improved = trial_fitness < self.fitness
        
        self.population[improved] = self._get_trial_if_improved(improved)
        self.fitness[improved] = trial_fitness[improved]
        
        if np.any(improved):
            self._update_best_tracking()
        
        self._update_archive_batched()
        self.stagnation_counter += 1
        if np.any(improved):
            self.stagnation_counter = 0
    
    def _get_trial_if_improved(self, improved_mask):
        """Get trial vectors where improvement occurred."""
        return self.population[improved_mask]
    
    def _update_best_tracking(self):
        """Update best solution found so far."""
        current_best_idx = np.argmin(self.fitness)
        current_best_fitness = self.fitness[current_best_idx]
        
        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_idx = current_best_idx
            self.best_solution = self.population[current_best_idx].copy()
    
    def _update_archive_batched(self):
        """Update non-dominated archive with current population."""
        new_candidates = self.population[self.fitness < self.best_fitness * 1.5]
        
        if len(new_candidates) == 0:
            return
        
        combined = np.vstack([self.archive, new_candidates])
        is_nondominated = self._find_nondominated_batched(combined)
        self.archive = combined[is_nondominated]
        
        max_archive = self.np * 2
        if len(self.archive) > max_archive:
            archive_fitness = self.func(self.archive)
            top_idx = np.argsort(archive_fitness)[:max_archive]
            self.archive = self.archive[top_idx]
    
    def _find_nondominated_batched(self, solutions):
        """Find non-dominated solutions using vectorized comparison."""
        n = len(solutions)
        if n <= 1:
            return np.ones(n, dtype=bool)
        
        is_nondom = np.ones(n, dtype=bool)
        
        for i in range(n):
            if not is_nondom[i]:
                continue
            diff = solutions - solutions[i]
            worse = np.all(diff >= 0, axis=1) & np.any(diff > 0, axis=1)
            is_nondom[worse] = False
        
        return is_nondom
    
    def _update_success_history(self, fitness, trial_fitness):
        """Track success rate for parameter adaptation."""
        improved = trial_fitness < fitness
        success_rate = np.mean(improved)
        self.success_history.append(success_rate)
        
        if len(self.success_history) > self.history_size:
            self.success_history.pop(0)
    
    def _adapt_parameters(self):
        """Adapt F and CR based on success history."""
        if len(self.success_history) < 3:
            return
        
        avg_success = np.mean(self.success_history)
        
        if avg_success > 0.3:
            self.F = max(0.3, self.F * 0.9)
            self.CR = min(0.95, self.CR * 1.05)
        elif avg_success < 0.1:
            self.F = min(1.5, self.F * 1.1)
            self.CR = max(0.1, self.CR * 0.95)
        
        self.F_adaptive = 0.8 + 0.4 * np.random.random()
        self.CR_adaptive = 0.5 + 0.4 * np.random.random()
    
    def _update_elite_indices(self):
        """Update indices of elite individuals for local search."""
        n_elite = min(self.n_elite_local, self.np)
        self.elite_indices = np.argsort(self.fitness)[:n_elite]
    
    def _apply_local_search_to_elite(self):
        """Apply Nelder-Mead local search to elite individuals."""
        if self.generation % self.local_search_interval != 0:
            return
        
        for idx in self.elite_indices:
            improved_solution = self._nelder_mead_refine(self.population[idx])
            
            if not np.any(np.isnan(improved_solution)):
                self.population[idx] = improved_solution
        
        self.fitness = self.func(self.population)
        self._update_best_tracking()
    
    def _nelder_mead_refine(self, x):
        """
        Nelder-Mead local search on single candidate.
        Returns refined solution vector.
        """
        simplex_size = self.dim + 1
        simplex = np.tile(x, (simplex_size, 1))
        
        for i in range(1, simplex_size):
            simplex[i] = x + np.random.uniform(-2.0, 2.0, self.dim)
            simplex[i] = np.clip(simplex[i], self.lb, self.ub)
        
        simplex_fitness = self.func(simplex)
        
        for _ in range(self.nm_max_iter):
            sorted_idx = np.argsort(simplex_fitness)
            simplex = simplex[sorted_idx]
            simplex_fitness = simplex_fitness[sorted_idx]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            reflected = centroid + self.nm_alpha * (centroid - simplex[-1])
            reflected = np.clip(reflected, self.lb, self.ub)
            reflected_fitness = self.func(reflected.reshape(1, -1))[0]
            
            if simplex_fitness[0] <= reflected_fitness < simplex_fitness[-2]:
                simplex[-1] = reflected
                simplex_fitness[-1] = reflected_fitness
                continue
            
            if reflected_fitness < simplex_fitness[0]:
                expanded = centroid + self.nm_gamma * (reflected - centroid)
                expanded = np.clip(expanded, self.lb, self.ub)
                expanded_fitness = self.func(expanded.reshape(1, -1))[0]
                
                if expanded_fitness < reflected_fitness:
                    simplex[-1] = expanded
                    simplex_fitness[-1] = expanded_fitness
                else:
                    simplex[-1] = reflected
                    simplex_fitness[-1] = reflected_fitness
                continue
            
            contracted = centroid - self.nm_rho * (centroid - simplex[-1])
            contracted = np.clip(contracted, self.lb, self.ub)
            contracted_fitness = self.func(contracted.reshape(1, -1))[0]
            
            if contracted_fitness < simplex_fitness[-1]:
                simplex[-1] = contracted
                simplex_fitness[-1] = contracted_fitness
            else:
                simplex[1:] = simplex[0] + self.nm_sigma * (simplex[1:] - simplex[0])
                simplex[1:] = np.clip(simplex[1:], self.lb, self.ub)
                simplex_fitness[1:] = self.func(simplex[1:])
        
        return simplex[0]
    
    def _check_stagnation(self):
        """Check if search has stagnated based on diversity."""
        if self.stagnation_counter < self.stagnation_limit:
            return
        
        diversity = self._compute_population_diversity()
        
        if diversity < self.min_diversity:
            self.F = min(1.2, self.F * 1.3)
            self.CR = max(0.2, self.CR * 0.8)
            self.stagnation_counter = 0
    
    def _compute_population_diversity(self):
        """Compute population diversity using average pairwise distance."""
        if len(self.population) < 2:
            return 0.0
        
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _restart_if_needed(self):
        """Perform adaptive restart with diversity injection."""
        if self.restart_count >= self.max_restarts:
            self.stagnation_counter = 0
            return
        
        best_keep = min(5, self.np // 10)
        best_indices = np.argsort(self.fitness)[:best_keep]
        elites = self.population[best_indices].copy()
        
        self.population = self._initialize_population()
        self.population[:best_keep] = elites
        
        self.fitness = self.func(self.population)
        self.archive = self._initialize_archive()
        
        self.restart_count += 1
        self.stagnation_counter = 0
        self.F = min(1.0, self.F * 1.2)
        
        self._update_best_tracking()
