import numpy as np


class AdaptiveMemeticDifferentialEvolution:
    """
    A memetic optimizer combining DE-style global search with Nelder-Mead
    local refinement. Features adaptive operator selection, diversity-driven
    restarts, and archive-based exploitation.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = max(50, min(300, 5 * dim))
        
        # Mutation operator pool with adaptive weights
        self.operator_weights = np.array([0.3, 0.25, 0.25, 0.2])
        self.operator_names = ['current_to_pbest_1', 'rand_1', 'current_to_rand_1', 'best_1_exp']
        
        # Archive for local search candidates
        self.archive_size = max(5, dim // 3)
        
        # Step size adaptation state
        self.f = 0.7
        self.cr = 0.85
        self.f_history = [self.f]
        self.cr_history = [self.cr]
        self.success_f = []
        self.success_cr = []
        
        # Operator reward tracking
        self.operator_successes = np.ones(4)
        self.operator_attempts = np.ones(4)
        
        # Diversity monitoring
        self.diversity_threshold = 1e-5 * dim
        self.stagnation_gen = 0
        self.max_stagnation = int(30 + dim * 0.5)
        
        # Nelder-Mead local search parameters
        self.nm_max_iter = 15
        self.nm_alpha = 1.0
        self.nm_gamma = 2.0
        self.nm_rho = 0.5
        self.nm_sigma = 0.5
        self.local_search_interval = max(3, self.np // 15)
        
        # Best solution tracking
        self.f_best = np.inf
        self.x_best = None
        self.gen_counter = 0
        
        # Bounds
        self.lower = -100.0
        self.upper = 100.0
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop with batched operations."""
        if stopping_condition():
            return self.f_best, self.x_best
        
        # Initialize population and best tracking
        population, fitness = self._initialize_population(func)
        self.f_best = np.min(fitness)
        self.x_best = population[np.argmin(fitness)].copy()
        
        # Main evolution loop
        while not stopping_condition():
            self.gen_counter += 1
            
            # Apply Nelder-Mead local search to archive periodically
            if self.gen_counter % self.local_search_interval == 0:
                population, fitness = self._apply_local_search_to_archive(
                    population, fitness, func, stopping_condition
                )
                if stopping_condition():
                    break
            
            # Generate mutant vectors using adaptive operator selection
            mutants = self._mutate_batch(population, np.arange(self.np))
            
            # Create trial vectors via crossover
            trials = self._crossover_population_batch(population, mutants)
            
            # Clip to bounds and evaluate
            trials = self._clip_to_bounds_batch(trials)
            trial_fitness = self._evaluate_batch(trials, func)
            
            # Handle budget exhaustion
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
                if len(trial_fitness) == 0:
                    break
            
            # Select survivors and update best
            population, fitness = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.f_best:
                self.f_best = fitness[current_best_idx]
                self.x_best = population[current_best_idx].copy()
            
            # Adapt step sizes based on successful mutations
            self._adapt_step_size()
            
            # Adapt crossover rate
            self._adapt_crossover_rate()
            
            # Adapt mutation operator weights
            self._adapt_operator_weights()
            
            # Check diversity and restart if stagnant
            if self._check_diversity_and_restart(population, func):
                population, fitness = self._initialize_population(func, restart=True)
            
            if stopping_condition():
                break
        
        return self.f_best, self.x_best
    
    def _initialize_population(self, func, restart=False):
        """Initialize random population within bounds, evaluating all at once."""
        population = np.random.uniform(
            self.lower, self.upper, size=(self.np, self.dim)
        )
        
        if restart and self.x_best is not None:
            n_replace = self.np // 2
            replace_indices = np.random.choice(self.np, n_replace, replace=False)
            for idx in replace_indices:
                population[idx] = self.x_best + np.random.uniform(
                    -10, 10, size=self.dim
                )
            population = np.clip(population, self.lower, self.upper)
        
        fitness = func(population)
        if len(fitness) < self.np:
            fitness = np.pad(fitness, (0, self.np - len(fitness)), constant_values=np.inf)
        
        fitness = np.where(np.isnan(fitness), np.inf, fitness)
        return population, fitness
    
    def _select_mutation_strategy(self):
        """Select mutation strategy based on adaptive weights."""
        return np.random.choice(len(self.operator_weights), p=self.operator_weights)
    
    def _select_indices_excluding(self, n, exclude_list, k):
        """Select k unique indices from [0, n) excluding specified indices."""
        valid = np.setdiff1d(np.arange(n), np.array(exclude_list))
        return np.random.choice(valid, size=k, replace=False)
    
    def _mutate_single(self, population, target_idx, strategy):
        """Generate one mutant vector using specified DE strategy."""
        n = len(population)
        p = 0.1
        pbest_n = max(1, int(p * n))
        pbest_indices = np.argpartition(population, pbest_n)[:pbest_n]
        pbest_idx = np.random.choice(pbest_indices)
        
        r1, r2 = self._select_indices_excluding(n, [target_idx], 2)
        
        if strategy == 0:  # current-to-pbest/1 with archive
            archive = getattr(self, 'local_archive', np.empty((0, self.dim)))
            all_indices = np.arange(n)
            valid_donors = np.setdiff1d(all_indices, [target_idx, pbest_idx])
            
            if len(archive) > 0:
                archive_indices = np.arange(len(archive)) + n
                all_donors = np.concatenate([valid_donors, archive_indices])
            else:
                all_donors = valid_donors
            
            if len(all_donors) >= 2:
                r_donors = np.random.choice(all_donors, size=2, replace=False)
                r2 = r_donors[0]
                r3 = r_donors[1]
                
                if r_donors[0] >= n:
                    donor1 = archive[r_donors[0] - n]
                else:
                    donor1 = population[r_donors[0]]
                    
                if r_donors[1] >= n:
                    donor2 = archive[r_donors[1] - n]
                else:
                    donor2 = population[r_donors[1]]
                
                mutant = population[target_idx] + self.f * (
                    population[pbest_idx] - population[target_idx]
                ) + self.f * (donor1 - donor2)
            else:
                mutant = population[target_idx] + self.f * (
                    population[pbest_idx] - population[target_idx]
                ) + self.f * (population[r1] - population[r2])
        
        elif strategy == 1:  # rand/1
            r3 = self._select_indices_excluding(n, [target_idx, r1, r2], 1)[0]
            mutant = population[r1] + self.f * (population[r2] - population[r3])
        
        elif strategy == 2:  # current-to-rand/1
            r3, r4 = self._select_indices_excluding(n, [target_idx], 2)
            mutant = (
                population[target_idx] +
                self.f * (population[r1] - population[target_idx]) +
                self.f * (population[r2] - population[r3])
            )
        
        else:  # best/1/exp
            best_idx = np.argmin(getattr(self, 'fitness', np.array([np.inf] * n)))
            r1, r2 = self._select_indices_excluding(n, [best_idx], 2)
            mutant = population[best_idx] + self.f * (population[r1] - population[r2])
        
        return mutant
    
    def _mutate_batch(self, population, target_indices):
        """Generate mutant vectors for all targets using adaptive operators."""
        n_trials = len(target_indices)
        mutants = np.empty((n_trials, self.dim))
        
        for i, idx in enumerate(target_indices):
            strategy = self._select_mutation_strategy()
            mutants[i] = self._mutate_single(population, idx, strategy)
            
            # Track operator attempts
            self.operator_attempts[strategy] += 1
        
        return mutants
    
    def _crossover_population_batch(self, population, mutants):
        """Apply binomial crossover to create trial vectors for entire population."""
        trials = np.empty_like(population)
        dim = self.dim
        
        for i in range(len(population)):
            j_rand = np.random.randint(dim)
            cr_mask = np.random.random(dim) < self.cr
            cr_mask[j_rand] = True
            trials[i] = np.where(cr_mask, mutants[i], population[i])
        
        return trials
    
    def _clip_to_bounds_batch(self, population):
        """Clip all candidates to search bounds."""
        return np.clip(population, self.lower, self.upper)
    
    def _evaluate_batch(self, population, func):
        """Evaluate fitness for entire population in one batched call."""
        fitness = func(population)
        
        if len(fitness) < len(population):
            return fitness[:len(fitness)]
        
        fitness = np.array(fitness)
        fitness = np.where(np.isnan(fitness), np.inf, fitness)
        return fitness
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Select better individual between target and trial for each position."""
        better_mask = trial_fitness < fitness
        
        new_population = np.where(better_mask[:, np.newaxis], trials, population)
        new_fitness = np.where(better_mask, trial_fitness, fitness)
        
        # Track successful operators for adaptation
        successful_trials = trials[better_mask]
        if len(successful_trials) > 0:
            self._record_successful_mutations(population[better_mask], successful_trials)
        
        return new_population, new_fitness
    
    def _record_successful_mutations(self, targets, trials):
        """Record successful mutations for step size adaptation."""
        diff = trials - targets
        nonzero_mask = np.abs(diff) > 1e-10
        
        for i in range(len(targets)):
            mask = nonzero_mask[i]
            if np.any(mask):
                successful_change = diff[i, mask]
                self.success_f.extend(np.abs(successful_change))
                self.success_cr.extend([np.sum(mask) / self.dim])
    
    def _apply_local_search_to_archive(self, population, fitness, func, stopping_condition):
        """Apply Nelder-Mead local search to elite archive members."""
        n_archive = min(self.archive_size, self.np // 4)
        sorted_indices = np.argsort(fitness)[:n_archive]
        
        self.local_archive = population[sorted_indices].copy()
        
        for archive_idx in range(len(self.local_archive)):
            if stopping_condition():
                break
            
            x_current = self.local_archive[archive_idx].copy()
            f_current = fitness[sorted_indices[archive_idx]]
            
            x_new, f_new = self._nelder_mead_step(x_current, f_current, population, func)
            
            if f_new < f_current and not np.isinf(f_new):
                self.local_archive[archive_idx] = x_new
                
                # Update population if improvement found
                target_idx = sorted_indices[archive_idx]
                if f_new < fitness[target_idx]:
                    population[target_idx] = x_new
                    fitness[target_idx] = f_new
        
        # Merge improved archive back into population
        population, fitness = self._merge_archive_to_population(
            population, fitness, func
        )
        
        return population, fitness
    
    def _nelder_mead_step(self, x_centroid, f_centroid, population, func):
        """Perform one iteration of Nelder-Mead local search."""
        # Build simplex from centroid and nearby population members
        n_simplex = self.dim + 1
        simplex_indices = np.argsort(
            np.sum((population - x_centroid) ** 2, axis=1)
        )[:n_simplex]
        
        simplex = np.vstack([x_centroid, population[simplex_indices]])
        simplex_f = np.array([f_centroid] + [fitness[i] for i in simplex_indices])
        
        # Sort simplex by fitness
        order = np.argsort(simplex_f)
        simplex = simplex[order]
        simplex_f = simplex_f[order]
        
        x_best = simplex[0]
        x_worst = simplex[-1]
        x_second_worst = simplex[-2]
        
        # Compute centroid of all points except worst
        x_centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_reflect = x_centroid + self.nm_alpha * (x_centroid - x_worst)
        x_reflect = np.clip(x_reflect, self.lower, self.upper)
        f_reflect = func(x_reflect.reshape(1, -1))[0]
        
        if np.isnan(f_reflect):
            f_reflect = np.inf
        
        # Expansion or contraction
        if f_reflect < simplex_f[0]:
            x_expand = x_centroid + self.nm_gamma * (x_reflect - x_centroid)
            x_expand = np.clip(x_expand, self.lower, self.upper)
            f_expand = func(x_expand.reshape(1, -1))[0]
            
            if np.isnan(f_expand):
                f_expand = np.inf
            
            if f_expand < f_reflect:
                return x_expand, f_expand
            return x_reflect, f_reflect
        
        if f_reflect < simplex_f[-2]:
            return x_reflect, f_reflect
        
        # Contraction
        x_contract = x_centroid - self.nm_rho * (x_centroid - x_worst)
        x_contract = np.clip(x_contract, self.lower, self.upper)
        f_contract = func(x_contract.reshape(1, -1))[0]
        
        if np.isnan(f_contract):
            f_contract = np.inf
        
        if f_contract < simplex_f[-1]:
            return x_contract, f_contract
        
        # Shrink
        x_best = simplex[0]
        new_simplex = np.empty_like(simplex)
        new_simplex[0] = x_best
        
        for i in range(1, len(simplex)):
            new_simplex[i] = x_best + self.nm_sigma * (simplex[i] - x_best)
        
        new_fitness = func(new_simplex.reshape(-1, self.dim))
        if len(new_fitness) < len(new_simplex):
            new_fitness = np.pad(
                new_fitness, (0, len(new_simplex) - len(new_fitness)), constant_values=np.inf
            )
        new_fitness = np.where(np.isnan(new_fitness), np.inf, new_fitness)
        
        best_idx = np.argmin(new_fitness)
        return new_simplex[best_idx], new_fitness[best_idx]
    
    def _merge_archive_to_population(self, population, fitness, func):
        """Merge locally-improved archive back into population."""
        for archive_x in self.local_archive:
            f_archive = func(archive_x.reshape(1, -1))[0]
            if np.isnan(f_archive):
                continue
            
            worst_idx = np.argmax(fitness)
            if f_archive < fitness[worst_idx]:
                population[worst_idx] = archive_x
                fitness[worst_idx] = f_archive
        
        return population, fitness
    
    def _adapt_step_size(self):
        """Adapt mutation scale factor based on successful mutations."""
        if len(self.success_f) >= 5:
            mean_successful_f = np.mean(self.success_f[-50:])
            self.f = 0.9 * self.f + 0.1 * mean_successful_f
            self.f = np.clip(self.f, 0.1, 1.0)
            self.success_f = self.success_f[-100:]
        
        self.f_history.append(self.f)
    
    def _adapt_crossover_rate(self):
        """Adapt crossover probability based on recent successful crossovers."""
        if len(self.success_cr) >= 5:
            mean_successful_cr = np.mean(self.success_cr[-50:])
            self.cr = 0.9 * self.cr + 0.1 * mean_successful_cr
            self.cr = np.clip(self.cr, 0.0, 1.0)
            self.success_cr = self.success_cr[-100:]
        
        self.cr_history.append(self.cr)
    
    def _adapt_operator_weights(self):
        """Adapt mutation operator weights based on success rates."""
        success_rates = self.operator_successes / (self.operator_attempts + 1e-10)
        success_rates = np.clip(success_rates, 0.01, 0.99)
        
        # Exponential moving average of success rates
        eta = 0.1
        self.operator_weights = (
            (1 - eta) * self.operator_weights + eta * success_rates
        )
        self.operator_weights /= self.operator_weights.sum()
    
    def _compute_diversity(self, population):
        """Compute population diversity as mean nearest-neighbor distance."""
        if len(population) < 2:
            return 0.0
        
        # Sample for efficiency on large populations
        sample_size = min(50, len(population))
        if sample_size < len(population):
            indices = np.random.choice(len(population), sample_size, replace=False)
            sample = population[indices]
        else:
            sample = population
        
        # Compute pairwise distances
        dists = np.linalg.norm(sample[:, np.newaxis] - sample[np.newaxis, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Mean of minimum distances
        mean_nn_dist = np.mean(np.min(dists, axis=1))
        return mean_nn_dist / np.sqrt(self.dim)
    
    def _check_diversity_and_restart(self, population, func):
        """Check diversity and trigger restart if population has stagnated."""
        diversity = self._compute_diversity(population)
        
        if diversity < self.diversity_threshold:
            self.stagnation_gen += 1
        else:
            self.stagnation_gen = 0
        
        if self.stagnation_gen >= self.max_stagnation:
            self.stagnation_gen = 0
            self._reset_adaptation_state()
            return True
        
        return False
    
    def _reset_adaptation_state(self):
        """Reset adaptation parameters during restart."""
        self.f = np.clip(0.5 + np.random.uniform(-0.2, 0.3), 0.1, 1.0)
        self.cr = np.clip(0.8 + np.random.uniform(-0.2, 0.2), 0.0, 1.0)
        self.operator_weights = np.ones(4) / 4
        self.success_f = []
        self.success_cr = []
        self.local_archive = np.empty((0, self.dim))
