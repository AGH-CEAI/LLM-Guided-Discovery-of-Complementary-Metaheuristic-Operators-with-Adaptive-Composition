import numpy as np


class HybridDESimplexOptimizer:
    """
    Hybrid Differential Evolution with Simplex-based local refinement.
    
    Key features:
    - DE/current-to-pbest/1 mutation with adaptive F and CR (SHADE-like)
    - Simplex-inspired local search applied to top candidates
    - Archive of successful parameters for adaptation
    - Stagnation detection with strategic restarts
    - Multiple mutation strategies with adaptive selection
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np_size = min(max(6 * dim, 60), 300)
        self.archive_size = self.np_size
        self.p_best_rate = 0.1
        self.memory_size = 6
        self.simplex_fraction = 0.15
        self.stagnation_limit = 15
        self.restart_count = 0
        self.max_restarts = 10

    def __call__(self, func, stopping_condition):
        self._init_tracking()
        population = self._initialize_population()
        fitness = func(population)
        if stopping_condition():
            return self._finalize(population, fitness)
        self._update_best(population, fitness)
        
        while not stopping_condition():
            # Main DE step
            F_vals, CR_vals = self._generate_parameters_batch()
            mutants = self._mutate_batch(population, fitness, F_vals)
            trials = self._crossover_batch(population, mutants, CR_vals)
            trials = self._clip_bounds(trials)
            
            trial_fitness = func(trials)
            if stopping_condition():
                self._update_best(trials, trial_fitness)
                break
            
            # Handle truncated returns
            valid = self._get_valid_mask(trial_fitness)
            if np.sum(valid) == 0:
                break
            
            population, fitness = self._select_survivors_batch(
                population, fitness, trials, trial_fitness,
                F_vals, CR_vals, valid
            )
            self._update_best(population, fitness)
            
            # Simplex-based local search on top candidates
            if not stopping_condition() and self.generation % 3 == 0:
                population, fitness = self._simplex_refinement(
                    population, fitness, func, stopping_condition
                )
                if stopping_condition():
                    break
            
            # Stagnation detection and restart
            stagnated = self._detect_stagnation()
            if stagnated and not stopping_condition():
                population, fitness = self._restart(
                    population, fitness, func, stopping_condition
                )
                if stopping_condition():
                    break
            
            self.generation += 1
        
        return self._finalize_result()

    def _init_tracking(self):
        self.best_x = None
        self.best_f = np.inf
        self.generation = 0
        self.archive = []
        self.memory_F = np.full(self.memory_size, 0.5)
        self.memory_CR = np.full(self.memory_size, 0.5)
        self.memory_idx = 0
        self.stagnation_counter = 0
        self.prev_best_f = np.inf
        self.strategy_success = np.ones(3)  # 3 strategies
        self.strategy_total = np.ones(3)
        self.restart_count = 0

    def _initialize_population(self):
        pop = np.random.uniform(
            self.lb, self.ub, size=(self.np_size, self.dim)
        )
        return pop

    def _generate_parameters_batch(self):
        """Generate F and CR values using Cauchy and Normal distributions (SHADE-like)."""
        indices = np.random.randint(0, self.memory_size, size=self.np_size)
        
        # F values from Cauchy distribution
        mu_F = self.memory_F[indices]
        F_vals = mu_F + 0.1 * np.random.standard_cauchy(self.np_size)
        F_vals = np.clip(F_vals, 0.01, 1.0)
        # Regenerate any that were <= 0
        bad_mask = F_vals <= 0.01
        while np.any(bad_mask):
            F_vals[bad_mask] = mu_F[bad_mask] + 0.1 * np.random.standard_cauchy(np.sum(bad_mask))
            F_vals = np.clip(F_vals, 0.01, 1.0)
            bad_mask = F_vals <= 0.01
        
        # CR values from Normal distribution
        mu_CR = self.memory_CR[indices]
        CR_vals = np.random.normal(mu_CR, 0.1)
        CR_vals = np.clip(CR_vals, 0.0, 1.0)
        
        return F_vals, CR_vals

    def _mutate_batch(self, population, fitness, F_vals):
        """DE/current-to-pbest/1 mutation with archive and strategy mixing."""
        NP = len(population)
        p = max(2, int(self.p_best_rate * NP))
        sorted_indices = np.argsort(fitness)
        
        # Select strategy per individual based on success rates
        probs = self.strategy_success / (self.strategy_total + 1e-30)
        probs = probs / probs.sum()
        strategies = np.random.choice(3, size=NP, p=probs)
        
        mutants = np.empty_like(population)
        
        # Strategy 0: current-to-pbest/1
        mask0 = strategies == 0
        if np.any(mask0):
            mutants[mask0] = self._mutation_current_to_pbest(
                population, fitness, F_vals, sorted_indices, p, mask0
            )
        
        # Strategy 1: rand/1
        mask1 = strategies == 1
        if np.any(mask1):
            mutants[mask1] = self._mutation_rand1(
                population, F_vals, mask1
            )
        
        # Strategy 2: best/2
        mask2 = strategies == 2
        if np.any(mask2):
            mutants[mask2] = self._mutation_best2(
                population, F_vals, sorted_indices, mask2
            )
        
        self._current_strategies = strategies
        return mutants

    def _mutation_current_to_pbest(self, population, fitness, F_vals, sorted_indices, p, mask):
        """Current-to-pbest/1 mutation with optional archive usage."""
        n_mask = np.sum(mask)
        NP = len(population)
        
        # Select pbest indices
        pbest_idx = sorted_indices[np.random.randint(0, p, size=n_mask)]
        
        # Select r1 != i
        r1 = np.random.randint(0, NP, size=n_mask)
        idx = np.where(mask)[0]
        same = r1 == idx
        while np.any(same):
            r1[same] = np.random.randint(0, NP, np.sum(same))
            same = r1 == idx
        
        # Select r2 from population + archive
        combined = population
        if len(self.archive) > 0:
            archive_arr = np.array(self.archive)
            combined = np.vstack([population, archive_arr])
        n_combined = len(combined)
        r2 = np.random.randint(0, n_combined, size=n_mask)
        same2 = (r2 == idx) | (r2 == r1)
        while np.any(same2):
            r2[same2] = np.random.randint(0, n_combined, np.sum(same2))
            same2 = (r2 == idx) | (r2 == r1)
        
        F_col = F_vals[mask].reshape(-1, 1)
        mutant = (population[idx] + 
                  F_col * (population[pbest_idx] - population[idx]) +
                  F_col * (population[r1] - combined[r2]))
        return mutant

    def _mutation_rand1(self, population, F_vals, mask):
        """DE/rand/1 mutation."""
        n_mask = np.sum(mask)
        NP = len(population)
        idx = np.where(mask)[0]
        
        r1 = np.random.randint(0, NP, size=n_mask)
        r2 = np.random.randint(0, NP, size=n_mask)
        r3 = np.random.randint(0, NP, size=n_mask)
        
        # Ensure distinctness
        for _ in range(10):
            bad = (r1 == idx) | (r2 == idx) | (r3 == idx) | (r1 == r2) | (r1 == r3) | (r2 == r3)
            if not np.any(bad):
                break
            r1[bad] = np.random.randint(0, NP, np.sum(bad))
            r2[bad] = np.random.randint(0, NP, np.sum(bad))
            r3[bad] = np.random.randint(0, NP, np.sum(bad))
        
        F_col = F_vals[mask].reshape(-1, 1)
        return population[r1] + F_col * (population[r2] - population[r3])

    def _mutation_best2(self, population, F_vals, sorted_indices, mask):
        """DE/best/2 mutation."""
        n_mask = np.sum(mask)
        NP = len(population)
        best_idx = sorted_indices[0]
        
        r1 = np.random.randint(0, NP, size=n_mask)
        r2 = np.random.randint(0, NP, size=n_mask)
        r3 = np.random.randint(0, NP, size=n_mask)
        r4 = np.random.randint(0, NP, size=n_mask)
        
        F_col = F_vals[mask].reshape(-1, 1)
        return (population[best_idx] + 
                F_col * (population[r1] - population[r2]) +
                F_col * (population[r3] - population[r4]))

    def _crossover_batch(self, population, mutants, CR_vals):
        """Binomial crossover applied in batch."""
        NP, dim = population.shape
        rand_matrix = np.random.random((NP, dim))
        j_rand = np.random.randint(0, dim, size=NP)
        
        CR_matrix = CR_vals.reshape(-1, 1) * np.ones((1, dim))
        mask = (rand_matrix < CR_matrix)
        # Ensure at least one dimension is from mutant
        mask[np.arange(NP), j_rand] = True
        
        trials = np.where(mask, mutants, population)
        return trials

    def _clip_bounds(self, candidates):
        """Clip candidates to search bounds with bounce-back for diversity."""
        # Simple clipping
        clipped = np.clip(candidates, self.lb, self.ub)
        return clipped

    def _get_valid_mask(self, fitness):
        """Return boolean mask of valid (non-NaN) fitness values."""
        return ~np.isnan(fitness)

    def _select_survivors_batch(self, population, fitness, trials, trial_fitness,
                                 F_vals, CR_vals, valid):
        """Greedy selection with archive update and parameter memory update."""
        NP = len(population)
        n_valid = min(len(trial_fitness), NP)
        
        improved = np.zeros(NP, dtype=bool)
        improved[:n_valid] = (valid[:n_valid]) & (trial_fitness[:n_valid] < fitness[:n_valid])
        
        # Update archive with replaced individuals
        replaced_individuals = population[improved]
        for ind in replaced_individuals:
            if len(self.archive) < self.archive_size:
                self.archive.append(ind.copy())
            else:
                replace_idx = np.random.randint(0, self.archive_size)
                self.archive[replace_idx] = ind.copy()
        
        # Track strategy success
        if hasattr(self, '_current_strategies'):
            for s in range(3):
                s_mask = self._current_strategies[:n_valid] == s
                self.strategy_total[s] += np.sum(s_mask)
                self.strategy_success[s] += np.sum(improved[:n_valid] & s_mask)
        
        # Update parameter memory
        self._update_parameter_memory(fitness, trial_fitness, F_vals, CR_vals, improved, n_valid)
        
        # Replace improved individuals
        new_pop = population.copy()
        new_fit = fitness.copy()
        new_pop[improved] = trials[improved]
        new_fit[improved] = trial_fitness[improved]
        
        return new_pop, new_fit

    def _update_parameter_memory(self, fitness, trial_fitness, F_vals, CR_vals, improved, n_valid):
        """SHADE-style parameter memory update using weighted Lehmer mean."""
        if not np.any(improved):
            return
        
        succ_F = F_vals[improved]
        succ_CR = CR_vals[improved]
        deltas = np.abs(fitness[improved] - trial_fitness[improved])
        
        if len(deltas) == 0 or np.sum(deltas) < 1e-30:
            return
        
        weights = deltas / (np.sum(deltas) + 1e-30)
        
        # Weighted Lehmer mean for F
        new_F = np.sum(weights * succ_F ** 2) / (np.sum(weights * succ_F) + 1e-30)
        # Weighted arithmetic mean for CR
        new_CR = np.sum(weights * succ_CR)
        
        self.memory_F[self.memory_idx] = np.clip(new_F, 0.1, 0.9)
        self.memory_CR[self.memory_idx] = np.clip(new_CR, 0.05, 0.95)
        self.memory_idx = (self.memory_idx + 1) % self.memory_size

    def _simplex_refinement(self, population, fitness, func, stopping_condition):
        """Apply simplex-like moves to top candidates for local refinement."""
        NP = len(population)
        n_simplex = max(2, int(self.simplex_fraction * NP))
        sorted_idx = np.argsort(fitness)
        top_idx = sorted_idx[:n_simplex]
        
        # Compute centroid of top candidates
        centroid = np.mean(population[top_idx], axis=0)
        
        # Reflection of worst top candidates through centroid
        worst_top = sorted_idx[n_simplex:2*n_simplex] if 2*n_simplex <= NP else sorted_idx[n_simplex:]
        n_reflect = len(worst_top)
        
        alpha = 1.0 + 0.5 * np.random.random(n_reflect).reshape(-1, 1)
        reflected = centroid + alpha * (centroid - population[worst_top])
        reflected = self._clip_bounds(reflected)
        
        # Also create contraction moves
        beta = 0.5 * np.random.random(n_simplex).reshape(-1, 1)
        contracted = centroid + beta * (population[top_idx] - centroid)
        contracted = self._clip_bounds(contracted)
        
        # Evaluate all simplex trials at once
        all_trials = np.vstack([reflected, contracted])
        trial_fit = func(all_trials)
        
        if stopping_condition():
            self._update_best(all_trials, trial_fit)
            return population, fitness
        
        valid = self._get_valid_mask(trial_fit)
        
        # Apply reflected improvements
        n_valid_ref = min(n_reflect, len(trial_fit))
        for i in range(n_valid_ref):
            if valid[i] and trial_fit[i] < fitness[worst_top[i]]:
                population[worst_top[i]] = all_trials[i]
                fitness[worst_top[i]] = trial_fit[i]
        
        # Apply contracted improvements
        for i in range(min(n_simplex, len(trial_fit) - n_reflect)):
            j = i + n_reflect
            if j < len(trial_fit) and valid[j] and trial_fit[j] < fitness[top_idx[i]]:
                population[top_idx[i]] = all_trials[j]
                fitness[top_idx[i]] = trial_fit[j]
        
        self._update_best(population, fitness)
        return population, fitness

    def _detect_stagnation(self):
        """Detect if optimization has stagnated."""
        improvement = self.prev_best_f - self.best_f
        relative_improvement = improvement / (np.abs(self.prev_best_f) + 1e-30)
        
        if relative_improvement < 1e-10:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        self.prev_best_f = self.best_f
        return self.stagnation_counter >= self.stagnation_limit

    def _restart(self, population, fitness, func, stopping_condition):
        """Partial restart: keep best individuals, reinitialize rest with varied strategies."""
        self.restart_count += 1
        self.stagnation_counter = 0
        NP = len(population)
        
        # Keep top 20% of population
        n_keep = max(2, NP // 5)
        sorted_idx = np.argsort(fitness)
        
        # Determine restart scale based on restart count
        scale = max(10.0, 100.0 * (0.5 ** self.restart_count))
        
        # Generate new individuals around the best with varying scales
        n_new = NP - n_keep
        n_local = n_new // 2
        n_random = n_new - n_local
        
        # Local restart: near best solution
        local_new = self.best_x + scale * 0.1 * np.random.randn(n_local, self.dim)
        
        # Random restart: fully random
        random_new = np.random.uniform(self.lb, self.ub, size=(n_random, self.dim))
        
        new_individuals = np.vstack([local_new, random_new])
        new_individuals = self._clip_bounds(new_individuals)
        
        new_fitness = func(new_individuals)
        if stopping_condition():
            self._update_best(new_individuals, new_fitness)
            return population, fitness
        
        # Merge kept and new
        population_new = np.vstack([population[sorted_idx[:n_keep]], new_individuals])
        fitness_new = np.concatenate([fitness[sorted_idx[:n_keep]], new_fitness])
        
        # Reset parameter memory partially
        self._soft_reset_memory()
        
        self._update_best(population_new, fitness_new)
        return population_new, fitness_new

    def _soft_reset_memory(self):
        """Partially reset parameter memory to encourage exploration."""
        # Blend current memory with default values
        self.memory_F = 0.5 * self.memory_F + 0.5 * 0.5
        self.memory_CR = 0.5 * self.memory_CR + 0.5 * 0.5
        self.archive = self.archive[len(self.archive)//2:]
        # Decay strategy statistics
        self.strategy_success = 0.5 * self.strategy_success + 0.5
        self.strategy_total = 0.5 * self.strategy_total + 0.5

    def _update_best(self, population, fitness):
        """Update global best tracking, robust to NaN."""
        if len(fitness) == 0:
            return
        valid = ~np.isnan(fitness)
        if not np.any(valid):
            return
        valid_fit = fitness[valid]
        valid_pop = population[valid]
        best_idx = np.argmin(valid_fit)
        if valid_fit[best_idx] < self.best_f:
            self.best_f = valid_fit[best_idx]
            self.best_x = valid_pop[best_idx].copy()

    def _compute_diversity(self, population):
        """Compute population diversity as mean pairwise distance to centroid."""
        centroid = np.mean(population, axis=0)
        diffs = population - centroid
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        return np.mean(distances)

    def _finalize(self, population, fitness):
        """Handle early termination."""
        self._update_best(population, fitness)
        return self._finalize_result()

    def _finalize_result(self):
        """Return the best found solution."""
        if self.best_x is None:
            return np.inf, np.zeros(self.dim)
        return self.best_f, self.best_x.copy()
