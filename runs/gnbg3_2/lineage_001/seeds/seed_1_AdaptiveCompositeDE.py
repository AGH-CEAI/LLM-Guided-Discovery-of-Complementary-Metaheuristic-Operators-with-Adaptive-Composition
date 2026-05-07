import numpy as np


class AdaptiveCompositeDE:
    """
    Differential Evolution with Adaptive Composite Mutation and Topology-based Restart.
    
    Key innovations:
    - Composite mutation blending 4 strategies with adaptive weights
    - Archive-based current-to-pbest/1 mutation for exploitation
    - Population topology adaptation based on diversity clustering
    - F/CR adaptation using weighted success history
    - Diversity-triggered partial reinitialization
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 6*dim bounded between 80 and 300
        self.np = min(max(6 * dim, 80), 300)
        
        # Internal state
        self.population = None
        self.fitness = None
        self.f_opt = np.inf
        self.x_opt = None
        
        # Adaptive parameters
        self.F = 0.7
        self.CR = 0.5
        self.p_best = 0.1  # Top p% for current-to-pbest
        
        # Composite mutation weights (one per strategy)
        self.mutation_weights = np.ones(4) / 4.0
        
        # Success history for adaptation
        self.success_weights = []
        self.success_F = []
        self.success_CR = []
        self.window_size = 15
        
        # Diversity and stagnation tracking
        self.diversity_history = []
        self.stagnation_threshold = 0.001
        self.stagnation_counter = 0
        self.max_stagnation = int(30 + dim * 0.5)
        
        # Mutation strategy pool
        self._mutation_funcs = [
            self._mutate_current_to_pbest_bin,
            self._mutate_current_to_random_bin,
            self._mutate_rand_1_bin,
            self._mutate_best_2_bin,
        ]
    
    def _initialize_population(self):
        """Initialize population uniformly in bounds and evaluate fitness."""
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, size=(self.np, self.dim)
        )
        self.fitness = func(self.population)
        if len(self.fitness) < self.np:
            self.fitness = np.array(self.fitness)
        self._update_best(self.population, self.fitness)
    
    def _update_best(self, pop, fit):
        """Update global best if new optimum found."""
        fit = np.asarray(fit).flatten()
        idx = np.nanargmin(fit)
        if fit[idx] < self.f_opt:
            self.f_opt = fit[idx]
            self.x_opt = pop[idx].copy()
    
    def _select_pbest_indices(self, n_select):
        """Select indices of top performers for current-to-pbest."""
        valid_fit = np.where(~np.isnan(self.fitness))[0]
        sorted_idx = valid_fit[np.argsort(self.fitness[valid_fit])]
        return sorted_idx[:max(1, int(n_select))].tolist()
    
    def _mutate_current_to_pbest_bin(self, targets, pbest_idx):
        """Current-to-pbest/1/bin with archive for diversity."""
        F_local = self.F * (0.9 + 0.2 * np.random.random())
        
        # Select base from top p% 
        base = self.population[pbest_idx]
        
        # Random donors from population
        perm_pop = np.random.permutation(self.np)
        mask_pop = ~np.isin(perm_pop, pbest_idx)
        donors_pop = perm_pop[mask_pop][:2]
        
        # Generate mutant
        mutant = (
            targets +
            F_local * (base - targets) +
            F_local * (self.population[donors_pop[0]] - self.population[donors_pop[1]])
        )
        return np.clip(mutant, self.lower_bound, self.upper_bound)
    
    def _mutate_current_to_random_bin(self, targets):
        """Current-to-random/1/bin for exploration."""
        F_local = self.F * (0.8 + 0.4 * np.random.random())
        
        # Three random distinct indices
        indices = np.random.choice(self.np, size=4, replace=False)
        r1, r2, r3, _ = indices
        
        # Generate mutant
        mutant = (
            self.population[r1] +
            F_local * (self.population[r2] - self.population[r3])
        )
        return np.clip(mutant, self.lower_bound, self.upper_bound)
    
    def _mutate_rand_1_bin(self, targets):
        """Classic DE/rand/1/bin for exploration."""
        F_local = self.F * (0.7 + 0.5 * np.random.random())
        
        # Three random distinct indices
        indices = np.random.choice(self.np, size=4, replace=False)
        r1, r2, r3, target_idx = indices
        
        # Generate mutant
        mutant = (
            self.population[r1] +
            F_local * (self.population[r2] - self.population[r3])
        )
        return np.clip(mutant, self.lower_bound, self.upper_bound)
    
    def _mutate_best_2_bin(self, targets):
        """DE/best/2/bin for exploitation with diversity."""
        F_local = self.F * (0.6 + 0.3 * np.random.random())
        
        # Use best as base
        base = self.x_opt if self.x_opt is not None else self.population[np.argmin(self.fitness)]
        
        # Two random pairs for difference
        indices = np.random.choice(self.np, size=4, replace=False)
        
        # Generate mutant
        mutant = (
            base +
            F_local * (self.population[indices[0]] - self.population[indices[1]]) +
            F_local * (self.population[indices[2]] - self.population[indices[3]])
        )
        return np.clip(mutant, self.lower_bound, self.upper_bound)
    
    def _mutate_batch(self):
        """Generate batch of mutants using weighted composite mutation."""
        # Select pbest indices
        n_pbest = max(1, int(self.np * self.p_best))
        pbest_candidates = self._select_pbest_indices(n_pbest)
        
        # Sample strategy for each individual
        strategy_indices = np.random.choice(
            len(self._mutation_funcs), size=self.np, p=self.mutation_weights
        )
        
        mutants = np.empty((self.np, self.dim))
        
        for i, strat_idx in enumerate(strategy_indices):
            if strat_idx == 0:  # current-to-pbest
                pbest = np.random.choice(pbest_candidates)
                mutants[i] = self._mutation_funcs[strat_idx](self.population[i], pbest)
            else:
                mutants[i] = self._mutation_funcs[strat_idx](self.population[i])
        
        return mutants, strategy_indices
    
    def _crossover_batch(self, targets, mutants):
        """Binomial crossover between targets and mutants."""
        CR_local = np.clip(self.CR + 0.1 * np.random.randn(self.np), 0.1, 0.95)
        
        # Determine which dimensions come from mutant
        j_rand = np.random.randint(0, self.dim, size=self.np)
        mask = np.random.random((self.np, self.dim)) < CR_local[:, np.newaxis]
        mask[j_rand, np.arange(self.np)] = True
        
        trials = np.where(mask, mutants, targets)
        return np.clip(trials, self.lower_bound, self.upper_bound)
    
    def _select_survivors_batch(self, targets, target_fitness, trials, trial_fitness):
        """Greedy selection between target and trial populations."""
        target_fitness = np.asarray(target_fitness).flatten()
        trial_fitness = np.asarray(trial_fitness).flatten()
        
        # Greedy: keep trial if better
        better_mask = trial_fitness < target_fitness
        
        new_pop = np.where(better_mask[:, np.newaxis], trials, targets)
        new_fit = np.where(better_mask, trial_fitness, target_fitness)
        
        return new_pop, new_fit, better_mask
    
    def _adapt_mutation_weights(self, strategy_indices, improvements):
        """Adapt mutation strategy weights based on recent success."""
        if len(self.success_weights) == 0:
            return
        
        # Aggregate success per strategy
        for strat_idx in range(len(self.mutation_weights)):
            mask = np.array(strategy_indices) == strat_idx
            if np.any(mask):
                strat_improvements = [imp for i, imp in zip(mask, improvements) if i]
                if strat_improvements:
                    avg_imp = np.mean(strat_improvements)
                    for _ in range(np.sum(mask)):
                        self.success_weights[strat_idx].append(avg_imp)
        
        # Compute new weights from recent history
        new_weights = np.zeros(len(self.mutation_weights))
        for strat_idx in range(len(self.mutation_weights)):
            if self.success_weights[strat_idx]:
                new_weights[strat_idx] = np.mean(self.success_weights[strat_idx][-self.window_size:])
        
        # Normalize and apply floor
        weight_sum = new_weights.sum()
        if weight_sum > 0:
            self.mutation_weights = new_weights / weight_sum
        self.mutation_weights = np.maximum(self.mutation_weights, 0.05)
        self.mutation_weights /= self.mutation_weights.sum()
    
    def _adapt_parameters(self, successful_F, successful_CR, successful_improvements):
        """Adapt F and CR using weighted success history."""
        if len(successful_improvements) == 0:
            return
        
        improvements = np.array(successful_improvements)
        improvements = improvements / (np.abs(improvements).max() + 1e-10)
        
        # Weighted mean for F
        if len(successful_F) > 0:
            F_arr = np.array(successful_F)
            self.F = np.dot(F_arr, improvements) / (len(improvements) + 1e-10)
            self.F = np.clip(self.F, 0.3, 1.2)
        
        # Weighted mean for CR
        if len(successful_CR) > 0:
            CR_arr = np.array(successful_CR)
            self.CR = np.dot(CR_arr, improvements) / (len(improvements) + 1e-10)
            self.CR = np.clip(self.CR, 0.1, 0.95)
    
    def _compute_diversity(self):
        """Compute normalized population diversity (average distance to centroid)."""
        valid_mask = ~np.isnan(self.fitness)
        if np.sum(valid_mask) < 2:
            return 0.0
        
        valid_pop = self.population[valid_mask]
        centroid = np.mean(valid_pop, axis=0)
        
        distances = np.linalg.norm(valid_pop - centroid, axis=1)
        search_diameter = np.sqrt(self.dim) * (self.upper_bound - self.lower_bound)
        
        return np.mean(distances) / search_diameter * np.sqrt(self.dim)
    
    def _check_stagnation(self, diversity):
        """Check if population has stagnated."""
        self.diversity_history.append(diversity)
        if len(self.diversity_history) > self.window_size:
            self.diversity_history.pop(0)
        
        if len(self.diversity_history) >= 5:
            recent_avg = np.mean(self.diversity_history[-5:])
            if abs(diversity - recent_avg) < self.stagnation_threshold:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        
        return self.stagnation_counter >= self.max_stagnation
    
    def _restart_diverse(self):
        """Partial reinitialization preserving best and diverse individuals."""
        # Keep best
        keep_best = self.x_opt.copy() if self.x_opt is not None else self.population[0].copy()
        
        # Compute pairwise distances
        valid_mask = ~np.isnan(self.fitness)
        if np.sum(valid_mask) > 10:
            valid_pop = self.population[valid_mask]
            centroid = np.mean(valid_pop, axis=0)
            distances = np.linalg.norm(valid_pop - centroid, axis=1)
            
            # Keep top 20% most diverse
            n_keep = max(2, self.np // 5)
            diverse_idx = np.argsort(distances)[-n_keep:]
            
            kept = [keep_best]
            for idx in diverse_idx:
                kept.append(self.population[valid_mask][idx])
            
            # Fill rest randomly
            n_new = self.np - len(kept)
            new_part = np.random.uniform(
                self.lower_bound, self.upper_bound, size=(n_new, self.dim)
            )
            
            self.population = np.vstack([np.array(kept), new_part])
        else:
            # Full reinit
            self.population = np.random.uniform(
                self.lower_bound, self.upper_bound, size=(self.np, self.dim)
            )
            self.population[0] = keep_best
        
        # Re-evaluate
        self.fitness = func(self.population)
        if len(self.fitness) < self.np:
            self.fitness = np.resize(self.fitness, self.np)
            self.fitness[len(func(self.population)):] = np.nan
        
        # Reset stagnation
        self.stagnation_counter = 0
        self.diversity_history = []
    
    def __call__(self, func, stopping_condition):
        """Run optimization and return (f_opt, x_opt)."""
        self.f_opt = np.inf
        self.x_opt = None
        
        # Initialize
        self._initialize_population()
        self.success_weights = [[] for _ in self.mutation_weights]
        
        # Main loop
        while not stopping_condition():
            # Generate mutants using composite mutation
            mutants, strategy_indices = self._mutate_batch()
            
            # Create trials via crossover
            trials = self._crossover_batch(self.population, mutants)
            
            # Evaluate trials
            trial_fitness = func(trials)
            if len(trial_fitness) < len(trials):
                actual_len = len(trial_fitness)
                trials = trials[:actual_len]
                trial_fitness = np.array(trial_fitness)
                if stopping_condition():
                    return self.f_opt, self.x_opt
            
            # Check budget exhaustion
            if stopping_condition():
                return self.f_opt, self.x_opt
            
            # Selection
            self.population, self.fitness, better_mask = self._select_survivors_batch(
                self.population, self.fitness, trials, trial_fitness
            )
            
            # Update best
            self._update_best(self.population, self.fitness)
            
            # Track success for adaptation
            successful_mask = better_mask & ~np.isnan(trial_fitness)
            if np.any(successful_mask):
                successful_trials = trials[successful_mask]
                successful_targets = self.population[successful_mask]
                
                # Compute improvements
                improvements = (
                    self.fitness[successful_mask] -
                    trial_fitness[successful_mask]
                )
                
                # Store success
                for i, (strat, imp) in enumerate(zip(
                    strategy_indices[successful_mask],
                    improvements
                )):
                    self.success_weights[strat].append(imp)
                
                # Adapt weights
                self._adapt_mutation_weights(
                    strategy_indices[successful_mask],
                    improvements.tolist()
                )
                
                # Adapt F and CR
                F_vals = self.F * (0.8 + 0.4 * np.random.random(sum(successful_mask)))
                CR_vals = np.clip(
                    self.CR + 0.1 * np.random.randn(sum(successful_mask)),
                    0.1, 0.95
                )
                self._adapt_parameters(F_vals, CR_vals, improvements.tolist())
            
            # Check stagnation
            diversity = self._compute_diversity()
            if self._check_stagnation(diversity):
                self._restart_diverse()
        
        return self.f_opt, self.x_opt
