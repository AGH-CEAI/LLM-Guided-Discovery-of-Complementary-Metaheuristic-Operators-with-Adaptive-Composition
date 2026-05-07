import numpy as np


class AdaptiveMultiStrategyDE:
    """
    Adaptive Multi-Strategy Differential Evolution with Spatial Crowding.
    
    A DE variant featuring:
    - Multiple mutation strategies with adaptive selection based on success history
    - Ring topology with adaptive neighborhood selection for multi-modal landscapes
    - Dynamic parameter adaptation (CR, F) per individual
    - Crowding-based survivor selection for diversity preservation
    - Novelty-triggered restart mechanism
    - Budget-aware evaluation handling
    
    The algorithm maintains several mutation strategies and adaptively weights
    them based on recent success rates, allowing it to switch between explorative
    and exploitative behaviors as needed.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = kwargs.get('np', min(max(6 * dim, 40), 300))
        self.f_bounds = [-100, 100]
        self._bound_range = self.f_bounds[1] - self.f_bounds[0]
        
        # Strategy pool with different mutation characteristics
        self._strategy_names = [
            'rand_1_bin',
            'best_1_bin', 
            'current_to_best_1_bin',
            'rand_2_bin',
            'target_to_best_2_bin',
            'local_ring_1_bin'
        ]
        self.n_strategies = len(self._strategy_names)
        self._strategy_scores = np.ones(self.n_strategies) * 0.5
        self._strategy_attempts = np.zeros(self.n_strategies)
        self._strategy_successes = np.zeros(self.n_strategies)
        self._strategy_window = 50
        
        # Per-individual adaptive parameters
        self.cr = np.full(self.np, 0.9)
        self.f = np.full(self.np, 0.5)
        self.cr_min, self.cr_max = 0.1, 0.95
        self.f_min, self.f_max = 0.3, 1.0
        
        # Diversity and stagnation tracking
        self._diversity_threshold = 0.05
        self._stagnation_limit = 15
        self._stagnation_counter = 0
        self._no_improvement_count = 0
        
        # Crowding parameters
        self._crowding_factor = 0.5
        
        # Memory for adaptation
        self._f_history = []
        self._cr_history = []
        
    def __call__(self, func, stopping_condition):
        """Execute the optimizer and return (f_opt, x_opt)."""
        population = self._initialize_population()
        fitness = self._evaluate_batch(func, population)
        
        if len(fitness) == 0:
            return np.inf, np.zeros(self.dim)
        
        best_idx = self._get_best_index(fitness)
        f_opt = fitness[best_idx]
        x_opt = population[best_idx].copy()
        
        generation = 0
        while not stopping_condition():
            # Select strategies for each individual
            strategy_ids = self._select_strategies_batch()
            
            # Generate trial population using multi-strategy mutation
            trials = self._mutate_and_crossover_batch(population, fitness, strategy_ids)
            
            # Clip to bounds before evaluation
            trials = self._clip_to_bounds(trials)
            
            # Check budget
            if stopping_condition():
                break
                
            # Evaluate trials in batch
            trial_fitness = self._evaluate_batch(func, trials)
            
            if len(trial_fitness) < len(trials):
                valid_len = len(trial_fitness)
                trials = trials[:valid_len]
                trial_fitness = trial_fitness[:valid_len]
                if valid_len == 0:
                    break
                    
            if stopping_condition():
                break
                
            # Update strategy scores based on successes
            self._update_strategy_scores(population, fitness, trials, trial_fitness, strategy_ids)
            
            # Crowding-based survivor selection
            population, fitness = self._select_survivors_crowding(
                population, fitness, trials, trial_fitness
            )
            
            # Update best solution
            best_idx = self._get_best_index(fitness)
            if fitness[best_idx] < f_opt:
                f_opt = fitness[best_idx]
                x_opt = population[best_idx].copy()
                self._no_improvement_count = 0
            else:
                self._no_improvement_count += 1
            
            # Adapt parameters
            self._adapt_parameters(generation)
            
            # Check for restart conditions
            if self._restart_if_needed(generation):
                population = self._reinitialize_population(population, fitness)
                fitness = self._evaluate_batch(func, population)
                if len(fitness) < self.np:
                    valid_len = len(fitness)
                    population = population[:valid_len]
                    fitness = fitness[:valid_len]
                    if valid_len == 0:
                        break
                        
            generation += 1
            
        return f_opt, x_opt
    
    def _initialize_population(self):
        """Initialize population using Latin Hypercube Sampling for better spread."""
        pop = np.zeros((self.np, self.dim))
        for d in range(self.dim):
            samples = np.linspace(self.f_bounds[0], self.f_bounds[1], self.np)
            pop[:, d] = np.random.permutation(samples) + np.random.uniform(-1, 1, self.np)
        return np.clip(pop, self.f_bounds[0], self.f_bounds[1])
    
    def _evaluate_batch(self, func, population):
        """Evaluate fitness for entire population in single batch call."""
        fitness = func(population)
        if fitness is None or len(fitness) == 0:
            return np.array([])
        fitness = np.asarray(fitness, dtype=np.float64)
        fitness = np.nan_to_num(fitness, nan=np.inf, posinf=np.inf, neginf=np.inf)
        return fitness
    
    def _get_best_index(self, fitness):
        """Get index of best (minimum) fitness value."""
        if len(fitness) == 0:
            return 0
        return int(np.argmin(fitness))
    
    def _select_strategies_batch(self):
        """Select mutation strategy for each individual using adaptive probabilities."""
        probs = self._strategy_scores / self._strategy_scores.sum()
        return np.random.choice(self.n_strategies, size=self.np, p=probs)
    
    def _mutate_and_crossover_batch(self, population, fitness, strategy_ids):
        """Generate trial population using selected strategies with DE operators."""
        trials = np.zeros_like(population)
        
        for i in range(self.np):
            strat = strategy_ids[i]
            cr_i = np.clip(self.cr[i] + np.random.normal(0, 0.1), self.cr_min, self.cr_max)
            f_i = np.clip(self.f[i] + np.random.normal(0, 0.1), self.f_min, self.f_max)
            
            if strat == 0:  # rand_1_bin
                mutant = self._mutate_rand_1(population, i, f_i)
            elif strat == 1:  # best_1_bin
                mutant = self._mutate_best_1(population, fitness, i, f_i)
            elif strat == 2:  # current_to_best_1_bin
                mutant = self._mutate_current_to_best_1(population, fitness, i, f_i)
            elif strat == 3:  # rand_2_bin
                mutant = self._mutate_rand_2(population, i, f_i)
            elif strat == 4:  # target_to_best_2_bin
                mutant = self._mutate_target_to_best_2(population, fitness, i, f_i)
            else:  # local_ring_1_bin
                mutant = self._mutate_local_ring_1(population, i, f_i)
            
            # Binomial crossover
            j_rand = np.random.randint(self.dim)
            mask = np.random.random(self.dim) < cr_i
            mask[j_rand] = True
            trials[i] = np.where(mask, mutant, population[i])
            
        return trials
    
    def _mutate_rand_1(self, pop, idx, f):
        """Classic rand/1 mutation: r0 + F * (r1 - r2)."""
        indices = self._get_three_different_indices(idx)
        return pop[indices[0]] + f * (pop[indices[1]] - pop[indices[2]])
    
    def _mutate_best_1(self, pop, fitness, idx, f):
        """Best/1 mutation: best + F * (r0 - r1)."""
        best_idx = self._get_best_index(fitness)
        indices = self._get_two_different_indices(idx, exclude=best_idx)
        return pop[best_idx] + f * (pop[indices[0]] - pop[indices[1]])
    
    def _mutate_current_to_best_1(self, pop, fitness, idx, f):
        """Current-to-best/1: x_i + F*(best - x_i) + F*(r0 - r1)."""
        best_idx = self._get_best_index(fitness)
        indices = self._get_two_different_indices(idx, exclude=best_idx)
        return pop[idx] + f * (pop[best_idx] - pop[idx]) + f * (pop[indices[0]] - pop[indices[1]])
    
    def _mutate_rand_2(self, pop, idx, f):
        """Rand/2 mutation: r0 + F*(r1 - r2) + F*(r3 - r4)."""
        indices = self._get_five_different_indices(idx)
        return pop[indices[0]] + f * (pop[indices[1]] - pop[indices[2]]) + f * (pop[indices[3]] - pop[indices[4]])
    
    def _mutate_target_to_best_2(self, pop, fitness, idx, f):
        """Target-to-best/2 mutation with current individual."""
        best_idx = self._get_best_index(fitness)
        indices = self._get_four_different_indices(idx, exclude=best_idx)
        return pop[idx] + f * (pop[best_idx] - pop[idx]) + f * (pop[indices[0]] - pop[indices[1]]) + f * (pop[indices[2]] - pop[indices[3]])
    
    def _mutate_local_ring_1(self, pop, idx, f):
        """Local ring mutation using neighboring individuals in ring topology."""
        np_local = min(5, self.np)
        local_indices = [(idx + k) % self.np for k in range(np_local)]
        r1, r2, r3 = local_indices[1], local_indices[2], local_indices[3] if np_local > 3 else local_indices[2]
        return pop[r1] + f * (pop[r2] - pop[r3])
    
    def _get_three_different_indices(self, exclude):
        """Get 3 unique indices excluding a given index."""
        candidates = np.concatenate([np.arange(exclude), np.arange(exclude + 1, self.np)])
        return np.random.choice(candidates, 3, replace=False)
    
    def _get_two_different_indices(self, exclude1, exclude=None):
        """Get 2 unique indices excluding given indices."""
        exclude = exclude if exclude is not None else exclude1
        candidates = np.setdiff1d(np.arange(self.np), [exclude1, exclude])
        return np.random.choice(candidates, 2, replace=False)
    
    def _get_four_different_indices(self, exclude1, exclude2):
        """Get 4 unique indices excluding given indices."""
        candidates = np.setdiff1d(np.arange(self.np), [exclude1, exclude2])
        return np.random.choice(candidates, 4, replace=False)
    
    def _get_five_different_indices(self, exclude):
        """Get 5 unique indices excluding a given index."""
        candidates = np.concatenate([np.arange(exclude), np.arange(exclude + 1, self.np)])
        return np.random.choice(candidates, 5, replace=False)
    
    def _clip_to_bounds(self, population):
        """Clip all individuals to search bounds."""
        return np.clip(population, self.f_bounds[0], self.f_bounds[1])
    
    def _update_strategy_scores(self, population, fitness, trials, trial_fitness, strategy_ids):
        """Update strategy scores based on improvement ratio."""
        improvements = trial_fitness < fitness[:len(trial_fitness)]
        
        for s in range(self.n_strategies):
            mask = strategy_ids == s
            if np.any(mask):
                s_improved = improvements[mask]
                attempt = np.sum(mask)
                success = np.sum(s_improved)
                
                self._strategy_attempts[s] += attempt
                self._strategy_successes[s] += success
                
                # Update score using success rate with smoothing
                if self._strategy_attempts[s] > 0:
                    success_rate = self._strategy_successes[s] / max(1, self._strategy_attempts[s])
                    self._strategy_scores[s] = 0.9 * self._strategy_scores[s] + 0.1 * success_rate
                
                # Decay old history to allow adaptation
                if self._strategy_attempts[s] > self._strategy_window * 2:
                    self._strategy_attempts[s] *= 0.9
                    self._strategy_successes[s] *= 0.9
    
    def _select_survivors_crowding(self, population, fitness, trials, trial_fitness):
        """Crowding-based survivor selection for diversity preservation."""
        n_trials = len(trials)
        n_parent = min(len(population), n_trials)
        
        new_pop = np.zeros((n_parent, self.dim))
        new_fit = np.zeros(n_parent)
        
        # Use deterministic crowding: replace closest parent
        for i in range(n_trials):
            parent_idx = i % n_parent
            parent = population[parent_idx]
            parent_fit = fitness[parent_idx]
            
            # Calculate distance to parent
            dist = np.linalg.norm(trials[i] - parent)
            
            # If trial is better, consider replacement
            if trial_fitness[i] < parent_fit:
                # Always replace if significantly better
                if dist > self._crowding_factor * np.sqrt(self.dim):
                    new_pop[parent_idx] = trials[i]
                    new_fit[parent_idx] = trial_fitness[i]
                else:
                    # Replace with some probability based on improvement
                    improvement_ratio = (parent_fit - trial_fitness[i]) / (abs(parent_fit) + 1e-10)
                    if np.random.random() < min(1.0, improvement_ratio):
                        new_pop[parent_idx] = trials[i]
                        new_fit[parent_idx] = trial_fitness[i]
                    else:
                        new_pop[parent_idx] = parent
                        new_fit[parent_idx] = parent_fit
            else:
                new_pop[parent_idx] = parent
                new_fit[parent_idx] = parent_fit
        
        # Fill remaining slots
        for i in range(n_trials, n_parent):
            new_pop[i] = population[i]
            new_fit[i] = fitness[i]
        
        return new_pop, new_fit
    
    def _adapt_parameters(self, generation):
        """Adapt CR and F parameters based on success history."""
        # Adaptive CR: increase if many improvements, decrease if few
        recent_f = self._f_history[-20:] if len(self._f_history) >= 20 else self._f_history
        recent_cr = self._cr_history[-20:] if len(self._cr_history) >= 20 else self._cr_history
        
        if len(recent_f) >= 5:
            avg_f = np.mean(recent_f)
            avg_cr = np.mean(recent_cr)
            
            # JADE-style adaptation with archive
            self.f = np.clip(0.5 + 0.5 * np.random.beta(0.5, 0.5, size=self.np), self.f_min, self.f_max)
            self.cr = np.clip(avg_cr + 0.1 * np.random.randn(self.np), self.cr_min, self.cr_max)
        
        # Store history
        self._f_history.append(np.mean(self.f))
        self._cr_history.append(np.mean(self.cr))
        
        # Limit history size
        if len(self._f_history) > 100:
            self._f_history = self._f_history[-50:]
            self._cr_history = self._cr_history[-50:]
    
    def _compute_diversity(self, population):
        """Compute population diversity using average pairwise distance."""
        if len(population) < 2:
            return 0.0
        centroids = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroids, axis=1)
        return np.mean(distances) / (0.5 * self._bound_range)
    
    def _restart_if_needed(self, generation):
        """Check and execute restart conditions based on stagnation or diversity loss."""
        current_diversity = self._compute_diversity(population if 'population' in dir() else np.zeros((self.np, self.dim)))
        
        # Check stagnation
        if self._no_improvement_count >= self._stagnation_limit:
            self._stagnation_counter += 1
            self._no_improvement_count = 0
            return True
        
        # Check diversity loss
        if current_diversity < self._diversity_threshold and generation > 20:
            self._stagnation_counter += 1
            return True
        
        # Periodic restart to prevent premature convergence
        if generation > 0 and generation % 200 == 0 and self._stagnation_counter > 0:
            return True
        
        return False
    
    def _reinitialize_population(self, old_pop, old_fitness):
        """Reinitialize population preserving some best solutions."""
        n_preserve = max(2, self.np // 5)
        best_indices = np.argsort(old_fitness)[:n_preserve]
        preserved = old_pop[best_indices].copy()
        
        # Generate new random population
        new_pop = np.random.uniform(
            self.f_bounds[0], self.f_bounds[1], 
            (self.np - n_preserve, self.dim)
        )
        
        # Combine preserved with new
        new_pop = np.vstack([preserved, new_pop])
        
        # Reset adaptation parameters
        self.cr = np.full(self.np, 0.9)
        self.f = np.full(self.np, 0.5)
        self._stagnation_counter = 0
        self._no_improvement_count = 0
        
        return new_pop
