import numpy as np


class AdaptiveMultiStrategyDE:
    """
    Adaptive Multi-Strategy Differential Evolution with Temporal Drift and Archive.
    
    Key innovations:
    - Multiple mutation strategies with adaptive selection probability
    - Temporal drift mechanism for momentum-based exploration
    - Archive of non-improving solutions for diversity
    - Adaptive F/CR based on generation-wise success rates
    - Diversity-triggered randomization for escaping local optima
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = kwargs.get('NP', min(6 * dim, 400))
        self.bounds = (-100.0, 100.0)
        
        # Core DE parameters
        self.F = kwargs.get('F', 0.5)
        self.CR = kwargs.get('CR', 0.9)
        self.p_best = kwargs.get('p_best', 0.1)
        
        # Adaptive components
        self.inertia_weight = 0.7
        self.temporal_drift = 0.2
        self.neighborhood_size = max(5, self.NP // 8)
        self.archive_size = self.NP
        
        # Strategy weights for multi-strategy mutation
        self.strategy_weights = np.ones(4) / 4.0
        
        # State variables
        self.population = None
        self.fitness = None
        self.trial_population = None
        self.trial_fitness = None
        self.archive = None
        self.drift_vector = None
        self.best_idx = None
        self.best_fitness = np.inf
        self.gen = 0
        self.success_history_F = []
        self.success_history_CR = []
        
    def _initialize_population(self, func):
        self.population = np.random.uniform(
            self.bounds[0], self.bounds[1], size=(self.NP, self.dim)
        )
        self.population = self._clip_to_bounds(self.population)
        self.fitness = func(self.population)
        
        if len(self.fitness) < self.NP:
            self.fitness = np.resize(self.fitness, self.NP)
            self.fitness[len(self.fitness):] = np.nan
        
        self.best_idx = np.nanargmin(self.fitness)
        self.best_fitness = self.fitness[self.best_idx]
        self.archive = self.population.copy()
        self.drift_vector = np.zeros(self.dim)
        self._reset_trial_arrays()
        
    def _reset_trial_arrays(self):
        self.trial_population = np.empty((self.NP, self.dim))
        self.trial_fitness = np.empty(self.NP)
        
    def _clip_to_bounds(self, pop):
        return np.clip(pop, self.bounds[0], self.bounds[1])
    
    def _get_best_individual(self):
        return self.population[self.best_idx]
    
    def _select_random_indices(self, size, exclude_indices=None):
        available = np.arange(self.NP)
        if exclude_indices is not None:
            available = np.setdiff1d(available, np.array(exclude_indices))
        if len(available) < size:
            available = np.arange(self.NP)
        return np.random.choice(available, size=size, replace=False)
    
    def _select_p_best_indices(self):
        n_select = max(1, int(self.p_best * self.NP))
        valid_fitness = np.where(~np.isnan(self.fitness))[0]
        sorted_idx = valid_fitness[np.argsort(self.fitness[valid_fitness])]
        return sorted_idx[:n_select]
    
    def _mutate_strategy_best(self, r1, r2):
        best = self._get_best_individual()
        return best + self.inertia_weight * (self.population[r1] - self.population[r2])
    
    def _mutate_strategy_rand(self, r1, r2, r3):
        return self.population[r1] + self.F * (self.population[r2] - self.population[r3])
    
    def _mutate_strategy_current_to_pbest(self, idx, pbest_idx, r1, r2):
        current = self.population[idx]
        best = self.population[pbest_idx]
        return current + self.inertia_weight * (best - current) + self.F * (self.population[r1] - self.population[r2])
    
    def _mutate_strategy_temporal_drift(self, r1, r2, r3):
        base_vector = self.population[r1] + self.F * (self.population[r2] - self.population[r3])
        drift_contribution = self.temporal_drift * self.drift_vector
        return base_vector + drift_contribution
    
    def _mutate_batch(self):
        n = self.NP
        mutants = np.empty((n, self.dim))
        
        # Get indices for all mutations
        all_indices = np.arange(n)
        rand_indices = [self._select_random_indices(n, exclude=[i]) for i in range(n)]
        
        pbest_candidates = self._select_p_best_indices()
        if len(pbest_candidates) == 0:
            pbest_candidates = [self.best_idx]
        
        for i in range(n):
            r = rand_indices[i]
            r1, r2, r3 = r[0], r[1], r[2]
            pbest = np.random.choice(pbest_candidates)
            
            # Select strategy based on adaptive weights
            strategy = np.random.choice(4, p=self.strategy_weights)
            
            if strategy == 0:
                mutants[i] = self._mutate_strategy_best(r1, r2)
            elif strategy == 1:
                mutants[i] = self._mutate_strategy_rand(r1, r2, r3)
            elif strategy == 2:
                mutants[i] = self._mutate_strategy_current_to_pbest(i, pbest, r1, r2)
            else:
                mutants[i] = self._mutate_strategy_temporal_drift(r1, r2, r3)
                
        return self._clip_to_bounds(mutants)
    
    def _crossover_batch(self, target, mutant):
        n = self.NP
        trials = np.empty((n, self.dim))
        
        for i in range(n):
            j_rand = np.random.randint(self.dim)
            cr_i = max(0.1, min(1.0, self.CR + np.random.normal(0, 0.1)))
            mask = np.random.random(self.dim) < cr_i
            mask[j_rand] = True
            trials[i] = np.where(mask, mutant[i], target[i])
            
        return self._clip_to_bounds(trials)
    
    def _evaluate_trial_batch(self, trials, func):
        trial_fit = func(trials)
        
        if len(trial_fit) < len(trials):
            missing = len(trials) - len(trial_fit)
            trial_fit = np.resize(trial_fit, len(trials))
            trial_fit[-missing:] = np.nan
            
        return trial_fit
    
    def _select_survivors_batch(self):
        improved_mask = self.trial_fitness < self.fitness
        better_count = np.sum(improved_mask & ~np.isnan(self.trial_fitness))
        
        # Update population
        for i in range(self.NP):
            if improved_mask[i] and not np.isnan(self.trial_fitness[i]):
                self.archive = np.vstack([self.archive, self.population[i]])
                if len(self.archive) > self.archive_size:
                    drop_idx = np.random.randint(len(self.archive))
                    self.archive = np.delete(self.archive, drop_idx, axis=0)
                self.population[i] = self.trial_population[i]
                self.fitness[i] = self.trial_fitness[i]
        
        # Update best
        current_best_idx = np.nanargmin(self.fitness)
        current_best_fit = self.fitness[current_best_idx]
        
        if current_best_fit < self.best_fitness:
            self.best_idx = current_best_idx
            self.best_fitness = current_best_fit
        
        return better_count
    
    def _adapt_inertia_weight(self, success_rate):
        if success_rate > 0.2:
            self.inertia_weight = min(1.0, self.inertia_weight * 1.05)
        elif success_rate < 0.1:
            self.inertia_weight = max(0.3, self.inertia_weight * 0.95)
            
    def _adapt_temporal_drift(self, success_rate):
        if success_rate > 0.25:
            self.temporal_drift = min(0.5, self.temporal_drift * 1.1)
        elif success_rate < 0.1:
            self.temporal_drift = max(0.05, self.temporal_drift * 0.9)
            
    def _adapt_F_CR(self, trial_pop, improved_mask):
        valid_improved = improved_mask & ~np.isnan(self.trial_fitness)
        
        if np.sum(valid_improved) > 0:
            improved_trials = trial_pop[valid_improved]
            improved_targets = self.population[valid_improved]
            
            diffs = improved_trials - improved_targets
            for dim_idx in range(self.dim):
                dim_diffs = diffs[:, dim_idx]
                valid_diffs = dim_diffs[~np.isnan(dim_diffs) & (dim_diffs != 0)]
                if len(valid_diffs) > 0:
                    self.success_history_F.append(np.std(valid_diffs) / (np.mean(np.abs(valid_diffs)) + 1e-10))
                    self.success_history_CR.append(np.mean(np.abs(improved_trials[:, dim_idx] - improved_targets[:, dim_idx]) > 1e-6))
        
        if len(self.success_history_F) > 5:
            self.F = np.clip(np.mean(self.success_history_F[-10:]), 0.3, 1.5)
            self.CR = np.clip(np.mean(self.success_history_CR[-10:]), 0.1, 1.0)
            
    def _adapt_strategy_weights(self, better_count):
        reward = better_count / (self.NP * 0.2)
        self.strategy_weights = self.strategy_weights * (1.0 - 0.1) + 0.1 * reward
        self.strategy_weights = self.strategy_weights / np.sum(self.strategy_weights)
        
    def _adapt_neighborhood_size(self, diversity):
        if diversity < 0.1:
            self.neighborhood_size = min(self.NP // 2, int(self.neighborhood_size * 1.2))
        elif diversity > 0.5:
            self.neighborhood_size = max(5, int(self.neighborhood_size * 0.9))
            
    def _position_update_temporal_drift(self):
        if self.best_idx is not None:
            old_best_pos = self.population[self.best_idx].copy()
            
        current_best = self.population[np.nanargmin(self.fitness)]
        
        if self.drift_vector is None or self.gen < 2:
            self.drift_vector = np.zeros(self.dim)
        else:
            movement = current_best - self.best_pos
            self.drift_vector = 0.8 * self.drift_vector + 0.2 * movement
            
        self.best_pos = current_best.copy()
        
    def _position_update_fitness_rank(self):
        valid_fitness = self.fitness.copy()
        valid_fitness[np.isnan(valid_fitness)] = np.nanmax(valid_fitness[~np.isnan(valid_fitness)]) * 2
        ranks = np.argsort(np.argsort(valid_fitness))
        rank_weights = 1.0 / (ranks + 1)
        return rank_weights / np.sum(rank_weights)
    
    def _update_drift_vector(self, improved_mask):
        if np.sum(improved_mask) > 0:
            improved_pop = self.trial_population[improved_mask]
            improved_orig = self.population[improved_mask]
            movement = np.mean(improved_pop - improved_orig, axis=0)
            self.drift_vector = 0.7 * self.drift_vector + 0.3 * movement
            
    def _compute_diversity(self):
        valid_pop = self.population[~np.isnan(self.fitness)]
        if len(valid_pop) < 2:
            return 0.0
        centroid = np.mean(valid_pop, axis=0)
        distances = np.linalg.norm(valid_pop - centroid, axis=1)
        return np.mean(distances) / (self.bounds[1] - self.bounds[0])
    
    def _restart_if_stagnant(self, diversity, gen_no_improve):
        if gen_no_improve > 50 or diversity < 0.01:
            n_replace = max(self.NP // 4, 1)
            replace_idx = self._select_random_indices(n_replace)
            
            for idx in replace_idx:
                self.population[idx] = np.random.uniform(
                    self.bounds[0], self.bounds[1], size=self.dim
                )
            
            self.fitness = func(self.population)
            if len(self.fitness) < self.NP:
                self.fitness = np.resize(self.fitness, self.NP)
                self.fitness[len(self.fitness):] = np.nan
                
            self.gen = 0
            return True
        return False
    
    def _randomize_stagnant_individuals(self, gen_no_improve):
        if gen_no_improve > 20:
            n_replace = max(self.NP // 8, 1)
            replace_idx = self._select_random_indices(n_replace)
            
            for idx in replace_idx:
                self.population[idx] = np.random.uniform(
                    self.bounds[0], self.bounds[1], size=self.dim
                )
                self.fitness[idx] = np.nan
                
    def __call__(self, func, stopping_condition):
        self._initialize_population(func)
        
        gen_no_improve = 0
        last_best = self.best_fitness
        
        while not stopping_condition():
            self.gen += 1
            
            mutants = self._mutate_batch()
            trials = self._crossover_batch(self.population, mutants)
            
            if stopping_condition():
                break
                
            self.trial_population = trials
            self.trial_fitness = self._evaluate_trial_batch(trials, func)
            
            if len(self.trial_fitness) < len(trials):
                break
                
            improved_mask = self.trial_fitness < self.fitness
            better_count = self._select_survivors_batch()
            success_rate = better_count / self.NP
            
            self._update_drift_vector(improved_mask)
            self._position_update_temporal_drift()
            
            self._adapt_inertia_weight(success_rate)
            self._adapt_temporal_drift(success_rate)
            self._adapt_F_CR(self.trial_population, improved_mask)
            self._adapt_strategy_weights(better_count)
            
            diversity = self._compute_diversity()
            self._adapt_neighborhood_size(diversity)
            
            if self.best_fitness < last_best:
                gen_no_improve = 0
                last_best = self.best_fitness
            else:
                gen_no_improve += 1
            
            self._randomize_stagnant_individuals(gen_no_improve)
            
            if self._restart_if_stagnant(diversity, gen_no_improve):
                gen_no_improve = 0
                
        return self.best_fitness, self.population[self.best_idx]
