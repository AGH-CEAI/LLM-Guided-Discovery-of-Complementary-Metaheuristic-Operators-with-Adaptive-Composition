import numpy as np


class ArchipelagoCMAES:
    """Island model CMA-ES with adaptive strategies and intelligent migration."""
    
    def __init__(self, dim, n_islands=5, np_per_island=None):
        self.dim = dim
        self.n_islands = n_islands
        self.np_per_island = np_per_island if np_per_island else max(15, dim * 2)
        self.NP = self.n_islands * self.np_per_island
        self.bounds = np.array([[-100.0, 100.0]] * dim)
        self._rng = np.random.default_rng()
        
    def __call__(self, func, stopping_condition):
        self._func = func
        population = self._initialize_population()
        self._init_islands(population)
        self._best_fitness = np.inf
        self._best_solution = None
        self._gen = 0
        self._func_eval = 0
        
        while not stopping_condition():
            trials = self._build_trials_batched(population)
            trial_fitness = self._func(trials)
            self._func_eval += len(trial_fitness)
            if len(trial_fitness) < len(trials):
                break
            if stopping_condition():
                break
            population = self._select_survivors_batch(population, trials, trial_fitness)
            self._adapt_step_size_batch(population)
            self._update_strategy_scores_batch(population)
            if self._gen % 8 == 0:
                self._migrate_between_islands(population)
            self._restart_if_stagnant(population)
            self._gen += 1
        
        return self._best_fitness, self._best_solution
    
    def _initialize_population(self):
        r = self.bounds[:, 1] - self.bounds[:, 0]
        pop = self._rng.uniform(0, 1, (self.NP, self.dim)) * r + self.bounds[:, 0]
        return pop
    
    def _init_islands(self, population):
        self._islands = []
        for i in range(self.n_islands):
            start, end = i * self.np_per_island, (i + 1) * self.np_per_island
            island_pop = population[start:end].copy()
            fitness = self._func(island_pop)
            self._func_eval += len(fitness)
            best_idx = np.argmin(fitness)
            island = {
                'pop': island_pop,
                'fit': fitness,
                'step': 0.5 * np.ones(self.np_per_island),
                'mean': island_pop[best_idx].copy(),
                'cov': np.eye(self.dim),
                'strat_probs': np.ones(4) / 4,
                'strat_success': np.zeros(4),
                'history': np.full(5 + self.dim // 10, np.inf),
                'best': fitness[best_idx]
            }
            self._islands.append(island)
            if fitness[best_idx] < self._best_fitness:
                self._best_fitness = fitness[best_idx]
                self._best_solution = island_pop[best_idx].copy()
    
    def _build_trials_batched(self, population):
        trials_list = []
        for i, island in enumerate(self._islands):
            island_pop = island['pop']
            island_trials = self._mutate_island_batch(island_pop, island)
            island_trials = self._crossover_batch(island_pop, island_trials)
            island_trials = self._clip_to_bounds(island_trials)
            trials_list.append(island_trials)
        return np.vstack(trials_list)
    
    def _mutate_island_batch(self, island_pop, island):
        np_island = len(island_pop)
        n_trials = np_island
        trials = np.empty((n_trials, self.dim))
        idx_range = np.arange(np_island)
        
        for j in range(n_trials):
            r1, r2, r3 = self._rng.choice(idx_range, 3, replace=False)
            base = island_pop[r1]
            diff = island_pop[r2] - island_pop[r3]
            step = island['step'][j % len(island['step'])]
            trials[j] = base + step * diff
        
        return trials
    
    def _crossover_batch(self, pop, trials):
        cr = 0.85
        mask = self._rng.uniform(0, 1, trials.shape) < cr
        mask[:, 0] = True
        return np.where(mask, trials, pop)
    
    def _clip_to_bounds(self, pop):
        return np.clip(pop, self.bounds[:, 0], self.bounds[:, 1])
    
    def _select_survivors_batch(self, population, trials, trial_fitness):
        start = 0
        new_population = []
        for i, island in enumerate(self._islands):
            np_island = self.np_per_island
            end = start + np_island
            island_trials = trials[start:end]
            island_trial_fit = trial_fitness[start:end]
            
            old_fit = island['fit']
            new_fit = np.minimum(old_fit, island_trial_fit)
            better_mask = island_trial_fit < old_fit
            
            new_pop = island['pop'].copy()
            new_pop[better_mask] = island_trials[better_mask]
            island['pop'] = new_pop
            island['fit'] = new_fit
            
            best_idx = np.argmin(new_fit)
            if new_fit[best_idx] < island['best']:
                island['best'] = new_fit[best_idx]
                island['mean'] = new_pop[best_idx].copy()
                island['history'] = np.roll(island['history'], -1)
                island['history'][-1] = new_fit[best_idx]
            
            if new_fit[best_idx] < self._best_fitness:
                self._best_fitness = new_fit[best_idx]
                self._best_solution = new_pop[best_idx].copy()
            
            new_population.append(new_pop)
            start = end
        
        return np.vstack(new_population)
    
    def _adapt_step_size_batch(self, population):
        for island in self._islands:
            old_best = island['history'][0]
            new_best = island['history'][-1]
            improvement = old_best - new_best
            mean_step = np.mean(island['step'])
            
            if improvement > 1e-12:
                island['step'] *= 1.1
                island['step'] = np.clip(island['step'], 0.01, 2.0)
            elif island['gen_without_improve'] > self.dim:
                island['step'] *= 0.85
            else:
                island['step'] *= 0.98 + 0.02 * self._rng.uniform(0, 1, len(island['step']))
            
            island['step'] = np.clip(island['step'], 0.001, 3.0)
            
            if not hasattr(island, 'gen_without_improve'):
                island['gen_without_improve'] = 0
            if new_best < old_best - 1e-10:
                island['gen_without_improve'] = 0
            else:
                island['gen_without_improve'] += 1
    
    def _update_strategy_scores_batch(self, population):
        for island in self._islands:
            island['strat_success'] *= 0.95
            island['strat_success'] += self._rng.uniform(0, 1, 4) * 0.1
            
            total = np.sum(island['strat_success']) + 1e-10
            island['strat_probs'] = island['strat_success'] / total
    
    def _migrate_between_islands(self, population):
        for i in range(self.n_islands):
            donor_island = self._islands[(i + 1) % self.n_islands]
            recipient_island = self._islands[i]
            best_donor_idx = np.argmin(donor_island['fit'])
            migrant = donor_island['pop'][best_donor_idx].copy()
            replace_idx = self._rng.integers(0, self.np_per_island)
            recipient_island['pop'][replace_idx] = migrant
    
    def _restart_if_stagnant(self, population):
        for island in self._islands:
            history_range = island['history'].max() - island['history'].min()
            stagnation_threshold = max(1.0, abs(island['history'].mean()) * 0.1)
            
            if history_range < stagnation_threshold and island['gen_without_improve'] > self.dim * 2:
                seed_count = min(3, self.np_per_island // 5)
                best_indices = np.argsort(island['fit'])[:seed_count]
                seeds = island['pop'][best_indices].copy()
                
                r = self.bounds[:, 1] - self.bounds[:, 0]
                new_pop = self._rng.uniform(0, 1, (self.np_per_island, self.dim)) * r + self.bounds[:, 0]
                for j in range(seed_count):
                    new_pop[j] = seeds[j]
                
                island['pop'] = new_pop
                island['step'] = 0.5 * np.ones(self.np_per_island)
                island['history'] = np.full(len(island['history']), np.inf)
                island['gen_without_improve'] = 0
    
    def _compute_diversity(self, population):
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances) / (self.dim * 10)
