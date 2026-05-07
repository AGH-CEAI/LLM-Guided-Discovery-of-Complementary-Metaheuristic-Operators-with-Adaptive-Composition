import numpy as np


class ProximityHybridDifferentialEvolution:
    """
    Hybrid DE with proximity-based neighborhoods and Nelder-Mead local search.
    Uses current-to-p-best/1 mutation with adaptive archive, combined with
    batched simplex-based local search on elite individuals.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(5 * dim, 40), 200)
        self.bounds = np.array([[-100.0, 100.0]])
        self.f_bounds = np.array([[-100.0] * dim, [100.0] * dim])
        
        self.p_best_rate = 0.15
        self.min_adapt_rate = 0.3
        self.max_adapt_rate = 0.9
        self.archive_rate = 0.5
        self.local_search_interval = 5
        self.local_search_fraction = 0.15
        self.stagnation_threshold = 20
        
        self._f_scale = 0.5
        self._cr = 0.7
        self._success_f_scale = 0.0
        self._success_cr = 0.0
        self._success_count = 0
        self._adapt_counter = 0
        
        self._f_history = []
        self._cr_history = []
        
    def __call__(self, func, stopping_condition):
        self._setup_initial_state()
        population = self._initialize_population()
        population = self._clip_to_bounds(population)
        fitness = self._evaluate_batch(func, population)
        self._update_best(population, fitness)
        
        gen = 0
        stagnation = 0
        prev_best = self._fitness_best
        
        while not stopping_condition():
            trials = self._generate_trials_batch(population, fitness)
            trials = self._clip_to_bounds(trials)
            trial_fitness = self._evaluate_batch(func, trials)
            
            if len(trial_fitness) < len(trials):
                valid = len(trial_fitness)
                trials = trials[:valid]
                trial_fitness = trial_fitness[:valid]
                if valid == 0:
                    break
            
            if stopping_condition():
                break
                
            population, fitness = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            self._update_best(population, fitness)
            
            self._update_archive(population, fitness)
            self._adapt_parameters_batch(fitness, trial_fitness)
            
            stagnation = self._check_stagnation(stagnation)
            if stagnation >= self.stagnation_threshold:
                population = self._diversify_population(population, fitness)
                stagnation = 0
            
            if gen > 0 and gen % self.local_search_interval == 0:
                population = self._apply_local_search_batch(
                    func, population, fitness
                )
                fitness = self._evaluate_batch(func, population)
                self._update_best(population, fitness)
            
            gen += 1
            
        return self._fitness_best, self._best_x.copy()
    
    def _setup_initial_state(self):
        self._population = None
        self._fitness = None
        self._fitness_best = np.inf
        self._best_x = None
        self._archive = []
        self._stagnation_count = 0
        self._generation_count = 0
        
    def _initialize_population(self):
        self._population = np.random.uniform(
            self.f_bounds[0], self.f_bounds[1],
            size=(self.np, self.dim)
        )
        return self._population
    
    def _clip_to_bounds(self, population):
        return np.clip(population, self.f_bounds[0], self.f_bounds[1])
    
    def _evaluate_batch(self, func, population):
        raw = func(population)
        if raw.ndim == 2:
            raw = raw.ravel()
        valid = ~np.isnan(raw)
        result = np.full(len(raw), np.inf)
        result[valid] = raw[valid]
        return result
    
    def _update_best(self, population, fitness):
        idx = np.argmin(fitness)
        if fitness[idx] < self._fitness_best:
            self._fitness_best = fitness[idx]
            self._best_x = population[idx].copy()
    
    def _update_archive(self, population, fitness):
        improved = fitness < self._fitness_best
        for i in range(len(population)):
            if np.random.rand() < self.archive_rate:
                self._archive.append(population[i].copy())
        
        max_archive = self.np * 3
        if len(self._archive) > max_archive:
            keep = np.random.choice(len(self._archive), max_archive, replace=False)
            self._archive = [self._archive[k] for k in keep]
    
    def _generate_trials_batch(self, population, fitness):
        donors = self._mutate_batch(population, fitness)
        trials = self._crossover_batch(population, donors)
        return trials
    
    def _mutate_batch(self, population, fitness):
        np_pop = len(population)
        sorted_idx = np.argsort(fitness)
        p_best_idx = sorted_idx[:max(1, int(self.p_best_rate * np_pop))]
        base_idx = np.arange(np_pop)
        
        f1_idx = self._sample_different_indices(np_pop, base_idx)
        f2_idx = self._sample_different_indices(np_pop, np.concatenate([base_idx.reshape(-1, 1), f1_idx.reshape(-1, 1)], axis=1))
        
        base = population[base_idx]
        f1 = population[f1_idx]
        f2 = population[f2_idx]
        
        if len(self._archive) > 0:
            archive_array = np.array(self._archive)
            archive_samples = archive_array[np.random.randint(0, len(archive_array), size=np_pop)]
        else:
            archive_samples = f2.copy()
        
        p_best = population[np.random.choice(p_best_idx)]
        
        scale1 = self._f_scale
        scale2 = self._f_scale * 0.5
        
        donors = base + scale1 * (p_best - base) + scale2 * (archive_samples - f1)
        donors = self._clip_to_bounds(donors)
        
        return donors
    
    def _sample_different_indices(self, size, exclude):
        exclude_set = set(exclude.ravel()) if exclude.ndim > 1 else set(exclude)
        valid = [i for i in range(size) if i not in exclude_set]
        if len(valid) == 0:
            return np.random.randint(0, size, size=size)
        return np.array([np.random.choice(valid) for _ in range(size)])
    
    def _crossover_batch(self, population, donors):
        np_pop, dim = population.shape
        mask = np.random.rand(np_pop, dim) < self._cr
        trials = np.where(mask, donors, population)
        return trials
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        improved = trial_fitness < fitness
        survivors = population.copy()
        survivors[improved] = trials[improved]
        
        new_fitness = fitness.copy()
        new_fitness[improved] = trial_fitness[improved]
        
        return survivors, new_fitness
    
    def _adapt_parameters_batch(self, fitness, trial_fitness):
        self._adapt_counter += 1
        
        improved = trial_fitness < fitness
        success_rate = np.mean(improved)
        
        if success_rate > 0.05:
            self._success_count += 1
            self._f_history.append(self._f_scale)
            self._cr_history.append(self._cr)
        else:
            self._success_count = max(0, self._success_count - 1)
        
        if self._adapt_counter >= 5:
            if len(self._f_history) > 0:
                self._f_scale = np.clip(
                    np.mean(self._f_history) * 0.9 + 0.1 * np.random.uniform(0.3, 0.9),
                    0.3, 1.0
                )
                self._cr = np.clip(
                    np.mean(self._cr_history) * 0.9 + 0.1 * np.random.uniform(0.5, 0.95),
                    0.3, 0.95
                )
            self._f_history = []
            self._cr_history = []
            self._adapt_counter = 0
        
        if success_rate > 0.2:
            self._cr = np.clip(self._cr + 0.05, self.min_adapt_rate, self.max_adapt_rate)
        elif success_rate < 0.05:
            self._cr = np.clip(self._cr - 0.05, self.min_adapt_rate, self.max_adapt_rate)
    
    def _check_stagnation(self, stagnation):
        if self._fitness_best >= prev_best if 'prev_best' in dir() else True:
            stagnation += 1
        else:
            stagnation = 0
        return stagnation
    
    def _diversify_population(self, population, fitness):
        sorted_idx = np.argsort(fitness)
        elite_count = max(2, self.np // 5)
        elite = population[sorted_idx[:elite_count]]
        
        diversity = self._compute_diversity(population)
        if diversity < 0.1 * self.dim:
            noise = np.random.uniform(-20, 20, size=population.shape)
            new_pop = population + 0.3 * noise
            new_pop = self._clip_to_bounds(new_pop)
        else:
            new_pop = population.copy()
        
        new_pop[sorted_idx[:elite_count]] = elite
        return new_pop
    
    def _compute_diversity(self, population):
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def _apply_local_search_batch(self, func, population, fitness):
        n_ls = max(1, int(self.local_search_fraction * self.np))
        sorted_idx = np.argsort(fitness)
        top_idx = sorted_idx[:n_ls]
        top_pop = population[top_idx]
        
        ls_trials = self._generate_simplex_trials_batch(top_pop)
        ls_trials = self._clip_to_bounds(ls_trials)
        ls_fitness = self._evaluate_batch(func, ls_trials)
        
        improved = ls_fitness < fitness[top_idx]
        population[top_idx[improved]] = ls_trials[improved]
        
        return population
    
    def _generate_simplex_trials_batch(self, individuals):
        n_ind, dim = individuals.shape
        reflections = self._reflect_batch(individuals)
        expansions = self._expand_batch(individuals, reflections)
        contractions = self._contract_batch(individuals)
        
        n_reflect = n_ind
        n_expand = n_ind // 2
        n_contract = n_ind // 2
        
        trials = np.vstack([
            reflections[:n_reflect],
            expansions[:n_expand],
            contractions[:n_contract]
        ])
        
        return trials
    
    def _reflect_batch(self, individuals):
        centroid = np.mean(individuals, axis=0)
        reflection_coeff = 1.2
        reflected = centroid + reflection_coeff * (individuals - centroid)
        return reflected
    
    def _expand_batch(self, individuals, reflections):
        centroid = np.mean(individuals, axis=0)
        expansion_coeff = 1.8
        expanded = centroid + expansion_coeff * (reflections - centroid)
        return expanded
    
    def _contract_batch(self, individuals):
        centroid = np.mean(individuals, axis=0)
        contraction_coeff = 0.5
        contracted = centroid + contraction_coeff * (individuals - centroid)
        return contracted
