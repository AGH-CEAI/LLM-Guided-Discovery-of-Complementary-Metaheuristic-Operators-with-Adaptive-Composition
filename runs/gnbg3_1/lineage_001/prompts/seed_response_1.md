```python
import numpy as np


class AdaptiveArchiveDifferentialEvolution:
    """
    Adaptive Archive DE with Novelty-Driven Mutation and Ring Topology.
    
    Key innovations:
    - Current-to-pbest mutation with archive-based diversity injection
    - Ring topology with adaptive neighborhood sampling
    - Novelty scores computed in batch for diversity maintenance
    - Dual-track parameter adaptation (F, Cr) with weighted success memory
    - Stagnation-triggered reinitialization with diversity preservation
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(150, max(60, 5 * dim))  # population size
        self.archive = []
        self.archive_max = 2 * self.np
        
        # DE control parameters
        self.f = 0.5 + 0.1 * np.random.randn()
        self.cr = 0.5 + 0.1 * np.random.randn()
        
        # Success history for adaptation
        self.f_history = []
        self.cr_history = []
        self.history_size = 10
        
        # Novelty detection
        self.k_nearest = max(5, self.np // 10)
        self.novelty_weights = np.ones(self.np) / self.np
        
        # Stagnation tracking
        self.stagnation_count = 0
        self.stagnation_threshold = 25
        self.best_fitness_history = []
        
        # Diversity thresholds
        self.diversity_threshold = 1e-4 * (200.0 / dim)
        
        # Bounds
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
    def __call__(self, func, stopping_condition):
        return self._optimize(func, stopping_condition)
    
    def _initialize_population(self, func):
        """Initialize population uniformly in bounds and evaluate."""
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, size=(self.np, self.dim)
        )
        population = self._clip_to_bounds(population)
        fitness = func(population)
        
        if len(fitness) < self.np:
            valid = len(fitness)
            fitness = fitness[:valid]
            population = population[:valid]
            self.np = valid
        
        self.archive = [population[i].copy() for i in range(self.np)]
        self._update_best(population, fitness)
        return population, fitness
    
    def _clip_to_bounds(self, population):
        """Clip all individuals to search bounds."""
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def _update_best(self, population, fitness):
        """Track global best solution."""
        valid_mask = ~np.isnan(fitness)
        if not np.any(valid_mask):
            return
        valid_fitness = fitness[valid_mask]
        valid_pop = population[valid_mask]
        best_idx = np.argmin(valid_fitness)
        self.f_opt = valid_fitness[best_idx]
        self.x_opt = valid_pop[best_idx].copy()
    
    def _compute_ring_neighbors(self, indices):
        """Get ring topology neighbors for given indices."""
        n = self.np
        half_span = max(1, n // 8)
        neighbors = []
        for idx in indices:
            start = (idx - half_span) % n
            end = (idx + half_span) % n
            if start < end:
                neighbor_set = set(range(start, end + 1))
            else:
                neighbor_set = set(range(start, n)) | set(range(0, end + 1))
            neighbor_set.discard(idx)
            neighbors.append(list(neighbor_set))
        return neighbors
    
    def _compute_novelty_scores(self, population):
        """Compute novelty scores based on k-nearest neighbor distances."""
        n = len(population)
        if n <= self.k_nearest:
            self.novelty_weights = np.ones(n) / n
            return np.ones(n)
        
        distances = np.zeros((n, n))
        for i in range(n):
            diffs = population - population[i]
            distances[i] = np.sqrt(np.sum(diffs ** 2, axis=1))
        
        np.fill_diagonal(distances, np.inf)
        sorted_distances = np.sort(distances, axis=1)
        k_distances = sorted_distances[:, :self.k_nearest]
        novelty = np.mean(k_distances, axis=1)
        
        novelty += 1e-10
        self.novelty_weights = novelty / np.sum(novelty)
        return novelty
    
    def _select_pbest_indices(self, fitness, n_select):
        """Select pbest indices weighted by fitness (lower is better)."""
        valid_mask = ~np.isnan(fitness)
        if not np.any(valid_mask):
            return np.random.choice(self.np, size=n_select, replace=True)
        
        inverted_fitness = np.where(valid_mask, 1.0 / (fitness + 1e-10), np.inf)
        weights = inverted_fitness / np.sum(inverted_fitness)
        weights = np.clip(weights, 0, 0.1)
        weights = weights / np.sum(weights)
        return np.random.choice(self.np, size=n_select, replace=True, p=weights)
    
    def _mutate_batch(self, population, fitness):
        """Generate mutant vectors using current-to-pbest/1/bin with archive."""
        np_pop = len(population)
        mutants = np.empty((np_pop, self.dim))
        
        pbest_indices = self._select_pbest_indices(fitness, np_pop)
        neighbors_list = self._compute_ring_neighbors(np.arange(np_pop))
        
        for i in range(np_pop):
            x_i = population[i]
            pbest = population[pbest_indices[i]]
            neighbors = neighbors_list[i]
            
            r1 = np.random.choice(neighbors)
            x_r1 = population[r1]
            
            # Add archive contribution with probability
            if np.random.random() < 0.3 and len(self.archive) > self.np:
                archive_sample = self.archive[np.random.randint(len(self.archive))]
                archive_contribution = 0.5 * (archive_sample - x_i)
            else:
                r2 = np.random.choice(neighbors)
                while r2 == r1:
                    r2 = np.random.choice(neighbors)
                archive_contribution = 0.5 * (population[r2] - x_i)
            
            # Current-to-pbest mutation with adaptive F
            f_adapted = self.f * (0.9 + 0.2 * np.random.random())
            mutants[i] = x_i + f_adapted * (pbest - x_i) + f_adapted * (x_r1 - x_i) + archive_contribution
        
        return mutants
    
    def _crossover_batch(self, population, mutants):
        """Binomial crossover with adaptive Cr."""
        np_pop = len(population)
        trials = np.empty((np_pop, self.dim))
        
        cr_adapted = np.clip(self.cr + 0.1 * np.random.randn(), 0.0, 1.0)
        j_rand = np.random.randint(0, self.dim, size=np_pop)
        
        for i in range(np_pop):
            cr_i = cr_adapted if np.random.random() < 0.8 else np.random.uniform(0.5, 0.9)
            mask = np.random.random(self.dim) < cr_i
            mask[j_rand[i]] = True
            trials[i] = np.where(mask, mutants[i], population[i])
        
        return trials
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Greedy selection with archive update."""
        np_pop = len(population)
        new_population = np.empty((np_pop, self.dim))
        new_fitness = np.empty(np_pop)
        
        improvements = trial_fitness < fitness
        n_improved = np.sum(improvements)
        
        for i in range(np_pop):
            if improvements[i]:
                new_population[i] = trials[i]
                new_fitness[i] = trial_fitness[i]
                if n_improved < len(self.archive):
                    self.archive.append(trials[i].copy())
            else:
                new_population[i] = population[i]
                new_fitness[i] = fitness[i]
        
        # Trim archive
        if len(self.archive) > self.archive_max:
            indices = np.random.choice(len(self.archive), self.archive_max, replace=False)
            self.archive = [self.archive[j] for j in indices]
        
        return new_population, new_fitness
    
    def _adapt_parameters(self, fitness_before, fitness_after, trials):
        """Adapt F and Cr based on successful mutations."""
        improvements = fitness_after < fitness_before
        n_success = np.sum(improvements)
        
        if n_success > 0:
            success_indices = np.where(improvements)[0]
            f_values = []
            cr_values = []
            
            for idx in success_indices:
                diff = np.abs(trials[idx] - fitness_before[idx])
                if diff > 1e-10:
                    f_i = np.random.uniform(0.4, 0.9)
                    cr_i = np.mean(np.abs(trials[idx] != fitness_before[idx]))
                    f_values.append(f_i)
                    cr_values.append(cr_i)
            
            if f_values:
                self.f_history.append(np.mean(f_values))
                self.cr_history.append(np.mean(cr_values))
            
            if len(self.f_history) > self.history_size:
                self.f_history.pop(0)
                self.cr_history.pop(0)
        
        if self.f_history:
            self.f = 0.9 * self.f + 0.1 * np.mean(self.f_history)
            self.cr = 0.9 * self.cr + 0.1 * np.mean(self.cr_history)
        else:
            self.f = np.clip(self.f + 0.1 * np.random.randn(), 0.2, 1.5)
            self.cr = np.clip(self.cr + 0.1 * np.random.randn(), 0.0, 1.0)
    
    def _compute_diversity(self, population):
        """Compute population diversity as average pairwise distance."""
        n = len(population)
        if n < 2:
            return 0.0
        
        centroid = np.mean(population, axis=0)
        distances = np.sqrt(np.sum((population - centroid) ** 2, axis=1))
        return np.mean(distances)
    
    def _restart_if_stagnant(self, population, fitness):
        """Reinitialize portion of population if stagnation detected."""
        self.best_fitness_history.append(self.f_opt)
        if len(self.best_fitness_history) > 20:
            self.best_fitness_history.pop(0)
        
        if len(self.best_fitness_history) >= 20:
            improvement = np.max(self.best_fitness_history) - np.min(self.best_fitness_history)
            if improvement < 1e-6:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
        
        diversity = self._compute_diversity(population)
        needs_restart = (self.stagnation_count >= 3) or (diversity < self.diversity_threshold)
        
        if needs_restart:
            n_replace = max(10, self.np // 4)
            replace_indices = np.random.choice(self.np, size=n_replace, replace=False)
            
            for idx in replace_indices:
                population[idx] = np.random.uniform(
                    self.lower_bound, self.upper_bound, size=self.dim
                )
            
            population = self._clip_to_bounds(population)
            fitness = func(population)
            
            if len(fitness) < self.np:
                valid = len(fitness)
                fitness = fitness[:valid]
                population = population[:valid]
                self.np = valid
            
            self.stagnation_count = 0
            self._update_best(population, fitness)
        
        return population, fitness
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop orchestrating all batch operations."""
        self.f_opt = np.inf
        self.x_opt = np.zeros(self.dim)
        
        if stopping_condition():
            return self.f_opt, self.x_opt
        
        population, fitness = self._initialize_population(func)
        self._update_best(population, fitness)
        
        generation = 0
        
        while not stopping_condition():
            generation += 1
            
            mutants = self._mutate_batch(population, fitness)
            mutants = self._clip_to_bounds(mutants)
            
            trials = self._crossover_batch(population, mutants)
            trials = self._clip_to_bounds(trials)
            
            if stopping_condition():
                break
            
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                valid = len(trial_fitness)
                trials = trials[:valid]
                trial_fitness = trial_fitness[:valid]
            
            if len(trial_fitness) == 0:
                break
            
            fitness_before = fitness[:len(trial_fitness)]
            population, fitness = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            self._update_best(population, fitness)
            self._adapt_parameters(fitness_before, trial_fitness, trials)
            self._compute_novelty_scores(population)
            
            if generation % 10 == 0:
                population, fitness = self._restart_if_stagnant(population, fitness)
            
            if stopping_condition():
                break
        
        return self.f_opt, self.x_opt
```