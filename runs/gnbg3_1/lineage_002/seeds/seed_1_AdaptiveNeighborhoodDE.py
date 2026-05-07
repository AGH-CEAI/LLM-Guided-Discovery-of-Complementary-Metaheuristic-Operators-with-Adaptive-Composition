import numpy as np


class AdaptiveNeighborhoodDE:
    """
    Adaptive Neighborhood Differential Evolution with Covariance Learning.
    
    Key innovations:
    - Non-standard mutation: blends current-to-pbest/1, ring-topology local mutation,
      and covariance-guided directional mutation
    - Adaptive strategy weights that evolve based on recent success
    - Covariance adaptation inspired by CMA-ES but integrated into DE framework
    - Ring topology with adaptive neighborhood radius
    - F and CR adaptation based on cumulative success history
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = kwargs.get('NP', max(60, min(300, 8 * dim)))
        self.lower = -100.0
        self.upper = 100.0
        self.p_best = kwargs.get('p_best', 0.15)
        self.arc_size = kwargs.get('arc_size', 2 * self.NP)
        
        self.F = 0.5
        self.CR = 0.9
        self.F_lb, self.F_ub = 0.1, 1.0
        self.CR_lb, self.CR_ub = 0.0, 1.0
        
        self.strategy_weights = np.array([0.4, 0.35, 0.25])
        self.success_history_F = []
        self.success_history_CR = []
        self.history_window = 15
        
        self.cov_matrix = np.eye(dim)
        self.cov_adapt_rate = 0.02
        self.cov_condition = 1e6
        
        self.neighbor_radius = 2
        self.diversity_threshold = 0.1
        
        self._best_fitness = np.inf
        self._best_position = None
        self._generation = 0
        self._stagnation_count = 0
        self._total_evals = 0
    
    def __call__(self, func, stopping_condition):
        self._reset_state()
        population = self._initialize_population()
        self._clip_to_bounds(population)
        fitness = self._evaluate_batch(func, population)
        self._update_best_tracking(population, fitness)
        
        while not stopping_condition():
            self._generation += 1
            
            trials = self._build_trials_batch(population, fitness)
            self._clip_to_bounds(trials)
            
            if len(trials) < len(population):
                trials = trials[:len(trials)]
            
            trial_fitness = self._evaluate_batch(func, trials)
            
            if len(trial_fitness) < len(trials):
                valid_len = len(trial_fitness)
                trials = trials[:valid_len]
                trial_fitness = trial_fitness[:valid_len]
                if stopping_condition():
                    break
            
            if stopping_condition():
                break
            
            population, fitness, archive = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            self._update_best_tracking(population, fitness)
            self._adapt_step_size(fitness, trial_fitness)
            self._adapt_crossover_rate(fitness, trial_fitness)
            self._adapt_strategy_weights(population, fitness, trials, trial_fitness)
            self._adapt_covariance(population, fitness)
            self._adapt_neighborhood_radius(population)
            self._restart_if_stagnant(population, fitness, stopping_condition)
        
        return self._best_fitness, self._best_position.copy()
    
    def _reset_state(self):
        self._best_fitness = np.inf
        self._best_position = None
        self._generation = 0
        self._stagnation_count = 0
        self._total_evals = 0
        self.cov_matrix = np.eye(self.dim)
        self.strategy_weights = np.array([0.4, 0.35, 0.25])
        self.archive = []
    
    def _initialize_population(self):
        rng = np.random.default_rng()
        population = self.lower + rng.uniform(0, 1, (self.NP, self.dim)) * (self.upper - self.lower)
        return population
    
    def _evaluate_batch(self, func, population):
        self._total_evals += len(population)
        fitness = func(population)
        if isinstance(fitness, list):
            fitness = np.array(fitness)
        return fitness
    
    def _clip_to_bounds(self, population):
        np.clip(population, self.lower, self.upper, out=population)
    
    def _update_best_tracking(self, population, fitness):
        valid_mask = ~np.isnan(fitness)
        if not np.any(valid_mask):
            return
        
        valid_fitness = fitness[valid_mask]
        valid_population = population[valid_mask]
        
        min_idx = np.argmin(valid_fitness)
        if valid_fitness[min_idx] < self._best_fitness:
            self._best_fitness = valid_fitness[min_idx]
            self._best_position = valid_population[min_idx].copy()
            self._stagnation_count = 0
        else:
            self._stagnation_count += 1
    
    def _build_trials_batch(self, population, fitness):
        rng = np.random.default_rng()
        trials = np.empty_like(population)
        
        valid_mask = ~np.isnan(fitness)
        if np.sum(valid_mask) < 3:
            return self._mutate_current_to_pbest(population, fitness)
        
        for i in range(self.NP):
            r = rng.random()
            if r < self.strategy_weights[0]:
                trial = self._mutate_current_to_pbest_single(population, fitness, i)
            elif r < self.strategy_weights[0] + self.strategy_weights[1]:
                trial = self._mutate_ring_local_single(population, i)
            else:
                trial = self._mutate_covariance_guided_single(population, fitness, i)
            
            trials[i] = self._crossover_single(population[i], trial)
        
        return trials
    
    def _mutate_current_to_pbest(self, population, fitness):
        rng = np.random.default_rng()
        n_best = max(1, int(self.p_best * self.NP))
        sorted_indices = np.argsort(fitness)
        pbest_indices = sorted_indices[:n_best]
        
        trials = np.empty_like(population)
        for i in range(self.NP):
            r1, r2 = self._select_distinct_indices(i, 2, self.NP)
            
            pbest_idx = rng.choice(pbest_indices)
            base = population[i]
            direction = population[pbest_idx] - population[i]
            
            F_adapted = self.F * rng.uniform(0.9, 1.1)
            trial = base + F_adapted * direction + F_adapted * (population[r1] - population[r2])
            trials[i] = trial
        
        return trials
    
    def _mutate_current_to_pbest_single(self, population, fitness, idx):
        rng = np.random.default_rng()
        n_best = max(1, int(self.p_best * self.NP))
        sorted_indices = np.argsort(fitness)
        pbest_idx = rng.choice(sorted_indices[:n_best])
        
        r1, r2 = self._select_distinct_indices(idx, 2, self.NP)
        
        F_var = self.F * rng.uniform(0.8, 1.2)
        base = population[idx]
        direction = population[pbest_idx] - base
        
        return base + F_var * direction + F_var * (population[r1] - population[r2])
    
    def _mutate_ring_local_single(self, population, idx):
        rng = np.random.default_rng()
        n_neighbors = 2 * self.neighbor_radius + 1
        
        indices = np.arange(idx - self.neighbor_radius, idx + self.neighbor_radius + 1)
        indices = ((indices % self.NP) + self.NP) % self.NP
        
        rng.shuffle(indices)
        neighbors = indices[:min(n_neighbors, len(indices))]
        
        r1, r2, r3 = neighbors[0], neighbors[1], neighbors[2] if len(neighbors) > 2 else neighbors[0]
        
        F_local = self.F * rng.uniform(0.7, 1.3)
        return population[r1] + F_local * (population[r2] - population[r3])
    
    def _mutate_covariance_guided_single(self, population, fitness, idx):
        rng = np.random.default_rng()
        r1 = rng.integers(self.NP)
        while r1 == idx:
            r1 = rng.integers(self.NP)
        
        base = population[r1]
        
        try:
            L = np.linalg.cholesky(self.cov_matrix)
            z = rng.standard_normal(self.dim)
            cov_direction = L @ z
        except np.linalg.LinAlgError:
            cov_direction = rng.standard_normal(self.dim)
        
        cov_direction = cov_direction / (np.linalg.norm(cov_direction) + 1e-10)
        
        F_cov = self.F * rng.uniform(0.5, 1.5)
        return base + F_cov * cov_direction
    
    def _crossover_single(self, target, mutant):
        rng = np.random.default_rng()
        dim = len(target)
        trial = np.empty(dim)
        
        j_rand = rng.integers(dim)
        
        for j in range(dim):
            if j == j_rand or rng.random() < self.CR:
                trial[j] = mutant[j]
            else:
                trial[j] = target[j]
        
        return trial
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        survived = np.empty_like(population)
        new_fitness = np.empty_like(fitness)
        
        improved = fitness > trial_fitness
        not_nan = ~np.isnan(trial_fitness)
        
        for i in range(len(population)):
            if not_nan[i] and improved[i]:
                survived[i] = trials[i]
                new_fitness[i] = trial_fitness[i]
                if len(self.archive) < self.arc_size:
                    self.archive.append(population[i].copy())
                elif np.random.random() < 0.1:
                    idx = np.random.randint(len(self.archive))
                    self.archive[idx] = population[i].copy()
            else:
                survived[i] = population[i]
                new_fitness[i] = fitness[i]
        
        return survived, new_fitness, self.archive
    
    def _adapt_step_size(self, fitness, trial_fitness):
        improved = (fitness > trial_fitness) & ~np.isnan(trial_fitness)
        success_rate = np.mean(improved)
        
        self.success_history_F.append(success_rate)
        if len(self.success_history_F) > self.history_window:
            self.success_history_F.pop(0)
        
        if len(self.success_history_F) >= 5:
            recent_avg = np.mean(self.success_history_F[-5:])
            if recent_avg > 0.2:
                self.F = min(self.F_ub, self.F * 1.1)
            elif recent_avg < 0.1:
                self.F = max(self.F_lb, self.F * 0.9)
    
    def _adapt_crossover_rate(self, fitness, trial_fitness):
        improved = (fitness > trial_fitness) & ~np.isnan(trial_fitness)
        
        if np.sum(improved) > 0:
            successful_CR = self.CR * np.ones(np.sum(improved))
            mean_CR = np.mean(successful_CR)
            self.success_history_CR.append(mean_CR)
        
        if len(self.success_history_CR) > self.history_window:
            self.success_history_CR.pop(0)
        
        if len(self.success_history_CR) >= 10:
            target_rate = 0.5
            recent_avg = np.mean(self.success_history_CR[-10:])
            if recent_avg > target_rate:
                self.CR = min(self.CR_ub, self.CR + 0.05)
            else:
                self.CR = max(self.CR_lb, self.CR - 0.05)
    
    def _adapt_strategy_weights(self, population, fitness, trials, trial_fitness):
        improved = (fitness > trial_fitness) & ~np.isnan(trial_fitness)
        
        if np.sum(improved) < 3:
            return
        
        n_best = max(1, int(self.p_best * self.NP))
        sorted_indices = np.argsort(fitness)
        pbest_indices = set(sorted_indices[:n_best])
        
        strategy_scores = np.zeros(3)
        
        for i in range(len(population)):
            if not improved[i]:
                continue
            
            original = population[i]
            mutant = trials[i]
            
            pbest_idx = None
            for idx in pbest_indices:
                if np.array_equal(original, population[idx]):
                    continue
                pbest_idx = idx
                break
            
            if pbest_idx is not None:
                diff = np.linalg.norm(mutant - (population[i] + self.F * (population[pbest_idx] - population[i])))
                if diff < 0.1 * np.linalg.norm(population[pbest_idx] - population[i]):
                    strategy_scores[0] += 1
                    continue
            
            ring_diff = 0
            for dr in range(-self.neighbor_radius, self.neighbor_radius + 1):
                neighbor_idx = (i + dr) % self.NP
                if neighbor_idx != i and neighbor_idx < len(population):
                    ring_diff += np.linalg.norm(mutant - population[neighbor_idx])
            
            if ring_diff > 0:
                base_diff = np.linalg.norm(population[(i + 1) % self.NP] - population[(i - 1) % self.NP])
                if ring_diff < base_diff * self.neighbor_radius * 0.8:
                    strategy_scores[1] += 1
                    continue
            
            strategy_scores[2] += 1
        
        total = np.sum(strategy_scores)
        if total > 0:
            new_weights = strategy_scores / total
            self.strategy_weights = 0.7 * self.strategy_weights + 0.3 * new_weights
            self.strategy_weights = np.clip(self.strategy_weights, 0.1, 0.9)
            self.strategy_weights /= np.sum(self.strategy_weights)
    
    def _adapt_covariance(self, population, fitness):
        valid_mask = ~np.isnan(fitness)
        if np.sum(valid_mask) < 5:
            return
        
        n_elite = max(3, self.NP // 10)
        sorted_indices = np.argsort(fitness)
        elite_indices = sorted_indices[:n_elite]
        
        elite_pop = population[elite_indices]
        mean_elite = np.mean(elite_pop, axis=0)
        
        cov_update = np.zeros((self.dim, self.dim))
        for x in elite_pop:
            diff = (x - mean_elite).reshape(-1, 1)
            cov_update += diff @ diff.T
        
        cov_update /= n_elite
        
        self.cov_matrix = (1 - self.cov_adapt_rate) * self.cov_matrix + self.cov_adapt_rate * cov_update
        
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        eigenvalues = np.minimum(eigenvalues, self.cov_condition)
        self.cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    def _adapt_neighborhood_radius(self, population):
        diversity = self._compute_diversity(population)
        
        if diversity < self.diversity_threshold and self.neighbor_radius < self.NP // 4:
            self.neighbor_radius += 1
        elif diversity > 2 * self.diversity_threshold and self.neighbor_radius > 1:
            self.neighbor_radius = max(1, self.neighbor_radius - 1)
    
    def _compute_diversity(self, population):
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances) / (self.upper - self.lower)
    
    def _restart_if_stagnant(self, population, fitness, stopping_condition):
        max_stagnation = 50 + self.dim
        
        if self._stagnation_count > max_stagnation:
            if stopping_condition():
                return
            
            n_replace = self.NP // 4
            rng = np.random.default_rng()
            replace_indices = rng.choice(self.NP, n_replace, replace=False)
            
            for idx in replace_indices:
                population[idx] = self.lower + rng.uniform(0, 1, self.dim) * (self.upper - self.lower)
            
            self._clip_to_bounds(population)
            self.F = min(self.F_ub, self.F * 1.5)
            self._stagnation_count = 0
    
    def _select_distinct_indices(self, exclude, count, pool_size):
        rng = np.random.default_rng()
        indices = []
        while len(indices) < count:
            idx = rng.integers(pool_size)
            if idx != exclude and idx not in indices:
                indices.append(idx)
        return indices
