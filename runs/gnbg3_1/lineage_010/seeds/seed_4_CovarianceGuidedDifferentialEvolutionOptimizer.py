import numpy as np


class CovarianceGuidedDifferentialEvolutionOptimizer:
    """
    Hybrid optimizer combining DE-style mutation guided by covariance adaptation
    with periodic Nelder-Mead local refinement. Uses adaptive operator selection
    and diversity-aware restarts for multi-modal problems.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 60), 300)
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        self.population = None
        self.fitness = None
        self.trial_population = None
        self.trial_fitness = None
        
        self.cov_matrix = None
        self.step_size = 0.5
        self.mean = None
        
        self.x_opt = None
        self.f_opt = np.inf
        
        self.generation = 0
        self.last_improvement_gen = 0
        self.stagnation_count = 0
        self.max_stagnation = 50
        
        self.num_operators = 4
        self.operator_weights = np.ones(self.num_operators)
        self.operator_success = np.zeros(self.num_operators)
        self.operator_attempts = np.zeros(self.num_operators) + 1e-6
        
        self.path_c = np.zeros(dim)
        self.path_sigma = np.zeros(dim)
        self.cov_factor = 1.0
        
        self.diversity_threshold = 1e-6
        self.local_search_interval = 5
        self.local_search_fraction = 0.2
    
    def _initialize_population(self, func):
        """Initialize population within bounds and evaluate fitness."""
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, size=(self.np, self.dim)
        )
        self.population = np.clip(self.population, self.lower_bound, self.upper_bound)
        self.fitness = func(self.population)
        
        best_idx = np.argmin(self.fitness)
        self.x_opt = self.population[best_idx].copy()
        self.f_opt = self.fitness[best_idx]
        self.last_improvement_gen = 0
    
    def _initialize_covariance(self):
        """Initialize covariance matrix to identity scaled by step size."""
        self.cov_matrix = np.eye(self.dim)
        self.mean = np.mean(self.population, axis=0)
        self.path_c = np.zeros(self.dim)
        self.path_sigma = np.zeros(self.dim)
        self.cov_factor = 1.0
    
    def _sample_mutation_vectors_batch(self):
        """Sample mutation vectors using current covariance structure."""
        L = np.linalg.cholesky(self.cov_matrix + 1e-8 * np.eye(self.dim))
        z = np.random.randn(self.np, self.dim)
        return z @ L.T * self.step_size
    
    def _mutate_current_to_best(self, indices, mutation_base):
        """DE/current-to-best/1 mutation strategy."""
        r1, r2 = self._select_random_indices(self.np, 2, exclude=indices)
        diff = self.x_opt - self.population[mutation_base]
        F = np.random.uniform(0.4, 0.9)
        mutation = self.population[mutation_base] + F * diff + self._sample_mutation_vectors_batch()[mutation_base]
        return np.clip(mutation, self.lower_bound, self.upper_bound), r1, r2
    
    def _mutate_rand_to_best(self, indices, mutation_base):
        """DE/rand-to-best/1 mutation strategy."""
        r1, r2 = self._select_random_indices(self.np, 2, exclude=indices)
        diff = self.x_opt - self.population[mutation_base]
        F = np.random.uniform(0.3, 1.0)
        mutation = self.population[mutation_base] + F * diff + self._sample_mutation_vectors_batch()[mutation_base]
        return np.clip(mutation, self.lower_bound, self.upper_bound), r1, r2
    
    def _mutate_de_current_to_pbest(self, indices, mutation_base, p=0.1):
        """DE/current-to-pbest/1 mutation with archive."""
        np_best = max(int(p * self.np), 2)
        pbest_indices = np.argsort(self.fitness)[:np_best]
        pbest = self.population[np.random.choice(pbest_indices)]
        r1, r2 = self._select_random_indices(self.np, 2, exclude=indices)
        F = np.random.uniform(0.4, 0.9)
        mutation = self.population[mutation_base] + F * (pbest - self.population[mutation_base]) + \
                   self._sample_mutation_vectors_batch()[mutation_base]
        return np.clip(mutation, self.lower_bound, self.upper_bound), r1, r2
    
    def _mutate_composite(self, indices, mutation_base):
        """Composite mutation: combination of multiple strategies."""
        r1, r2, r3 = self._select_random_indices(self.np, 3, exclude=indices)
        F1 = np.random.uniform(0.4, 0.9)
        F2 = np.random.uniform(0.3, 0.7)
        mutation = self.population[r1] + F1 * (self.population[r2] - self.population[r3]) + \
                   F2 * (self.x_opt - self.population[mutation_base]) + \
                   self._sample_mutation_vectors_batch()[mutation_base]
        return np.clip(mutation, self.lower_bound, self.upper_bound), r1, r2
    
    def _select_random_indices(self, count, num_indices, exclude=None):
        """Select random unique indices excluding specified ones."""
        all_indices = set(range(self.np))
        if exclude is not None:
            all_indices -= set(exclude) if isinstance(exclude, (list, np.ndarray)) else {exclude}
        all_indices = list(all_indices)
        return np.random.choice(all_indices, size=num_indices, replace=False)
    
    def _select_mutation_strategy(self):
        """Select mutation strategy based on adaptive weights."""
        weights = self.operator_weights / np.sum(self.operator_weights)
        return np.random.choice(self.num_operators, p=weights)
    
    def _build_trial_population_batch(self):
        """Build trial population using selected mutation strategies."""
        trials = np.empty((self.np, self.dim))
        strategy_used = np.zeros(self.np, dtype=int)
        
        mutation_funcs = [
            self._mutate_current_to_best,
            self._mutate_rand_to_best,
            self._mutate_de_current_to_pbest,
            self._mutate_composite
        ]
        
        for i in range(self.np):
            strategy = self._select_mutation_strategy()
            strategy_used[i] = strategy
            mutation_base = i
            trials[i], _, _ = mutation_funcs[strategy](i, mutation_base)
        
        return trials, strategy_used
    
    def _crossover_batch(self, trials):
        """Binomial crossover between population and trials."""
        cr = np.random.uniform(0.5, 0.9)
        j_rand = np.random.randint(0, self.dim)
        
        mask = np.random.rand(self.np, self.dim) < cr
        mask[np.arange(self.np), j_rand] = True
        
        cross_trials = np.where(mask, trials, self.population)
        return np.clip(cross_trials, self.lower_bound, self.upper_bound)
    
    def _select_survivors_batch(self, trials, trial_fitness, strategy_used):
        """Greedy selection and update operator rewards."""
        better_mask = trial_fitness < self.fitness
        
        self.population = np.where(better_mask[:, np.newaxis], trials, self.population)
        self.fitness = np.where(better_mask, trial_fitness, self.fitness)
        
        for op_idx in range(self.num_operators):
            op_better = better_mask & (strategy_used == op_idx)
            self.operator_success[op_idx] += np.sum(op_better)
            self.operator_attempts[op_idx] += np.sum(strategy_used == op_idx)
        
        improvements = np.sum(better_mask)
        if improvements > 0:
            self.last_improvement_gen = self.generation
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.f_opt:
                self.x_opt = self.population[best_idx].copy()
                self.f_opt = self.fitness[best_idx]
        
        return improvements
    
    def _update_operator_weights(self):
        """Update operator selection weights based on success rates."""
        success_rates = self.operator_success / (self.operator_attempts + 1e-6)
        normalized_rates = success_rates / (np.sum(success_rates) + 1e-6)
        self.operator_weights = 0.9 * self.operator_weights + 0.1 * (normalized_rates * self.num_operators)
        self.operator_weights = np.clip(self.operator_weights, 0.1, 2.0)
        
        self.operator_success.fill(0.0)
        self.operator_attempts.fill(1e-6)
    
    def _adapt_step_size(self):
        """Adapt step size based on evolution path."""
        z = np.linalg.solve(np.linalg.cholesky(self.cov_matrix + 1e-8 * np.eye(self.dim)),
                           (self.mean - self.x_opt))
        self.path_sigma = (1 - self.cov_factor) * self.path_sigma + \
                         np.sqrt(self.cov_factor * (2 - self.cov_factor)) * z
        
        sigma_diff = np.linalg.norm(self.path_sigma) - self.dim ** 0.5
        adaptation_rate = 0.1
        self.step_size *= np.exp(adaptation_rate * sigma_diff / (self.dim ** 0.5 + 1e-8))
        self.step_size = np.clip(self.step_size, 1e-6, 50.0)
    
    def _adapt_covariance(self):
        """CMA-ES style covariance matrix adaptation."""
        old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        
        y = (self.mean - old_mean) / (self.step_size + 1e-8)
        self.path_c = (1 - self.cov_factor) * self.path_c + \
                     np.sqrt(self.cov_factor * (2 - self.cov_factor)) * y
        
        rank_one_update = self.cov_factor * np.outer(self.path_c, self.path_c)
        
        weights = np.ones(self.np) / self.np
        z = (self.population - old_mean) / (self.step_size + 1e-8)
        rank_mu_update = np.sum(weights[:, np.newaxis] * (z ** 2), axis=0)
        
        self.cov_matrix = (1 - self.cov_factor) * self.cov_matrix + rank_one_update + \
                         self.cov_factor * np.diag(rank_mu_update)
        
        self.cov_matrix = (self.cov_matrix + self.cov_matrix.T) / 2
        eigenvalues = np.linalg.eigvalsh(self.cov_matrix)
        if np.any(eigenvalues < 1e-10):
            self.cov_matrix += 1e-6 * np.eye(self.dim)
    
    def _compute_diversity(self):
        """Compute population diversity using average Euclidean distance."""
        if self.population is None or len(self.population) < 2:
            return 1.0
        
        centroids = np.mean(self.population, axis=0)
        distances = np.sqrt(np.sum((self.population - centroids) ** 2, axis=1))
        return np.mean(distances) / (self.upper_bound - self.lower_bound + 1e-8)
    
    def _local_search_batch(self, func):
        """Apply Nelder-Mead simplex refinement to top candidates."""
        num_refine = max(int(self.local_search_fraction * self.np), 1)
        sorted_indices = np.argsort(self.fitness)[:num_refine]
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        for idx in sorted_indices:
            if np.random.rand() > 0.3:
                continue
            
            x_curr = self.population[idx].copy()
            f_curr = self.fitness[idx]
            
            simplex = np.tile(x_curr, (self.dim + 1, 1))
            for j in range(self.dim):
                simplex[j + 1, j] += sigma * self.step_size * np.random.randn()
            simplex = np.clip(simplex, self.lower_bound, self.upper_bound)
            
            simplex_fitness = func(simplex)
            
            for _ in range(10):
                sorted_order = np.argsort(simplex_fitness)
                simplex = simplex[sorted_order]
                simplex_fitness = simplex_fitness[sorted_order]
                
                x_o = np.mean(simplex[:-1], axis=0)
                x_r = x_o + alpha * (x_o - simplex[-1])
                x_r = np.clip(x_r, self.lower_bound, self.upper_bound)
                f_r = func(x_r.reshape(1, -1))[0]
                
                if f_r < simplex_fitness[0]:
                    x_e = x_o + gamma * (x_r - x_o)
                    x_e = np.clip(x_e, self.lower_bound, self.upper_bound)
                    f_e = func(x_e.reshape(1, -1))[0]
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_fitness[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_fitness[-1] = f_r
                else:
                    if f_r < simplex_fitness[-2]:
                        simplex[-1] = x_r
                        simplex_fitness[-1] = f_r
                    else:
                        x_c = x_o + rho * (simplex[-1] - x_o)
                        x_c = np.clip(x_c, self.lower_bound, self.upper_bound)
                        f_c = func(x_c.reshape(1, -1))[0]
                        if f_c < simplex_fitness[-1]:
                            simplex[-1] = x_c
                            simplex_fitness[-1] = f_c
                
                if simplex_fitness[0] < f_curr:
                    self.population[idx] = simplex[0]
                    self.fitness[idx] = simplex_fitness[0]
                    if simplex_fitness[0] < self.f_opt:
                        self.x_opt = simplex[0].copy()
                        self.f_opt = simplex_fitness[0]
                    break
    
    def _check_stagnation(self):
        """Check if algorithm has stagnated."""
        stagnation = self.generation - self.last_improvement_gen > self.max_stagnation
        if stagnation:
            self._restart_population()
        return stagnation
    
    def _restart_population(self):
        """Reinitialize population preserving best solution."""
        new_population = np.random.uniform(
            self.lower_bound, self.upper_bound, size=(self.np, self.dim)
        )
        new_population = np.clip(new_population, self.lower_bound, self.upper_bound)
        
        new_population[0] = self.x_opt.copy()
        self.population = new_population
        
        self.last_improvement_gen = self.generation
        self.stagnation_count += 1
        self.step_size = min(self.step_size * 2, 10.0)
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        self._initialize_population(func)
        self._initialize_covariance()
        
        while not stopping_condition():
            self.generation += 1
            
            trials, strategy_used = self._build_trial_population_batch()
            crossed_trials = self._crossover_batch(trials)
            crossed_trials = np.clip(crossed_trials, self.lower_bound, self.upper_bound)
            
            trial_fitness = func(crossed_trials)
            if len(trial_fitness) < len(crossed_trials):
                trial_fitness = np.pad(trial_fitness, (0, len(crossed_trials) - len(trial_fitness)),
                                      mode='constant', constant_values=np.inf)
                if stopping_condition():
                    break
            
            self._select_survivors_batch(crossed_trials, trial_fitness, strategy_used)
            
            if self.generation % 3 == 0:
                self._update_operator_weights()
            
            self._adapt_covariance()
            self._adapt_step_size()
            
            diversity = self._compute_diversity()
            if diversity < self.diversity_threshold:
                self._restart_population()
            
            if self.generation % self.local_search_interval == 0:
                self._local_search_batch(func)
            
            self._check_stagnation()
            
            if stopping_condition():
                break
        
        return self.f_opt, self.x_opt
