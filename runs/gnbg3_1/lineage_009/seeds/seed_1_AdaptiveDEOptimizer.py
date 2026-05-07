import numpy as np


class AdaptiveDEOptimizer:
    """
    Differential Evolution with adaptive mutation strategies, covariance-guided
    exploration, and operator reward tracking. Designed for robust performance
    across diverse benchmark problems.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(8 * dim, 240)
        self.lb = -100.0
        self.ub = 100.0
        self.bounds = np.array([self.lb, self.ub])
        
        # DE control parameters
        self.F = 0.6
        self.CR = 0.5
        self.F_adapt = 0.9
        self.CR_adapt = 0.1
        
        # Strategy pool with 6 distinct mutation operators
        self.num_strategies = 6
        self.strategy_weights = np.ones(self.num_strategies) / self.num_strategies
        self.strategy_rewards = np.zeros(self.num_strategies)
        self.strategy_counts = np.zeros(self.num_strategies, dtype=int)
        
        # Covariance adaptation (simplified CMA-ES like)
        self.cov_matrix = np.eye(dim)
        self.cov_scale = 0.5
        self.p_c = np.zeros(dim)
        self.p_s = np.zeros(dim)
        
        # Diversity and stagnation tracking
        self.diversity_history = []
        self.stagnation_tol = 50
        self.min_diversity = 1e-8
        self.generation = 0
        
    def _initialize_population(self):
        """Initialize population with orthogonal Latin hypercube sampling."""
        pop = np.zeros((self.NP, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(self.NP)
            pop[:, d] = (perm + np.random.rand(self.NP)) / self.NP
        pop = self.lb + pop * (self.ub - self.lb)
        return pop
    
    def _select_elite_batch(self, population, fitness, p=0.2):
        """Select top p portion as elite pool for guided mutation."""
        n_elite = max(1, int(self.NP * p))
        sorted_idx = np.argsort(fitness)
        elite_pool = population[sorted_idx[:n_elite]]
        elite_idx = sorted_idx[:n_elite]
        return elite_pool, elite_idx
    
    def _mutation_rand_1(self, population, indices, F):
        """Classic DE/rand/1 mutation."""
        n, d = population.shape
        r1, r2, r3 = indices[:, 0], indices[:, 1], indices[:, 2]
        return population[r1] + F * (population[r2] - population[r3])
    
    def _mutation_best_1(self, population, best, indices, F):
        """DE/best/1 mutation - greedy exploitation."""
        r1, r2 = indices[:, 0], indices[:, 1]
        return best + F * (population[r1] - population[r2])
    
    def _mutation_current_to_pbest(self, population, fitness, indices, F):
        """DE/current-to-pbest/1 with adaptive p selection."""
        n = len(population)
        p = np.clip(0.05 + 0.15 * np.random.rand(), 0.01, 0.3)
        elite_pool, _ = self._select_elite_batch(population, fitness, p=p)
        r1, r2 = indices[:, 0], indices[:, 1]
        base = population[r1]
        best_rand = elite_pool[np.random.randint(len(elite_pool), size=n)]
        return base + F * (best_rand - base) + F * (best_rand - population[r2])
    
    def _mutation_rand_2(self, population, indices, F):
        """DE/rand/2 - more diversity via two difference vectors."""
        r1, r2, r3, r4, r5 = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3], indices[:, 4]
        return (population[r1] + F * (population[r2] - population[r3]) +
                F * (population[r4] - population[r5]))
    
    def _mutation_cov_guided(self, population, best, indices, F):
        """Covariance-guided mutation - uses covariance structure."""
        r1, r2 = indices[:, 0], indices[:, 1]
        cov_noise = np.random.multivariate_normal(np.zeros(self.dim), self.cov_matrix)
        return best + F * (population[r1] - population[r2]) + 0.3 * cov_noise
    
    def _mutation_local_best(self, population, fitness, indices, F):
        """Ring-topology local best mutation."""
        n = len(population)
        ring_idx = np.roll(np.arange(n), -1)
        rev_ring_idx = np.roll(np.arange(n), 1)
        local_best_idx = np.argmin([fitness, fitness[ring_idx], fitness[rev_ring_idx]], axis=0)
        local_best = np.where(local_best_idx == 0, population,
                             np.where(local_best_idx == 1, population[ring_idx], population[rev_ring_idx]))
        r1 = indices[:, 0]
        return local_best + F * (population[r1] - population[r1])
    
    def _generate_indices_batch(self, n, k):
        """Generate unique random indices for mutation base and vectors."""
        indices = np.zeros((n, k), dtype=int)
        for i in range(n):
            indices[i] = np.random.choice(self.NP, k, replace=False)
        return indices
    
    def _mutate_batch(self, population, fitness, best, best_fitness):
        """Apply selected mutation strategies adaptively to whole population."""
        mutants = np.empty_like(population)
        n = len(population)
        
        # Sample strategy for each individual
        strategies = np.random.choice(self.num_strategies, n, p=self.strategy_weights)
        
        for s in range(self.num_strategies):
            mask = strategies == s
            count = np.sum(mask)
            if count == 0:
                continue
                
            # Dynamic F based on strategy
            F_s = self.F * (1.0 + 0.1 * np.random.randn()) if s % 2 == 0 else self.F * 0.8
            
            if s == 0:
                idx = self._generate_indices_batch(count, 3)
                mutants[mask] = self._mutation_rand_1(population[mask], idx, F_s)
            elif s == 1:
                idx = self._generate_indices_batch(count, 2)
                mutants[mask] = self._mutation_best_1(population[mask], best, idx, F_s)
            elif s == 2:
                idx = self._generate_indices_batch(count, 2)
                mutants[mask] = self._mutation_current_to_pbest(population[mask], fitness, idx, F_s)
            elif s == 3:
                idx = self._generate_indices_batch(count, 5)
                mutants[mask] = self._mutation_rand_2(population[mask], idx, F_s)
            elif s == 4:
                idx = self._generate_indices_batch(count, 2)
                mutants[mask] = self._mutation_cov_guided(population[mask], best, idx, F_s)
            else:
                idx = self._generate_indices_batch(count, 1)
                mutants[mask] = self._mutation_local_best(population[mask], fitness, idx, F_s)
        
        mutants = np.clip(mutants, self.lb, self.ub)
        return mutants, strategies
    
    def _crossover_batch(self, population, mutants, CR):
        """Binomial crossover with dimension-wise adaptation."""
        n, d = population.shape
        cr_vec = np.clip(CR + 0.1 * np.random.randn(n), 0.1, 0.95)
        j_rand = np.random.randint(d, size=n)
        
        mask = np.random.rand(n, d) < cr_vec[:, np.newaxis]
        mask[np.arange(n), j_rand] = True
        
        trials = np.where(mask, mutants, population)
        trials = np.clip(trials, self.lb, self.ub)
        return trials
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Greedy (mu+lambda) survivor selection with truncation for budget."""
        n_valid = len(trial_fitness)
        if n_valid < len(trials):
            trials = trials[:n_valid]
            trial_fitness = np.nan_to_num(trial_fitness, nan=np.inf)
        
        improved = trial_fitness < fitness[:n_valid]
        new_pop = population.copy()
        new_fit = fitness.copy()
        new_pop[:n_valid][improved] = trials[improved]
        new_fit[:n_valid][improved] = trial_fitness[improved]
        
        return new_pop, new_fit, improved
    
    def _compute_diversity(self, population):
        """Compute population diversity via average pairwise distance."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_step_size(self, improved_mask, trial_fitness, fitness):
        """Adapt F and CR based on success history."""
        n_imp = np.sum(improved_mask)
        if n_imp == 0:
            return
        
        imp_rate = n_imp / len(improved_mask)
        self.F = np.clip(self.F * (1 - 0.1 * (1 - imp_rate)), 0.3, 1.5)
        self.CR = np.clip(self.CR * (1 + 0.05 * (imp_rate - 0.5)), 0.1, 0.95)
    
    def _update_covariance(self, population, best, improved_mask):
        """Simplified covariance adaptation similar to CMA-ES."""
        alpha = 0.01
        n = len(population)
        
        # Evolution path for covariance
        self.p_c = (1 - 0.05) * self.p_c + np.sqrt(0.05 * (2 - 0.05)) * (best - self.p_c) / self.cov_scale
        
        # Rank-1 update
        self.cov_matrix = (1 - alpha) * self.cov_matrix + alpha * np.outer(self.p_c, self.p_c)
        
        # Ensure positive definiteness
        eigvals = np.linalg.eigvalsh(self.cov_matrix)
        if np.min(eigvals) < 1e-10:
            self.cov_matrix += (1e-9 - np.min(eigvals)) * np.eye(self.dim)
        
        # Adapt scale
        if np.any(improved_mask):
            self.cov_scale = np.clip(self.cov_scale * 1.02, 0.1, 5.0)
        else:
            self.cov_scale *= 0.98
    
    def _update_operator_rewards(self, strategies, improved_mask):
        """Update strategy weights using multi-armed bandit approach."""
        for s in range(self.num_strategies):
            mask = strategies == s
            n_used = np.sum(mask)
            if n_used > 0:
                n_improved = np.sum(improved_mask[mask])
                reward = n_improved / n_used
                self.strategy_rewards[s] = 0.9 * self.strategy_rewards[s] + 0.1 * reward
                self.strategy_counts[s] += n_used
        
        # Update weights using softmax
        if self.generation % 5 == 0:
            rewards = self.strategy_rewards - np.max(self.strategy_rewards)
            exp_rewards = np.exp(rewards * 10)
            self.strategy_weights = exp_rewards / np.sum(exp_rewards)
            self.strategy_weights = np.clip(self.strategy_weights, 0.05, 0.5)
            self.strategy_weights /= np.sum(self.strategy_weights)
    
    def _restart_if_stagnant(self, population, fitness, best_fitness, best):
        """Restart population if diversity is lost or stagnation detected."""
        diversity = self._compute_diversity(population)
        self.diversity_history.append(diversity)
        
        if len(self.diversity_history) > self.stagnation_tol:
            recent_div = np.mean(self.diversity_history[-self.stagnation_tol:])
            if diversity < self.min_diversity or recent_div < self.min_diversity * 2:
                return True, self._initialize_population()
        
        if len(self.diversity_history) > self.stagnation_tol * 2:
            fit_range = np.ptp(self.diversity_history[-self.stagnation_tol * 2:])
            if fit_range < 1e-10:
                return True, self._initialize_population()
        
        return False, population
    
    def _track_best(self, population, fitness):
        """Track and return best solution in population."""
        valid = ~np.isnan(fitness)
        if not np.any(valid):
            return None, np.inf
        best_idx = np.argmin(fitness[valid])
        return population[valid][best_idx], fitness[valid][best_idx]
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop - orchestrates all batched operations."""
        population = self._initialize_population()
        fitness = func(population)
        fitness = np.nan_to_num(fitness, nan=np.inf)
        
        best, best_fitness = self._track_best(population, fitness)
        self.generation = 0
        
        while not stopping_condition():
            self.generation += 1
            
            # Mutation phase
            mutants, strategies = self._mutate_batch(population, fitness, best, best_fitness)
            
            # Crossover phase
            trials = self._crossover_batch(population, mutants, self.CR)
            
            # Check budget before evaluation
            if stopping_condition():
                break
            
            # Evaluate trials in batch
            trial_fitness = func(trials)
            trial_fitness = np.nan_to_num(trial_fitness, nan=np.inf)
            
            # Check budget after evaluation
            if len(trial_fitness) < len(trials):
                break
            
            # Selection phase
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness)
            
            # Update tracking
            new_best, new_best_fitness = self._track_best(population, fitness)
            if new_best_fitness < best_fitness:
                best, best_fitness = new_best, new_best_fitness
            
            # Adaptation methods
            self._adapt_step_size(improved_mask, trial_fitness, fitness)
            self._update_covariance(population, best, improved_mask)
            self._update_operator_rewards(strategies, improved_mask)
            
            # Check for restart
            should_restart, population = self._restart_if_stagnant(
                population, fitness, best_fitness, best)
            if should_restart:
                fitness = func(population)
                fitness = np.nan_to_num(fitness, nan=np.inf)
                self.diversity_history = []
        
        return best_fitness, best
