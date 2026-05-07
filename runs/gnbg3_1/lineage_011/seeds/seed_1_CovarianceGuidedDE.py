import numpy as np

class CovarianceGuidedDE:
    """
    Covariance-Guided Differential Evolution
    A DE variant with adaptive mutation strategies, covariance learning,
    and diversity-aware restarts for robust multi-modal optimization.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower = -100.0
        self.upper = 100.0
        
        # Population sizing: moderate NP for dim=30
        self.NP = kwargs.get('NP', min(max(4 * dim, 120), 300))
        
        # DE control parameters
        self.F = 0.5
        self.CR = 0.9
        self.p_best_rate = 0.11
        self.F_min, self.F_max = 0.1, 1.0
        self.CR_min, self.CR_max = 0.3, 1.0
        
        # Covariance adaptation
        self.cov = np.eye(dim)
        self.cov_decay = 0.99
        self.cov_eps = 1e-8
        self.use_cov = True
        self.cov_weight = 0.02
        self.cov_history = np.zeros((5, dim))
        self.cov_history_idx = 0
        
        # Strategy pool with adaptive selection
        self.strategy_names = ['de_rand_1', 'de_best_1', 'de_current_to_pbest', 'cov_guided']
        self.strategy_probs = np.ones(4) / 4.0
        self.strategy_rewards = np.zeros(4)
        self.strategy_success = {i: [] for i in range(4)}
        self.epsilon = 0.01
        
        # Archive for current-to-pbest
        self.archive = np.zeros((self.NP, dim))
        self.archive_fitness = np.full(self.NP, np.inf)
        self.archive_count = 0
        
        # Diversity and stagnation
        self.diversity_history = []
        self.stagnation_tol = 1e-8
        self.stagnation_count = 0
        self.max_stagnation = 30
        self.diversity_threshold = 0.01
        
        # Best tracking
        self.best_fitness = np.inf
        self.best_position = None
        self.gen = 0
        
    def _initialize_population(self, func):
        """Initialize population with Latin Hypercube Sampling for better spread."""
        pop = np.zeros((self.NP, self.dim))
        for d in range(self.dim):
            pop[:, d] = np.linspace(self.lower, self.upper, self.NP)
            if self.NP > 1:
                perm = np.random.permutation(self.NP)
                pop[:, d] = pop[perm, d]
        pop += np.random.uniform(-1, 1, (self.NP, self.dim)) * (self.upper - self.lower) / self.NP
        pop = np.clip(pop, self.lower, self.upper)
        
        fitness = func(pop)
        self.best_idx = np.argmin(fitness)
        self.best_fitness = fitness[self.best_idx]
        self.best_position = pop[self.best_idx].copy()
        
        self.archive = pop.copy()
        self.archive_fitness = fitness.copy()
        self.archive_count = self.NP
        
        return pop, fitness
    
    def _select_strategy(self):
        """Adaptive strategy selection using softmax-like probabilities."""
        exp_probs = np.exp(self.strategy_rewards - np.max(self.strategy_rewards))
        self.strategy_probs = exp_probs / (np.sum(exp_probs) + self.epsilon)
        return np.random.choice(4, p=self.strategy_probs)
    
    def _mutate_batch(self, population, fitness, strategy_idx):
        """Generate mutant vectors using selected strategy."""
        NP, dim = population.shape
        mutants = np.empty((NP, dim))
        
        # Select indices for mutation
        all_indices = np.arange(NP)
        
        for i in range(NP):
            if strategy_idx == 0:
                # DE/rand/1
                r1, r2, r3 = self._select_three_different(i, all_indices)
                mutants[i] = population[r1] + self.F * (population[r2] - population[r3])
                
            elif strategy_idx == 1:
                # DE/best/1
                r1, r2 = self._select_two_different(i, all_indices)
                best_idx = self.best_idx
                mutants[i] = self.best_position + self.F * (population[r1] - population[r2])
                
            elif strategy_idx == 2:
                # Current-to-pbest/1 with archive
                pbest_count = max(1, int(self.p_best_rate * NP))
                pbest_indices = np.argsort(fitness)[:pbest_count]
                pbest_idx = np.random.choice(pbest_indices)
                r1, r2 = self._select_two_different(i, all_indices)
                
                # Include archive in selection
                if self.archive_count > 0 and np.random.rand() < 0.5:
                    arch_idx = np.random.randint(self.archive_count)
                    donor = self.archive[arch_idx]
                else:
                    donor = population[r1]
                
                mutants[i] = population[i] + self.F * (self.best_position - population[i]) + self.F * (population[pbest_idx] - donor)
                
            else:  # strategy_idx == 3
                # Covariance-guided mutation
                r1, r2 = self._select_two_different(i, all_indices)
                direction = population[r1] - population[r2]
                
                # Add covariance-guided perturbation
                if self.use_cov and np.random.rand() < 0.3:
                    z = np.random.randn(dim)
                    cov_dir = self.cov @ z
                    cov_dir = cov_dir / (np.linalg.norm(cov_dir) + self.cov_eps)
                    mutants[i] = population[i] + self.F * direction + self.cov_weight * cov_dir * (self.upper - self.lower)
                else:
                    mutants[i] = population[i] + self.F * direction
        
        mutants = np.clip(mutants, self.lower, self.upper)
        return mutants
    
    def _select_three_different(self, exclude, pool):
        """Select three indices different from exclude."""
        candidates = np.delete(pool, np.where(pool == exclude)[0])
        selected = np.random.choice(candidates, size=3, replace=False)
        return selected[0], selected[1], selected[2]
    
    def _select_two_different(self, exclude, pool):
        """Select two indices different from exclude."""
        candidates = np.delete(pool, np.where(pool == exclude)[0])
        selected = np.random.choice(candidates, size=2, replace=False)
        return selected[0], selected[1]
    
    def _crossover_batch(self, population, mutants):
        """Binomial crossover between target and mutant vectors."""
        NP, dim = population.shape
        mask = np.random.rand(NP, dim) < self.CR
        
        # Ensure at least one dimension is different
        enforce_idx = np.random.randint(dim, size=NP)
        mask[np.arange(NP), enforce_idx] = True
        
        trials = np.where(mask, mutants, population)
        return trials
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness, strategy_idx):
        """Select survivors using greedy selection and update strategy rewards."""
        NP = len(fitness)
        improved = trial_fitness < fitness
        
        # Count improvements per strategy
        n_improved = np.sum(improved)
        if n_improved > 0:
            self.strategy_success[strategy_idx].append(n_improved / NP)
        else:
            self.strategy_success[strategy_idx].append(0)
        
        # Keep only recent history
        for s in self.strategy_success:
            if len(self.strategy_success[s]) > 10:
                self.strategy_success[s].pop(0)
        
        # Update strategy rewards
        if len(self.strategy_success[strategy_idx]) >= 3:
            recent = self.strategy_success[strategy_idx][-3:]
            reward = np.mean(recent)
            self.strategy_rewards[strategy_idx] += 0.1 * (reward - 0.1)
            self.strategy_rewards[strategy_idx] = np.clip(self.strategy_rewards[strategy_idx], -2, 2)
        
        # Greedy selection
        new_pop = np.where(improved[:, np.newaxis], trials, population)
        new_fitness = np.where(improved, trial_fitness, fitness)
        
        # Update archive
        self._update_archive_batch(population, trials, fitness, trial_fitness, improved)
        
        # Update best
        best_in_trial = np.argmin(trial_fitness)
        if trial_fitness[best_in_trial] < self.best_fitness:
            self.best_fitness = trial_fitness[best_in_trial]
            self.best_position = trials[best_in_trial].copy()
        
        return new_pop, new_fitness
    
    def _update_archive_batch(self, population, trials, fitness, trial_fitness, improved):
        """Update the archive with improved solutions."""
        if self.archive_count < self.NP:
            add_count = min(np.sum(improved), self.NP - self.archive_count)
            add_indices = np.where(improved)[0][:add_count]
            self.archive[self.archive_count:self.archive_count + len(add_indices)] = trials[add_indices]
            self.archive_fitness[self.archive_count:self.archive_count + len(add_indices)] = trial_fitness[add_indices]
            self.archive_count += len(add_indices)
        else:
            # Replace worst archive members with improvements
            worst_arch_indices = np.argsort(self.archive_fitness)[-np.sum(improved):]
            improved_indices = np.where(improved)[0]
            for i, (arch_i, imp_i) in enumerate(zip(worst_arch_indices, improved_indices)):
                if trial_fitness[imp_i] < self.archive_fitness[arch_i]:
                    self.archive[arch_i] = trials[imp_i]
                    self.archive_fitness[arch_i] = trial_fitness[imp_i]
    
    def _adapt_F_CR(self, fitness, trial_fitness, improved):
        """Adapt F and CR based on success rate."""
        if len(improved) == 0:
            return
        
        success_rate = np.mean(improved)
        
        # Adapt F: decrease if success is low, increase if high
        if success_rate > 0.3:
            self.F = min(self.F * 1.1, self.F_max)
        elif success_rate < 0.1:
            self.F = max(self.F * 0.9, self.F_min)
        
        # Adapt CR: slightly perturbed
        if success_rate > 0.2:
            self.CR = np.clip(self.CR + np.random.randn() * 0.1, self.CR_min, self.CR_max)
    
    def _adapt_covariance(self, population, fitness, improved):
        """Adapt covariance matrix based on successful mutations."""
        if not self.use_cov or np.sum(improved) < 2:
            return
        
        # Store best directions
        improved_pop = population[improved]
        if len(improved_pop) >= 2:
            mean_pop = np.mean(population, axis=0)
            mean_improved = np.mean(improved_pop, axis=0)
            direction = mean_improved - mean_pop
            
            self.cov_history[self.cov_history_idx] = direction
            self.cov_history_idx = (self.cov_history_idx + 1) % 5
            
            # Update covariance matrix
            if self.gen % 5 == 0:
                directions = self.cov_history
                valid_dirs = np.any(directions != 0, axis=1)
                if np.sum(valid_dirs) > 0:
                    dir_matrix = directions[valid_dirs]
                    dir_matrix = dir_matrix - np.mean(dir_matrix, axis=0)
                    
                    if len(dir_matrix) > 1:
                        new_cov = (dir_matrix.T @ dir_matrix) / (len(dir_matrix) + self.cov_eps)
                        new_cov = new_cov / (np.max(np.abs(new_cov)) + self.cov_eps)
                        self.cov = self.cov_decay * self.cov + (1 - self.cov_decay) * new_cov
    
    def _compute_diversity(self, population):
        """Compute population diversity using average Euclidean distance."""
        NP = len(population)
        if NP < 2:
            return 0.0
        
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances) / (self.upper - self.lower)
    
    def _check_stagnation(self, fitness, trial_fitness):
        """Check if the optimization is stagnating."""
        current_best = np.min(fitness)
        best_improvement = self.best_fitness - current_best
        
        if best_improvement < self.stagnation_tol:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        
        return self.stagnation_count >= self.max_stagnation
    
    def _restart_if_needed(self, population, fitness):
        """Restart population if stagnated or diversity is too low."""
        diversity = self._compute_diversity(population)
        self.diversity_history.append(diversity)
        
        if len(self.diversity_history) > 10:
            self.diversity_history.pop(0)
        
        recent_diversity_drop = len(self.diversity_history) >= 5 and np.mean(self.diversity_history[-5:]) < self.diversity_threshold
        
        if self.stagnation_count >= self.max_stagnation or recent_diversity_drop:
            # Reinitialize worst half
            n_replace = self.NP // 2
            worst_indices = np.argsort(fitness)[-n_replace:]
            
            # Generate new individuals around the best
            for i in worst_indices:
                population[i] = self.best_position + np.random.randn(self.dim) * (self.upper - self.lower) * 0.1
                population[i] = np.clip(population[i], self.lower, self.upper)
            
            fitness[worst_indices] = np.inf
            
            # Reset stagnation counter
            self.stagnation_count = 0
            self.diversity_history = []
            
            # Reset covariance
            self.cov = np.eye(self.dim)
    
    def _update_best_idx(self, population, fitness):
        """Update the index of the best individual."""
        self.best_idx = np.argmin(fitness)
    
    def __call__(self, func, stopping_condition):
        """Execute the optimizer."""
        population, fitness = self._initialize_population(func)
        self.gen = 0
        self.stagnation_count = 0
        
        while not stopping_condition():
            # Select mutation strategy
            strategy_idx = self._select_strategy()
            
            # Generate mutants
            mutants = self._mutate_batch(population, fitness, strategy_idx)
            
            # Crossover
            trials = self._crossover_batch(population, mutants)
            trials = np.clip(trials, self.lower, self.upper)
            
            # Evaluate trials
            trial_fitness = func(trials)
            
            # Handle budget exhaustion
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                mutants = mutants[:len(trial_fitness)]
                trial_fitness = func(trials)
            
            if stopping_condition():
                break
            
            # Selection
            improved = trial_fitness < fitness
            population, fitness = self._select_survivors_batch(population, fitness, trials, trial_fitness, strategy_idx)
            
            # Update best index
            self._update_best_idx(population, fitness)
            
            # Adaptation
            self._adapt_F_CR(fitness, trial_fitness, improved)
            self._adapt_covariance(population, fitness, improved)
            
            # Check stagnation and restart if needed
            self._restart_if_needed(population, fitness)
            
            self.gen += 1
            
            if self.gen % 50 == 0:
                self.use_cov = self._compute_diversity(population) > 0.05
        
        return self.best_fitness, self.best_position
