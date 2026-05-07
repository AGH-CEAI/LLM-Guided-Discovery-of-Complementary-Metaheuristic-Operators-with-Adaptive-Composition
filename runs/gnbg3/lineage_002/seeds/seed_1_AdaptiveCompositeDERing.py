import numpy as np


class AdaptiveCompositeDERing:
    """
    Adaptive Composite Differential Evolution with Ring Topology.
    
    Key features:
    - Ring neighborhood topology for mutation (reduces premature convergence)
    - Multiple mutation strategies selected adaptively based on success rates
    - Composite mutation combining multiple differential vectors
    - Adaptive F and CR parameters per strategy
    - Diversity-triggered restart mechanism
    - Batched operations for efficiency
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(150, max(60, 5 * dim))
        self.bounds = np.array([-100.0, 100.0])
        
        # Strategy pool: 0=rand/1, 1=current-to-pbest/1, 2=composite/3
        self.n_strategies = 3
        self.strategy_scores = np.ones(self.n_strategies)
        self.strategy_counts = np.ones(self.n_strategies)
        
        # Adaptive parameters
        self.F_base = 0.5
        self.CR_base = 0.5
        self.p_best_rate = 0.1
        
        # Diversity and stagnation tracking
        self.diversity_threshold = 1e-6
        self.stagnation_counter = 0
        self.max_stagnation = 30
        
        # Archive for composite mutation
        self.archive = None
        
    def __call__(self, func, stopping_condition):
        return self._optimize(func, stopping_condition)
    
    def _initialize_population(self):
        """Initialize population uniformly in bounds."""
        low, high = self.bounds
        pop = np.random.uniform(low, high, (self.np, self.dim))
        return pop
    
    def _clip_to_bounds(self, population):
        """Clip population to search bounds."""
        low, high = self.bounds
        return np.clip(population, low, high)
    
    def _evaluate_batch(self, func, population):
        """Evaluate entire population in one batch call."""
        pop = self._clip_to_bounds(population)
        fitness = func(pop)
        
        # Handle budget exhaustion gracefully
        if len(fitness) < len(population):
            actual_len = len(fitness)
            pop = pop[:actual_len]
            fitness = fitness[:actual_len]
        
        # Replace NaN with worst fitness
        if np.any(np.isnan(fitness)):
            nan_mask = np.isnan(fitness)
            fitness[nan_mask] = np.nanmax(fitness) if not np.all(nan_mask) else 1e10
        
        return pop, fitness
    
    def _get_ring_neighbors(self, indices, ring_size=3):
        """Get ring neighborhood indices for given positions."""
        n = len(indices)
        neighbors = np.zeros((n, ring_size), dtype=int)
        
        for i in range(n):
            idx = indices[i]
            # Ring neighbors: i-2, i-1, i+1, i+2
            neighbors[i, 0] = (idx - 2) % n
            neighbors[i, 1] = (idx - 1) % n
            neighbors[i, 2] = (idx + 1) % n
        
        return neighbors
    
    def _mutate_rand_1(self, population, indices, F):
        """Classic rand/1 mutation on ring neighborhood."""
        neighbors = self._get_ring_neighbors(indices)
        
        r0 = population[indices]
        r1 = population[neighbors[:, 0]]
        r2 = population[neighbors[:, 1]]
        
        mutant = r0 + F * (r1 - r2)
        return mutant
    
    def _mutate_current_to_pbest(self, population, indices, fitness, F):
        """Current-to-pbest/1 mutation with adaptive p."""
        n = len(indices)
        p_size = max(1, int(self.p_best_rate * self.np))
        pbest_indices = np.argsort(fitness)[:p_size]
        pbest = population[np.random.choice(pbest_indices, n)]
        
        neighbors = self._get_ring_neighbors(indices)
        r1 = population[neighbors[:, 0]]
        r2 = population[neighbors[:, 1]]
        
        current = population[indices]
        mutant = current + F * (pbest - current) + F * (r1 - r2)
        return mutant
    
    def _mutate_composite_3(self, population, indices, F):
        """Composite mutation using 3 differential vectors."""
        neighbors = self._get_ring_neighbors(indices, ring_size=4)
        
        current = population[indices]
        r0 = population[neighbors[:, 0]]
        r1 = population[neighbors[:, 1]]
        r2 = population[neighbors[:, 2]]
        r3 = population[neighbors[:, 3]]
        
        # Three differential vectors with decreasing weights
        mutant = current + F * (r0 - r1) + 0.5 * F * (r2 - r3)
        return mutant
    
    def _mutate_batch(self, population, fitness, strategy):
        """Select and apply mutation strategy."""
        indices = np.arange(self.np)
        F = np.clip(self.F_base + np.random.randn(self.np) * 0.1, 0.3, 1.5)
        
        if strategy == 0:
            mutants = self._mutate_rand_1(population, indices, F)
        elif strategy == 1:
            mutants = self._mutate_current_to_pbest(population, indices, fitness, F)
        else:
            mutants = self._mutate_composite_3(population, indices, F)
        
        return self._clip_to_bounds(mutants)
    
    def _select_mutation_strategy(self):
        """Select mutation strategy using softmax on scores."""
        scores = self.strategy_scores / (self.strategy_counts + 1e-6)
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)
        return np.random.choice(self.n_strategies, p=probs)
    
    def _update_strategy_scores(self, strategy, improved):
        """Update strategy scores based on success."""
        self.strategy_counts[strategy] += 1
        
        if improved:
            # Increase score for successful strategy
            self.strategy_scores[strategy] += 1.0
            # Decay other scores slightly
            self.strategy_scores *= 0.95
            self.strategy_scores[strategy] /= 0.95  # Restore this one
        else:
            # Small decay for unsuccessful
            self.strategy_scores[strategy] *= 0.9
        
        # Ensure minimum scores
        self.strategy_scores = np.maximum(self.strategy_scores, 0.1)
    
    def _crossover_batch(self, target, mutant, CR):
        """Binomial crossover with dimension-wise adaptation."""
        CR_adapted = np.clip(CR + np.random.randn(self.np) * 0.1, 0.0, 1.0)
        
        n = self.np
        dims = self.dim
        
        # Generate mask: each row has at least one crossover
        mask = np.random.rand(n, dims) < CR_adapted[:, np.newaxis]
        
        # Ensure at least one dimension is crossed
        enforce_idx = np.random.randint(0, dims, n)
        mask[np.arange(n), enforce_idx] = True
        
        trial = np.where(mask, mutant, target)
        return trial
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """One-to-one selection with greedy replacement."""
        improved = trial_fitness < fitness
        
        # Create new population
        new_population = np.copy(population)
        new_fitness = np.copy(fitness)
        
        # Update improved individuals
        improve_idx = np.where(improved)[0]
        if len(improve_idx) > 0:
            new_population[improve_idx] = trials[improve_idx]
            new_fitness[improve_idx] = trial_fitness[improve_idx]
        
        return new_population, new_fitness, np.sum(improved)
    
    def _compute_diversity(self, population):
        """Compute population diversity using average Euclidean distance."""
        if len(population) < 2:
            return 0.0
        
        # Sample for efficiency
        sample_size = min(50, len(population))
        indices = np.random.choice(len(population), sample_size, replace=False)
        sample = population[indices]
        
        # Pairwise distances
        diffs = sample[:, np.newaxis, :] - sample[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diffs ** 2, axis=2))
        
        # Exclude diagonal
        mask = ~np.eye(sample_size, dtype=bool)
        return np.mean(distances[mask])
    
    def _adapt_parameters(self, improvement_rate):
        """Adapt F and CR based on recent improvement rate."""
        # Adapt F: increase if improving well, decrease if stuck
        if improvement_rate > 0.2:
            self.F_base = min(0.9, self.F_base * 1.05)
        elif improvement_rate < 0.05:
            self.F_base = max(0.3, self.F_base * 0.95)
        
        # Adapt CR: increase exploration tendency when stuck
        if improvement_rate < 0.1:
            self.CR_base = min(0.9, self.CR_base * 1.02)
        else:
            self.CR_base = max(0.3, self.CR_base * 0.98)
    
    def _restart_if_stagnant(self, population, fitness, best_fitness, gen):
        """Restart population if stagnating or losing diversity."""
        current_best = np.min(fitness)
        
        if current_best >= best_fitness - 1e-10:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        diversity = self._compute_diversity(population)
        
        should_restart = (
            self.stagnation_counter >= self.max_stagnation or
            diversity < self.diversity_threshold or
            gen > 100 and current_best > best_fitness * 1.5
        )
        
        if should_restart:
            # Keep best individual, reinitialize others
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx].copy()
            
            low, high = self.bounds
            new_pop = np.random.uniform(low, high, (self.np, self.dim))
            new_pop[0] = best_individual
            
            # Add some variants of best
            for i in range(1, min(10, self.np)):
                new_pop[i] = best_individual + np.random.randn(self.dim) * 5
                new_pop[i] = np.clip(new_pop[i], low, high)
            
            self.stagnation_counter = 0
            return new_pop
        
        return population
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop."""
        # Initialize
        population = self._initialize_population()
        population, fitness = self._evaluate_batch(func, population)
        
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx].copy()
        best_solution = population[best_idx].copy()
        
        generation = 0
        
        while not stopping_condition():
            # Select mutation strategy
            current_strategy = self._select_mutation_strategy()
            
            # Generate mutants
            mutants = self._mutate_batch(population, fitness, current_strategy)
            
            # Generate trials via crossover
            trials = self._crossover_batch(population, mutants, self.CR_base)
            
            # Check stopping condition before evaluation
            if stopping_condition():
                break
            
            # Evaluate trials in batch
            trials, trial_fitness = self._evaluate_batch(func, trials)
            
            # Check budget exhaustion
            if len(trial_fitness) < len(trials):
                break
            
            # Check stopping after evaluation
            if stopping_condition():
                break
            
            # Selection
            population, fitness, n_improved = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Update best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_solution = population[current_best_idx].copy()
            
            # Update strategy scores
            improvement_rate = n_improved / self.np
            self._update_strategy_scores(current_strategy, improvement_rate > 0.1)
            
            # Adapt parameters
            self._adapt_parameters(improvement_rate)
            
            # Check for restart
            population = self._restart_if_stagnant(
                population, fitness, best_fitness, generation
            )
            
            # Re-evaluate if population changed
            if np.any(population != trials):
                _, fitness = self._evaluate_batch(func, population)
            
            generation += 1
        
        return best_fitness, best_solution
