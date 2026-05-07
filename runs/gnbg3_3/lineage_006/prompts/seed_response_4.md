```python
import numpy as np

class AdaptiveNichingDEOptimizer:
    """
    A population-based metaheuristic combining DE-style mutation with
    Nelder-Mead local search, ring topology, and adaptive strategy selection.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(8 * dim, 400)
        self.bounds = np.array([-100.0, 100.0])
        
        # DE parameters with adaptation
        self.f = 0.5
        self.cr = 0.9
        self.f_history = []
        self.cr_history = []
        
        # Strategy pool for adaptive selection
        self.strategies = ['de_rand_1', 'de_best_1', 'de_target_best_1', 'de_rand_2']
        self.strategy_scores = {s: 0.0 for s in self.strategies}
        self.current_strategy = 'de_rand_1'
        
        # Local search parameters
        self.ls_interval = 5
        self.ls_top_k = max(2, self.np // 20)
        self.ls_reflection = 1.0
        self.ls_contraction = 0.5
        self.ls_expansion = 2.0
        self.ls_shrink = 0.5
        
        # Diversity and stagnation
        self.diversity_history = []
        self.stagnation_gen = 0
        self.max_stagnation = 15
        
        # Tracking
        self.func_calls = 0
        self.gen = 0
    
    def _initialize_population(self):
        """Create uniformly distributed initial population within bounds."""
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.np, self.dim))
        return np.clip(pop, self.bounds[0], self.bounds[1])
    
    def _get_ring_indices(self, idx):
        """Get indices of neighbors in ring topology (2 on each side)."""
        prev2 = (idx - 2) % self.np
        prev1 = (idx - 1) % self.np
        next1 = (idx + 1) % self.np
        next2 = (idx + 2) % self.np
        return prev2, prev1, next1, next2
    
    def _mutate_de_rand_1(self, pop, target_idx):
        """DE/rand/1: pick 3 random (non-target) indices from ring neighborhood."""
        p2, p1, n1, n2 = self._get_ring_indices(target_idx)
        candidates = [p2, p1, n1, n2]
        r1, r2, r3 = candidates[:3]
        return pop[r1] + self.f * (pop[r2] - pop[r3])
    
    def _mutate_de_best_1(self, pop, target_idx, best_idx):
        """DE/best/1: use best and 2 random ring neighbors."""
        p2, p1, n1, n2 = self._get_ring_indices(target_idx)
        candidates = [p2, p1, n1, n2]
        r1, r2 = candidates[:2]
        return pop[best_idx] + self.f * (pop[r1] - pop[r2])
    
    def _mutate_de_target_best_1(self, pop, target_idx, best_idx):
        """DE/target-to-best/1: direction toward best + ring differential."""
        p2, p1, n1, n2 = self._get_ring_indices(target_idx)
        candidates = [p2, p1, n1, n2]
        r1, r2 = candidates[:2]
        target = pop[target_idx]
        return target + self.f * (pop[best_idx] - target) + self.f * (pop[r1] - pop[r2])
    
    def _mutate_de_rand_2(self, pop, target_idx):
        """DE/rand/2: use 4 random ring neighbors for larger diversity step."""
        p2, p1, n1, n2 = self._get_ring_indices(target_idx)
        candidates = [p2, p1, n1, n2]
        r1, r2, r3, r4 = candidates
        return pop[r1] + self.f * (pop[r2] - pop[r3]) + self.f * (pop[r4] - pop[p2])
    
    def _select_mutation_batch(self, pop, best_idx):
        """Select mutation strategy adaptively based on accumulated scores."""
        # Exploration: occasionally try a random different strategy
        if np.random.random() < 0.1:
            other_strats = [s for s in self.strategies if s != self.current_strategy]
            self.current_strategy = np.random.choice(other_strats)
        
        mutants = np.empty((self.np, self.dim))
        for i in range(self.np):
            if self.current_strategy == 'de_rand_1':
                mutants[i] = self._mutate_de_rand_1(pop, i)
            elif self.current_strategy == 'de_best_1':
                mutants[i] = self._mutate_de_best_1(pop, i, best_idx)
            elif self.current_strategy == 'de_target_best_1':
                mutants[i] = self._mutate_de_target_best_1(pop, i, best_idx)
            else:
                mutants[i] = self._mutate_de_rand_2(pop, i)
        
        return np.clip(mutants, self.bounds[0], self.bounds[1])
    
    def _crossover_batch(self, targets, mutants):
        """Binomial crossover with per-dimension random mask."""
        mask = np.random.random((self.np, self.dim)) < self.cr
        # Ensure at least one dimension comes from mutant
        enforce_mask = np.zeros((self.np, self.dim), dtype=bool)
        enforce_idx = np.random.randint(0, self.dim, size=self.np)
        enforce_mask[np.arange(self.np), enforce_idx] = True
        mask = mask | enforce_mask
        return np.where(mask, mutants, targets)
    
    def _evaluate_batch(self, pop, func):
        """Evaluate entire population in a single batch call."""
        fitness = func(pop)
        self.func_calls += len(pop)
        return fitness
    
    def _select_survivors_batch(self, pop, fitness, trials, trial_fitness):
        """Greedy selection: keep better of target vs trial for each individual."""
        improved = trial_fitness < fitness
        new_pop = np.where(improved[:, np.newaxis], trials, pop)
        new_fitness = np.where(improved, trial_fitness, fitness)
        return new_pop, new_fitness, improved
    
    def _update_strategy_scores(self, improved_mask):
        """Credit assignment for the current mutation strategy."""
        success_rate = np.mean(improved_mask)
        self.strategy_scores[self.current_strategy] += success_rate
        
        # Decay all scores slightly for recency bias
        for s in self.strategies:
            self.strategy_scores[s] *= 0.95
        
        # Switch strategy if current is performing poorly
        if np.random.random() < 0.2:
            best_strat = max(self.strategy_scores, key=self.strategy_scores.get)
            if best_strat != self.current_strategy:
                self.current_strategy = best_strat
    
    def _adapt_f_batch(self, improved_mask):
        """Adapt scaling factor using weighted success rate (JADE-style)."""
        if len(improved_mask) == 0:
            return
        success_rate = np.mean(improved_mask)
        self.f_history.append(success_rate)
        
        if len(self.f_history) > 5:
            self.f_history.pop(0)
        
        # Increase F on success, decrease on failure
        if success_rate > 0.2:
            self.f = min(1.5, self.f * 1.1)
        elif success_rate < 0.1:
            self.f = max(0.1, self.f * 0.9)
        
        # Add random jitter
        self.f = np.clip(self.f + np.random.normal(0, 0.1), 0.1, 1.5)
    
    def _adapt_cr_batch(self, improved_mask):
        """Adapt crossover rate based on success."""
        success_rate = np.mean(improved_mask)
        self.cr_history.append(success_rate)
        
        if len(self.cr_history) > 5:
            self.cr_history.pop(0)
        
        if success_rate > 0.25:
            self.cr = min(0.95, self.cr * 1.05)
        elif success_rate < 0.1:
            self.cr = max(0.3, self.cr * 0.95)
        
        self.cr = np.clip(self.cr + np.random.normal(0, 0.05), 0.3, 0.95)
    
    def _apply_local_search_batch(self, pop, fitness, func):
        """
        Nelder-Mead inspired local search on top-k performers.
        Evaluates simplex operations in batches for efficiency.
        """
        top_indices = np.argsort(fitness)[:self.ls_top_k]
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        
        for idx in top_indices:
            center = pop[idx].copy()
            centroid = np.mean(pop, axis=0)
            
            # Build simplex: center + small random offsets
            simplex_points = [center]
            for _ in range(self.dim):
                offset = np.random.uniform(-0.5, 0.5, self.dim)
                simplex_points.append(np.clip(center + offset * 10.0, self.bounds[0], self.bounds[1]))
            
            simplex = np.array(simplex_points)
            simplex_fitness = self._evaluate_batch(simplex, func)
            
            if len(simplex_fitness) < len(simplex):
                break
            
            sorted_idx = np.argsort(simplex_fitness)
            best_pt = simplex[sorted_idx[0]]
            best_f = simplex_fitness[sorted_idx[0]]
            
            # Reflection
            reflected = np.clip(2 * centroid - center, self.bounds[0], self.bounds[1])
            reflected_batch = np.array([reflected])
            reflected_f = self._evaluate_batch(reflected_batch, func)
            
            if len(reflected_f) < 1:
                continue
                
            if simplex_fitness[sorted_idx[-1]] <= reflected_f[0] < simplex_fitness[sorted_idx[-2]]:
                # Accept reflection
                if reflected_f[0] < best_f:
                    best_pt = reflected
                    best_f = reflected_f[0]
            
            # Contraction
            contracted = np.clip(centroid + self.ls_contraction * (center - centroid), 
                               self.bounds[0], self.bounds[1])
            contracted_batch = np.array([contracted])
            contracted_f = self._evaluate_batch(contracted_batch, func)
            
            if len(contracted_f) < 1:
                continue
                
            if contracted_f[0] < best_f:
                best_pt = contracted
                best_f = contracted_f[0]
            
            # Accept improvement if found
            if best_f < fitness[idx]:
                new_pop[idx] = best_pt
                new_fitness[idx] = best_f
        
        return new_pop, new_fitness
    
    def _compute_diversity(self, pop):
        """Compute population diversity as average pairwise distance."""
        centroid = np.mean(pop, axis=0)
        distances = np.sqrt(np.sum((pop - centroid) ** 2, axis=1))
        return np.mean(distances)
    
    def _restart_if_stagnant(self, pop, fitness):
        """Reinitialize portion of population if diversity or stagnation detected."""
        diversity = self._compute_diversity(pop)
        self.diversity_history.append(diversity)
        
        if len(self.diversity_history) > 10:
            self.diversity_history.pop(0)
        
        # Detect stagnation
        if len(self.diversity_history) >= 5:
            recent = self.diversity_history[-5:]
            if np.std(recent) < 1.0 and diversity < 10.0:
                self.stagnation_gen += 1
            else:
                self.stagnation_gen = max(0, self.stagnation_gen - 1)
        
        # Perform partial restart
        if self.stagnation_gen >= self.max_stagnation:
            n_replace = max(self.np // 4, 1)
            worst_indices = np.argsort(fitness)[-n_replace:]
            new_portion = self._initialize_population()[:n_replace]
            pop[worst_indices] = new_portion
            fitness[worst_indices] = np.inf
            self.stagnation_gen = 0
            self.diversity_history.clear()
        
        return pop, fitness
    
    def _check_budget_exhausted(self, n_evaluated):
        """Check if function budget is exhausted."""
        return n_evaluated < self.np
    
    def _clip_to_bounds(self, pop):
        """Ensure all candidates are within search bounds."""
        return np.clip(pop, self.bounds[0], self.bounds[1])
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer.
        
        Args:
            func: Fitness function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        # Initialization
        pop = self._initialize_population()
        pop = self._clip_to_bounds(pop)
        fitness = self._evaluate_batch(pop, func)
        
        if self._check_budget_exhausted(len(fitness)):
            return np.min(fitness), pop[np.argmin(fitness)]
        
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_x = pop[best_idx].copy()
        
        # Main evolution loop
        while not stopping_condition():
            self.gen += 1
            
            # Mutation with adaptive strategy selection
            mutants = self._select_mutation_batch(pop, best_idx)
            
            # Crossover
            trials = self._crossover_batch(pop, mutants)
            trials = self._clip_to_bounds(trials)
            
            # Evaluate trials
            trial_fitness = self._evaluate_batch(trials, func)
            
            if self._check_budget_exhausted(len(trial_fitness)):
                break
            
            # Selection
            pop, fitness, improved = self._select_survivors_batch(pop, fitness, trials, trial_fitness)
            
            # Check stopping condition inside loop
            if stopping_condition():
                break
            
            # Adaptation
            self._update_strategy_scores(improved)
            self._adapt_f_batch(improved)
            self._adapt_cr_batch(improved)
            
            # Local search on top performers
            if self.gen % self.ls_interval == 0:
                pop, fitness = self._apply_local_search_batch(pop, fitness, func)
                
                if self._check_budget_exhausted(len(fitness)):
                    break
            
            # Diversity maintenance and restart
            pop, fitness = self._restart_if_stagnant(pop, fitness)
            
            # Update best solution
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_fitness = fitness[best_idx]
                best_x = pop[best_idx].copy()
        
        return best_fitness, best_x
```