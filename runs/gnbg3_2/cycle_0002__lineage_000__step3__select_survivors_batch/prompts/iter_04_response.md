**Idea: Rank-Momentum Survivor Selection**

Category D (Fitness-landscape / rank-based): Track rank transitions across generations using exponential moving averages of rank improvements. Instead of pure greedy selection, compute a composite score combining immediate fitness improvement with historical rank momentum to identify individuals that show consistent improvement patterns, not just lucky mutations.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Rank-momentum based selection: favor individuals with consistent rank improvement."""
    np_pop = len(fitness)
    
    # Compute ranks within current population (lower rank = better fitness)
    current_ranks = np.argsort(np.argsort(fitness))
    trial_ranks = np.argsort(np.argsort(trial_fitness))
    
    # Rank improvement: positive means trial moved up in rank
    rank_improvement = current_ranks - trial_ranks
    
    # Initialize rank momentum history if needed
    if not hasattr(self, 'rank_momentum_history') or len(self.rank_momentum_history) != np_pop:
        self.rank_momentum_history = np.zeros(np_pop)
        self.rank_momentum_count = np.zeros(np_pop)
    
    # Update per-individual rank momentum with EMA
    alpha = 0.4
    for i in range(np_pop):
        if self.rank_momentum_count[i] > 0:
            self.rank_momentum_history[i] = (alpha * rank_improvement[i] + 
                                              (1 - alpha) * self.rank_momentum_history[i])
        else:
            self.rank_momentum_history[i] = rank_improvement[i] * alpha
        self.rank_momentum_count[i] = min(self.rank_momentum_count[i] + 1, 20)
    
    # Composite score: immediate improvement (scaled by rank) + momentum
    # Normalize ranks to [0, 1]
    rank_norm = current_ranks / max(np_pop - 1, 1)
    momentum_norm = self.rank_momentum_history / max(np_pop - 1, 1)
    
    # Immediate fitness gain (normalized)
    fitness_range = np.max(fitness) - np.min(fitness) + 1e-10
    immediate_gain = (fitness - trial_fitness) / fitness_range
    
    # Combined score: exploit immediate gains but also track momentum
    # Higher momentum = consistently improving rank = more valuable
    combined_score = immediate_gain + 0.3 * momentum_norm
    
    # Selection: replace if combined_score > 0 (good immediate OR momentum)
    improved_mask = combined_score > 0
    
    # Fallback: if nothing has positive combined score, take best by fitness
    if not np.any(improved_mask):
        improved_mask = trial_fitness < fitness
    
    new_population = population.copy()
    new_fitness = fitness.copy()
    
    new_population[improved_mask] = trials[improved_mask]
    new_fitness[improved_mask] = trial_fitness[improved_mask]
    
    return new_population, new_fitness, improved_mask
```