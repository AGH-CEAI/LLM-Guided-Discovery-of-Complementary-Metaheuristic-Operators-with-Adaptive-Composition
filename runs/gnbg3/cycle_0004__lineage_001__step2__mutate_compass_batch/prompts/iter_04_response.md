**Idea: Rank-Gradient Success-History Compass**

Category D: Uses fitness rank gradients and exponential weighted averaging of directional rank improvements to learn which mutation directions historically lead to better fitness ranks, then adapts mutation strength based on accumulated success.

```python
def _mutate_compass_batch(self, population, fitness):
    """Rank-gradient compass with success-history learning."""
    NP, dim = population.shape
    
    # Compute fitness ranks (D: fitness-landscape signal)
    fitness_ranks = self._compute_fitness_ranking(fitness)
    
    # Initialize success-history tracking (D: temporal rank-based)
    if not hasattr(self, '_rank_improvement_history'):
        self._rank_improvement_history = {}
        self._ema_rank_improvement = {}
    
    # Get current individual's rank
    i_rank = fitness_ranks  # shape (NP,)
    
    mutants = np.empty_like(population)
    
    for i in range(NP):
        # Get other indices
        others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
        np.random.shuffle(others)
        
        # Define 3 candidate directions (D: rank-based scoring)
        # Direction A: toward random individual
        r1, r2, r3 = others[:3]
        dir_A_target = r1
        rank_gain_A = i_rank[i] - fitness_ranks[r1]  # positive if r1 is better
        
        # Direction B: toward best individual
        best_idx = np.argmin(fitness)
        rank_gain_B = i_rank[i] - fitness_ranks[best_idx]
        dir_B_target = best_idx
        
        # Direction C: toward p-best (top 20%)
        top_p = max(1, int(0.2 * NP))
        elite_indices = np.argsort(fitness_ranks)[:top_p]
        pbest = np.random.choice(elite_indices)
        rank_gain_C = i_rank[i] - fitness_ranks[pbest]
        dir_C_target = pbest
        
        # Select best direction based on rank gain (D: rank-based, no distances)
        candidates = [(dir_A_target, rank_gain_A), (dir_B_target, rank_gain_B), (dir_C_target, rank_gain_C)]
        best_target, best_gain = max(candidates, key=lambda x: x[1])
        
        # Direction key for history tracking
        direction_key = (min(i, best_target), max(i, best_target))
        
        # Initialize history for new direction (D: success-history)
        if direction_key not in self._rank_improvement_history:
            self._rank_improvement_history[direction_key] = []
            self._ema_rank_improvement[direction_key] = 0.0
        
        # Update EMA of rank improvement for this direction (D: temporal rank-based)
        alpha = 0.3
        self._ema_rank_improvement[direction_key] = (1 - alpha) * self._ema_rank_improvement[direction_key] + alpha * max(0, best_gain)
        self._rank_improvement_history[direction_key].append(max(0, best_gain))
        if len(self._rank_improvement_history[direction_key]) > 10:
            self._rank_improvement_history[direction_key].pop(0)
        
        # Compute momentum factor from historical rank improvement (D: rank-based)
        ema_improvement = self._ema_rank_improvement[direction_key]
        momentum_factor = np.clip(0.5 + ema_improvement * 2.0, 0.3, 1.5)
        
        # Build weighted centroid from elite individuals (D: fitness percentiles)
        top_elite = max(1, NP // 5)
        elite_idx = np.argsort(fitness_ranks)[:top_elite]
        elite_ranks = fitness_ranks[elite_idx]
        # Inverse-rank weighting: better individuals get higher weight (D: rank-based)
        inv_ranks = 1.0 / (elite_ranks + 0.01)
        weights = inv_ranks / inv_ranks.sum()
        weighted_centroid = np.sum(population[elite_idx] * weights[:, np.newaxis], axis=0)
        
        # Compute base direction toward best target
        direction = population[best_target] - population[i]
        
        # Compute centroid direction (D: fitness-percentile weighted)
        centroid_direction = weighted_centroid - population[i]
        
        # Combine directions with momentum-adaptive scaling (D: rank-based adaptation)
        mutation = direction + momentum_factor * self.F * centroid_direction
        
        mutants[i] = population[i] + mutation
    
    return np.clip(mutants, -100.0, 100.0)
```