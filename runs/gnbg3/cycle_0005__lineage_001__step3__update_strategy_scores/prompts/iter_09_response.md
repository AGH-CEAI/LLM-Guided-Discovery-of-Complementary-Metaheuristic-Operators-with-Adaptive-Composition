**Idea: k-NN Density-Weighted Geometric Strategy Scoring**

This approach computes local density via k-NN distances to identify underexplored regions of the search space. Strategies that generate improvements in low-density (novel) regions receive higher rewards, since such successes indicate the strategy is probing new geometric territory. Additionally, strategies whose improvements are widely spread across the population (high convex hull coverage) are rewarded for exploration breadth.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """
    Geometry-based strategy scoring using k-NN density and spatial spread.
    
    Strategies that generate improvements in underexplored (low-density) regions
    OR in geometrically diverse locations receive higher rewards.
    """
    if not hasattr(self, '_current_population') or self._current_population is None:
        return
    
    pop = self._current_population
    NP, dim = pop.shape
    
    if NP < 4:
        return
    
    # Compute k-NN distances for local density estimation
    k = max(2, min(5, NP // 4))
    
    # Pairwise squared distances (vectorized)
    diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Find k nearest neighbors for each individual
    nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
    nearest_sq_dists = sq_dists[np.arange(NP)[:, np.newaxis], nearest_indices]
    
    # Local density: inverse of average k-NN distance (higher = denser)
    avg_k_dist = np.mean(np.sqrt(nearest_sq_dists), axis=1)
    local_density = 1.0 / (avg_k_dist + 1e-8)
    
    # Normalize density to sum to NP
    density_sum = local_density.sum()
    if density_sum > 1e-10:
        local_density *= NP / density_sum
    
    # Compute centroid of all improving individuals
    if improved.any():
        improved_pop = pop[improved]
        global_centroid = improved_pop.mean(axis=0)
    else:
        global_centroid = pop.mean(axis=0)
    
    # Score each strategy based on geometric properties of its successes
    num_strategies = len(self.strategy_names)
    strategy_rewards = np.zeros(num_strategies)
    
    for s in range(num_strategies):
        # Individuals that used this strategy
        strategy_mask = (strategy_used == s)
        if not strategy_mask.any():
            continue
        
        # Among those, which improved?
        success_mask = strategy_mask & improved
        if not success_mask.any():
            continue
        
        # Geographic spread: mean distance from global centroid of improvements
        success_pop = pop[success_mask]
        centroid_dist = np.linalg.norm(success_pop - global_centroid, axis=1)
        mean_spread = np.mean(centroid_dist) + 1e-8
        
        # Density of successful region (lower density = more novel exploration)
        success_density = local_density[success_mask].sum()
        density_bonus = 1.0 / (success_density + 1.0)
        
        # Convex hull area proxy (for spread in multiple dimensions)
        if len(success_pop) >= 3:
            # Use pairwise distance span as hull proxy
            hull_dists = sq_dists[np.ix_(success_mask, success_mask)]
            np.fill_diagonal(hull_dists, 0)
            hull_span = np.mean(hull_dists) + 1e-8
        else:
            hull_span = 1.0
        
        # Combined geometric score: spread * novelty * hull coverage
        strategy_rewards[s] = mean_spread * density_bonus * np.sqrt(hull_span)
    
    # Normalize rewards
    total_reward = strategy_rewards.sum()
    if total_reward > 1e-10:
        strategy_rewards /= total_reward
    
    # Update scores using exponential moving average with momentum
    momentum = 0.7
    learning_rate = 0.3
    
    for s in range(num_strategies):
        if strategy_rewards[s] > 0:
            reward_signal = np.log1p(strategy_rewards[s] * NP)
        else:
            reward_signal = -2.0
        
        self.strategy_scores[s] = (momentum * self.strategy_scores[s] + 
                                   learning_rate * reward_signal)
    
    # Softmax normalization for final probabilities
    scores_exp = np.exp(self.strategy_scores - np.max(self.strategy_scores))
    self.strategy_scores = np.log(scores_exp / (scores_exp.sum() + 1e-10))
```