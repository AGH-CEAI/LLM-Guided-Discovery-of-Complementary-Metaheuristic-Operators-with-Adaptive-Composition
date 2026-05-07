**Idea: k-NN Graph Centrality Credit Assignment**

Use a k-nearest neighbor graph over the population to weight strategy credits by individual centrality and local neighborhood improvement rates. Individuals in sparse regions (low centrality) that improve receive higher credit since they're harder to find. Individuals in dense clusters that improve get lower credit since they're easier to discover.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Credit assignment using k-NN graph centrality and local improvement rates."""
    if not hasattr(self, '_current_population') or self._current_population is None:
        # No population available yet — decay all scores uniformly
        self.strategy_scores *= 0.95
        return
    
    pop = self._current_population
    NP, dim = pop.shape
    
    if NP < 4:
        self.strategy_scores *= 0.95
        return
    
    # Build k-NN graph (k scales with population size)
    k = max(2, min(5, NP // 8))
    
    # Compute pairwise squared distances
    diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Find k nearest neighbors for each individual
    nearest = np.argsort(sq_dists, axis=1)[:, :k]
    nearest_dists = sq_dists[np.arange(NP)[:, np.newaxis], nearest]
    
    # Compute centrality: inverse of average distance to k-NN (higher = more isolated)
    avg_knn_dist = np.mean(np.sqrt(nearest_dists), axis=1)
    centrality = 1.0 / (avg_knn_dist + 1e-8)
    centrality /= (centrality.max() + 1e-8)  # Normalize to [0, 1]
    
    # Compute local improvement rate for each individual's neighborhood
    local_improvement = np.zeros(NP)
    for i in range(NP):
        neighbors = nearest[i]
        neighbor_improved = improved[neighbors]
        # Weight by inverse distance (closer neighbors matter more)
        weights = 1.0 / (np.sqrt(nearest_dists[i]) + 1e-8)
        weights /= (weights.sum() + 1e-8)
        local_improvement[i] = np.sum(weights * neighbor_improved.astype(float))
    
    # Aggregate credits per strategy using graph-based weighting
    strategy_credits = np.zeros(len(self.strategy_scores))
    for s in range(len(self.strategy_scores)):
        mask = strategy_used == s
        if np.sum(mask) == 0:
            continue
        # Credit = mean of (centrality * local_improvement) for individuals using this strategy
        weighted_scores = centrality[mask] * (local_improvement[mask] + 0.1)
        strategy_credits[s] = np.mean(weighted_scores)
    
    # Apply credits via softmax-scaled updates (prevents runaway scores)
    if strategy_credits.max() > 1e-10:
        # Scale: stronger updates for higher credited strategies
        scaled = strategy_credits * 10.0
        exp_scaled = np.exp(scaled - scaled.max())
        weights = exp_scaled / (exp_scaled.sum() + 1e-8)
        self.strategy_scores += weights * 0.5
    
    # Penalize strategies that were used but produced no local improvements
    used_strategies = np.unique(strategy_used)
    for s in used_strategies:
        mask = strategy_used == s
        if np.sum(mask) > 0 and not improved[mask].any():
            self.strategy_scores[s] *= 0.85
    
    # Soft bounds to prevent score collapse
    self.strategy_scores = np.clip(self.strategy_scores, 0.1, 20.0)
```