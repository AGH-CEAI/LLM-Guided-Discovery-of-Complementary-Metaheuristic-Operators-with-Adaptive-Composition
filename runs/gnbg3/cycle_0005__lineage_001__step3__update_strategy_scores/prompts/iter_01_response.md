**Idea: Geometric Strategy Credit via Centroid Distance and k-NN Density**

Use spatial metrics (centroid distance, k-NN density, population spread ratio) to score strategies based on whether they explore underexplored geometric regions — purely geometric reasoning, no fitness signals.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Geometry-based strategy credit: spatial analysis of where strategies operate."""
    if not hasattr(self, '_current_population') or self._current_population is None:
        return
    
    pop = self._current_population
    NP, dim = pop.shape
    
    # Global population metrics
    global_centroid = pop.mean(axis=0)
    
    # Pairwise distances (vectorized)
    diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Global spread
    global_spread = np.mean(np.sqrt(sq_dists))
    global_spread = max(global_spread, 1e-10)
    
    n_strategies = len(self.strategy_scores)
    
    for s in range(n_strategies):
        mask = (strategy_used == s)
        count = np.sum(mask)
        
        if count < 2:
            continue
        
        # Strategy centroid
        centroid = pop[mask].mean(axis=0)
        
        # Strategy spread: mean distance from centroid
        centroid_dists = np.linalg.norm(pop[mask] - centroid, axis=1)
        spread = np.mean(centroid_dists)
        spread_ratio = spread / global_spread
        
        # Distance from strategy centroid to global centroid
        centroid_to_global = np.linalg.norm(centroid - global_centroid)
        normalized_dist = centroid_to_global / global_spread
        
        # k-NN local density within strategy group
        k = min(3, count - 1)
        if k > 0:
            group_sq_dists = sq_dists[np.ix_(mask, mask)]
            np.fill_diagonal(group_sq_dists, np.inf)
            knn_dists = np.sort(group_sq_dists, axis=1)[:, :k]
            avg_knn = np.mean(np.sqrt(knn_dists))
            density_score = 1.0 / (avg_knn + 1e-10)
        else:
            density_score = 0.0
        
        # Geometric score: reward strategies in underexplored sparse regions
        # Components: (1) spread ratio, (2) inverse density, (3) distance from global centroid
        geometric_score = spread_ratio * 0.35 + density_score * 0.35 + normalized_dist * 0.30
        
        # Clamp to prevent extreme values
        geometric_score = np.clip(geometric_score, 0.01, 10.0)
        
        # Exponential moving average update
        self.strategy_scores[s] = 0.85 * self.strategy_scores[s] + 0.15 * geometric_score
```