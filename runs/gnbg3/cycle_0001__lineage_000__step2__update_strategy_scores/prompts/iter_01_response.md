**Idea: Centroid-Hull Credit Assignment**
Use geometric displacement metrics (centroid shift and convex hull volume proxy) to assign credit to strategies that produce spatially significant improvements, rather than raw fitness-based counts.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """Credit assignment using geometric displacement: centroid shift and hull volume change."""
    # Access population from instance (set by __call__ before this method)
    if not hasattr(self, 'population') or self.population is None:
        return
    
    # Initialize geometric state tracking on first call
    if not hasattr(self, '_prev_centroid'):
        self._prev_centroid = None
        self._prev_hull_vol = None
    
    # Compute current population geometry
    centroid = self.population.mean(axis=0)
    pop_range = self.population.max(axis=0) - self.population.min(axis=0) + 1e-10
    hull_proxy = np.prod(pop_range)
    
    # Compute geometric change from previous generation
    if self._prev_centroid is not None and self._prev_hull_vol is not None:
        centroid_shift = np.linalg.norm(centroid - self._prev_centroid)
        hull_change_ratio = hull_proxy / (self._prev_hull_vol + 1e-10)
    else:
        centroid_shift = 0.0
        hull_change_ratio = 1.0
    
    # Population spread for normalization
    pop_spread = np.linalg.norm(pop_range)
    pop_spread = max(pop_spread, 1e-10)
    
    # Credit assignment per strategy using geometric signals
    for s in range(len(self.strategy_scores)):
        mask = strategy_used == s
        if not np.any(mask):
            continue
        
        improved_idx = np.where(mask & improved)[0]
        n_improved = len(improved_idx)
        
        if n_improved == 0:
            # No improvements: geometric penalty based on hull contraction
            if hull_change_ratio < 1.0:
                self.strategy_scores[s] *= (0.95 + 0.05 * hull_change_ratio)
            continue
        
        # Geometric displacement of improved solutions from centroid
        improved_centroid = self.population[improved_idx].mean(axis=0)
        displacement = np.linalg.norm(improved_centroid - centroid)
        norm_displacement = displacement / pop_spread
        
        # Credit = normalized displacement * sqrt(success count) * hull expansion factor
        # sqrt(n_improved) balances between having many small wins vs few large jumps
        credit = norm_displacement * np.sqrt(n_improved) * max(hull_change_ratio, 0.5)
        
        # Exponential moving average update with momentum
        momentum = 0.1
        self.strategy_scores[s] = (1 - momentum) * self.strategy_scores[s] + momentum * credit
    
    # Prevent pathological score values
    self.strategy_scores = np.clip(self.strategy_scores, 0.01, 100.0)
    
    # Store current geometric state for next iteration
    self._prev_centroid = centroid
    self._prev_hull_vol = hull_proxy
```