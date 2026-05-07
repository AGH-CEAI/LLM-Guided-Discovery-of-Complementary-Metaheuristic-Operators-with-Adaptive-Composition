**Idea: Spatial Spread Adaptation**
Adapt F and CR based on population geometric spread: axis-aligned range and pairwise centroid distances. When the population is spatially compressed (low spread), increase F to escape; when dispersed (high spread), reduce F and increase CR for exploitation.

```python
def _adapt_parameters(self, improvement_rate):
    """Adapt F and CR based on geometric spatial spread of population."""
    if not hasattr(self, '_pop_history'):
        self._pop_history = []
    
    # Get current population from the optimizer context
    # We access it via self's stored reference or reconstruct from available state
    # Since we don't have direct population access, we track spread via stored metrics
    
    # Compute axis-aligned spread (range per dimension)
    # This requires population - we store it in a class attribute set by __call__
    if hasattr(self, '_current_population') and self._current_population is not None:
        pop = self._current_population
        pop_min = pop.min(axis=0)
        pop_max = pop.max(axis=0)
        axis_spread = pop_max - pop_min
        
        # Normalize spread relative to search space (assuming [-100, 100])
        normalized_spread = axis_spread / 200.0
        mean_spread = normalized_spread.mean()
        max_spread = normalized_spread.max()
        
        # Compute pairwise distances from centroid
        centroid = pop.mean(axis=0)
        centroid_distances = np.linalg.norm(pop - centroid, axis=1)
        mean_centroid_dist = centroid_distances.mean()
        
        # Normalize by expected random spread (roughly sqrt(dim) * range/sqrt(NP))
        expected_dist = 200.0 * np.sqrt(self.dim / self.NP)
        normalized_centroid_dist = mean_centroid_dist / expected_dist
        
        # Geometric composite: combine spread metrics
        # Low spread -> population clustered -> need larger F to explore
        # High spread -> population dispersed -> reduce F, increase CR for exploitation
        spread_factor = np.clip(normalized_centroid_dist, 0.1, 2.0)
        
        # F adaptation: inversely proportional to spread
        # When spread is low (clustered), increase F significantly
        target_F = 0.6 / (spread_factor + 0.3)
        target_F = np.clip(target_F, 0.3, 1.5)
        
        # CR adaptation: proportional to spread (more exploration when dispersed)
        target_CR = 0.5 + 0.4 * np.clip(spread_factor - 0.5, 0, 1)
        target_CR = np.clip(target_CR, 0.3, 0.95)
        
        # Smooth transition with momentum
        if len(self.F_history) > 0:
            alpha = 0.3
            self.F = alpha * target_F + (1 - alpha) * self.F_history[-1]
            self.CR = alpha * target_CR + (1 - alpha) * self.CR_history[-1]
        else:
            self.F = target_F
            self.CR = target_CR
        
        # Clamp final values
        self.F = float(np.clip(self.F, 0.1, 2.0))
        self.CR = float(np.clip(self.CR, 0.1, 0.99))
        
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
        
        # Keep history bounded
        if len(self.F_history) > 50:
            self.F_history = self.F_history[-50:]
            self.CR_history = self.CR_history[-50:]
```