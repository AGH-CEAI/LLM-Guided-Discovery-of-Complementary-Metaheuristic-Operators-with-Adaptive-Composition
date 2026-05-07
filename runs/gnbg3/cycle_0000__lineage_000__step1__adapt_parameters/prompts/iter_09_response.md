**Idea: Pairwise Distance Momentum Adaptation**
Adapt F and CR using geometric statistics from pairwise Euclidean distances (median spread and distance variance), with exponential smoothing for temporal stability. This differs from the previous Category A approach by using explicit pairwise distance geometry rather than centroid-based measures.

```python
def _adapt_parameters(self, improvement_rate):
    """Adapt F and CR using pairwise distance geometry with momentum."""
    # Compute pairwise Euclidean distances (geometric feature)
    # Use broadcasting to get all pairwise distances efficiently
   膨胀 = self.dim * (100.0 ** 2)  # Approximate scale normalization
    
    # Pairwise distance matrix
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    pairwise_dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    
    # Extract upper triangle (avoid self-distances)
    n = self.NP
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    flat_dists = pairwise_dists[upper_mask]
    
    if len(flat_dists) < 10:
        return  # Not enough pairs
    
    # Geometric features: median spread and distance variance
    median_dist = np.median(flat_dists)
    dist_variance = np.var(flat_dists)
    
    # Normalize by dimension-appropriate scale
    median_norm = median_dist / 50.0
    variance_norm = dist_variance / 膨胀
    
    # Compute geometric momentum (temporal smoothing)
    geo_signal = 0.5 * median_norm + 0.5 * variance_norm
    self.geo_momentum = 0.7 * getattr(self, 'geo_momentum', geo_signal) + 0.3 * geo_signal
    
    # Adaptation signals
    spread_signal = np.clip(self.geo_momentum, 0.01, 5.0)
    improvement_signal = np.clip(improvement_rate, 0.0, 1.0)
    
    # Map signals to parameters via geometric reasoning:
    # High spread → reduce F (already exploring well)
    # Low spread → increase F (need larger steps to explore)
    # High improvement → increase CR (exploit good directions)
    # Low improvement → decrease CR (explore more)
    target_F = 0.3 + 0.7 * (1.0 / (1.0 + spread_signal))
    target_CR = 0.5 + 0.4 * improvement_signal
    
    # Apply momentum to updates
    self.F = 0.6 * self.F + 0.4 * target_F
    self.CR = 0.6 * self.CR + 0.4 * target_CR
    
    # Store history
    self.F_history.append(self.F)
    self.CR_history.append(self.CR)
    
    # Bound parameters
    self.F = np.clip(self.F, 0.1, 1.5)
    self.CR = np.clip(self.CR, 0.3, 0.98)
```