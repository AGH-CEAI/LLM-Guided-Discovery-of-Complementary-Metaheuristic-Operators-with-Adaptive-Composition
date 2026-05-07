**Idea: Convex Hull + k-NN Density Adaptation**

Adapt neighborhood size based on convex hull volume (population spread) combined with k-NN density statistics. This uses geometric structure of the population rather than simple centroid distance.

```python
def _adapt_neighborhood_size(self):
    """Adapt ring topology neighborhood size based on convex hull geometry and k-NN density."""
    # --- Geometric Feature 1: Convex Hull Volume Proxy ---
    # Use axis-aligned bounding box volume as proxy (exact convex hull is expensive in high-D)
    pop_min = np.min(self.population, axis=0)
    pop_max = np.max(self.population, axis=0)
    bbox_extent = pop_max - pop_min
    bbox_volume = np.prod(bbox_extent + 1e-10)
    expected_volume = (self.upper_bound - self.lower_bound) ** self.dim
    volume_ratio = np.clip(bbox_volume / (expected_volume + 1e-10), 0.0, 1.0)
    
    # --- Geometric Feature 2: k-NN Density ---
    k = min(5, self.np - 1)
    sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    knn_dists = np.sqrt(np.sort(sq_dists, axis=1)[:, :k])
    mean_knn_dist = np.mean(knn_dists)
    std_knn_dist = np.std(knn_dist) if len(knn_dists.ravel()) > 1 else 0.0
    
    # Normalize by expected search space diagonal
    search_diagonal = np.sqrt(self.dim) * (self.upper_bound - self.lower_bound)
    knn_ratio = mean_knn_dist / (search_diagonal + 1e-10)
    
    # --- Geometric Feature 3: Axis-Aligned Spread Anisotropy ---
    # Compute coefficient of variation across dimensions
    spread_per_dim = bbox_extent / (np.mean(bbox_extent) + 1e-10)
    spread_cv = np.std(spread_per_dim) / (np.mean(spread_per_dim) + 1e-10)
    
    # --- Combine Geometric Signals ---
    # Low volume_ratio = compressed population → increase neighborhood (more social influence)
    # High knn_ratio = sparse particles → increase neighborhood (need more connectivity)
    # High spread_cv = anisotropic spread → moderate neighborhood (balance exploration)
    
    volume_signal = 1.0 - volume_ratio  # 0 = well-spread, 1 = compressed
    knn_signal = knn_ratio * 10.0  # Scale to reasonable range
    anisotropy_signal = np.clip(spread_cv, 0.0, 2.0)
    
    combined_signal = 0.4 * volume_signal + 0.4 * knn_signal + 0.2 * anisotropy_signal
    
    # --- Decision Logic ---
    # Map combined signal to neighborhood adjustment
    if combined_signal < 0.3:
        # Population is well-spread and connected → decrease neighborhood (exploit)
        self.neighborhood_size = max(1, self.neighborhood_size - 1)
    elif combined_signal > 0.8:
        # Population is compressed or fragmented → increase neighborhood (explore)
        self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
    # else: maintain current neighborhood_size
```