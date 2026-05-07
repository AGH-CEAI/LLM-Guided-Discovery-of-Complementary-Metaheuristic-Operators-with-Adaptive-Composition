**Idea: k-NN Density Ratio Adaptation**
One-line description: Adapt neighborhood size using k-NN average nearest-neighbor distance ratio to global spread, capturing local density structure that centroid-based diversity misses.

```python
def _adapt_neighborhood_size(self):
    """Adapt ring topology neighborhood size based on k-NN local density structure.
    
    Category A (Geometry / spatial): Uses k-NN nearest-neighbor distances and
    axis-aligned spread ratio to detect clustering vs dispersion. Different from
    centroid-distance approach in prior variant — focuses on LOCAL neighborhood
    density rather than global centroid scatter.
    """
    if self.np < 5:
        return
    
    # k-NN structure: average distance to k nearest neighbors per particle
    k = min(3, self.np - 1)
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    sorted_dists = np.sort(sq_dists, axis=1)
    knn_avg_dist = np.mean(sorted_dists[:, :k], axis=1)
    
    # Global spread via axis-aligned bounding box extent
    pop_min = np.min(self.population, axis=0)
    pop_max = np.max(self.population, axis=0)
    bbox_extent = np.mean(pop_max - pop_min) + 1e-10
    
    # Local density ratio: small knn_avg_dist → dense/clustered
    density_ratio = knn_avg_dist.mean() / bbox_extent
    
    # Spread signal: how much of the bbox is occupied
    occupancy = np.std(self.population) / (bbox_extent * 0.5 + 1e-10)
    occupancy = np.clip(occupancy, 0.0, 2.0)
    
    # Cluster detection: variance of knn distances (high variance = fragmented)
    knn_variance = np.var(knn_avg_dist) / (np.mean(knn_avg_dist) ** 2 + 1e-10)
    knn_variance = np.clip(knn_variance, 0.0, 10.0)
    
    # Determine adjustment based on density structure
    # Dense population (low ratio) → shrink neighborhood to maintain local search
    # Sparse/fragmented (high ratio or high variance) → expand neighborhood
    if density_ratio < 0.1 or knn_variance > 2.0:
        # Clustered or fragmented: shrink to favor local exploitation
        self.neighborhood_size = max(1, self.neighborhood_size - 1)
    elif density_ratio > 0.5 and occupancy < 0.5:
        # Sparse and under-spread: expand to encourage exploration
        self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
    elif density_ratio > 0.3:
        # Moderately sparse: gradual expansion
        self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
    # else: maintain current neighborhood_size
```