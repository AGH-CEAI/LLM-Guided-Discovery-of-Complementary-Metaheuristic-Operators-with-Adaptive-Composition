**Idea: Axis-Aligned Spread + k-NN Distance Adaptation**
Category A: Use geometric spread (axis-aligned range per dimension) and k-nearest-neighbor distances to drive F and Cr adaptation, replacing fitness-based success history with literal spatial structure analysis.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr based on geometric layout of population.
    
    Category A: Uses axis-aligned spread and k-NN structure instead of
    fitness-based success history to determine exploration vs exploitation.
    """
    pop = self.population if hasattr(self, 'population') else None
    if pop is None or len(pop) < 3:
        return
    
    # Compute axis-aligned spread per dimension (sum of ranges)
    dim_mins = np.min(pop, axis=0)
    dim_maxs = np.max(pop, axis=0)
    dim_ranges = dim_maxs - dim_mins
    total_spread = np.sum(dim_ranges)
    
    # Normalize by search space size for scale-invariance
    search_space_size = (self.upper - self.lower) * self.dim
    normalized_spread = total_spread / (search_space_size + 1e-30)
    
    # Compute k-NN distances for each individual (k=3)
    k = min(3, len(pop) - 1)
    knn_distances = np.zeros(len(pop))
    for i in range(len(pop)):
        dists = np.linalg.norm(pop - pop[i], axis=1)
        dists[i] = np.inf  # Exclude self
        sorted_dists = np.sort(dists)
        knn_distances[i] = np.mean(sorted_dists[:k])
    
    mean_knn = np.mean(knn_distances)
    knn_normalized = mean_knn / (search_space_size ** 0.5 + 1e-30)
    
    # Geometric adaptation rules:
    # Low spread = tightly clustered = need MORE exploration (higher F)
    # High spread = well-dispersed = need exploitation (lower F)
    # High k-NN distance = sparse local neighborhoods = higher Cr for mixing
    # Low k-NN distance = dense neighborhoods = lower Cr for local search
    
    if normalized_spread < 0.01:  # Very tight clustering
        self.F = np.clip(self.F * 1.15, 0.1, 2.0)
    elif normalized_spread > 0.5:  # Well dispersed
        self.F = np.clip(self.F * 0.9, 0.1, 2.0)
    
    if knn_normalized > 0.3:  # Sparse neighborhoods
        self.Cr = np.clip(self.Cr * 1.08, 0.1, 0.9)
    elif knn_normalized < 0.1:  # Dense neighborhoods
        self.Cr = np.clip(self.Cr * 0.92, 0.1, 0.9)
```