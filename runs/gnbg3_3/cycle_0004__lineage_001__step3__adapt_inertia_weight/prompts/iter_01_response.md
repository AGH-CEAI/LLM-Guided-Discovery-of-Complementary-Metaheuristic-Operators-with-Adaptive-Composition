**Idea: k-NN Density and Pairwise Distance Geometry**
Adapt inertia weight using k-NN local density ratio and pairwise distance spread — distinctly geometric measures of particle clustering vs dispersion.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight based on k-NN density ratio and pairwise distance geometry."""
    n = self.np
    
    # 1. Global pairwise distance spread (upper triangle sample)
    if n > 1:
        indices = np.triu_indices(n, k=1)
        diffs = self.population[indices[0]] - self.population[indices[1]]
        pairwise_dists = np.linalg.norm(diffs, axis=1)
        avg_pairwise = np.mean(pairwise_dists)
        std_pairwise = np.std(pairwise_dists)
    else:
        avg_pairwise = 1.0
        std_pairwise = 0.0
    
    # 2. k-NN local density for each particle
    k = min(5, n - 1)
    if k > 0 and n > 1:
        diffs_knn = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists_knn = np.sum(diffs_knn ** 2, axis=2)
        np.fill_diagonal(sq_dists_knn, np.inf)
        sorted_knn = np.sort(sq_dists_knn, axis=1)[:, :k]
        knn_avg_dist = np.mean(np.sqrt(sorted_knn)) + 1e-10
    else:
        knn_avg_dist = 1.0
    
    # 3. Geometric ratio: local density vs global spread
    # High ratio → particles in dense neighborhoods → lower inertia (exploit)
    # Low ratio → sparse particles → higher inertia (explore)
    density_spread_ratio = knn_avg_dist / (avg_pairwise + 1e-10)
    density_ratio_normalized = np.clip(density_spread_ratio / 2.0, 0.0, 1.0)
    
    # 4. Coefficient of variation of pairwise distances (anisotropy signal)
    cv_normalized = np.clip(std_pairwise / (avg_pairwise + 1e-10) / 0.5, 0.0, 1.0)
    
    # 5. Combine signals: higher inertia when sparse or anisotropic
    target_inertia = 0.4 + (1.0 - density_ratio_normalized) * 0.35 + cv_normalized * 0.2
    
    # 6. Blend with generation-based schedule
    generation_factor = 1.0 - 0.15 * (self.generation / 1000)
    target_inertia = np.clip(target_inertia * 0.6 + generation_factor * 0.4, 0.4, 0.95)
    
    # 7. EMA update
    self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
    self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```