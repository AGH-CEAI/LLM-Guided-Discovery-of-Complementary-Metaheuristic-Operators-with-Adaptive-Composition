Looking at the task, I need a fundamentally different geometric approach. The previous Category A variant (variant_01) used convex hull concepts and performed poorly (5.16e+02). I should try a different mechanism within geometry.

My approach: **k-NN isolation-weighted directional mutation** — use the geometric structure to identify isolated individuals and scale mutation strength inversely with local density, while computing mutation direction from the k-NN centroid. This is purely geometric (distances, neighbor structure) with no fitness information.

**Idea: kNN Isolation-Weighted Compass**
Category A geometry mutation using k-nearest neighbor structure to detect geometrically isolated individuals and weight mutation direction/scale accordingly.

```python
def _mutate_compass_batch(self, population, fitness):
    NP, dim = population.shape
    mutants = np.empty_like(population)
    
    # Compute pairwise Euclidean distances (purely geometric)
    diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # k-NN structure: find k nearest geometric neighbors per individual
    k = max(3, min(8, NP // 3))
    nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
    nearest_dists = np.sqrt(sq_dists[np.arange(NP)[:, np.newaxis], nearest_indices])
    
    # Compute isolation score: mean distance to k-NN (higher = more isolated)
    isolation = np.mean(nearest_dists, axis=1)
    isolation_norm = isolation / (isolation.max() + 1e-10)
    
    # Compute population geometric centroid
    centroid = population.mean(axis=0)
    
    # For each individual, compute geometric mutation
    for i in range(NP):
        neighbors = nearest_indices[i]
        neighbor_points = population[neighbors]
        
        # Direction A: toward centroid of k-NN (local structure)
        knn_centroid = neighbor_points.mean(axis=0)
        dir_knn = knn_centroid - population[i]
        dir_knn_norm = np.linalg.norm(dir_knn)
        if dir_knn_norm > 1e-10:
            dir_knn = dir_knn / dir_knn_norm
        
        # Direction B: toward population centroid (global structure)
        dir_global = centroid - population[i]
        dir_global_norm = np.linalg.norm(dir_global)
        if dir_global_norm > 1e-10:
            dir_global = dir_global / dir_global_norm
        
        # Direction C: principal axis of k-NN neighborhood
        centered = neighbor_points - knn_centroid
        if k > 1 and dim > 1:
            cov = (centered.T @ centered) / max(k - 1, 1)
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
                principal_dir = eigvecs[:, -1]
                # Project onto principal direction
                principal_component = np.dot(dir_knn, principal_dir) * principal_dir
                dir_principal = dir_knn - principal_component
                dir_principal_norm = np.linalg.norm(dir_principal)
                if dir_principal_norm > 1e-10:
                    dir_principal = dir_principal / dir_principal_norm
            except:
                dir_principal = np.zeros(dim)
        else:
            dir_principal = np.zeros(dim)
        
        # Weight directions by isolation: isolated individuals follow local structure more
        local_weight = 0.6 + 0.3 * (1.0 - isolation_norm[i])
        global_weight = 0.2 + 0.2 * isolation_norm[i]
        principal_weight = 0.2
        
        combined_dir = (local_weight * dir_knn + 
                       global_weight * dir_global + 
                       principal_weight * dir_principal)
        
        dir_norm = np.linalg.norm(combined_dir)
        if dir_norm > 1e-10:
            combined_dir = combined_dir / dir_norm
        
        # Scale mutation: isolated = larger step, clustered = smaller step
        # Use isolation to drive exploration for isolated points
        scale = self.F * (1.0 + 0.5 * isolation_norm[i])
        
        mutants[i] = population[i] + scale * combined_dir
    
    return mutants
```