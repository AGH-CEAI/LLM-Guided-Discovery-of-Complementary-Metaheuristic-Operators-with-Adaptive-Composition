**Idea: Pairwise Distance Diversity**
Use mean pairwise Euclidean distances between population members, normalized by search space extent. This directly measures population spread and clustering, triggering restarts when the population collapses or clusters around deceptive local optima.

```python
def _compute_diversity(self):
    """Compute diversity using pairwise distances in population space.

    Directly measures spatial distribution of population via mean/median
    Euclidean distances between individuals. Normalized by search space
    diameter gives scale-invariant metric that detects:
      - Population collapse (all individuals clustered)
      - Deceptive local optima trapping (clustering away from global optimum)
      - Loss of exploration (low pairwise distances)
    """
    # Pairwise distance computation (vectorized for efficiency)
    # Compute all pairwise squared distances: ||x_i - x_j||^2
    pop = self.population[:len(self.fitness)]
    n = len(pop)
    
    if n < 2:
        return float(1e-15)
    
    # Use broadcasting: sq_dists[i,j] = ||pop[i] - pop[j]||^2
    # Shape: (n, n)
    sq_dists = np.sum((pop[:, np.newaxis, :] - pop[np.newaxis, :, :]) ** 2, axis=2)
    
    # Extract upper triangle (excluding diagonal) for unique pairs
    upper_tri_indices = np.triu_indices(n, k=1)
    pairwise_sq_dists = sq_dists[upper_tri_indices]
    
    if len(pairwise_sq_dists) == 0:
        return float(1e-15)
    
    # Statistics of pairwise distances
    mean_sq_dist = np.mean(pairwise_sq_dists)
    median_sq_dist = np.median(pairwise_sq_dists)
    
    # Avoid sqrt of large arrays when possible
    mean_dist = np.sqrt(mean_sq_dist)
    median_dist = np.sqrt(median_sq_dist)
    
    # Normalize by search space diameter
    search_diameter = np.sqrt(self.dim) * (self.ub[0] - self.lb[0])
    expected_dist = search_diameter / np.sqrt(2.0)  # Expected distance in uniform cube
    
    # Normalized metrics (0 to 1 range)
    mean_dist_norm = mean_dist / max(expected_dist, 1e-10)
    median_dist_norm = median_dist / max(expected_dist, 1e-10)
    
    # Also compute distance from population to best solution found
    # Low distance means population is converging toward (possibly local) optimum
    dists_to_best = np.sqrt(np.sum((pop - self.x_opt) ** 2, axis=1))
    mean_dist_to_best = np.mean(dists_to_best)
    std_dist_to_best = np.std(dists_to_best)
    
    # Diversity is high when population is spread AND not all clustered at best
    # Penalize if all individuals are close to the current best (possible trap)
    clustering_factor = mean_dist_to_best / max(expected_dist, 1e-10)
    clustering_penalty = np.clip(clustering_factor, 0.0, 1.0)
    
    # Combine metrics: geometric mean of spread and inverse-clustering
    spread_metric = (mean_dist_norm + median_dist_norm) / 2.0
    spread_metric = np.clip(spread_metric, 1e-15, 1.0)
    
    # Final diversity: geometric mean balances spread and avoids clustering
    diversity = spread_metric * np.sqrt(1.0 - 0.5 * clustering_penalty)
    
    return float(np.clip(diversity, 1e-15, None))
```