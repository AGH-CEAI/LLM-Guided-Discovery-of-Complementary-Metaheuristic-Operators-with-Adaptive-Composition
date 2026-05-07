**Idea: k-NN Density & Centroid Gravity with Axis-Aligned Spread**

Category A geometry-based position update using k-NN local density, centroid-directed gravity, and axis-aligned spread to modulate exploration/exploitation purely from spatial structure.

```python
def _position_update_fitness_rank(self):
    """Geometry-based position update using k-NN density and centroid distance.
    
    Category A: Uses pairwise distances, k-NN structure, centroid distances,
    axis-aligned spread. No fitness signals, no eigenvalues.
    """
    centroid = np.mean(self.population, axis=0)
    
    # k-NN density: compute average distance to k nearest neighbors
    k = min(5, self.np - 1)
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    knn_dists = np.sort(sq_dists, axis=1)[:, :k]
    knn_avg_dist = np.mean(np.sqrt(knn_dists), axis=1) + 1e-10
    
    # Global spread from pairwise distances
    global_spread = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10
    
    # Axis-aligned spread per dimension
    axis_spreads = np.ptp(self.population, axis=0) + 1e-10
    
    # Normalize k-NN density
    density_modulation = np.clip(knn_avg_dist / global_spread, 0.2, 2.0)
    
    # Distance to centroid
    dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True) + 1e-10
    
    # Centroid attraction (particles farther from centroid get more pull)
    centroid_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread))
    
    # Direction to centroid (normalized)
    to_centroid_dir = centroid - self.population
    to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
    to_centroid_dir = to_centroid_dir / to_centroid_dist
    
    # Axis-aligned exploration: dimensions with smaller spread get boosted exploration
    axis_explore = axis_spreads / (np.sum(axis_spreads) + 1e-10)
    per_dim_explore = np.where(
        axis_spreads < global_spread * 0.5,
        1.5,
        1.0
    )
    
    # Combined geometric correction
    correction = centroid_attraction * to_centroid_dir * density_modulation
    correction = correction * per_dim_explore
    
    # Convergence detection from axis-aligned spread
    total_spread = np.sum(axis_spreads)
    convergence_factor = np.clip(total_spread / (self.dim * global_spread + 1e-10), 0.5, 1.5)
    
    # Exploration factor for random perturbation
    exploration_factor = np.clip(
        density_modulation.mean() * convergence_factor,
        0.5, 2.0
    )
    
    # Random geometric perturbation scaled by density
    random_perturb = exploration_factor * 0.3 * np.random.randn(self.np, self.dim)
    
    # Apply velocity with geometric correction
    new_population = self.population + self.inertia_weight * self.velocity + 0.3 * correction + random_perturb
    
    self.population = self._clip_to_bounds(new_population)
```