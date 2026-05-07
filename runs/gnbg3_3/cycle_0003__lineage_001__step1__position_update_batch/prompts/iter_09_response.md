**Idea: Centroid-Gravity with k-NN Repulsion Modulation**

One-line description: Uses geometric centroid attraction weighted by inverse distance, modulated by k-NN local density to prevent crowding — avoiding spectral decomposition entirely (Category A: Geometry / spatial).

```python
def _position_update_batch(self):
    """Update positions using centroid gravity and k-NN density modulation."""
    centroid = np.mean(self.population, axis=0)
    
    # Compute distance to centroid for each particle
    dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)
    
    # Compute k-NN local density (average distance to k nearest neighbors)
    k = min(5, self.np - 1)
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    knn_dists = np.sort(sq_dists, axis=1)[:, :k]
    knn_avg_dist = np.mean(np.sqrt(knn_dists), axis=1, keepdims=True) + 1e-10
    
    # Global spread: average pairwise distance as normalization factor
    global_spread = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10
    
    # Centroid attraction strength: stronger for distant particles
    # Use inverse distance weighting, clipped to avoid explosion
    centroid_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))
    
    # Direction toward centroid
    to_centroid_dir = centroid - self.population
    to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
    to_centroid_dir = to_centroid_dir / to_centroid_dir
    
    # k-NN density modulation: repulse from dense regions
    # Particles in sparse regions get more centroid pull
    density_modulation = np.clip(knn_avg_dist / global_spread, 0.2, 2.0)
    
    # Combined geometric correction
    correction = centroid_attraction * to_centroid_dir * density_modulation
    
    # Base position update + geometric correction, scaled by inertia
    new_population = self.population + self.inertia_weight * self.velocity + 0.3 * correction
    
    self.population = self._clip_to_bounds(new_population)
```