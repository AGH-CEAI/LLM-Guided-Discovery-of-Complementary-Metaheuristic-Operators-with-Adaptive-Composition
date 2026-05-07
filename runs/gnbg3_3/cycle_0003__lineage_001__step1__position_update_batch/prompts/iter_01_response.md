**Idea: k-NN Density-Modulated Geometric Update**
Use k-nearest-neighbor distances to estimate local density and modulate position updates based on geometric spread and centroid attraction/repulsion, without spectral decomposition.

```python
def _position_update_batch(self):
    """Update positions using k-NN density and centroid geometry."""
    # Compute centroid and bounding box (purely geometric)
    centroid = np.mean(self.population, axis=0)
    min_pos = np.min(self.population, axis=0)
    max_pos = np.max(self.population, axis=0)
    bbox = max_pos - min_pos + 1e-10  # axis-aligned spread
    
    # k-NN structure: compute local density for each particle
    k = min(5, self.np - 1)
    sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    knn_dists = np.sort(sq_dists, axis=1)[:, :k]
    knn_mean_dist = np.mean(knn_dists, axis=1, keepdims=True) + 1e-10
    
    # Normalize knn density: low distance = high density
    global_mean_dist = np.mean(knn_dists) + 1e-10
    density_factor = knn_mean_dist / global_mean_dist
    density_factor = np.clip(density_factor, 0.2, 3.0)
    
    # Centroid direction and distance
    to_centroid = centroid - self.population
    dist_to_centroid = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
    centroid_dir = to_centroid / dist_to_centroid
    
    # Axis-aligned spread modulation: scale velocity components by inverse spread
    # Low spread in a dimension -> stronger movement allowed in that direction
    spread_modulation = np.mean(bbox) / (bbox + 1e-10)
    spread_modulation = np.clip(spread_modulation, 0.1, 5.0)
    
    # Velocity update: density-weighted centroid attraction + spread scaling
    # Particles in low-density regions (far from neighbors) are attracted more to centroid
    # Particles in high-density regions are attracted less
    attraction_strength = 0.4 / density_factor
    centroid_component = attraction_strength * centroid_dir * dist_to_centroid
    
    # Apply spread modulation to velocity and centroid component
    new_population = self.population + self.velocity * spread_modulation + centroid_component
    
    self.population = self._clip_to_bounds(new_population)
```