**Idea: Convex Hull Boundary + k-NN Density Modulation**
Category A geometry/spatial approach using convex hull proximity and k-NN density to modulate velocities without any spectral decomposition.

```python
def _position_update_original(self):
    """Geometry A: convex hull boundary + k-NN density modulation."""
    centroid = np.mean(self.population, axis=0)
    centered = self.population - centroid
    
    # --- Signal 1: Convex hull boundary distance ---
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(self.population)
        hull_vertices = self.population[hull.vertices]
        
        # Distance from each particle to nearest hull vertex
        min_hull_dist = np.min(
            np.linalg.norm(centered[:, np.newaxis, :] - hull_vertices[np.newaxis, :, :], axis=2),
            axis=1
        )
        hull_extent = np.max(np.linalg.norm(centered, axis=1)) + 1e-10
        boundary_proximity = np.clip(min_hull_dist / hull_extent, 0.0, 1.0)
        
        # Near boundary = high anisotropy risk → modulate velocity
        boundary_scale = 0.7 + 0.6 * boundary_proximity
        boundary_scale = np.clip(boundary_scale, 0.5, 1.3)
    except Exception:
        boundary_scale = 1.0
    
    # --- Signal 2: k-NN local density ---
    k = min(5, self.np - 1)
    sq_dists = np.sum(centered[:, np.newaxis, :] ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    knn_sorted = np.sort(sq_dists, axis=1)[:, :k]
    knn_mean_dist = np.mean(np.sqrt(knn_sorted), axis=1) + 1e-10
    
    global_knn_mean = np.mean(knn_mean_dist) + 1e-10
    density_ratio = knn_mean_dist / global_knn_mean
    
    # Sparse (low density) = may be trapped → boost exploration
    # Dense (high density) = good convergence → reduce exploration
    density_scale = np.clip(1.0 / density_ratio, 0.5, 2.0)
    
    # --- Signal 3: Centroid direction magnitude (axis-aligned spread proxy) ---
    axis_spread = np.std(centered, axis=0) + 1e-10
    axis_spread_max = np.max(axis_spread)
    axis_spread_normalized = axis_spread / axis_spread_max
    
    # Per-dimension spread: low spread direction = anisotropic → scale up
    dim_scale = 1.0 + 0.4 * (1.0 - axis_spread_normalized)
    
    # --- Combine signals into per-particle velocity scaling ---
    vel_scale = boundary_scale * np.mean(density_scale) * 1.0
    vel_scale = np.clip(vel_scale, 0.5, 1.8)
    
    # --- Compute geometric correction from centroid direction ---
    to_centroid = centroid - self.population
    to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
    to_centroid_dir = to_centroid / to_centroid_dist
    
    # Particles far from centroid get stronger pull toward center
    centroid_attraction = 0.3 * (to_centroid_dist / (to_centroid_dist + 50.0))
    
    # --- Apply: velocity scaled by geometry + centroid correction ---
    new_population = self.population + self.velocity * vel_scale
    new_population += centroid_attraction * to_centroid_dir
    
    self.population = self._clip_to_bounds(new_population)
```