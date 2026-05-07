**Idea: Convex Hull Boundary Correction**

Category A geometry variant using convex hull analysis to distinguish boundary (exploration) vs interior (exploitation) particles. Hull vertices get inward corrections; interior particles get outward corrections weighted by local convexity. This targets the worst tasks (17, 16, 6) by explicitly managing the spatial extent of the swarm through geometric hull reasoning rather than density or spectral signals.

```python
def _position_update_centroid_knn(self):
    """Convex hull boundary correction (Category A).
    
    Targets worst tasks (17, 16, 6, 11) with large errors by explicitly
    managing exploration/exploitation via hull geometry. Particles on the
    convex hull are at the boundary of explored space — they need inward
    correction to maintain cohesion. Interior particles with low local
    convexity get outward nudges. Uses ONLY geometric computations."""
    from scipy.spatial import ConvexHull
    
    centroid = np.mean(self.population, axis=0)
    
    # --- Convex hull analysis ---
    try:
        hull = ConvexHull(self.population)
        hull_vertices = hull.vertices
        is_hull_vertex = np.zeros(self.np, dtype=bool)
        is_hull_vertex[hull_vertices] = True
        
        # Compute distance to hull for each particle using outward normal
        # For hull vertices, signed distance is positive (outside or on)
        # For interior points, we need to project to nearest hull facet
        hull_points = self.population[hull_vertices]
        hull_centroid = np.mean(hull_points, axis=0)
        
        # Distance from centroid to hull centroid (measure of swarm spread)
        dist_hull_centroid_to_global = np.linalg.norm(hull_centroid - centroid) + 1e-10
        
        # For each particle: project to hull and measure penetration depth
        # Simple proxy: distance from particle to hull centroid normalized by hull radius
        hull_radius = np.mean(np.linalg.norm(hull_points - hull_centroid, axis=1)) + 1e-10
        dist_to_hull_centroid = np.linalg.norm(self.population - hull_centroid, axis=1, keepdims=True)
        hull_penetration = dist_to_hull_centroid / (hull_radius + 1e-10)
        
        # Convexity signal: positive = outside hull centroid (exploration zone)
        # negative = inside hull centroid (exploitation zone)
        convexity_signal = (dist_to_hull_centroid - hull_radius) / (hull_radius + 1e-10)
        
    except Exception:
        # Fallback: no hull available (degenerate case)
        is_hull_vertex = np.zeros(self.np, dtype=bool)
        hull_penetration = np.zeros((self.np, 1))
        convexity_signal = np.zeros((self.np, 1))
        dist_hull_centroid_to_global = 1.0
    
    # --- k-NN local density (geometry-only) ---
    k = min(5, self.np - 1)
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    knn_dists = np.sort(sq_dists, axis=1)[:, :k]
    knn_avg_dist = np.mean(np.sqrt(knn_dists), axis=1, keepdims=True) + 1e-10
    
    global_spread = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10
    
    # --- Direction to centroid ---
    to_centroid_dir = centroid - self.population
    to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
    to_centroid_dir_norm = to_centroid_dir / to_centroid_dist
    
    # --- Geometry-based correction ---
    # Hull vertices: inward correction (reduce exploration, increase cohesion)
    # Interior with low convexity: outward correction (increase exploration)
    # Strong density modulation from k-NN
    
    density_modulation = np.clip(knn_avg_dist / global_spread, 0.2, 2.0)
    
    # Base centroid attraction (normalized)
    centroid_attraction = 0.5 * (to_centroid_dist / (to_centroid_dist + global_spread + 1e-10))
    
    # Convexity-based exploration/exploitation split
    # convexity_signal > 0: outside typical hull radius -> more exploitation (inward)
    # convexity_signal < 0: inside hull radius -> more exploration (outward)
    convexity_weight = np.clip(convexity_signal, -1.0, 1.0)
    
    # Hull vertex correction: strong inward pull for boundary particles
    hull_correction = np.where(
        is_hull_vertex[:, np.newaxis],
        -0.8 * to_centroid_dir_norm,  # Inward for hull vertices
        convexity_weight * to_centroid_dir_norm  # Outward/inward for interior
    )
    
    # Combine: centroid attraction + hull correction + density modulation
    correction = (
        0.3 * centroid_attraction * to_centroid_dir_norm +
        0.5 * hull_correction * density_modulation
    )
    
    new_population = self.population + self.inertia_weight * self.velocity + 0.4 * correction
    
    self.population = self._clip_to_bounds(new_population)
```