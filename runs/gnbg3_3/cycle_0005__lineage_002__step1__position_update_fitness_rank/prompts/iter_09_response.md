**Idea: Convex-Hull Volume & Axis-Spread Compression Correction**

Uses convex hull volume ratio and per-axis spread analysis to detect population collapse, then applies corrective expansion along compressed spatial directions. Purely geometric: no fitness signals, no covariance matrices.

```python
def _position_update_fitness_rank(self):
    """Convex-hull volume + axis-spread compression correction (variant_10, Cat A).
    
    Purely geometric: convex hull boundary detection, pairwise distance
    statistics, axis-aligned bounding box, per-axis spread analysis.
    Detects when population collapses into local optima and applies
    corrective expansion along compressed spatial directions.
    """
    # 1. Centroid and axis-aligned bounding box
    centroid = np.mean(self.population, axis=0)
    bb_min = np.min(self.population, axis=0)
    bb_max = np.max(self.population, axis=0)
    bb_sizes = bb_max - bb_min
    bb_volume = np.prod(bb_sizes + 1e-10)
    mean_spread = np.mean(bb_sizes) + 1e-10
    
    # 2. Pairwise distance statistics
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    min_pairwise = np.min(np.sqrt(sq_dists + 1e-20), axis=1)
    mean_pairwise = np.mean(np.sqrt(sq_dists[sq_dists < np.inf]) + 1e-20)
    
    # 3. Convex hull volume (boundary detection)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(self.population)
        hull_vol = hull.volume if hasattr(hull, 'volume') else hull.area
    except Exception:
        hull_vol = bb_volume * 0.5
    hull_vol = max(hull_vol, 1e-10)
    
    # 4. k-NN density (k=3) for local sparsity detection
    k = min(3, self.np - 1)
    sorted_sq_dists = np.sort(sq_dists, axis=1)
    knn_dists = np.sqrt(sorted_sq_dists[:, 1:k+1] + 1e-20)
    knn_density = 1.0 / (np.mean(knn_dists, axis=1) + 1e-10)
    knn_density_norm = knn_density / (np.mean(knn_density) + 1e-10)
    
    # 5. Per-axis spread analysis
    axis_spreads = bb_sizes
    axis_compression = np.zeros(self.dim)
    for d in range(self.dim):
        if mean_spread > 1e-10 and axis_spreads[d] < 0.3 * mean_spread:
            axis_compression[d] = 1.0 - (axis_spreads[d] / (0.3 * mean_spread + 1e-10))
    compressed_dims = np.sum(axis_compression > 0.5)
    
    # 6. Geometric penalty: combine hull volume ratio + pairwise compression + axis compression
    hull_volume_ratio = hull_vol / (bb_volume + 1e-10)
    hull_penalty = np.clip(1.0 - hull_volume_ratio, 0.0, 1.0)
    
    pairwise_compression = np.clip(1.0 - (mean_pairwise / (mean_spread + 1e-10)), 0.0, 1.0)
    axis_compression_score = np.mean(axis_compression)
    
    # Combined geometric penalty
    geometric_penalty = np.clip(0.5 * hull_penalty + 0.3 * pairwise_compression + 0.2 * axis_compression_score, 0.0, 1.0)
    
    # 7. Velocity scaling based purely on geometry
    vel_scale = 1.0 + 0.5 * geometric_penalty
    vel_scale = np.clip(vel_scale, 0.5, 2.5)
    
    # 8. Spatial correction: boundary particles pulled inward, compressed axes expanded
    # k-NN density: sparse particles get outward push, dense get inward pull
    density_force = (knn_density_norm - 1.0)[:, np.newaxis] * (self.population - centroid) * 0.15
    
    # Convex hull boundary detection: hull vertices get centripetal correction
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(self.population)
        hull_vertices = set(hull.vertices)
        boundary_correction = np.zeros_like(self.population)
        for i in range(self.np):
            if i in hull_vertices:
                to_centroid = centroid - self.population[i]
                dist_c = np.linalg.norm(to_centroid) + 1e-10
                boundary_correction[i] = (to_centroid / dist_c) * 0.1 * geometric_penalty
    except Exception:
        boundary_correction = np.zeros_like(self.population)
    
    # Axis compression correction: expand along compressed dimensions
    axis_correction = np.zeros_like(self.population)
    if compressed_dims > 0 and mean_spread > 1e-10:
        for d in range(self.dim):
            if axis_compression[d] > 0.3:
                # Direction: away from centroid along axis d
                direction = np.sign(self.population[:, d] - centroid[d])
                magnitude = (mean_spread - axis_spreads[d]) / mean_spread * axis_compression[d] * 0.2 * geometric_penalty
                axis_correction[:, d] = direction * magnitude
    
    # 9. Combine all geometric corrections
    total_correction = density_force + boundary_correction + axis_correction
    
    # 10. Position update: velocity scaled by geometry + geometric corrections
    new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + total_correction
    
    self.population = self._clip_to_bounds(new_population)
```