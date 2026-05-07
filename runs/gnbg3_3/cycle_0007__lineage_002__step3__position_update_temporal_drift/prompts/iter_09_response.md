**Idea: Convex Hull Volume + Principal Axis Expansion**

This variant replaces temporal-drift signals (EMA, autocorrelation) with geometric analysis: convex hull volume in PC-space detects population collapse, while principal axis projections modulate expansion direction. This targets the worst tasks (17, 16, 6) where the swarm may be geometrically trapped in a small region far from the optimum.

```python
def _position_update_temporal_drift(self):
    """Convex-hull volume + principal-axis expansion (Category A: Geometry).
    
    Key insight: Detect population collapse via convex hull volume in PC-space;
    use principal axis projections to direct expansion away from centroid.
    Targets tasks where swarm is geometrically trapped (e.g., Task 17: 4.5e+04 error).
    """
    # --- Geometric Signal 1: Convex Hull Volume ---
    try:
        from scipy.spatial import ConvexHull
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        pc_population = self.population @ eigenvectors[:, :min(3, self.dim)]
        hull = ConvexHull(pc_population)
        hull_volume = hull.volume if hasattr(hull, 'volume') else hull.area
        hull_volume = max(hull_volume, 1e-30)
    except:
        hull_volume = 1.0

    search_space_vol = (self.upper_bound - self.lower_bound) ** min(3, self.dim)
    volume_ratio = hull_volume / (search_space_vol + 1e-30)
    volume_ratio = np.clip(volume_ratio, 1e-10, 1.0)

    # Small volume_ratio = collapsed population → strong expansion
    if volume_ratio < 1e-6:
        expansion_strength = 2.5
    elif volume_ratio < 1e-3:
        expansion_strength = 1.8
    elif volume_ratio < 0.01:
        expansion_strength = 1.3
    else:
        expansion_strength = 1.0

    # --- Geometric Signal 2: Principal Axis Projection Analysis ---
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]
        total_var = np.sum(eigenvalues)
        var_ratios = eigenvalues / (total_var + 1e-30)
        effective_dims = np.sum(var_ratios > 0.01)
        elongation_factor = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
    except:
        effective_dims = self.dim
        elongation_factor = 1.0

    elongation_modulation = np.clip(elongation_factor / 50.0, 0.5, 2.0)

    # --- Geometric Signal 3: Axis-Aligned Spread Ratio ---
    pop_range = np.ptp(self.population, axis=0)
    search_range = self.upper_bound - self.lower_bound
    spread_ratios = pop_range / (search_range + 1e-30)
    min_spread = np.min(spread_ratios)
    max_spread = np.max(spread_ratios)
    spread_aspect = max_spread / (min_spread + 1e-30)
    spread_aspect = np.clip(spread_aspect, 1.0, 100.0)

    if spread_aspect > 50.0:
        spread_modulation = 1.8
    elif spread_aspect > 20.0:
        spread_modulation = 1.4
    else:
        spread_modulation = 1.0

    # --- Geometric Signal 4: k-NN Density Gradient ---
    k = min(5, self.np - 1)
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    knn_dists = np.sort(sq_dists, axis=1)[:, :k]
    knn_mean = np.mean(np.sqrt(knn_dists), axis=1) + 1e-10
    global_knn_mean = np.mean(knn_mean)
    density_signal = knn_mean / (global_knn_mean + 1e-10)
    density_signal = np.clip(density_signal, 0.5, 2.0)

    # --- Combine Geometric Signals into Spatial Scale ---
    spatial_scale = expansion_strength * elongation_modulation * spread_modulation
    spatial_scale = spatial_scale * (0.7 + 0.3 * np.mean(density_signal))
    spatial_scale = np.clip(spatial_scale, 0.4, 2.5)

    # --- Apply Position Update ---
    new_population = self.population + spatial_scale * self.velocity
    self.population = self._clip_to_bounds(new_population)
```