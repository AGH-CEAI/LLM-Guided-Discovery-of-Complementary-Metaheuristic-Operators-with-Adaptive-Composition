**Idea: Convex Hull + k-NN Geometric Drift**
Uses literal geometric features: convex hull volume/area, pairwise distance distribution, centroid drift magnitude, axis-aligned spread, and k-NN component analysis. No eigenvalues or entropy.
```python
def _position_update_temporal_drift(self):
    """Geometry-based temporal drift modulation (Category A).

    Uses literal geometric layout: centroid drift magnitude, convex hull
    volume/area, pairwise distance distribution, axis-aligned spread, and
    k-NN connected components. Detects convergence from population geometry
    rather than spectral decomposition.
    """
    # --- Centroid temporal drift ---
    current_centroid = np.mean(self.population, axis=0)
    if not hasattr(self, '_prev_centroid'):
        self._prev_centroid = current_centroid.copy()
        self._centroid_drift_mag = 0.0
        self._centroid_ema = 0.0
    else:
        raw_drift = np.linalg.norm(current_centroid - self._prev_centroid)
        self._centroid_drift_mag = raw_drift
        self._prev_centroid = current_centroid.copy()
        self._centroid_ema = 0.7 * self._centroid_ema + 0.3 * raw_drift

    # --- Pairwise distance statistics ---
    sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    pairwise_dists = np.sqrt(sq_dists)
    valid_dists = pairwise_dists[pairwise_dists < np.inf]
    
    if len(valid_dists) > 0:
        mean_dist = np.mean(valid_dists)
        std_dist = np.std(valid_dists) + 1e-10
        dist_cv = std_dist / mean_dist  # coefficient of variation
    else:
        mean_dist, std_dist, dist_cv = 1.0, 1.0, 1.0

    # --- Axis-aligned spread per dimension ---
    mins = np.min(self.population, axis=0)
    maxs = np.max(self.population, axis=0)
    axis_spreads = maxs - mins + 1e-10
    max_spread = np.max(axis_spreads)
    min_spread = np.min(axis_spreads)
    spread_ratio = min_spread / max_spread  # low = anisotropic collapse

    # --- k-NN connected components (geometric clustering) ---
    k = min(5, self.np - 1)
    knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
    visited = np.zeros(self.np, dtype=bool)
    n_components = 0
    component_sizes = []
    for start in range(self.np):
        if visited[start]:
            continue
        component = []
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            for neighbor in knn_indices[node]:
                if not visited[neighbor]:
                    stack.append(neighbor)
        n_components += 1
        component_sizes.append(len(component))
    
    component_sizes = np.array(component_sizes)
    largest_component_ratio = np.max(component_sizes) / self.np if len(component_sizes) > 0 else 1.0
    fragmentation = 1.0 - largest_component_ratio  # 0 = unified, 1 = fully fragmented

    # --- Convex hull approximation using axis-aligned bounding box ---
    hull_volume = np.prod(axis_spreads)
    hull_volume_log = np.log(hull_volume + 1e-10)
    hull_volume_ema_key = '_hull_volume_ema'
    if not hasattr(self, hull_volume_ema_key):
        setattr(self, hull_volume_ema_key, hull_volume_log)
    else:
        setattr(self, hull_volume_ema_key, 0.9 * getattr(self, hull_volume_ema_key) + 0.1 * hull_volume_log)
    hull_shrink_rate = hull_volume_log - getattr(self, hull_volume_ema_key)

    # --- Geometric convergence signal ---
    # Low centroid drift + high fragmentation + shrinking hull = converged
    drift_signal = np.clip(self._centroid_ema / (mean_dist + 1e-10), 0.0, 2.0)
    spread_signal = np.clip(spread_ratio, 0.0, 1.0)
    component_signal = 1.0 - fragmentation
    hull_signal = np.clip(-hull_shrink_rate / 10.0, 0.0, 1.0)  # shrinking hull = convergence

    # Combined geometric signal: high when population is scattered/exploring, low when converged
    geo_signal = (drift_signal * 0.25 + spread_signal * 0.25 + 
                  component_signal * 0.25 + hull_signal * 0.25)

    # --- Temporal scale from geometric state ---
    if self._centroid_ema > mean_dist * 0.05:
        base_scale = 1.2  # population moving → can explore more
    elif hull_shrink_rate < -0.01:
        base_scale = 0.7  # hull shrinking rapidly → converged, slow down
    else:
        base_scale = 1.0

    temporal_scale = base_scale * (0.8 + 0.4 * geo_signal)

    # --- Per-particle scaling from local density ---
    knn_avg_dist = np.mean(pairwise_dists[np.arange(self.np)[:, None], knn_indices], axis=1, keepdims=True) + 1e-10
    density_factor = np.clip(knn_avg_dist / (mean_dist + 1e-10), 0.5, 2.0)

    # --- Directional correction toward centroid ---
    to_centroid = current_centroid - self.population
    to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
    to_centroid_dir = to_centroid / to_centroid_dist
    centroid_pull = 0.2 * to_centroid_dir * density_factor

    # --- Apply position update ---
    new_population = (self.population + 
                      self.inertia_weight * self.velocity * temporal_scale * density_factor + 
                      centroid_pull)
    
    self.population = self._clip_to_bounds(new_population)
```