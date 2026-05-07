**Idea: Convex-Hull Centroid-Drift Modulation**

Geometry-based variant using convex hull volume, centroid-relative distance distribution, k-NN radial density, and axis-aligned bounding box elongation to modulate velocity scaling. Detects convergence via hull shrinkage and radial clustering; detects oscillation via centroid drift autocorrelation. All decisions derive from literal point geometry, not spectral decomposition.
```python
def _position_update_temporal_drift(self):
    """Geometry-modulated velocity scaling (Category A: Geometry / spatial).
    
    Uses literal geometric properties of the swarm layout:
    - Convex hull volume (detects convergence/divergence)
    - Centroid-relative distance distribution (spread signal)
    - k-NN radial density (local clustering)
    - Axis-aligned bounding box elongation (shape anisotropy)
    - Temporal tracking of centroid drift (oscillation detection)
    """
    centroid = np.mean(self.population, axis=0)
    
    # --- GEOMETRIC PROPERTY 1: Convex Hull ---
    # Use hull volume as convergence signal (shrinking hull = converging)
    try:
        from scipy.spatial import ConvexHull
        if self.dim <= 3 and self.np >= self.dim + 2:
            hull = ConvexHull(self.population)
            hull_volume = hull.volume
        else:
            # Fallback for high-dim: use centroid-radius sphere volume proxy
            dists = np.linalg.norm(self.population - centroid, axis=1)
            max_dist = np.max(dists) + 1e-10
            hull_volume = max_dist ** self.dim
    except:
        dists = np.linalg.norm(self.population - centroid, axis=1)
        max_dist = np.max(dists) + 1e-10
        hull_volume = max_dist ** self.dim
    
    # Initialize hull volume history for temporal tracking
    if not hasattr(self, '_hull_volume_history'):
        self._hull_volume_history = []
    self._hull_volume_history.append(hull_volume)
    if len(self._hull_volume_history) > 20:
        self._hull_volume_history.pop(0)
    
    # Hull shrinkage rate (negative = converging, positive = diverging)
    if len(self._hull_volume_history) >= 3:
        recent_vol = np.array(self._hull_volume_history[-3:])
        hull_shrink_rate = (recent_vol[-1] - recent_vol[0]) / (recent_vol[0] + 1e-10)
    else:
        hull_shrink_rate = 0.0
    
    # --- GEOMETRIC PROPERTY 2: Distance to Centroid Distribution ---
    dists_to_centroid = np.linalg.norm(self.population - centroid, axis=1)
    
    # Interquartile range as robust spread measure
    q25, q50, q75 = np.percentile(dists_to_centroid, [25, 50, 75])
    iqr = q75 - q25 + 1e-10
    median_dist = q50 + 1e-10
    
    # Normalized spread: compare IQR to median distance
    spread_normalized = iqr / median_dist
    spread_signal = np.clip(spread_normalized / 10.0, 0.0, 2.0)
    
    # --- GEOMETRIC PROPERTY 3: k-NN Radial Density ---
    k = min(5, self.np - 1)
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    knn_sorted = np.sort(sq_dists, axis=1)
    knn_dists = np.sqrt(knn_sorted[:, :k])
    knn_mean_dist = np.mean(knn_dists, axis=1) + 1e-10  # Local density proxy
    
    # Density ratio: high = sparse region, low = dense region
    global_knn_mean = np.mean(knn_mean_dist) + 1e-10
    density_ratio = knn_mean_dist / global_knn_mean
    
    # --- GEOMETRIC PROPERTY 4: Axis-Aligned Bounding Box Elongation ---
    pop_min = np.min(self.population, axis=0)
    pop_max = np.max(self.population, axis=0)
    spans = pop_max - pop_min + 1e-10
    
    max_span = np.max(spans)
    min_span = np.min(spans)
    elongation = max_span / min_span
    elongation = np.clip(elongation / 10.0, 0.5, 2.0)
    
    # --- COMBINE GEOMETRIC SIGNALS INTO VELOCITY SCALE ---
    # Base scale from spread: low spread (converged) → reduce velocity
    if median_dist < 5.0:
        base_scale = 0.7
    elif median_dist < 20.0:
        base_scale = 1.0
    else:
        base_scale = 1.3
    
    # Modulate by hull shrinkage: converging → reduce further
    hull_factor = 1.0 - np.clip(hull_shrink_rate * 2.0, -0.5, 0.5)
    
    # Modulate by elongation: elongated → anisotropic-like correction
    elongation_factor = 2.0 - elongation  # High elongation → lower factor
    elongation_factor = np.clip(elongation_factor, 0.5, 1.5)
    
    # Combine: geometric_scale = base * hull_factor * elongation_factor
    geometric_scale = base_scale * hull_factor * elongation_factor
    
    # Per-particle adjustment from k-NN density
    # Sparse regions (high density_ratio) → more velocity, dense → less
    density_modulation = np.clip(density_ratio, 0.5, 2.0)
    
    # --- TEMPORAL TRACKING: Centroid Drift for Oscillation Detection ---
    if not hasattr(self, '_ema_centroid'):
        self._ema_centroid = centroid.copy()
        self._ema_centroid_velocity = np.zeros(self.dim)
        self._centroid_vel_history = []
    
    # EMA of centroid position
    alpha_pos = 0.1
    self._ema_centroid = alpha_pos * centroid + (1 - alpha_pos) * self._ema_centroid
    
    # Centroid velocity (rate of centroid movement)
    current_centroid_vel = centroid - self._ema_centroid
    alpha_vel = 0.2
    self._ema_centroid_velocity = alpha_vel * current_centroid_vel + (1 - alpha_vel) * self._ema_centroid_velocity
    
    # Track velocity history for autocorrelation
    self._centroid_vel_history.append(self._ema_centroid_velocity.copy())
    if len(self._centroid_vel_history) > 20:
        self._centroid_vel_history.pop(0)
    
    # Oscillation detection via autocorrelation
    if len(self._centroid_vel_history) >= 5:
        recent = np.array(self._centroid_vel_history[-5:])
        norm_a = np.linalg.norm(recent[:-1]) + 1e-10
        norm_b = np.linalg.norm(recent[1:]) + 1e-10
        autocorr = np.sum(recent[:-1] * recent[1:]) / (norm_a * norm_b)
    else:
        autocorr = 0.0
    
    # Oscillation factor: high positive autocorr = oscillating → reduce velocity
    oscillation_factor = 1.0 + 0.4 * np.clip(autocorr, -1.0, 1.0)
    
    # --- FINAL VELOCITY UPDATE ---
    # Apply geometric scale with temporal oscillation modulation
    final_scale = geometric_scale * oscillation_factor
    final_scale = np.clip(final_scale, 0.5, 2.5)
    
    # Per-particle velocity adjustment from density
    vel_adjustment = density_modulation[:, np.newaxis]
    
    new_population = self.population + final_scale * self.velocity * vel_adjustment
    self.population = self._clip_to_bounds(new_population)
```