**Idea: Centroid Momentum & Convergence Rate Tracker**

Temporal/dynamical approach: Track centroid drift velocity (EMA) across generations and k-NN distance rate-of-change to detect premature convergence. When centroid stalls or population clusters too rapidly, inject momentum-based perturbations. This is fundamentally different from the spatial centroid_knn which only uses single-generation snapshots.

```python
def _position_update_centroid_knn(self):
    """Centroid momentum + convergence rate tracking (Category F - Temporal).
    
    Key insight: Track centroid DRIFT across generations (EMA) and k-NN
    distance RATE OF CHANGE. When centroid stalls or population clusters
    too fast, inject momentum-based exploration. Single-generation spatial
    snapshots miss convergence dynamics that temporal tracking captures.
    """
    # --- TEMPORAL: Centroid drift velocity EMA across generations ---
    current_centroid = np.mean(self.population, axis=0)
    
    if not hasattr(self, '_ema_centroid'):
        self._ema_centroid = current_centroid.copy()
        self._ema_drift_velocity = np.zeros(self.dim)
    
    # Current centroid drift velocity
    current_drift = current_centroid - self._ema_centroid
    
    # Update EMA of drift velocity (momentum = 0.7)
    self._ema_drift_velocity = 0.7 * self._ema_drift_velocity + 0.3 * current_drift
    self._ema_centroid = 0.7 * self._ema_centroid + 0.3 * current_centroid
    
    drift_magnitude = np.linalg.norm(self._ema_drift_velocity) + 1e-10
    
    # --- TEMPORAL: k-NN convergence rate (rate of change) ---
    k = min(5, self.np - 1)
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    knn_dists = np.sort(sq_dists, axis=1)[:, :k]
    current_knn_mean = np.mean(np.sqrt(knn_dists))
    
    if not hasattr(self, '_prev_knn_mean'):
        self._prev_knn_mean = current_knn_mean
    
    # Rate of change: negative = converging (clusters forming)
    knn_rate = current_knn_mean - self._prev_knn_mean
    self._prev_knn_mean = current_knn_mean
    
    # EMA of convergence rate (detect sustained convergence)
    if not hasattr(self, '_ema_knn_rate'):
        self._ema_knn_rate = 0.0
    self._ema_knn_rate = 0.8 * self._ema_knn_rate + 0.2 * knn_rate
    
    # --- Spatial components (for baseline correction) ---
    centroid = current_centroid
    dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)
    knn_avg_dist = np.mean(np.sqrt(knn_dists), axis=1, keepdims=True) + 1e-10
    global_spread = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10
    
    centroid_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))
    to_centroid_dir = centroid - self.population
    to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
    to_centroid_dir = to_centroid_dir / to_centroid_dist
    density_modulation = np.clip(knn_avg_dist / global_spread, 0.2, 2.0)
    
    # --- TEMPORAL: Convergence-driven exploration boost ---
    # If k-NN distance shrinking (ema_knn_rate < 0), population is clustering
    # Boost exploration to counteract premature convergence
    if self._ema_knn_rate < -0.01:
        convergence_pressure = np.clip(-self._ema_knn_rate * 10.0, 0.0, 1.0)
        explore_boost = 1.0 + 0.5 * convergence_pressure
    else:
        explore_boost = 1.0
    
    # --- TEMPORAL: Drift-based directional perturbation ---
    # If centroid drift is small, population may be stagnant → inject drift momentum
    drift_threshold = 0.1 * np.sqrt(self.dim)
    if drift_magnitude < drift_threshold:
        # Stagnation detected: use historical drift direction with boost
        drift_correction = self._ema_drift_velocity / (drift_magnitude + 1e-10)
        temporal_boost = 1.0 + 0.3 * (drift_threshold / (drift_magnitude + 1e-10))
        drift_correction = drift_correction * temporal_boost
    else:
        drift_correction = self._ema_drift_velocity / (drift_magnitude + 1e-10)
    
    # --- Combine spatial correction with temporal signals ---
    spatial_correction = centroid_attraction * to_centroid_dir * density_modulation * explore_boost
    
    # Temporal correction: moves along historical centroid drift direction
    temporal_correction = 0.2 * drift_correction * explore_boost
    
    total_correction = spatial_correction + temporal_correction
    
    new_population = self.population + self.inertia_weight * self.velocity + 0.3 * total_correction
    
    self.population = self._clip_to_bounds(new_population)
```