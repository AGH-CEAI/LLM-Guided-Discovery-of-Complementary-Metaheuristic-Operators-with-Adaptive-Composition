**Idea: Temporal Stagnation-Drift Modulation**

Track centroid movement across generations using EMA of centroid velocity and k-NN density to detect stagnation and oscillation. Modulate correction strength dynamically — increase pull when stagnant, damp when oscillating, adapt to convergence signals. This temporal approach targets tasks 5, 6, 11, 16, 17 where the swarm gets trapped in poor local optima.

```python
def _position_update_centroid_knn(self):
    """Temporal stagnation-drift modulation (Category F).
    
    Tracks centroid movement ACROSS generations using EMA of velocity
    and k-NN density to detect stagnation/oscillation. Modulates
    correction strength dynamically rather than using static values.
    """
    centroid = np.mean(self.population, axis=0)

    dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)

    # --- TEMPORAL: Centroid velocity EMA (across generations) ---
    if not hasattr(self, '_ema_centroid'):
        self._ema_centroid = centroid.copy()
        self._centroid_vel_ema = 0.0
        self._prev_accel_sign = np.zeros(self.dim)
        self._knn_density_ema = 1.0
        self._oscillation_count = 0

    # Centroid velocity (instantaneous)
    centroid_vel = np.linalg.norm(centroid - self._ema_centroid)
    # EMA of centroid velocity
    self._centroid_vel_ema = 0.7 * self._centroid_vel_ema + 0.3 * centroid_vel
    self._ema_centroid = 0.7 * self._ema_centroid + 0.3 * centroid

    # Acceleration from velocity change
    acceleration = centroid_vel - self._centroid_vel_ema

    # --- TEMPORAL: k-NN density EMA ---
    k = min(5, self.np - 1)
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    knn_dists = np.sort(sq_dists, axis=1)[:, :k]
    knn_avg_dist = np.mean(np.sqrt(knn_dists), axis=1, keepdims=True) + 1e-10
    knn_density = np.mean(knn_avg_dist)

    # EMA of k-NN density
    self._knn_density_ema = 0.7 * self._knn_density_ema + 0.3 * knn_density

    # Rate of change of density (convergence detection)
    density_change = knn_density - self._knn_density_ema + 1e-10
    density_rate = density_change / (self._knn_density_ema + 1e-10)

    # --- TEMPORAL: Stagnation detection via centroid velocity EMA ---
    stagnation_signal = np.clip(1.0 - self._centroid_vel_ema / (1.0 + self._centroid_vel_ema), 0.0, 1.0)

    # --- TEMPORAL: Oscillation detection via acceleration sign changes ---
    curr_accel_sign = np.sign(acceleration) if centroid_vel > 1e-10 else np.zeros(self.dim)
    oscillation = np.mean(curr_accel_sign != self._prev_accel_sign) / max(1, self.dim)
    self._prev_accel_sign = curr_accel_sign

    # --- TEMPORAL: Convergence/divergence signals from density ---
    is_converging = knn_density < 0.8 * self._knn_density_ema
    is_diverging = knn_density > 1.2 * self._knn_density_ema

    # --- TEMPORAL: Adaptive correction strength ---
    base_correction = 0.3
    stagnation_boost = 0.2 * stagnation_signal
    oscillation_damping = 0.5 if oscillation > 0.3 else 1.0
    convergence_modulation = 0.7 if is_converging else (1.3 if is_diverging else 1.0)
    correction_strength = base_correction * (1.0 + stagnation_boost) * oscillation_damping * convergence_modulation

    global_spread = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10

    centroid_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))

    to_centroid_dir = centroid - self.population
    to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
    to_centroid_dir = to_centroid_dir / to_centroid_dist

    # Density modulation enhanced with temporal rate
    density_modulation = np.clip(knn_avg_dist / global_spread, 0.2, 2.0)
    density_modulation = density_modulation * (1.0 + 0.2 * density_rate)

    correction = centroid_attraction * to_centroid_dir * density_modulation

    new_population = self.population + self.inertia_weight * self.velocity + correction_strength * correction

    self.population = self._clip_to_bounds(new_population)
```