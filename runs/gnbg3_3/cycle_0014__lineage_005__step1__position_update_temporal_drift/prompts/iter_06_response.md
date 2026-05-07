**Idea: Autocorrelation-Stagnation Momentum (ASMo)**

One-line description: Pure Category F approach using fitness improvement autocorrelation, EMA-based centroid momentum tracking, windowed stagnation detection, and acceleration/deceleration signals to detect when the population is stuck in correlated drift vs. actively exploring.

```python
def _position_update_temporal_drift(self):
    """Pure temporal/dynamical: Autocorrelation-stagnation momentum (Category F).

    Tracks quantities ACROSS generations — not single-generation snapshots:
      1. Fitness improvement autocorrelation (lag-1): detects correlated drift vs. exploration
      2. EMA-based centroid momentum: tracks population movement direction and magnitude
      3. Windowed stagnation detection: how long since last meaningful improvement
      4. Acceleration trend of momentum: detecting convergence phases
      5. Diversity EMA trend: detecting premature collapse

    Combined "stuckness" signal drives adaptive velocity scaling and directional kicks.
    Targets worst tasks (17, 16, 6, 5) where temporal signals reveal premature convergence.
    """
    try:
        # === MECHANISM 1: FITNESS IMPROVEMENT AUTOCORRELATION ===
        if not hasattr(self, '_fitness_history'):
            self._fitness_history = []
        self._fitness_history.append(float(np.min(self.current_fitness)))
        if len(self._fitness_history) > 30:
            self._fitness_history.pop(0)

        if len(self._fitness_history) >= 10:
            improvements = np.diff(self._fitness_history)
            mean_imp = np.mean(improvements)
            var_imp = np.var(improvements) + 1e-10
            autocorr_lag1 = np.sum(improvements[:-1] * improvements[1:]) / ((len(improvements) - 1) * var_imp)
            autocorr_lag1 = np.clip(autocorr_lag1, -1.0, 1.0)
        else:
            autocorr_lag1 = 0.0

        # === MECHANISM 2: EMA-BASED CENTROID MOMENTUM ===
        current_centroid = np.mean(self.population, axis=0)
        if not hasattr(self, '_ema_centroid'):
            self._ema_centroid = current_centroid.copy()
        self._ema_centroid = 0.7 * self._ema_centroid + 0.3 * current_centroid
        centroid_vel = current_centroid - self._ema_centroid
        centroid_vel_mag = np.linalg.norm(centroid_vel)

        if not hasattr(self, '_centroid_vel_ema_mag'):
            self._centroid_vel_ema_mag = 0.0
        self._centroid_vel_ema_mag = 0.7 * self._centroid_vel_ema_mag + 0.3 * centroid_vel_mag

        # === MECHANISM 3: WINDOWED STAGNATION DETECTION ===
        if not hasattr(self, '_stagnation_counter_temporal'):
            self._stagnation_counter_temporal = 0
        if self.global_best is not None and hasattr(self, '_prev_global_best'):
            if abs(self._prev_global_best - self.global_best_fitness) < 1e-10:
                self._stagnation_counter_temporal += 1
            else:
                self._stagnation_counter_temporal = 0
        self._prev_global_best = self.global_best_fitness if self.global_best is not None else 0.0

        stagnation_depth = min(self._stagnation_counter_temporal / 50.0, 1.0)

        # === MECHANISM 4: ACCELERATION TREND OF MOMENTUM ===
        if not hasattr(self, '_accel_history'):
            self._accel_history = []
        self._accel_history.append(centroid_vel_mag)
        if len(self._accel_history) > 30:
            self._accel_history.pop(0)

        if len(self._accel_history) >= 5:
            accel_array = np.array(self._accel_history)
            accel_trend = np.mean(np.diff(accel_array))
        else:
            accel_trend = 0.0
        accel_signal = -np.sign(accel_trend) * min(abs(accel_trend), 1.0)

        # === MECHANISM 5: DIVERSITY EMA TREND ===
        current_diversity = self._compute_diversity()
        if not hasattr(self, '_ema_diversity'):
            self._ema_diversity = current_diversity
        self._ema_diversity = 0.95 * self._ema_diversity + 0.05 * current_diversity
        diversity_ratio = current_diversity / (self._ema_diversity + 1e-10)
        diversity_signal = 1.0 - np.clip(diversity_ratio, 0.0, 2.0)

        # === COMBINED TEMPORAL "STUCKNESS" SIGNAL ===
        stuckness = (
            0.35 * max(0.0, autocorr_lag1) +
            0.25 * stagnation_depth +
            0.20 * max(0.0, accel_signal) +
            0.20 * max(0.0, diversity_signal)
        )
        stuckness = np.clip(stuckness, 0.0, 1.0)

        # === ADAPTIVE VELOCITY SCALING ===
        if stuckness > 0.7:
            base_scale = 1.6
        elif stuckness > 0.4:
            base_scale = 1.2
        else:
            base_scale = 1.0

        # Diversity-aware boost
        if diversity_ratio < 0.8:
            diversity_boost = 1.4
        elif diversity_ratio < 1.0:
            diversity_boost = 1.2
        else:
            diversity_boost = 1.0

        # === DIRECTIONAL KICK FOR STUCK POPULATION ===
        if stuckness > 0.5:
            if centroid_vel_mag > 1e-6:
                vel_dir = centroid_vel / centroid_vel_mag
            else:
                vel_dir = np.zeros(self.dim)
            kick_dir = np.random.uniform(-1, 1, self.dim)
            if centroid_vel_mag > 1e-6:
                kick_dir -= vel_dir * np.dot(kick_dir, vel_dir)
            kick_mag = stuckness * 2.0
            directional_kick = kick_dir * kick_mag
        else:
            directional_kick = np.zeros(self.dim)

        # === FINAL UPDATE ===
        new_population = self.population + base_scale * self.velocity * diversity_boost + directional_kick

    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity

    self.population = self._clip_to_bounds(new_population)
```