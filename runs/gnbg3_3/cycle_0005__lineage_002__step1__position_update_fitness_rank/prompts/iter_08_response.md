**Idea: FDC-Temporal Drift Hybrid**
Combine fitness-distance correlation (FDC) exploration/exploitation with temporal centroid-drift convergence detection for principled switching. When stagnation is detected, temporal drift dominates to escape basins; otherwise FDC guides fine-grained navigation.

```python
def _position_update_fitness_rank(self):
    """Hybrid: FDC exploration/exploitation + temporal centroid-drift for convergence sensing.
    
    Two distinct mechanisms with data-driven switching:
      1. FDC-based: fitness-distance correlation modulates exploration (from variant_04)
      2. Temporal-drift: EMA centroid dynamics detect convergence/oscillation (from variant_06)
    
    Switching signal: stagnation_counter drives drift_weight — when stuck, temporal
    dynamics take over to escape basins. Principled because stagnation directly
    indicates the FDC mechanism has failed to make progress.
    """
    # --- MECHANISM 1: FDC-based exploration/exploitation ---
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

    if self.global_best is not None:
        dists = np.linalg.norm(self.population - self.global_best, axis=1)
        dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
        if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
            fdc_corr = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
        else:
            fdc_corr = 0.0
    else:
        fdc_corr = 0.0

    exploration_factor = np.clip(1.0 - fdc_corr, 0.3, 2.0)

    # --- MECHANISM 2: Temporal centroid-drift ---
    centroid = np.mean(self.population, axis=0)

    if not hasattr(self, '_ema_centroid'):
        self._ema_centroid = centroid.copy()
        self._ema_centroid_velocity = np.zeros(self.dim)

    alpha_pos = 0.1
    self._ema_centroid = alpha_pos * centroid + (1 - alpha_pos) * self._ema_centroid

    current_centroid_vel = centroid - self._ema_centroid
    alpha_vel = 0.2
    self._ema_centroid_velocity = alpha_vel * current_centroid_vel + (1 - alpha_vel) * self._ema_centroid_velocity

    # --- PRINCIPLED SWITCHING: stagnation drives drift_weight ---
    stagnation_threshold = 20
    stagnation_normalized = np.clip(self.stagnation_counter / max(stagnation_threshold, 1), 0.0, 1.0)
    drift_weight = stagnation_normalized ** 0.5  # concave: gradual rise, then accelerate
    fdc_weight = 1.0 - drift_weight

    # --- FDC-based velocity scaling ---
    if not hasattr(self, '_success_count'):
        self._success_count = np.zeros(self.np)
    improved = self.personal_best_fitness >= self.current_fitness
    self._success_count[improved] += 1
    self._success_count[~improved] *= 0.9
    success_norm = self._success_count / (np.max(self._success_count) + 1.0)

    vel_scale = exploration_factor * (1.0 + 0.3 * success_norm)
    vel_scale = np.clip(vel_scale, 0.5, 2.0)

    # --- TEMPORAL-DRIFT correction ---
    to_ema_dir = self._ema_centroid - self.population
    to_ema_dist = np.linalg.norm(to_ema_dir, axis=1, keepdims=True) + 1e-10
    drift_correction = (to_ema_dir / to_ema_dist) * np.linalg.norm(self._ema_centroid_velocity)

    # --- COMBINE MECHANISMS ---
    drift_correction_scaled = drift_correction * drift_weight

    # FDC-based random perturbation
    fitness_perturb = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
    exploration_perturb = exploration_factor * fitness_perturb
    random_perturb = exploration_perturb * np.random.uniform(-0.5, 0.5, (self.np, self.dim))
    fdc_perturb = random_perturb * fdc_weight

    # Combined position update
    new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + drift_correction_scaled + fdc_perturb

    self.population = self._clip_to_bounds(new_population)
```