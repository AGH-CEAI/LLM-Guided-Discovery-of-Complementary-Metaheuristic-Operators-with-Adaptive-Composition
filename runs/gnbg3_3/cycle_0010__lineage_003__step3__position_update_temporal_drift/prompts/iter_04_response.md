**Idea: Fitness-Rank Gradient with FDC-Driven Exploration**

Rank-based velocity modulation using fitness percentiles, success-history, and fitness-distance correlation (FDC) as a landscape ruggedness signal. Worsest particles receive stronger exploratory velocity and directed gradient pushes toward top-performing regions. Targets tasks 17/16/6/11 with large deceptive basins by detecting ruggedness via FDC and boosting exploration accordingly.

```python
def _position_update_temporal_drift(self):
    """Fitness-landscape rank-based velocity modulation (Category D).

    Uses fitness ranks, success-history, and fitness-distance correlation
    to modulate velocity. No raw distances or covariance - only fitness signals.
    Targets worst tasks (17, 16, 6, 11) with large deceptive basins where
    population may collapse to local optima.
    """
    # 1. Fitness ranks (0 = best, 1 = worst)
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

    # 2. Success history with exponential decay
    if not hasattr(self, '_success_count'):
        self._success_count = np.zeros(self.np)
    improved = self.personal_best_fitness > self.current_fitness
    self._success_count[improved] += 1
    self._success_count[~improved] *= 0.9
    success_norm = self._success_count / (np.max(self._success_count) + 1.0)

    # 3. Fitness-percentile velocity scaling
    # Worse particles (higher rank) get more exploration velocity
    vel_scale = 0.5 + 1.5 * fitness_ranks
    vel_scale = np.clip(vel_scale, 0.5, 2.0)

    # 4. Fitness-distance correlation (FDC) as landscape ruggedness signal
    # Compute rank-based correlation between fitness and distance to global best
    if self.global_best is not None and self.np > 2:
        dists = np.linalg.norm(self.population - self.global_best, axis=1)
        dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
        if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
            fdc = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
        else:
            fdc = 0.0
    else:
        fdc = 0.0

    # 5. FDC-driven exploration factor
    # High FDC (>0) = smooth landscape = more exploitation
    # Low/negative FDC = rugged/deceptive = more exploration
    fdc_explore_factor = np.clip(2.0 - fdc, 0.5, 2.5)

    # 6. Success-based boost for recently improved particles
    success_boost = 1.0 + 0.5 * success_norm
    success_boost = np.clip(success_boost, 1.0, 2.0)

    # 7. Combined scale
    combined_scale = vel_scale * fdc_explore_factor * success_boost
    combined_scale = np.clip(combined_scale, 0.3, 3.0)

    # 8. Compute fitness gradient direction
    # Direction toward centroid of top 20% fittest particles
    top_k = max(1, self.np // 5)
    top_indices = np.argsort(self.current_fitness)[:top_k]
    top_centroid = np.mean(self.population[top_indices], axis=0)

    to_top = top_centroid - self.population
    to_top_dist = np.linalg.norm(to_top, axis=1, keepdims=True) + 1e-10
    to_top_dir = to_top / to_top_dist

    # Gradient strength proportional to fitness rank (worse particles get stronger pull)
    gradient_strength = 0.5 * fitness_ranks[:, np.newaxis]

    # 9. Position update
    scaled_velocity = self.velocity * combined_scale[:, np.newaxis]
    gradient_push = gradient_strength * to_top_dir
    random_perturb = fdc_explore_factor * np.random.uniform(-0.5, 0.5, (self.np, self.dim))

    new_population = self.population + scaled_velocity + gradient_push + random_perturb
    self.population = self._clip_to_bounds(new_population)
```