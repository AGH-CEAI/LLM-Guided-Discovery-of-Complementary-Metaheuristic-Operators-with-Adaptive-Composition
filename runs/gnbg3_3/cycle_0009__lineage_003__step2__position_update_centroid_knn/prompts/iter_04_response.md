**Idea: Fitness-Rank Centroid Gravity**
Category D variant replacing geometric k-NN density with fitness-rank and success-history signals. Uses fitness-distance correlation (FDC) to detect landscape smoothness, fitness percentiles to weight global-best attraction, and per-particle success rates for velocity scaling.
```python
def _position_update_centroid_knn(self):
    """Fitness-rank centroid gravity (Category D variant).
    
    Replaces k-NN geometry with fitness signals: FDC for landscape
    ruggedness, fitness percentiles for directional modulation, and
    success-history for velocity scaling. Targets worst tasks (17,16,6)
    where deceptive landscapes trap geometry-based operators.
    """
    # Initialize success tracking
    if not hasattr(self, '_fdc_success_rate'):
        self._fdc_success_rate = np.zeros(self.np)
    
    # 1. Fitness ranks and percentiles (Category D core signal)
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
    fitness_percentile = 1.0 - fitness_ranks
    
    # 2. Fitness-distance correlation: measure landscape smoothness
    if self.global_best is not None and self.np > 2:
        dists = np.linalg.norm(self.population - self.global_best, axis=1)
        dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
        if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
            fdc = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
        else:
            fdc = 0.0
    else:
        fdc = 0.0
    fdc = np.clip(fdc, -1.0, 1.0)
    
    # 3. Success history per particle (Category D)
    improved = self.personal_best_fitness >= self.current_fitness
    self._fdc_success_rate[improved] += 1
    self._fdc_success_rate[~improved] *= 0.95
    self._fdc_success_rate = np.clip(self._fdc_success_rate, 0.0, 100.0)
    success_norm = self._fdc_success_rate / (np.max(self._fdc_success_rate) + 1.0)
    
    # 4. Centroid attraction modulated by fitness signals
    centroid = np.mean(self.population, axis=0)
    to_centroid = centroid - self.population
    to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
    to_centroid_dir = to_centroid / to_centroid_dist
    
    # Fitness-based centroid modulation: high-fitness particles get more pull
    fitness_centroid_mod = 0.5 + 0.5 * fitness_percentile
    centroid_attraction = 0.3 * fitness_centroid_mod
    
    # 5. Global best attraction weighted by fitness percentile
    if self.global_best is not None:
        to_best = self.global_best - self.population
        to_best_dist = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
        to_best_dir = to_best / to_best_dist
        # Stronger pull for top fitness; weaker for bottom (exploration)
        to_best_weight = fitness_percentile ** 0.5
        to_best_weight = np.clip(to_best_weight, 0.0, 1.0)
        global_best_pull = 0.5 * to_best_weight * to_best_dir
    else:
        global_best_pull = 0.0
    
    # 6. Exploration when landscape is rugged (low FDC) or stagnant
    ruggedness = 1.0 - np.clip(fdc, 0.0, 1.0)
    max_success = np.max(self._fdc_success_rate)
    stagnation = (max_success < 3.0) if max_success > 0 else True
    explore_strength = 0.3 * ruggedness + (0.3 if stagnation else 0.0)
    explore_perturb = explore_strength * np.random.uniform(-1, 1, (self.np, self.dim))
    
    # 7. Combine corrections
    correction = centroid_attraction[:, np.newaxis] * to_centroid_dir + global_best_pull + explore_perturb
    
    # 8. Success-history velocity scaling
    vel_scale = 1.0 + 0.5 * success_norm
    vel_scale = np.clip(vel_scale, 0.3, 2.0)
    
    new_population = self.population + vel_scale[:, np.newaxis] * self.velocity + 0.3 * correction
    self.population = self._clip_to_bounds(new_population)
```