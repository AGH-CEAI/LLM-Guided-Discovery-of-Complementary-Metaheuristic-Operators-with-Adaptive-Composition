**Idea: Fitness-Rank Velocity Modulation**
Rank-based velocity scaling using per-particle success history and fitness percentiles (Category D).
```python
def _position_update_temporal_drift(self):
    """Fitness-rank-modulated velocity scaling (Category D).
    
    Key insight: Use ONLY fitness signals (ranks, success history) to modulate
    velocity. Per-particle success tracking identifies exploit/explore regimes.
    Fitness-distance correlation detects funnel vs deceptive landscapes.
    """
    # Per-particle success history (fitness signal, not distance)
    if not hasattr(self, '_particle_success_count'):
        self._particle_success_count = np.zeros(self.np)
    
    improved = self.personal_best_fitness >= self.current_fitness
    self._particle_success_count[improved] += 1
    self._particle_success_count[~improved] *= 0.9
    
    # Per-particle success rate (0 to 1)
    max_success = np.max(self._particle_success_count) + 1e-10
    success_rate = self._particle_success_count / max_success
    
    # Fitness percentile rank (Category D: purely fitness-based)
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
    
    # Fitness-distance correlation: Spearman-like (Category D)
    if self.global_best is not None and self.np > 2:
        # Use squared distance rank (proxy, but computed from positions)
        # Then correlate with fitness rank
        dists_sq = np.sum((self.population - self.global_best) ** 2, axis=1)
        dist_ranks = np.argsort(np.argsort(dists_sq)) / max(1, self.np - 1)
        
        # Spearman correlation between fitness rank and distance rank
        if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
            fdc_corr = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
        else:
            fdc_corr = 0.0
    else:
        fdc_corr = 0.0
    
    # Track global improvement for regime detection
    if self.global_best is not None and hasattr(self, '_prev_global_fitness'):
        improvement = max(0.0, self._prev_global_fitness - self.global_best_fitness)
    else:
        improvement = 0.0
    self._prev_global_fitness = self.global_best_fitness
    
    if not hasattr(self, '_ema_improvement'):
        self._ema_improvement = 0.0
    self._ema_improvement = 0.3 * improvement + 0.7 * self._ema_improvement
    
    # Regime detection from fitness signals only (Category D)
    if self._ema_improvement > 1e-6:
        regime = 'improving'
    elif self._ema_improvement > 1e-10:
        regime = 'slow'
    else:
        regime = 'stagnant'
    
    # Fitness-distance correlation informs exploration need
    # fdc_corr > 0: funnel landscape (good), fdc_corr < 0: deceptive
    if fdc_corr > 0.3:
        landscape_quality = 1.2  # funnel - exploit more
    elif fdc_corr < -0.1:
        landscape_quality = 0.7  # deceptive - explore more
    else:
        landscape_quality = 1.0
    
    # Per-particle velocity scaling (Category D: fitness-based only)
    # High success + high fitness rank = exploit (lower scale)
    # Low success + low fitness rank = explore (higher scale)
    fitness_exploit = 1.0 - 0.5 * fitness_ranks  # 0.5 to 1.0
    success_explore = 0.8 + 0.4 * (1.0 - success_rate)  # 0.8 to 1.2
    
    vel_scale = fitness_exploit * success_explore
    
    # Regime-dependent global modifier
    if regime == 'improving':
        regime_mod = 1.1 * landscape_quality
    elif regime == 'slow':
        regime_mod = 1.0
    else:  # stagnant
        regime_mod = 0.8 * landscape_quality
    
    vel_scale = vel_scale * regime_mod
    vel_scale = np.clip(vel_scale, 0.3, 2.0)
    
    # Apply velocity with per-particle scaling
    new_population = self.population + self.velocity * vel_scale[:, np.newaxis]
    self.population = self._clip_to_bounds(new_population)
```