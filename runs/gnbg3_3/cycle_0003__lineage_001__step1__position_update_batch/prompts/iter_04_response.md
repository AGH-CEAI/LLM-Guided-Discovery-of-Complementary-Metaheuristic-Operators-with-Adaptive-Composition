**Idea: Rank-Correlation Fitness Modulation**
Modulate position updates using Spearman-like rank correlations between fitness and distance, plus fitness percentiles to guide exploration intensity and success-history weighting.

```python
def _position_update_batch(self):
    """Update positions using fitness-landscape rank signals only (no distances/covariance)."""
    # Compute fitness rank percentiles (0=worst, 1=best)
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
    
    # Compute Spearman-like correlation between fitness rank and distance to global best
    if self.global_best is not None:
        dists = np.linalg.norm(self.population - self.global_best, axis=1)
        dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
        if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
            fdc_corr = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
        else:
            fdc_corr = 0.0
    else:
        fdc_corr = 0.0
    
    # Detect landscape ruggedness: low correlation = rugged = more exploration
    exploration_factor = np.clip(1.0 - fdc_corr, 0.3, 2.0)
    
    # Track success-history: count recent improvements per particle
    if not hasattr(self, '_success_count'):
        self._success_count = np.zeros(self.np)
    improved = self.personal_best_fitness >= self.current_fitness
    self._success_count[improved] += 1
    self._success_count[~improved] *= 0.9  # Decay
    success_norm = self._success_count / (np.max(self._success_count) + 1.0)
    
    # Scale velocity by exploration factor and success history
    vel_scale = exploration_factor * (1.0 + 0.3 * success_norm)
    vel_scale = np.clip(vel_scale, 0.5, 2.0)
    
    # Fitness-percentile perturbation: push lower-ranked particles more
    # High percentile (close to 1) = good fitness = small perturbation
    # Low percentile (close to 0) = bad fitness = large perturbation
    fitness_perturb = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
    exploration_perturb = exploration_factor * fitness_perturb
    random_perturb = exploration_perturb * np.random.uniform(-0.5, 0.5, (self.np, self.dim))
    
    # Apply rank-based velocity modulation
    new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + random_perturb
    
    self.population = self._clip_to_bounds(new_population)
```