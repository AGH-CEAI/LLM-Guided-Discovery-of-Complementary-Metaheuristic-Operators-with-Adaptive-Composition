Looking at the situation:

**Analysis of worst unsolved tasks (17, 16, 6, 5, 11):** Errors stuck at 10^2-10^4 range suggest the swarm is getting trapped in local optima or failing to explore high-fitness regions effectively. The current centroid_knn uses geometric density (distances), which doesn't directly sense the fitness landscape.

**Category D approach:** Replace geometric centroid/density signals with **fitness-weighted centroid** — positions weighted by inverse fitness rank, so particles are pulled toward high-fitness regions purely through rank signals.

**Key differentiator from existing Category D (`temporal_drift`):** `temporal_drift` uses temporal dynamics (Kendall tau, EMA, rank stability). This new variant uses **spatial fitness-centroid attraction** — the direction is computed from population positions weighted by fitness rank, not from temporal history.

**Idea: Fitness-Rank Centroid Attraction**
Uses fitness percentiles and rank-based weighting to create a "gravity toward good solutions" force, with success-history reinforcement.

```python
def _position_update_centroid_knn(self):
    """Fitness-rank centroid attraction + success-history modulation (Category D).
    
    Key insight: Replace geometric centroid with FITNESS CENTROID - the
    weighted average position where weights are inversely proportional to
    fitness rank. Particles are pulled toward high-fitness (low-rank) regions.
    Uses ONLY fitness signals: ranks, percentiles, success-history.
    NO raw distances, NO covariance.
    """
    # Compute fitness ranks (0 = best, np-1 = worst)
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
    
    # Fitness percentile signals for adaptive modulation
    p10 = np.percentile(self.current_fitness, 10)
    p50 = np.percentile(self.current_fitness, 50)
    p90 = np.percentile(self.current_fitness, 90)
    fitness_range = p90 - p10 + 1e-10
    fitness_spread = fitness_range / (np.abs(p50) + 1e-10)
    
    # Success history (Category D signal)
    if not hasattr(self, '_success_count'):
        self._success_count = np.zeros(self.np)
    improved = self.personal_best_fitness >= self.current_fitness
    self._success_count[improved] += 1
    self._success_count[~improved] *= 0.9
    success_norm = self._success_count / (np.max(self._success_count) + 1.0)
    
    # Fitness centroid: position weighted by inverse fitness rank
    # Better (lower) fitness = higher weight = stronger pull toward that region
    inv_rank_weights = 1.0 - fitness_ranks  # Range [0, 1], 1 = best
    weight_sum = np.sum(inv_rank_weights) + 1e-10
    fitness_centroid = np.sum(self.population * inv_rank_weights[:, np.newaxis], axis=0) / weight_sum
    
    # Direction toward fitness centroid (purely positional, weights from fitness)
    to_centroid = fitness_centroid - self.population
    to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
    to_centroid_dir = to_centroid / to_centroid_dist
    
    # Rank-based attraction strength: worse ranks get stronger pull
    rank_attraction = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
    
    # Success modulation: successful particles get slightly stronger pull
    success_modulation = 1.0 + 0.3 * success_norm[:, np.newaxis]
    
    # Convergence detection: reduce pull when spread is small (near optimum)
    if fitness_spread < 0.1:
        convergence_factor = 0.5
    elif fitness_spread > 1.0:
        convergence_factor = 1.5
    else:
        convergence_factor = 1.0
    
    # Combine into final correction
    correction = rank_attraction * to_centroid_dir * success_modulation * convergence_factor
    
    # Velocity scaling based on fitness rank
    vel_scale = 0.7 + 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
    vel_scale = np.clip(vel_scale, 0.3, 2.0)
    
    # Update population
    new_population = self.population + vel_scale * self.velocity + 0.3 * correction
    self.population = self._clip_to_bounds(new_population)
```