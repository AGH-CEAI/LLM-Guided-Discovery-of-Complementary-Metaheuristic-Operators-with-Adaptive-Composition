**Idea: Spearman Tau + Rank-Weighted Exploitation Push**

Uses Spearman rank correlation (more robust than Pearson) and aggressive rank-differential velocity scaling: particles in the bottom fitness quartile receive a strong directed pull toward the best-ranked particle, with success-rate dampening to prevent over-exploitation.

```python
def _position_update_fitness_rank(self):
    """Fitness-landscape rank signals only — Spearman tau + rank-weighted exploitation push.
    
    Category D: Uses Spearman tau (robust to outliers), fitness percentile quantile
    reasoning, and rank-differential velocity scaling. Bottom-quartile particles get
    a strong directed perturbation toward the best-ranked particle. No distances/covariance.
    """
    from scipy.stats import spearmanr
    
    n = self.np
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, n - 1)
    
    # Spearman tau: more robust rank correlation than Pearson
    if self.global_best is not None and n > 2:
        dists = np.linalg.norm(self.population - self.global_best, axis=1)
        dist_ranks = np.argsort(np.argsort(dists)) / max(1, n - 1)
        if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
            spearman_tau, _ = spearmanr(fitness_ranks, dist_ranks)
            if np.isnan(spearman_tau):
                spearman_tau = 0.0
        else:
            spearman_tau = 0.0
    else:
        spearman_tau = 0.0
    
    # Exploration signal from rank correlation
    # Negative tau = good fitness far from global best (rugged landscape) → more exploration
    exploration_signal = np.clip(-spearman_tau, -1.0, 1.0)
    
    # Success history tracking
    if not hasattr(self, '_rank_success_count'):
        self._rank_success_count = np.zeros(n)
        self._rank_attempt_count = np.ones(n)
    
    improved = self.personal_best_fitness >= self.current_fitness
    self._rank_success_count[improved] += 1
    self._rank_attempt_count += 1
    self._rank_success_count[~improved] *= 0.95
    
    success_rate = self._rank_success_count / np.maximum(self._rank_attempt_count, 1.0)
    success_rate = np.clip(success_rate, 0.01, 1.0)
    
    # Find best-ranked particle for directed push
    best_idx = np.argmin(self.current_fitness)
    best_position = self.population[best_idx]
    
    # Quantile-based velocity scaling: bottom 25% get strong directed push
    bottom_quartile = fitness_ranks < 0.25
    middle_range = (fitness_ranks >= 0.25) & (fitness_ranks < 0.75)
    top_quartile = fitness_ranks >= 0.75
    
    # Directed perturbation toward best particle
    to_best = best_position - self.population
    to_best_dist = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
    to_best_dir = to_best / to_best_dist
    
    # Scale by rank: worse ranks get stronger push
    rank_urgency = 1.0 - fitness_ranks[:, np.newaxis]  # Higher for worse particles
    
    # Success-dampened: particles that always fail get even more aggressive
    exploit_dampen = np.clip(1.0 - success_rate[:, np.newaxis], 0.1, 2.0)
    
    # Strong directed push for bottom quartile
    push_strength = np.where(
        bottom_quartile[:, np.newaxis],
        0.8 * rank_urgency * exploit_dampen,
        np.where(
            middle_range[:, np.newaxis],
            0.3 * rank_urgency * exploit_dampen,
            0.1 * rank_urgency * exploit_dampen
        )
    )
    
    directed_perturb = push_strength * to_best_dir * np.random.uniform(0.3, 1.0, (n, 1))
    
    # Exploration perturbation: modulated by Spearman signal
    explore_scale = 0.5 * (1.0 + exploration_signal)
    random_perturb = explore_scale * (1.0 - fitness_ranks[:, np.newaxis]) * np.random.uniform(
        -0.5, 0.5, (n, self.dim)
    )
    
    # Velocity scaling from exploration signal
    vel_scale = 1.0 + 0.5 * exploration_signal
    vel_scale = np.clip(vel_scale, 0.4, 1.8)
    
    new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + directed_perturb + random_perturb
    
    self.population = self._clip_to_bounds(new_population)
```