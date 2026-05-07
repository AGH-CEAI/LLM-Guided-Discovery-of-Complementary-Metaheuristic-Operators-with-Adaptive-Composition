Looking at this problem, I need to replace the SVD-based whitening (Category B) with a pure fitness-landscape/rank-based approach (Category D). The key insight is that the worst unsolved tasks (17, 16, 6, 11, 19, 9) have massive errors, suggesting the algorithm is failing to properly exploit fitness landscape structure. 

My approach: Use **Spearman fitness-distance correlation** to detect landscape smoothness, combined with **per-particle fitness percentiles** and **success history** to modulate velocity scaling. When FDC is high (smooth landscape), exploit aggressively with top performers; when FDC is low (rugged/deceptive), boost exploration for poorly-ranked particles.

**Idea: FDC-Guided Fitness Percentile Modulation**
Uses Spearman correlation between fitness rank and global-best distance to detect landscape type, then applies percentile-stratified velocity scaling with success-history weighting — NO raw distances or covariance matrices.

```python
def _position_update_svd_whitening(self):
    """Fitness-landscape rank-based velocity modulation (Category D).
    
    Uses Spearman fitness-distance correlation to detect landscape smoothness,
    fitness percentiles for per-particle scaling, and success history for
    momentum. Target: tasks 17, 16, 6, 11, 19, 9 (worst unsolved).
    """
    # Compute fitness ranks (0=best, 1=worst)
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
    
    # Spearman-like fitness-distance correlation via Kendall tau proxy
    if self.global_best is not None and self.np > 3:
        # Distance to global best (only for ranking, not magnitude)
        dists = np.linalg.norm(self.population - self.global_best, axis=1)
        dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
        
        # Kendall tau: count concordant vs discordant pairs
        n = self.np
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                fit_diff = (fitness_ranks[i] - fitness_ranks[j]) * (dist_ranks[i] - dist_ranks[j])
                if fit_diff > 0:
                    concordant += 1
                elif fit_diff < 0:
                    discordant += 1
        total_pairs = n * (n - 1) / 2
        kendall_tau = (concordant - discordant) / (total_pairs + 1e-10)
        fdc_signal = np.clip(kendall_tau, -1.0, 1.0)
    else:
        fdc_signal = 0.0
    
    # Fitness-distance correlation guides exploration/exploitation balance
    # High positive FDC = smooth landscape → exploit (lower scaling for top performers)
    # Low/negative FDC = rugged/deceptive → explore (boost poorly-ranked)
    exploit_weight = 0.5 + 0.5 * fdc_signal  # [0, 1]
    
    # Success history tracking
    if not hasattr(self, '_fitness_success_count'):
        self._fitness_success_count = np.zeros(self.np)
    
    improved = self.personal_best_fitness > self.current_fitness
    self._fitness_success_count[improved] += 1
    self._fitness_success_count[~improved] *= 0.9
    success_norm = self._fitness_success_count / (np.max(self._fitness_success_count) + 1.0)
    
    # Per-particle velocity scale based on fitness percentile
    # Top performers (low rank): moderate scale to refine
    # Middle: higher scale for active search
    # Bottom (high rank): highest scale to escape local optima
    vel_scale = np.ones(self.np)
    
    top_mask = fitness_ranks < 0.25
    mid_mask = (fitness_ranks >= 0.25) & (fitness_ranks < 0.75)
    bot_mask = fitness_ranks >= 0.75
    
    # Base scales differ by FDC-driven exploit_weight
    vel_scale[top_mask] = 0.6 + 0.4 * exploit_weight
    vel_scale[mid_mask] = 1.0 + 0.3 * (1.0 - exploit_weight)
    vel_scale[bot_mask] = 1.5 + 0.5 * (1.0 - exploit_weight)
    
    # Success history boosts: particles with recent wins get momentum
    vel_scale = vel_scale * (1.0 + 0.4 * success_norm)
    vel_scale = np.clip(vel_scale, 0.3, 2.5)
    
    # Fitness-guided directional perturbation (no raw distances)
    # Poor fitness → stronger pull toward better regions
    pull_strength = 0.3 * (fitness_ranks ** 1.5)  # Non-linear: worse particles get much stronger pull
    if self.global_best is not None:
        to_best = self.global_best - self.population
        to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
        to_best_dir = to_best / to_best_norm
        directional = pull_strength[:, np.newaxis] * to_best_dir
    else:
        # Fallback: random perturbation weighted by fitness rank
        directional = pull_strength[:, np.newaxis] * np.random.uniform(-1, 1, (self.np, self.dim))
    
    # Random perturbation scaled by fitness uncertainty (high rank = more noise)
    random_perturb = (0.2 + 0.4 * fitness_ranks[:, np.newaxis]) * np.random.uniform(-0.5, 0.5, (self.np, self.dim))
    
    # Combine: velocity scaling + directional pull + noise
    new_population = self.population + vel_scale[:, np.newaxis] * self.velocity + directional + random_perturb
    self.population = self._clip_to_bounds(new_population)
```