**Idea: Kendall Tau Rank-Stability Modulation with Fitness Percentile Adaptive Exploration**

Category D pure fitness-rank approach: Uses Kendall tau rank correlation between consecutive generations to detect stagnation in rank ordering (not raw fitness values), combined with fitness percentile convergence signals. When rank stability is high (particles stuck in same relative positions), this triggers aggressive rank-based exploration independent of fitness magnitude.

```python
def _position_update_fitness_rank(self):
    """Fitness-percentile & Kendall-tau rank stability modulation (Category D).

    Uses ONLY fitness signals — no distances, no graph structures, no covariance:
    - Fitness percentile ranks per particle (scale-invariant)
    - Kendall tau rank correlation between consecutive generations (stagnation detection)
    - Fitness spread ratio (p90-p10)/p50 for convergence detection
    - Success-history EMA per particle
    - Rank-based directional perturbation toward best fitness region
    """
    try:
        # === FITNESS RANK COMPUTATION ===
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
        
        # === FITNESS PERCENTILE SIGNALS ===
        p10 = np.percentile(self.current_fitness, 10)
        p50 = np.percentile(self.current_fitness, 50)
        p90 = np.percentile(self.current_fitness, 90)
        fitness_spread = (p90 - p10) / (np.abs(p50) + 1e-10)
        
        # === KENDALL TAU RANK STABILITY (temporal rank correlation) ===
        if hasattr(self, '_prev_fitness') and len(self._prev_fitness) == len(self.current_fitness):
            # Concordant pairs: both ranks improved or both worsened
            f_improved = self.current_fitness < self._prev_fitness
            # Count discordant swaps in rank order
            n = len(self.current_fitness)
            concordant = 0
            discordant = 0
            for i in range(n):
                for j in range(i + 1, n):
                    curr_order = self.current_fitness[i] < self.current_fitness[j]
                    prev_order = self._prev_fitness[i] < self._prev_fitness[j]
                    if curr_order == prev_order:
                        concordant += 1
                    else:
                        discordant += 1
            n_pairs = n * (n - 1) / 2
            kendall_tau = (concordant - discordant) / (n_pairs + 1e-10)
        else:
            kendall_tau = 0.5  # Neutral on first evaluation
        self._prev_fitness = self.current_fitness.copy()
        
        # === SUCCESS HISTORY PER PARTICLE ===
        if not hasattr(self, '_success_count'):
            self._success_count = np.zeros(self.np)
        improved = self.personal_best_fitness >= self.current_fitness
        self._success_count[improved] += 1
        self._success_count[~improved] *= 0.85
        success_norm = self._success_count / (np.max(self._success_count) + 1.0)
        
        # === CONVERGENCE PHASE DETECTION ===
        is_converged = fitness_spread < 0.05
        is_stagnant = kendall_tau > 0.75  # High tau = ranks barely changing = trapped
        
        # === GLOBAL EXPLORATION SCALE FROM RANK SIGNALS ===
        if is_stagnant and is_converged:
            global_scale = 1.8  # Both stuck and converged = deep trap
        elif is_stagnant:
            global_scale = 1.4  # Stuck but spread = exploring badly
        elif is_converged:
            global_scale = 0.7  # Converging well = exploit
        else:
            global_scale = 1.0  # Normal operation
        
        # === PER-PARTICLE VELOCITY MODULATION FROM FITNESS RANK ===
        # Higher rank (worse fitness) = more exploration boost
        rank_explore = 1.0 + 0.6 * fitness_ranks
        # Particles with low success = trapped = extra exploration
        success_explore = 1.0 + 0.4 * (1.0 - success_norm)
        # Combine: rank signal drives direction, success signal drives magnitude
        vel_scale = rank_explore[:, np.newaxis] * global_scale * success_explore[:, np.newaxis]
        vel_scale = np.clip(vel_scale, 0.2, 3.0)
        
        # === RANK-BASED DIRECTIONAL PERTURBATION ===
        # Compute rank-based best direction (not spatial)
        best_rank_idx = np.argmin(fitness_ranks)
        # Attraction strength inversely proportional to rank (best particles attracted less)
        attract_strength = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
        
        # Random rank-based direction using random projection (no spatial info)
        random_dir = np.random.randn(self.np, self.dim)
        random_dir = random_dir / (np.linalg.norm(random_dir, axis=1, keepdims=True) + 1e-10)
        
        # Blend: attract toward best region + random exploration
        if self.global_best is not None and kendall_tau < 0.5:
            # Only attract when not too stagnant
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            directional = attract_strength * to_best_dir
        else:
            # Stagnant: pure random exploration in rank space
            directional = attract_strength * random_dir
        
        # === RANDOM PERTURBATION SCALED BY STAGNATION ===
        random_strength = 0.3 * (1.0 + kendall_tau)  # More random when stagnant
        random_perturb = random_strength * np.random.uniform(-1.0, 1.0, (self.np, self.dim))
        
        # === COMBINE UPDATES ===
        new_population = self.population + vel_scale * self.velocity + directional + random_perturb
        
    except:
        new_population = self.population + self.velocity
    
    self.population = self._clip_to_bounds(new_population)
```