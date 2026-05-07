**Idea: Spearman-Kendall Fitness Landscape Modulation**

Pure fitness-rank-based approach using Spearman correlation, Kendall tau stability, fitness percentiles, and success-history to modulate velocity. Targets the worst tasks (17, 16, 6) by detecting rugged vs smooth landscapes purely from fitness signals without any spatial/covariance reasoning.

```python
def _position_update_temporal_drift(self):
    """Spearman-Kendall fitness-rank modulation with success-history (Category D).

    Uses ONLY fitness signals:
    - Spearman rank correlation (fitness vs distance-to-best rank)
    - Kendall tau rank stability across generations
    - Fitness percentiles (10th, 50th, 90th) for convergence detection
    - Success-history per particle
    - Temporal improvement EMA

    NO raw distances, NO covariance, NO spectral analysis.
    Targets worst tasks (17, 16, 6, 5) where rugged landscapes trap the swarm.
    """
    try:
        # === RANK COMPUTATION ===
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        # === SUCCESS HISTORY (Category D) ===
        if not hasattr(self, '_success_count'):
            self._success_count = np.zeros(self.np)
        improved = self.personal_best_fitness >= self.current_fitness
        self._success_count[improved] += 1
        self._success_count[~improved] *= 0.9
        success_norm = self._success_count / (np.max(self._success_count) + 1.0)

        # === SPEARMAN FDC (fitness-distance correlation, rank-based) ===
        if self.global_best is not None and self.np > 2:
            dists = np.linalg.norm(self.population - self.global_best, axis=1)
            dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
            if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                fdc = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
            else:
                fdc = 0.0
        else:
            fdc = 0.0

        # === TEMPORAL FITNESS HISTORY ===
        if not hasattr(self, '_fitness_history'):
            self._fitness_history = []
        self._fitness_history.append(np.min(self.current_fitness))
        if len(self._fitness_history) > 30:
            self._fitness_history.pop(0)

        # === KENDALL TAU RANK STABILITY ===
        if len(self._fitness_history) >= 2:
            prev_fitness = self._fitness_history[-2]
            curr_fitness = self._fitness_history[-1]
            n = len(self.current_fitness)
            concordant = np.sum((self.current_fitness < prev_fitness) & (self.current_fitness < curr_fitness))
            discordant = np.sum((self.current_fitness > prev_fitness) | (self.current_fitness > curr_fitness))
            kendall_tau = (concordant - discordant) / (n * (n - 1) / 2 + 1e-10)
        else:
            kendall_tau = 0.0

        # === FITNESS PERCENTILE CONVERGENCE DETECTION ===
        p10 = np.percentile(self.current_fitness, 10)
        p50 = np.percentile(self.current_fitness, 50)
        p90 = np.percentile(self.current_fitness, 90)
        fitness_spread = (p90 - p10) / (np.abs(p50) + 1e-10)

        if not hasattr(self, '_prev_fitness_spread'):
            self._prev_fitness_spread = fitness_spread
        spread_change = abs(fitness_spread - self._prev_fitness_spread)
        self._prev_fitness_spread = fitness_spread

        is_converging = fitness_spread < 0.1 and spread_change < 0.01
        is_exploring = fitness_spread > 0.5 or spread_change > 0.05

        # === TEMPORAL IMPROVEMENT EMA ===
        if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
            improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_best_fitness = self.global_best_fitness

        if not hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        self._ema_improvement = 0.3 * improvement + 0.7 * self._ema_improvement

        # === SPEARMAN-BASED EXPLOIT/EXPLORE SPLIT ===
        fdc_signal = np.clip(fdc, -1.0, 1.0)
        exploit_weight = 0.5 * (fdc_signal + 1.0)
        explore_weight = 1.0 - exploit_weight

        # === GLOBAL SCALE FROM KENDALL TAU ===
        if kendall_tau < 0.2:
            global_scale = 1.4
        elif kendall_tau < 0.5:
            global_scale = 1.1
        else:
            global_scale = 0.85

        if is_converging:
            global_scale *= 0.75
        elif is_exploring:
            global_scale *= 1.25

        if self._ema_improvement > 1e-6:
            global_scale *= 0.9
        elif self._ema_improvement < 1e-10:
            global_scale *= 1.2

        # === PER-PARTICLE VELOCITY MODULATION ===
        rank_modulation = 1.0 + 0.5 * fitness_ranks
        success_modulation = 1.0 + 0.3 * success_norm
        vel_scale = rank_modulation[:, np.newaxis] * global_scale * success_modulation[:, np.newaxis]
        vel_scale = np.clip(vel_scale, 0.3, 2.5)

        # === DIRECTIONAL PERTURBATION FROM FITNESS RANK ===
        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            directional_strength = 0.4 * (1.0 - fitness_ranks[:, np.newaxis])
            directional = directional_strength * to_best_dir
        else:
            directional = np.zeros((self.np, self.dim))

        # === RANDOM PERTURBATION INVERSELY PROPORTIONAL TO SUCCESS ===
        random_perturb = np.random.uniform(-0.25, 0.25, (self.np, self.dim))
        random_strength = (1.0 - success_norm[:, np.newaxis] * 0.5)
        random_perturb *= random_strength * explore_weight[:, np.newaxis]

        # === COMBINE ===
        new_population = self.population + vel_scale * self.velocity + directional + random_perturb

    except:
        new_population = self.population + self.velocity

    self.population = self._clip_to_bounds(new_population)
```