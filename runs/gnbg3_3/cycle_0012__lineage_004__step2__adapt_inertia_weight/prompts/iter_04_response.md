Looking at the priority targets (Tasks 17, 16, 6 with errors ~1e+02 to ~1e+04), the algorithm is failing to escape local optima. The current inertia adaptation uses spectral/covariance signals (Category B), which misses the actual fitness landscape structure.

**Category D Analysis**: The worst tasks likely have deceptive fitness-distance correlations (fitness improving but particles far from global optimum) or rugged landscapes where rank improvements stall. I need to use fitness percentiles, rank correlations, and success-history — not geometric/spatial signals.

**Idea: Spearman FDC + Success-Rate Inertia**
Compute Spearman correlation between fitness rank and distance-to-global-best (fitness-distance correlation, FDC) plus per-particle success rates. High FDC = smooth landscape → low inertia exploit. Low/negative FDC = deceptive/rugged → high inertia explore. Success-rate EMA detects stagnation phases.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using fitness-distance correlation and success-history (Category D)."""
    try:
        # === MECHANISM 1: Spearman Fitness-Distance Correlation (FDC) ===
        if self.global_best is not None and self.np > 3:
            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
            dists = np.linalg.norm(self.population - self.global_best, axis=1)
            dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
            
            if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                # Spearman correlation: +1 = fitness improves with proximity (smooth)
                # -1 = deceptive/rugged (fitness and distance uncorrelated)
                corr_matrix = np.corrcoef(fitness_ranks, dist_ranks)
                spearman_fdc = np.clip(corr_matrix[0, 1], -1.0, 1.0)
            else:
                spearman_fdc = 0.0
        else:
            spearman_fdc = 0.0
        
        # === MECHANISM 2: Fitness Percentile Spread ===
        # Wide spread = diverse fitness landscape, need exploration
        fitness_sorted = np.sort(self.current_fitness)
        p90 = fitness_sorted[int(0.9 * self.np)] if self.np > 0 else 0.0
        p10 = fitness_sorted[int(0.1 * self.np)] if self.np > 0 else 0.0
        percentile_spread = (p90 - p10) / (np.abs(self.global_best_fitness) + 1e-10 + p90 - p10 + 1e-10)
        percentile_spread = np.clip(percentile_spread, 0.0, 1.0)
        
        # === MECHANISM 3: Success History (EMA of per-particle improvement) ===
        if not hasattr(self, '_success_ema'):
            self._success_ema = 0.0
        if not hasattr(self, '_prev_fitness_median'):
            self._prev_fitness_median = np.median(self.current_fitness)
        
        median_fitness = np.median(self.current_fitness)
        median_improvement = max(0.0, self._prev_fitness_median - median_fitness)
        self._prev_fitness_median = median_fitness
        
        # EMA of improvement rate
        self._success_ema = 0.2 * median_improvement + 0.8 * self._success_ema
        success_signal = np.clip(self._success_ema / (np.abs(self.global_best_fitness) + 1.0), 0.0, 1.0)
        
        # === COMBINE SIGNALS ===
        # FDC signal: +1 → smooth → low inertia (exploit)
        #             -1 → deceptive → high inertia (explore)
        fdc_exploit_signal = (spearman_fdc + 1.0) / 2.0  # Map [-1,1] to [0,1]
        
        # Landscape roughness = 1 - FDC_exploit
        landscape_roughness = 1.0 - fdc_exploit_signal
        
        # Final exploration signal
        exploration_signal = (
            0.5 * landscape_roughness +
            0.3 * percentile_spread +
            0.2 * (1.0 - success_signal)
        )
        exploration_signal = np.clip(exploration_signal, 0.0, 1.0)
        
        # Map to inertia range [0.4, 0.95]
        target_inertia = 0.4 + 0.55 * exploration_signal
        
        # Blend with time-based baseline
        decay = 0.729 - 0.15 * (self.generation / 1000)
        target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)
        
        # Smooth EMA update
        self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
        
    except Exception:
        decay = 0.729 - 0.15 * (self.generation / 1000)
        self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```