**Idea: Adaptive Hybrid Stagnation Detection via Temporal Fitness & Spatial Dispersion**

Combine temporal fitness-improvement-rate tracking (F: rate of change) with spatial dispersion analysis (A: population spread), weighted inversely by their empirical variance to let the more reliable signal dominate. Uses a rolling-window confidence scheme to avoid premature restart decisions.

```python
def _check_stagnation(self, best_fitness):
    """Hybrid stagnation: temporal fitness + spatial dispersion, variance-weighted."""
    # --- Mechanism 1: Temporal fitness improvement rate ---
    if not hasattr(self, '_stagn_history_fitness'):
        self._stagn_history_fitness = []
        self._stagn_ema_fitness = float('inf')
        self._stagn_ema_alpha = 0.1
    
    self._stagn_history_fitness.append(float(best_fitness))
    if len(self._stagn_history_fitness) > 20:
        self._stagn_history_fitness.pop(0)
    
    # EMA of best fitness for trend detection
    if np.isfinite(self._stagn_ema_fitness) and np.isfinite(best_fitness):
        self._stagn_ema_fitness = (1 - self._stagn_ema_alpha) * self._stagn_ema_fitness + self._stagn_ema_alpha * best_fitness
    else:
        self._stagn_ema_fitness = best_fitness
    
    # Compute normalized fitness improvement rate (relative to initial gap)
    fitness_improvement = abs(self._stagn_ema_fitness - float(best_fitness)) / (abs(self._stagn_ema_fitness) + 1e-12)
    
    # --- Mechanism 2: Spatial dispersion (population centroid drift) ---
    if not hasattr(self, '_stagn_history_centroid'):
        self._stagn_history_centroid = []
    
    if hasattr(self, '_current_population') and self._current_population is not None:
        centroid = self._current_population.mean(axis=0)
        self._stagn_history_centroid.append(centroid.copy())
        if len(self._stagn_history_centroid) > 15:
            self._stagn_history_centroid.pop(0)
        
        # Compute centroid displacement over recent window
        if len(self._stagn_history_centroid) >= 3:
            recent = np.array(self._stagn_history_centroid[-3:])
            centroid_drift = np.mean(np.linalg.norm(np.diff(recent, axis=0), axis=1))
        elif len(self._stagn_history_centroid) >= 2:
            centroid_drift = np.linalg.norm(self._stagn_history_centroid[-1] - self._stagn_history_centroid[0])
        else:
            centroid_drift = 0.0
        
        # Normalize by dimension and population spread
        pop_spread = self._compute_diversity(self._current_population) + 1e-12
        spatial_signal = centroid_drift / (pop_spread * np.sqrt(self.dim) + 1e-12)
    else:
        spatial_signal = 1.0
    
    # --- Mechanism 3: Adaptive weighting via inverse-variance ---
    if not hasattr(self, '_stagn_fitness_variance'):
        self._stagn_fitness_variance = 1.0
        self._stagn_spatial_variance = 1.0
        self._stagn_fitness_signal_history = []
        self._stagn_spatial_signal_history = []
    
    self._stagn_fitness_signal_history.append(fitness_improvement)
    self._stagn_spatial_signal_history.append(spatial_signal)
    
    if len(self._stagn_fitness_signal_history) > 10:
        self._stagn_fitness_signal_history.pop(0)
    if len(self._stagn_spatial_signal_history) > 10:
        self._stagn_spatial_signal_history.pop(0)
    
    # Compute variance of each signal
    if len(self._stagn_fitness_signal_history) >= 3:
        self._stagn_fitness_variance = max(np.var(self._stagn_fitness_signal_history), 1e-8)
    if len(self._stagn_spatial_signal_history) >= 3:
        self._stagn_spatial_variance = max(np.var(self._stagn_spatial_signal_history), 1e-8)
    
    # Inverse-variance weighting: consistent signal gets higher weight
    eps = 1e-10
    w_fitness = 1.0 / (self._stagn_fitness_variance + eps)
    w_spatial = 1.0 / (self._stagn_spatial_variance + eps)
    total_w = w_fitness + w_spatial + eps
    w_fitness /= total_w
    w_spatial /= total_w
    
    # --- Hybrid stagnation score ---
    hybrid_score = w_fitness * fitness_improvement + w_spatial * spatial_signal
    
    # --- Adaptive threshold based on generation count and problem scale ---
    if not hasattr(self, '_generation_count'):
        self._generation_count = 0
    self._generation_count += 1
    
    # Early generations: more lenient; later generations: stricter
    gen_factor = min(1.0, self._generation_count / 100.0)
    base_threshold = 1e-4 * gen_factor
    threshold = base_threshold + 1e-6 * (self.dim / 100.0)
    
    # --- Decision with minimum confidence requirement ---
    stagnation_detected = (hybrid_score < threshold) and (len(self._stagn_fitness_signal_history) >= 5)
    
    return stagnation_detected
```