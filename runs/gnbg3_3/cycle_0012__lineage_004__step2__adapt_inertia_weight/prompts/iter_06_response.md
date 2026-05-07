**Idea: Temporal Momentum Inertia**
One-line description: Track exponential moving averages of fitness improvement rate and stagnation depth across generations to dynamically modulate inertia — higher EMA improvement → lower inertia (exploit), stagnation → higher inertia (explore).

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using temporal signals: EMA of improvement and stagnation."""
    # --- Temporal Signal 1: Fitness improvement EMA ---
    if self.global_best is not None and hasattr(self, '_prev_best_fitness'):
        raw_improvement = max(0.0, self._prev_best_fitness - self.global_best_fitness)
    else:
        raw_improvement = 1.0  # Initialise on first call
    self._prev_best_fitness = self.global_best_fitness if self.global_best is not None else np.inf

    # EMA of improvement rate (alpha=0.2 → ~5-gen effective window)
    if not hasattr(self, '_ema_improvement'):
        self._ema_improvement = raw_improvement
    self._ema_improvement = 0.2 * raw_improvement + 0.8 * self._ema_improvement

    # Normalise by current fitness magnitude to be scale-independent
    scale = max(abs(self.global_best_fitness), 1e-10)
    norm_ema_improvement = self._ema_improvement / scale

    # --- Temporal Signal 2: Stagnation depth EMA ---
    if not hasattr(self, '_ema_stagnation'):
        self._ema_stagnation = 0.0
    self._ema_stagnation = 0.15 * self.stagnation_counter + 0.85 * self._ema_stagnation
    norm_stagnation = np.clip(self._ema_stagnation / 100.0, 0.0, 1.0)

    # --- Temporal Signal 3: Diversity drift EMA ---
    current_diversity = self._compute_diversity()
    if not hasattr(self, '_ema_diversity'):
        self._ema_diversity = current_diversity
    self._ema_diversity = 0.1 * current_diversity + 0.9 * self._ema_diversity

    # Diversity ratio (current / EMA) > 1 means recovering, < 1 means collapsing
    diversity_ratio = current_diversity / (self._ema_diversity + 1e-10)
    norm_diversity = np.clip(diversity_ratio - 1.0, -1.0, 1.0)  # centred at 0

    # --- Combine temporal signals into target inertia ---
    # Primary driver: improvement EMA (high improvement → low inertia for exploitation)
    # Secondary driver: stagnation (high stagnation → high inertia for exploration)
    # Tertiary driver: diversity drift (collapsing diversity → higher inertia)

    # Map improvement to inertia: high improvement → low inertia [0.3, 0.7]
    improvement_signal = np.clip(norm_ema_improvement, 0.0, 2.0) / 2.0
    inertia_from_improvement = 0.7 - 0.4 * improvement_signal

    # Stagnation adds exploration pressure
    stagnation_boost = 0.3 * norm_stagnation

    # Diversity correction (collapsing = bad → boost inertia)
    diversity_correction = -0.1 * norm_diversity  # negative ratio → positive correction

    target_inertia = inertia_from_improvement + stagnation_boost + diversity_correction
    target_inertia = np.clip(target_inertia, 0.3, 0.95)

    # Time-based floor prevents complete lock-in on easy tasks
    time_floor = 0.4 - 0.1 * min(self.generation / 500, 1.0)
    target_inertia = max(target_inertia, time_floor)

    # Smooth update (EMA with alpha=0.2)
    self.inertia_weight = 0.8 * self.inertia_weight + 0.2 * target_inertia
    self.inertia_weight = np.clip(self.inertia_weight, 0.3, 0.95)
```