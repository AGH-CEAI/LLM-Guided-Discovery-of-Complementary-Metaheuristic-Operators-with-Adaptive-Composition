**Idea: EMA-Driven Stagnation-Responsive Inertia**

Adapt inertia weight using exponential moving averages of global-best improvement rate and stagnation detection across generations — when improvement stalls, boost inertia to escape local traps; when converging well, reduce inertia for fine-grained exploitation.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using temporal signals: EMA of improvement rate + stagnation detection."""
    # --- Temporal signal: improvement rate (EMA) ---
    if self.global_best is not None and hasattr(self, '_prev_global_fitness'):
        improvement = max(0.0, self._prev_global_fitness - self.global_best_fitness)
    else:
        improvement = 0.0
    self._prev_global_fitness = self.global_best_fitness

    if not hasattr(self, '_ema_improvement'):
        self._ema_improvement = 0.0
    alpha_imp = 0.15
    self._ema_improvement = alpha_imp * improvement + (1.0 - alpha_imp) * self._ema_improvement

    # --- Temporal signal: stagnation detection ---
    stagnation_window = 25
    is_stagnant = self.stagnation_counter >= stagnation_window

    # --- Temporal signal: generation-scaled exploration pressure ---
    gen_factor = min(self.generation / 500.0, 1.0)

    # --- Temporal signal: rate-of-change of EMA improvement (acceleration) ---
    if not hasattr(self, '_ema_improvement_prev'):
        self._ema_improvement_prev = self._ema_improvement
    imp_accel = self._ema_improvement - self._ema_improvement_prev
    self._ema_improvement_prev = self._ema_improvement

    # --- Compute target inertia based on temporal signals ---
    if is_stagnant:
        # Stagnation: raise inertia to escape local traps
        target_inertia = 0.92 + 0.03 * gen_factor
    elif imp_accel < -1e-6 and self._ema_improvement > 1e-8:
        # Improving but decelerating: start fine-tuning
        target_inertia = 0.5 - 0.1 * min(imp_accel * 1e5, 1.0)
    elif self._ema_improvement > 1e-5:
        # Strong improvement phase: keep inertia moderate for exploitation
        target_inertia = 0.55
    elif self._ema_improvement > 1e-10:
        # Weak improvement: balance exploration/exploitation
        target_inertia = 0.65
    else:
        # No improvement: increase exploration
        target_inertia = 0.78 + 0.07 * gen_factor

    target_inertia = np.clip(target_inertia, 0.4, 0.95)

    # --- EMA smoothing to avoid jitter across generations ---
    if not hasattr(self, '_inertia_ema'):
        self._inertia_ema = self.inertia_weight
    alpha_inertia = 0.2
    self._inertia_ema = alpha_inertia * target_inertia + (1.0 - alpha_inertia) * self._inertia_ema

    self.inertia_weight = np.clip(self._inertia_ema, 0.4, 0.95)
```