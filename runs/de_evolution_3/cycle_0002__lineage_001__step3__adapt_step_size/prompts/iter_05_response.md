**Idea: Fitness-Landscape Adaptive Sigma**
Replace evolution-path based sigma adaptation with one driven by fitness dynamics: if best AND median improve → reduce sigma (refine); if only best improves → more aggressive reduction (converging); if median improves but best worsens → increase sigma (escaping local optima); if neither improves → big sigma increase (break stagnation).

```python
def _adapt_step_size(self):
    if not hasattr(self, '_prev_best_fit'):
        self._prev_best_fit = np.inf
    if not hasattr(self, '_prev_median_fit'):
        self._prev_median_fit = np.inf

    current_best = self.best_f
    valid_fit = np.array([f for f in self.pop_fitness if np.isfinite(f)])
    current_median = np.median(valid_fit) if len(valid_fit) > 0 else np.inf

    best_improved = current_best < self._prev_best_fit - 1e-15
    median_improved = current_median < self._prev_median_fit - 1e-15

    if best_improved and median_improved:
        self.sigma *= np.exp(-0.3)
    elif best_improved and not median_improved:
        self.sigma *= np.exp(-0.5)
    elif not best_improved and median_improved:
        self.sigma *= np.exp(0.3)
    else:
        self.sigma *= np.exp(0.5)

    self._prev_best_fit = current_best
    self._prev_median_fit = current_median
    self.sigma = np.clip(self.sigma, 1e-20, 1e5)
```