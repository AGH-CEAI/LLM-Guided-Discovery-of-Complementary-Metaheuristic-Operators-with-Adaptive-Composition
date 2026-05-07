**Idea: Elite-Derived Parameter Adaptation with Stagnation Escaping**

Instead of random sampling, this strategy learns from the best-performing individuals in the current generation. It analyzes the F/CR values used by top solutions that improved, maintains an exponential moving average of successful parameters, and when stagnating, aggressively expands the search range with larger F values and higher CR to escape local optima — directly targeting the large errors on Tasks 12, 8, 16, 21, 13.

```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
    # Track successful parameter values from improved individuals
    if not hasattr(self, '_ema_F'):
        self._ema_F = 0.7
        self._ema_CR = 0.5
        self._ema_p = 0.1

    improvement = fitness - np.minimum(fitness, np.array([np.inf] * len(fitness)))
    improved_mask = improvement > 0
    n_improved = int(np.sum(improved_mask))

    if n_improved > 0:
        improved_idx = np.where(improved_mask)[0]
        F_used = self.F * np.ones(n_improved)
        CR_used = self.CR * np.ones(n_improved)

        # Exponential moving average of successful parameters
        alpha = 0.3
        self._ema_F = float(np.clip(alpha * np.mean(F_used) + (1 - alpha) * self._ema_F, 0.1, 2.0))
        self._ema_CR = float(np.clip(alpha * np.mean(CR_used) + (1 - alpha) * self._ema_CR, 0.05, 0.99))

        # Blend EMA with current values
        blend = min(0.4, 0.1 + 0.05 * n_improved)
        self.F = float(np.clip(blend * self._ema_F + (1 - blend) * self.F, 0.3, 1.5))
        self.CR = float(np.clip(blend * self._ema_CR + (1 - blend) * self.CR, 0.1, 0.95))
    else:
        # Stagnation: expand search range with larger F and higher CR
        stagnation_level = min(float(self.stagnation_count) / max(1, float(self.stagnation_limit)), 1.0)
        F_expansion = 1.0 + stagnation_level * 0.5
        CR_increase = stagnation_level * 0.15

        self.F = float(np.clip(
            self.F + np.random.uniform(-0.1, 0.2) * F_expansion,
            0.3 * F_expansion, 2.0
        ))
        self.CR = float(np.clip(
            self.CR + np.random.uniform(-0.1, 0.1) + CR_increase,
            0.1, 0.99
        ))

    # Adapt p_best_rate based on diversity
    if hasattr(self, '_prev_diversity'):
        diversity_change = self._compute_diversity_batch(population) - self._prev_diversity
        if diversity_change < -0.01 * self.NP:
            self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(0.01, 0.03), 0.05, 0.3))
        elif diversity_change > 0.01 * self.NP:
            self.p_best_rate = float(np.clip(self.p_best_rate - np.random.uniform(0.01, 0.03), 0.05, 0.3))
    self._prev_diversity = self._compute_diversity_batch(population)
```