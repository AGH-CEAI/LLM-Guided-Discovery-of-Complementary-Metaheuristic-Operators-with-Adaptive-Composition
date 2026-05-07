Looking at the benchmark results, I see a pattern where the worst unsolved tasks (21, 17, 18, 19, 22, 20, 15, 14) have errors stuck at 0.5-4.1, suggesting the optimizer is getting trapped in local optima. The current `_adapt_step_size` relies solely on the evolution path norm, which may not respond well when the algorithm stagnates. I need a fundamentally different approach that combines multiple feedback signals: fitness momentum, stagnation detection, and population diversity monitoring.

**Idea: Momentum-Stagnation Diversity Step Size Adaptation**

This approach uses three signals: (1) fitness improvement momentum over recent generations, (2) explicit stagnation counter, and (3) population spread monitoring. When momentum is poor and stagnation is detected, sigma is boosted significantly. When diversity collapses (population spread shrinking), sigma is also increased. This multi-signal approach can break through local optima traps that the evolution-path-only adaptation misses.
```python
def _adapt_step_size(self):
    # Track fitness history for momentum
    if not hasattr(self, '_fitness_momentum'):
        self._fitness_momentum = []
    self._fitness_momentum.append(self.best_f)
    if len(self._fitness_momentum) > 5:
        self._fitness_momentum.pop(0)
    
    # Compute momentum from recent fitness improvements
    momentum = 0.0
    if len(self._fitness_momentum) >= 2:
        improvements = []
        for i in range(1, len(self._fitness_momentum)):
            delta = self._fitness_momentum[i-1] - self._fitness_momentum[i]
            if delta > 0:
                improvements.append(delta)
        if improvements:
            momentum = np.mean(improvements) / (abs(self.best_f) + 1e-20)
    
    # Standard CMA-ES adaptation
    p_sigma_norm = np.linalg.norm(self.p_sigma)
    c_sigma_adapt = (self.c_sigma / self.d_sigma) * (p_sigma_norm / self.chi_n - 1.0)
    
    # Determine sigma update based on momentum and stagnation
    if self.stagnation_counter > 10:
        # Strong stagnation: aggressive sigma increase
        self.sigma *= 2.5
    elif self.stagnation_counter > 5:
        # Moderate stagnation: moderate sigma increase
        self.sigma *= 1.5
    elif momentum < 1e-6:
        # Low momentum but not stagnating: slight increase
        self.sigma *= np.exp(c_sigma_adapt * 1.5)
    elif momentum > 0.1:
        # Good momentum: reduce step size for refinement
        self.sigma *= np.exp(c_sigma_adapt * 0.5)
    else:
        # Normal adaptation
        self.sigma *= np.exp(c_sigma_adapt)
    
    # Diversity-based correction: detect population collapse
    if hasattr(self, '_last_pop_spread'):
        pop_spread = np.std(self.mean) if self.dim > 0 else 1.0
        if pop_spread < 0.5 * self._last_pop_spread and self.stagnation_counter > 2:
            self.sigma *= 2.0
    self._last_pop_spread = np.std(self.mean) if self.dim > 0 else 1.0
    
    self.sigma = np.clip(self.sigma, 1e-20, 1e5)
```