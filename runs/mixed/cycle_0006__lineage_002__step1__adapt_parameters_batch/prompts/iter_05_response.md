**Idea: Fitness-Feedback Stagnation-Driven Adaptation**

This variant replaces random parameter averaging with a feedback-driven approach that tracks improvement rate and stagnation depth. When the algorithm is stuck (stagnation_count > 0), it exponentially increases F and CR to force exploration. This fundamentally differs from all prior variants which use decorrelated random sampling.

```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
    # Track improvement history for adaptive feedback
    if not hasattr(self, '_improvement_history'):
        self._improvement_history = []
    
    # Compute recent improvement rate
    if len(fitness) > 0:
        best_fit = float(np.min(fitness))
        if hasattr(self, '_prev_best_fitness') and np.isfinite(self._prev_best_fitness):
            improvement = self._prev_best_fitness - best_fit
        else:
            improvement = 0.0
        self._improvement_history.append(improvement)
        if len(self._improvement_history) > 10:
            self._improvement_history.pop(0)
        self._prev_best_fitness = best_fit
    
    # Stagnation depth drives exploration
    stagnation_depth = min(self.stagnation_count, 50)
    
    # Adaptive F: low when improving, high when stagnant
    if stagnation_depth == 0:
        # Normal mode: keep F moderate
        F_base = 0.6
        F_spread = 0.15
    else:
        # Stagnation mode: scale F exponentially with stagnation depth
        F_base = min(0.6 + 0.03 * stagnation_depth, 1.8)
        F_spread = min(0.15 + 0.02 * stagnation_depth, 0.4)
    
    F_candidates = np.random.uniform(
        max(0.0, F_base - F_spread),
        min(2.0, F_base + F_spread),
        size=5
    )
    self.F = float(np.clip(np.mean(F_candidates), 0.1, 2.0))
    
    # Adaptive CR: scale with stagnation for broader search
    if stagnation_depth == 0:
        CR_base = 0.5
        CR_spread = 0.2
    else:
        # Deeper stagnation → higher CR for wider exploration
        CR_base = min(0.5 + 0.02 * stagnation_depth, 0.95)
        CR_spread = min(0.2 + 0.01 * stagnation_depth, 0.3)
    
    CR_candidates = np.random.uniform(
        max(0.05, CR_base - CR_spread),
        min(0.99, CR_base + CR_spread),
        size=5
    )
    self.CR = float(np.clip(np.mean(CR_candidates), 0.05, 0.99))
    
    # Adaptive p_best_rate: lower p when stagnant (more greedy search)
    if stagnation_depth > 15:
        p_target = max(0.05, self.p_best_rate - 0.02)
    elif stagnation_depth == 0 and len(self._improvement_history) >= 3:
        recent = self._improvement_history[-3:]
        if sum(recent) > 0:
            p_target = min(0.25, self.p_best_rate + 0.01)
        else:
            p_target = self.p_best_rate
    else:
        p_target = self.p_best_rate
    
    if np.random.rand() < 0.3:
        self.p_best_rate = float(np.clip(
            self.p_best_rate + np.random.uniform(-0.05, 0.05) + 0.5 * (p_target - self.p_best_rate),
            0.05, 0.3
        ))
```