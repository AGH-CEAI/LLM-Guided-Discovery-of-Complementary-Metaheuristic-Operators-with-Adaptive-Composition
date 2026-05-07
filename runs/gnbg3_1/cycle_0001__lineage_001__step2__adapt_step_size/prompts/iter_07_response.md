Looking at the problem: all 6 previous variants use the same CSA (Cumulative Step Size Adaptation) based on evolution path norm. The worst unsolved tasks (17, 16, 20, 18, 12, 10) have errors ~10x-250x, indicating stagnation/plateaus where CSA gets confused by the decorrelation in the evolution path.

**Key insight**: CSA assumes unimodal progress, but the worst tasks likely have deceptive local optima or rugged fitness landscapes where the evolution path becomes misleading. A **success-history based** approach (like IPOP-CMA-ES uses for restarts) directly measures improvement rate and adapts faster without path decorrelation artifacts.

**Idea: Success-Rate Step Size Adaptation**

This fundamentally differs from CSA by using a sliding window of recent fitness improvements instead of the cumulative evolution path, allowing faster, more direct response to local optima.

```python
def _adapt_step_size(self):
    """Adapt step-size using success-history (IPOP-style), not evolution path."""
    if not hasattr(self, '_step_fitness_history'):
        self._step_fitness_history = []
    self._step_fitness_history.append(self.f_opt)
    if len(self._step_fitness_history) > max(5, self.dim // 4):
        self._step_fitness_history.pop(0)
    
    history = self._step_fitness_history
    n = len(history)
    
    if n >= 2:
        improvements = [max(0.0, history[i] - history[i + 1]) for i in range(n - 1)]
        success_rate = sum(1 for imp in improvements if imp > 0) / max(1, len(improvements))
        avg_improvement = np.mean(improvements) if improvements else 0.0
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        target_success = 0.2 + 0.3 * diversity
        adaptation = 1.0 + (success_rate - target_success) * 0.5 * self.cs
        
        if avg_improvement > 1e-12:
            adaptation *= (1.0 + np.log1p(avg_improvement * 1e6) * 0.1)
        
        if success_rate < 0.1 and n >= 3:
            adaptation *= 0.5
        elif success_rate > 0.5:
            adaptation *= 1.2
        
        self.sigma *= np.clip(adaptation, 0.1, 10.0)
    
    self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```