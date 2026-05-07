**Idea: Success-Rate Step-Size Adaptation**

Replaces cumulative evolution path (long-term memory) with short-term success-history feedback. When recent samples consistently improve fitness, sigma grows aggressively to escape local optima; when stuck, sigma shrinks. This directly addresses the huge errors (10^1–10^3) on worst tasks by breaking the cumulative-path staleness that causes exploration failure on multimodal/ill-conditioned landscapes.

```python
def _adapt_step_size(self):
    """Adapt step-size using short-term success history (no cumulation)."""
    recent_improved = 0
    recent_total = 0
    
    for i in range(self.NP):
        if i < len(self.trial_fitness) and i < len(self.fitness):
            if self.trial_fitness[i] < self.fitness[i]:
                recent_improved += 1
            recent_total += 1
    
    if recent_total == 0:
        recent_total = max(1, self.NP // 4)
    
    success_rate = recent_improved / recent_total
    success_rate = np.clip(success_rate, 0.0, 1.0)
    
    target_rate = 0.25
    adaptation = (success_rate - target_rate) / target_rate
    adaptation = np.clip(adaptation, -0.5, 0.5)
    
    damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim))
    damping_adaptive = np.clip(damping_adaptive, 0.1, 100.0)
    
    self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
    self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```