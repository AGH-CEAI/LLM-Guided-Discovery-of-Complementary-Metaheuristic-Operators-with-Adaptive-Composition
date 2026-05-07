**Idea: Success-Rate Step-Size Adaptation**
Adapt sigma based on recent success/failure history with diversity weighting, fundamentally replacing the evolution path (CSA) with a direct improvement-rate approach that responds faster to local minima traps common in the worst unsolved tasks.

```python
def _adapt_step_size(self):
    """Adapt step-size using success history and diversity-weighted control."""
    improvement = self.f_opt_prev - self.f_opt
    
    if not hasattr(self, 'success_history'):
        self.success_history = []
    if not hasattr(self, 'sigma_history'):
        self.sigma_history = []
    
    is_success = improvement > 0
    self.success_history.append(1.0 if is_success else 0.0)
    self.sigma_history.append(self.sigma)
    
    max_history = max(5, self.dim // 10 + 5)
    if len(self.success_history) > max_history:
        self.success_history.pop(0)
        self.sigma_history.pop(0)
    
    success_rate = np.mean(self.success_history)
    
    pop_spread = np.mean(np.std(self.population, axis=0))
    expected_spread = (self.ub[0] - self.lb[0]) / 6.0
    diversity_ratio = np.clip(pop_spread / (expected_spread + 1e-10), 0.1, 3.0)
    
    target_success = 0.25
    
    if success_rate > target_success:
        adjustment = 1.0 + 0.3 * (success_rate - target_success) / (1.0 - target_success + 1e-10)
    else:
        adjustment = 0.7 + 0.3 * success_rate / (target_success + 1e-10)
    
    if diversity_ratio < 0.3:
        adjustment *= 1.5
    elif diversity_ratio < 0.6:
        adjustment *= 1.2
    
    if len(self.sigma_history) >= 3:
        sigma_trend = self.sigma_history[-1] / (np.mean(self.sigma_history[-3:]) + 1e-10)
        if sigma_trend < 0.8:
            adjustment *= 1.1
        elif sigma_trend > 1.2:
            adjustment *= 0.9
    
    self.sigma *= adjustment
    self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```