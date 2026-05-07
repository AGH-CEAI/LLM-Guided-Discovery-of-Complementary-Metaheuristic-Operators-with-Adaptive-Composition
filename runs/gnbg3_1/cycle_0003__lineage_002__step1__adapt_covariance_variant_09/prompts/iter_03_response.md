**Idea: Exploration Temperature with Fitness Gradient Tracking**

This variant introduces an "exploration temperature" that rises when fitness improvement stalls or reverses, forcing aggressive exploration when the algorithm gets stuck. Unlike previous variants that only scale learning rates, this approach uses the temperature to *simultaneously* control both the rank-one (pc) update AND the rank-mu update, creating a fundamentally different exploration-exploitation balance. The fitness gradient magnitude (delta_f_opt) directly modulates temperature, allowing the algorithm to automatically escape local optima on the hardest tasks.

```python
def _adapt_covariance_variant_09(self):
    """Exploration temperature with fitness gradient tracking for escaping local optima."""
    if not hasattr(self, 'prev_f_opt'):
        self.prev_f_opt = self.f_opt
        self.exploration_temp = 1.0
    
    delta_f_opt = self.prev_f_opt - self.f_opt
    self.prev_f_opt = self.f_opt
    
    fitness_gradient = max(abs(delta_f_opt), 1e-15)
    temp_target = np.clip(1.0 / (1.0 + np.log1p(fitness_gradient * 1e5)), 0.05, 5.0)
    self.exploration_temp = 0.95 * self.exploration_temp + 0.05 * temp_target
    
    base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
    ccov_adaptive = base_ccov * self.exploration_temp
    ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 0.5)
    
    cc_adaptive = self.cc * (1.0 / max(self.exploration_temp, 0.5))
    cc_adaptive = np.clip(cc_adaptive, 0.01, 0.3)
    
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - cc_adaptive) * self.pc + np.sqrt(cc_adaptive * (2.0 - cc_adaptive)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    total_w = 0.0
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        w = self.weights[i]
        rank_mu += w * np.outer(diff, diff)
        total_w += abs(w)
    
    if total_w > 0:
        rank_mu /= total_w
    
    self.C = ((1.0 - ccov_adaptive) * self.C + 
              ccov_adaptive * rank_one + 
              (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)
    
    self.C = self._ensure_positive_definite(self.C)
```