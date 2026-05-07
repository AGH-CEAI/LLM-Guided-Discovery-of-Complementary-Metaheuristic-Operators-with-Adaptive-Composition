**Idea: Success-Rate Step-Size Adaptation**
One-line description.
```python
def _adapt_step_size(self):
        """Adapt step-size using explicit success-rate tracking (no CSA evolution path)."""
        if not hasattr(self, '_step_history'):
            self._step_history = []
        
        n_successful = sum(1 for f in self.trial_fitness if f < self.f_opt_prev)
        current_success_rate = n_successful / max(len(self.trial_fitness), 1)
        self._step_history.append(current_success_rate)
        
        if len(self._step_history) > 5:
            self._step_history.pop(0)
        
        smoothed_success = sum(self._step_history) / len(self._step_history)
        
        expected_success = 0.2
        success_diff = smoothed_success - expected_success
        
        damping_adjusted = self.damping * np.sqrt(self.dim)
        
        adapt_rate = self.cs * 1.5
        sigma_multiplier = np.exp(adapt_rate * success_diff / damping_adjusted)
        
        self.sigma *= sigma_multiplier
        
        pop_spread = np.mean(np.std(self.population, axis=0))
        if pop_spread > 1e-10:
            target_spread = (self.ub[0] - self.lb[0]) * 0.05
            if self.sigma * pop_spread > target_spread * 10:
                self.sigma *= 0.9
            elif self.sigma * pop_spread < target_spread * 0.1:
                self.sigma *= 1.1
        
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```