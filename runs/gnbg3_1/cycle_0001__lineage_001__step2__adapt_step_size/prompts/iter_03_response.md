Looking at the benchmark data, I need to analyze why the worst tasks (17, 16, 10, 18, 20) are stuck at errors of 10^1 to 10^2.5 — roughly 8-11 orders of magnitude above target. The current CSA-based step-size adaptation relies on evolution path length, which can get trapped in suboptimal attractors on highly multi-modal or ill-conditioned landscapes.

**Idea: Success-Rate Step-Size Adaptation**

Instead of using the evolution path direction (which can mislead in deceptive landscapes), directly track the empirical success rate of steps and adapt sigma based on how well recent steps correlated with fitness improvements. This is fundamentally different from CSA's theoretical cumulation approach — it's data-driven adaptation that responds directly to observed progress.

```python
def _adapt_step_size(self):
    """Adapt step-size using success-rate feedback (data-driven approach)."""
    if not hasattr(self, 'step_history'):
        self.step_history = {'success': [], 'norms': [], 'fitness_improvements': []}
    if not hasattr(self, 'success_rate_ema'):
        self.success_rate_ema = 0.5
    
    actual_step = self.mean - self.old_mean
    step_norm = np.linalg.norm(actual_step)
    
    is_success = self.f_opt < getattr(self, 'f_opt_prev', float('inf'))
    fitness_delta = getattr(self, 'f_opt_prev', self.f_opt) - self.f_opt
    self.f_opt_prev = self.f_opt
    
    history = self.step_history
    max_history = min(15, self.dim + 5)
    
    history['success'].append(float(is_success))
    history['norms'].append(float(max(step_norm / max(self.sigma, 1e-15), 1e-15)))
    history['fitness_improvements'].append(float(max(fitness_delta, 0.0)))
    
    if len(history['success']) > max_history:
        history['success'].pop(0)
        history['norms'].pop(0)
        history['fitness_improvements'].pop(0)
    
    recent_window = min(5, len(history['success']))
    recent_success = history['success'][-recent_window:]
    recent_norms = history['norms'][-recent_window:]
    recent_fitness = history['fitness_improvements'][-recent_window:]
    
    observed_success_rate = sum(recent_success) / max(len(recent_success), 1)
    self.success_rate_ema = 0.7 * self.success_rate_ema + 0.3 * observed_success_rate
    
    target_success_rate = 0.25
    success_ratio = observed_success_rate / max(target_success_rate, 0.01)
    
    mean_recent_norm = np.mean(recent_norms) if recent_norms else 1.0
    norm_ratio = mean_recent_norm / max(np.sqrt(self.dim) * 0.5, 0.1)
    
    if len(history['success']) >= 3:
        total_improvement = sum(recent_fitness)
        improvement_factor = 1.0 + 0.2 * np.log1p(total_improvement)
        improvement_factor = min(improvement_factor, 3.0)
    else:
        improvement_factor = 1.0
    
    if success_ratio > 1.5:
        sigma_mult = 1.0 + 0.15 * np.log(success_ratio + 1e-10) * improvement_factor
    elif success_ratio < 0.5:
        sigma_mult = max(0.5, 1.0 - 0.2 * np.log(1.0 / (success_ratio + 1e-10)))
    else:
        sigma_mult = 1.0 + 0.05 * (success_ratio - 1.0)
    
    if self.success_rate_ema < 0.15:
        sigma_mult *= 0.7
    elif self.success_rate_ema > 0.4:
        sigma_mult *= 1.1
    
    self.sigma *= sigma_mult
    self.sigma = np.clip(self.sigma, 1e-12, 50.0)
```