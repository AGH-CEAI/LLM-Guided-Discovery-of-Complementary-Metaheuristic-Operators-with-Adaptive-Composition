Looking at the unsolved tasks, the worst ones have errors in the 10-100 range, indicating the algorithm is essentially stuck and not converging. The current step-size adaptation is too conservative - it uses a fixed target rate of 0.25 and makes incremental changes, which cannot break through the optimization barriers on these difficult tasks.

**Idea: Magnitude-Aware Aggressive Step-Size with Stagnation Detection**

This approach fundamentally differs by:
1. Using log-scale improvement magnitude instead of just success rate
2. Detecting stagnation and making aggressive sigma increases (up to 10x) to escape local optima
3. Adapting the target rate dynamically based on dim and phase of optimization
4. Using exponential damping for more responsive sigma control

```python
def _adapt_step_size(self):
    """Adapt step-size using magnitude-aware improvement with stagnation detection."""
    # Calculate improvement metrics
    if len(self.trial_fitness) > 0 and len(self.fitness) > 0:
        improvements = np.maximum(0.0, self.fitness - self.trial_fitness)
        recent_improved = np.sum(improvements > 0)
        total_improvement = np.sum(improvements)
        max_improvement = np.max(improvements)
    else:
        recent_improved = 0
        total_improvement = 0.0
        max_improvement = 0.0

    recent_total = max(1, len(self.trial_fitness))
    success_rate = recent_improved / recent_total
    
    # Dynamic target rate based on dimension
    target_rate = 0.2 + 0.1 * np.tanh(self.dim / 20.0)
    
    # Magnitude-aware adaptation: use log-scale improvement
    expected_improvement = max(abs(self.f_opt), 1.0) * 0.01
    if total_improvement > 0 and expected_improvement > 0:
        improvement_ratio = total_improvement / (expected_improvement * recent_total + 1e-15)
        magnitude_factor = np.tanh(np.log1p(improvement_ratio) / 5.0)
    else:
        magnitude_factor = -0.5
    
    # Stagnation detection: if no significant improvement, increase sigma aggressively
    is_stagnant = (max_improvement < abs(self.f_opt) * 1e-6 + 1e-10) or (self.f_opt > 1e3)
    
    if is_stagnant:
        # Aggressive expansion to escape local optima
        stagnation_factor = 2.0
        self.sigma = np.clip(self.sigma * stagnation_factor, 1e-10, 50.0)
    else:
        # Combine success rate and magnitude for adaptation
        rate_adaptation = (success_rate - target_rate) / (target_rate + 1e-15)
        combined_adaptation = 0.7 * rate_adaptation + 0.3 * magnitude_factor
        combined_adaptation = np.clip(combined_adaptation, -0.8, 0.8)
        
        # Adaptive damping based on dimension and current sigma
        damping_adaptive = self.damping * (0.3 + 0.7 * np.log1p(self.dim) / np.log1p(100))
        damping_adaptive = np.clip(damping_adaptive, 0.05, 50.0)
        
        # Exponential update with combined factors
        self.sigma *= np.exp(combined_adaptation * self.cs / damping_adaptive)
        self.sigma = np.clip(self.sigma, 1e-10, 50.0)
```