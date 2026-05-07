**Idea: Momentum-Gradient Step Size Adaptation**

This variant replaces the binary success-rate approach with a momentum-based method that tracks fitness improvement gradients and adapts the step size using exponential moving averages. It detects stagnation and automatically increases exploration by boosting sigma when improvement stalls. This fundamentally differs from the current approach which only counts discrete wins without considering improvement magnitude or momentum.

```python
def _adapt_step_size(self):
    """Adapt step-size using momentum-based gradient tracking with stagnation detection."""
    # Compute fitness improvement for each individual
    total_improvement = 0.0
    improving_count = 0
    
    for i in range(min(self.NP, len(self.trial_fitness), len(self.fitness))):
        delta = self.fitness[i] - self.trial_fitness[i]
        if delta > 0:
            total_improvement += delta
            improving_count += 1
    
    # Normalize improvement by current fitness scale
    fitness_scale = max(abs(self.f_opt), 1.0)
    normalized_improvement = total_improvement / (fitness_scale * max(improving_count, 1))
    
    # Exponential moving average of improvement (momentum)
    beta = 0.7
    if not hasattr(self, 'improvement_ema'):
        self.improvement_ema = normalized_improvement
    self.improvement_ema = beta * self.improvement_ema + (1.0 - beta) * normalized_improvement
    
    # Stagnation detection: check if best fitness hasn't improved
    if not hasattr(self, 'stagnation_steps'):
        self.stagnation_steps = 0
    if hasattr(self, 'f_opt_prev'):
        if self.f_opt >= self.f_opt_prev - 1e-12:
            self.stagnation_steps += 1
        else:
            self.stagnation_steps = 0
    
    # Target success rate adapts based on momentum (more aggressive when improving)
    base_target = 0.25
    momentum_factor = min(self.improvement_ema * 10.0, 1.5)
    target_rate = base_target / momentum_factor
    target_rate = np.clip(target_rate, 0.05, 0.5)
    
    # Compute actual success rate
    success_rate = improving_count / max(self.NP, 1)
    success_rate = np.clip(success_rate, 0.0, 1.0)
    
    # Adaptation factor: how far actual is from target
    adaptation = (success_rate - target_rate) / max(target_rate, 0.01)
    adaptation = np.clip(adaptation, -0.8, 0.8)
    
    # Stagnation boost: increase exploration when stuck
    stagnation_boost = 1.0
    if self.stagnation_steps > 10:
        stagnation_boost = 1.5 + 0.1 * min(self.stagnation_steps, 50)
    
    # Adaptive damping based on dimension and momentum
    damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim))
    damping_adaptive *= (1.0 + 2.0 * max(0.0, 0.1 - self.improvement_ema))
    damping_adaptive = np.clip(damping_adaptive, 0.05, 200.0)
    
    # Apply adaptation with momentum and stagnation boost
    self.sigma *= np.exp(adaptation * self.cs / damping_adaptive) * stagnation_boost
    self.sigma = np.clip(self.sigma, 1e-10, 50.0)
```