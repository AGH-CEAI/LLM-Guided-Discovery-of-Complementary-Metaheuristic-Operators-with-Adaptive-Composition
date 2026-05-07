**Idea: Momentum-Driven Sigma with Fitness Gradient and Diversity Awareness**

This approach replaces the stateless success-rate counting with a momentum-based system that tracks fitness improvement momentum AND population diversity. When improvement stalls, positive momentum boosts sigma for exploration. When converging well, negative momentum reduces sigma. Diversity monitoring prevents premature convergence by forcing sigma increase when population collapses.

```python
def _adapt_step_size(self):
    """Adapt step-size using momentum, fitness gradient, and diversity awareness."""
    # Compute fitness improvement
    f_mean_prev = getattr(self, 'f_mean_prev', float(np.mean(self.fitness)))
    f_mean_curr = float(np.mean(self.fitness))
    f_improvement = f_mean_prev - f_mean_curr
    
    # Initialize momentum if needed
    if not hasattr(self, 'sigma_momentum'):
        self.sigma_momentum = 0.0
        self.sigma_history = []
    
    # Update sigma history
    self.sigma_history.append(float(self.sigma))
    if len(self.sigma_history) > 10:
        self.sigma_history.pop(0)
    
    # Compute momentum: positive when improving, negative when stagnating
    improvement_rate = f_improvement / (abs(f_mean_prev) + 1e-10)
    target_momentum = float(np.clip(improvement_rate * 10.0, -1.0, 1.0))
    
    # Smooth momentum update with inertia
    momentum_alpha = 0.3
    self.sigma_momentum = (1.0 - momentum_alpha) * self.sigma_momentum + momentum_alpha * target_momentum
    self.sigma_momentum = float(np.clip(self.sigma_momentum, -0.8, 0.8))
    
    # Compute population diversity (normalized spread)
    pop_spread = float(np.mean(np.std(self.population, axis=0)))
    expected_spread = (self.ub[0] - self.lb[0]) / 6.0
    diversity_ratio = pop_spread / (expected_spread + 1e-10)
    diversity_ratio = float(np.clip(diversity_ratio, 0.0, 1.0))
    
    # Compute success rate for additional signal
    recent_improved = 0
    recent_total = 0
    for i in range(min(self.NP, len(self.trial_fitness), len(self.fitness))):
        if self.trial_fitness[i] < self.fitness[i]:
            recent_improved += 1
        recent_total += 1
    
    if recent_total == 0:
        recent_total = max(1, self.NP // 4)
    
    success_rate = float(recent_improved / recent_total)
    success_rate = float(np.clip(success_rate, 0.0, 1.0))
    
    # Diversity-aware damping: low diversity -> high damping (force exploration)
    min_diversity_thresh = 0.05
    if diversity_ratio < min_diversity_thresh:
        diversity_factor = 0.3  # Low damping to boost sigma
    else:
        diversity_factor = 1.0
    
    # Combined adaptation signal
    # Momentum dominates, success rate and diversity provide correction
    adaptation = self.sigma_momentum + 0.2 * (success_rate - 0.25)
    
    # Force sigma increase when diversity is critically low
    if diversity_ratio < min_diversity_thresh:
        adaptation = max(adaptation, 0.3)
    
    adaptation = float(np.clip(adaptation, -0.6, 0.6))
    
    # Adaptive damping with dimension scaling
    damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim)) * diversity_factor
    damping_adaptive = float(np.clip(damping_adaptive, 0.05, 100.0))
    
    # Apply momentum-driven sigma update
    self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
    self.sigma = float(np.clip(self.sigma, 1e-12, 15.0))
    
    # Prevent sigma collapse: if sigma is too small relative to population spread
    if pop_spread > 1e-10 and self.sigma < pop_spread * 1e-4:
        self.sigma = pop_spread * 1e-3
    
    # Store current mean fitness for next iteration
    self.f_mean_prev = f_mean_curr
```