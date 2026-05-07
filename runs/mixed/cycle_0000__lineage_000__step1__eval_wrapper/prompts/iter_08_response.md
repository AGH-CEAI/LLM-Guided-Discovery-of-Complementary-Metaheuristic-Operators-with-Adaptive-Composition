**Idea: Gradient-Approximated Tunnel Perturbation**

Uses numerical gradient estimation and adaptive step sizing during tunnel evaluations to escape local optima on ill-conditioned, non-separable, and deceptive tasks. By computing approximate gradients from nearby samples and perturbing along gradient directions with controlled step sizes, this enables more directed exploration on the worst tasks (12, 21, 13, 8, 16, 2) that are stuck at errors >100.

```python
def _eval_wrapper(self, x):
    """Gradient-approximated evaluation for tunnel phase with adaptive step sizing."""
    epsilon = 1e-5
    n_samples = min(5, self.dim)
    
    # Compute numerical gradient via coordinate perturbations
    grad = np.zeros(self.dim)
    f0 = self._user_func(x.reshape(1, -1))[0]
    
    for j in range(n_samples):
        coord = j % self.dim
        x_plus = x.copy()
        x_plus[coord] += epsilon
        x_plus = np.clip(x_plus, self.lower, self.upper)
        f_plus = self._user_func(x_plus.reshape(1, -1))[0]
        grad[coord] = (f_plus - f0) / epsilon
    
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e-10:
        grad = grad / grad_norm
    
    # Adaptive step sizing from history
    step = self.step_size
    if hasattr(self, 'fitness_history') and len(self.fitness_history) > 2:
        recent_improvement = abs(self.fitness_history[-1] - self.fitness_history[-2]) if len(self.fitness_history) >= 2 else 1.0
        if recent_improvement < 1e-6:
            step *= 2.0
        elif recent_improvement > 1.0:
            step *= 0.5
    step = np.clip(step, 0.01, 2.0)
    
    # Perturb along gradient direction with adaptive step
    x_perturbed = x - 0.5 * grad * step * (self.upper - self.lower)
    
    # Add bounded random exploration
    explore_scale = 0.1 * (self.upper - self.lower) * step
    x_perturbed += np.random.uniform(-explore_scale, explore_scale, size=self.dim)
    
    # Clip to bounds
    x_perturbed = np.clip(x_perturbed, self.lower, self.upper)
    
    return self._user_func(x_perturbed.reshape(1, -1))[0]
```