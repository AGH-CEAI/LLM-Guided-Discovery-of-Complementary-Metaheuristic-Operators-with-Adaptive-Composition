**Idea: Adaptive Path Momentum with Fitness-Guided Scaling**

This approach fundamentally differs from previous variants by making the evolution path update **responsive to fitness landscape progress**. Instead of a fixed exponential decay rate, it computes a smoothed fitness improvement ratio and uses it to scale the path update magnitude. When fitness is improving, the scaling factor > 1 makes the path more aggressive (larger steps, faster covariance adaptation). When stuck, scaling factor < 1 makes the path more conservative, preventing momentum from carrying the search away from local optima. This directly targets the worst unsolved tasks (16, 19, 20, 23, 18) which are stuck at O(10^0) to O(10^2) — classic signs of premature convergence where the evolution paths have too much momentum in unpromising directions.

```python
def _update_evolution_paths(self, old_mean):
    """
    Update evolution paths with fitness-guided adaptive scaling.
    The path momentum is modulated by recent fitness improvement.
    """
    mean_shift = (self.mean - old_mean) / self.sigma
    transformed = self.invsqrtC @ mean_shift

    # Track smoothed fitness for improvement detection
    if not hasattr(self, 'smoothed_fitness'):
        self.smoothed_fitness = self.best_f
    self.smoothed_fitness = 0.9 * self.smoothed_fitness + 0.1 * self.best_f

    # Compute normalized fitness improvement
    if self.last_best_f < np.inf and self.last_best_f > 1e-30:
        improvement = (self.last_best_f - self.best_f) / (abs(self.last_best_f) + 1e-30)
        improvement = max(-5.0, min(5.0, improvement))
        improvement_factor = 1.0 + 0.5 * improvement
        improvement_factor = max(0.1, min(3.0, improvement_factor))
    else:
        improvement_factor = 1.0

    # Update step-size path with adaptive scaling
    c_s_complement = np.sqrt(1.0 - (1.0 - self.c_sigma) ** (2 * (self.generation + 1)))
    base_update = np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * transformed
    self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + improvement_factor * base_update

    # Heaviside function with chi_n-based threshold
    p_sigma_norm = np.linalg.norm(self.p_sigma)
    threshold = (1.4 + 2.0 / (self.dim + 1.0)) * self.chi_n * c_s_complement
    h_sigma = 1.0 if p_sigma_norm < threshold else 0.0

    # Update covariance path with adaptive scaling
    base_cov_update = np.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * mean_shift
    self.p_c = (1.0 - self.c_c) * self.p_c + h_sigma * improvement_factor * base_cov_update

    return h_sigma
```