**Idea: Multi-Trial Adaptive Mutation in Eval Wrapper**

The current wrapper is a passive pass-through. The priority tasks (errors 10^2-10^3) are stuck in deep local optima. The quantum tunnel only modifies dim//5=6 dims with fixed scale, which is insufficient to escape. This wrapper will **internally generate and rank multiple adaptive mutation trials** before committing to a single evaluation result — turning the wrapper into an active escape mechanism rather than a passive evaluator.

```python
def _eval_wrapper(self, x):
    """Multi-trial adaptive mutation wrapper for tunneling phase."""
    # Start with the proposed tunneled position
    best_x = x.copy()
    best_f = self._user_func(x.reshape(1, -1))[0]
    
    # Adaptive mutation scale based on current best fitness magnitude
    scale_factor = max(1.0, abs(self.global_best_fit) / (abs(self.global_best_fit) + 1.0))
    base_scale = (self.upper - self.lower) * self.tunnel_scale * self.step_size * scale_factor
    
    # Generate multiple trial mutations with varying scales (conservative to aggressive)
    n_trials = 5
    for trial in range(n_trials):
        trial_scale = base_scale * (0.3 + 0.7 * trial / (n_trials - 1)) if n_trials > 1 else base_scale
        n_dims = max(1, self.dim // 3)  # Modify more dimensions than default (10 instead of 6)
        
        dims = np.random.choice(self.dim, n_dims, replace=False)
        trial_x = best_x.copy()
        trial_x[dims] += np.random.uniform(-trial_scale, trial_scale, size=n_dims)
        trial_x = np.clip(trial_x, self.lower, self.upper)
        
        trial_f = self._user_func(trial_x.reshape(1, -1))[0]
        if trial_f < best_f:
            best_x = trial_x
            best_f = trial_f
    
    return best_f
```