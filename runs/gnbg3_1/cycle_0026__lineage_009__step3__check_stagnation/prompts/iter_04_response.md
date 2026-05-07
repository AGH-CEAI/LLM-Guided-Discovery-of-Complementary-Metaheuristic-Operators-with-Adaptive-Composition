**Idea: Multi-Dimensional Stagnation Detection with Diversity-Aware Restart Triggers**

This approach replaces the simple fitness-only stagnation check with a multi-dimensional detector that tracks fitness improvement rate, population diversity, and covariance condition. It triggers restarts earlier when the algorithm gets stuck in local optima on multimodal tasks (the primary failure mode for Tasks 17, 16, 23, 20, 10, 11, 18, 12 with errors ~10^1-10^2).

```python
def _check_stagnation(self):
    """Multi-dimensional stagnation detection with diversity and condition awareness."""
    # Track fitness improvement
    if self.f_opt < self.f_opt_prev - 1e-12:
        self.stagnation_counter = 0
        self.stagnation_start_f = self.f_opt
    else:
        self.stagnation_counter += 1
    
    self.f_opt_prev = self.f_opt
    
    # Initialize stagnation tracking
    if not hasattr(self, 'stagnation_start_f'):
        self.stagnation_start_f = self.f_opt
    if not hasattr(self, 'diversity_stagnation_counter'):
        self.diversity_stagnation_counter = 0
    if not hasattr(self, 'condition_stagnation_counter'):
        self.condition_stagnation_counter = 0
    
    # Compute current diversity and condition
    pop_diffs = self.population - self.mean
    max_sq_dist = np.max(np.sum(pop_diffs ** 2, axis=1))
    expected_max_sq = self.dim * ((self.ub[0] - self.lb[0]) / 3.0) ** 2
    current_diversity = np.sqrt(max_sq_dist) / np.sqrt(max(expected_max_sq, 1e-10))
    current_diversity = np.clip(current_diversity, 0.0, 2.0)
    
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.min(eigvals)
    eig_max = np.max(eigvals)
    current_cond = eig_max / max(eig_min, 1e-15)
    
    # Track diversity stagnation
    if current_diversity < 0.05:
        self.diversity_stagnation_counter += 1
    else:
        self.diversity_stagnation_counter = max(0, self.diversity_stagnation_counter - 1)
    
    # Track condition stagnation
    if current_cond > 1e6:
        self.condition_stagnation_counter += 1
    else:
        self.condition_stagnation_counter = max(0, self.condition_stagnation_counter - 1)
    
    # Compute relative improvement rate (for detecting slow progress on multimodal tasks)
    f_range = abs(self.stagnation_start_f) + 1e-10
    relative_stagnation = abs(self.f_opt - self.stagnation_start_f) / f_range
    
    # Store metrics for restart logic (accessed by _restart_if_needed)
    self.relative_stagnation = relative_stagnation
    self.current_diversity = current_diversity
    self.current_condition = current_cond
```