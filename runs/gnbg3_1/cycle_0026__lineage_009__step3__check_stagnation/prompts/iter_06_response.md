**Idea: Adaptive Multi-Signal Stagnation Detection**

Instead of a simple absolute improvement threshold, this approach combines fitness improvement rate, population diversity, and covariance conditioning into an adaptive stagnation score that scales with problem difficulty. This is fundamentally different because it detects "local optimum trapping" (not just slow improvement) by monitoring population collapse and matrix degeneracy, enabling much earlier restarts on the hardest tasks where the algorithm gets stuck at errors 10-100.

```python
def _check_stagnation(self):
    """Adaptive multi-signal stagnation detection for early restart on hard tasks."""
    # Track improvement magnitude (relative to current scale)
    abs_improvement = max(0.0, self.f_opt_prev - self.f_opt)
    rel_improvement = abs_improvement / max(abs(self.f_opt), 1.0)
    
    # Compute population diversity (max-normalized spread)
    pop_diffs = self.population - self.mean
    max_sq_dist = np.max(np.sum(pop_diffs ** 2, axis=1))
    expected_max_sq = self.dim * ((self.ub[0] - self.lb[0]) / 3.0) ** 2
    spread_norm = np.sqrt(max_sq_dist) / np.sqrt(max(expected_max_sq, 1e-10))
    spread_norm = np.clip(spread_norm, 0.0, 2.0)
    
    # Compute covariance condition number
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.maximum(np.min(eigvals), 1e-15)
    eig_max = np.maximum(np.max(eigvals), 1e-15)
    cond = eig_max / eig_min
    cond_good = 1.0 / np.log1p(cond)
    
    # Adaptive thresholds scale with problem difficulty
    fit_scale = max(abs(self.f_opt), 1.0)
    improvement_threshold = max(1e-12, 1e-6 * fit_scale)
    diversity_threshold = max(0.01, 0.05 / np.log1p(fit_scale + 1.0))
    condition_threshold = max(0.05, 0.2 / np.log1p(cond))
    
    # Multi-signal stagnation: fitness stalled, population collapsed, or matrix degenerate
    fitness_stalled = abs_improvement < improvement_threshold
    population_collapsed = spread_norm < diversity_threshold
    matrix_degenerate = cond_good < condition_threshold
    
    # Stagnation score (0 = healthy, 1 = fully stagnant)
    stagnation_score = 0.0
    if fitness_stalled:
        stagnation_score += 0.5
    if population_collapsed:
        stagnation_score += 0.3
    if matrix_degenerate:
        stagnation_score += 0.2
    
    # Reset counter on meaningful progress OR healthy state
    if abs_improvement > improvement_threshold or stagnation_score < 0.3:
        self.stagnation_counter = 0
    else:
        self.stagnation_counter += 1
    
    self.f_opt_prev = self.f_opt
```