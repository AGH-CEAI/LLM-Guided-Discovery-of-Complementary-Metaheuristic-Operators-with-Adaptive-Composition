Looking at the problem, the worst unsolved tasks (16, 17, 20, 23, 19, 18, 10, 11) have errors stuck at ~1e+0 to ~1e+2. The current `_compute_diversity` uses simple mean per-dimension std, which doesn't capture the **shape** of the covariance matrix. The algorithm likely gets stuck in local optima because the diversity metric fails to detect when the population has collapsed into a degenerate subspace.

**Idea: Eigenvalue-Shape Diversity with Effective Sample Size**

This approach computes diversity using the covariance matrix's eigenvalue distribution. It captures:
1. The effective sample size (ratio of geometric to arithmetic mean of eigenvalues) - detects when population is concentrated in few eigen-directions
2. The condition number - detects extreme ill-conditioning
3. Mean distance from centroid - captures overall spread

This is fundamentally different because it analyzes the **shape** of the population distribution, not just its marginal spreads. When eigenvalues are highly skewed (few large, many small), the population is exploring only a small subspace, triggering needed restarts.

```python
def _compute_diversity(self):
    """Compute diversity using covariance eigenvalue shape (effective sample size)."""
    if self.population.shape[0] < 2:
        return 0.0
    
    # Signal 1: Mean distance from centroid (overall spread)
    centroid = np.mean(self.population, axis=0)
    dists = np.linalg.norm(self.population - centroid, axis=1)
    mean_dist = np.mean(dists)
    
    # Signal 2: Effective sample size from eigenvalue distribution
    # ESS = geometric_mean(eigenvalues) / arithmetic_mean(eigenvalues)
    # ESS = 1 when all eigenvalues equal (good), ESS -> 0 when skewed (bad)
    eigvals = np.linalg.eigvalsh(self.C)
    eigvals = np.maximum(eigvals, 1e-15)
    geometric_mean = np.exp(np.mean(np.log(eigvals)))
    arithmetic_mean = np.mean(eigvals)
    ess = geometric_mean / (arithmetic_mean + 1e-15)
    
    # Signal 3: Condition number (bounded for stability)
    cond = np.min(eigvals) / (np.max(eigvals) + 1e-15)
    cond_score = 1.0 / (1.0 + np.log1p(max(cond, 1.0)))
    
    # Combine signals: shape-aware diversity
    expected_range = (self.ub[0] - self.lb[0]) / 6.0
    spread_normalized = mean_dist / (expected_range + 1e-10)
    
    # Weight by ESS and condition score for shape-awareness
    diversity = spread_normalized * (0.4 + 0.4 * ess + 0.2 * cond_score)
    
    return float(np.clip(diversity, 0.0, 1e6))
```