**Idea: Evolution Path Step-Size Control**

This approach mirrors CMA-ES's path length control mechanism. Instead of counting discrete successes, it tracks the cumulative evolution path `p_sigma` — the running sum of normalized mean displacements. When the path length deviates from the expected 1/sqrt(dim) under random sampling, sigma is adjusted accordingly. This is fundamentally different from success-rate counting: it captures directional momentum and can detect stagnation or erratic oscillation, enabling more aggressive exploration when the algorithm gets stuck in local optima (the primary failure mode for the worst tasks with errors >10).

```python
def _adapt_step_size(self):
    """Adapt step-size using evolution path length control (CMA-ES style)."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Evolution path for step-size: cumulative sum of normalized mean steps
    if not hasattr(self, 'p_sigma') or self.p_sigma is None:
        self.p_sigma = np.zeros(self.dim)
    
    self.p_sigma = (1.0 - self.cs) * self.p_sigma + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean
    
    path_norm = np.linalg.norm(self.p_sigma)
    expected_norm = np.sqrt(self.dim) * (1.0 - (1.0 - self.cs) ** (2.0 * max(1, self.generation)))
    expected_norm = max(expected_norm, 0.1)
    
    # Control signal: path too short -> increase sigma, path too long -> decrease
    control = path_norm / expected_norm - 1.0
    control = np.clip(control, -0.5, 0.5)
    
    # Adaptive damping based on dimension and stagnation
    stagnation_boost = 1.0 + 0.5 * min(self.stagnation_counter / max(1, self.max_stagnation), 1.0)
    damping_eff = self.damping * stagnation_boost
    damping_eff = np.clip(damping_eff, 0.1, 200.0)
    
    # Exponential update with controlled damping
    self.sigma *= np.exp(control * self.cs / damping_eff)
    self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```