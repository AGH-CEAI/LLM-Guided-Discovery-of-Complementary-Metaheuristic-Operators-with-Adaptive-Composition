**Idea: CMA-ES Cumulative Path Adaptation**
Uses the evolution path length (ps) to adapt step-size, following the classic CMA-ES Cumulative Step-size Adaptation (CSA) approach. This is fundamentally different from counting immediate successes—it cumulates step information over time and compares the path length to its expected value under random selection.
```python
def _adapt_step_size(self):
    """Adapt step-size using CMA-ES Cumulative Step-size Adaptation (CSA).
    
    This approach uses an evolution path ps that accumulates successful steps.
    If ||ps|| is larger than expected under random selection, sigma increases;
    if smaller, sigma decreases. This provides smoother, more robust adaptation
    than immediate success counting.
    """
    # Compute mean shift in sigma-scaled coordinates
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Update evolution path ps (cumulative conjugate gradient direction)
    # This path should behave like a random walk under success
    self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * y_mean
    
    # Expected length of ps under random normal distribution (chi_d)
    # chi_d ≈ sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim^2))
    chi_d = self.dim ** 0.5 * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim ** 2))
    
    # Compute norm of evolution path
    ps_norm = np.linalg.norm(self.ps)
    
    # Avoid division by zero or extreme values
    ps_norm_safe = max(ps_norm, 1e-10)
    chi_d_safe = max(chi_d, 1e-10)
    
    # Step-size adaptation: if path is long, increase sigma; if short, decrease
    adaptation = (ps_norm_safe / chi_d_safe) - 1.0
    adaptation = np.clip(adaptation, -1.0, 1.0)
    
    # Apply exponential update with damping
    self.sigma *= np.exp((self.cs / self.damping) * adaptation)
    self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```