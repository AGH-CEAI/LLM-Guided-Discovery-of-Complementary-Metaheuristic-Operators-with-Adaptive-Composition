**Idea: Evolution Path Momentum Adaptation**

Use CMA-ES-style cumulative evolution path tracking for step-size adaptation instead of simple success counting. This provides momentum-based, noise-robust step-size control that can escape local optima and maintain search direction over many generations.

```python
def _adapt_step_size(self):
    """Adapt step-size using CMA-ES-style cumulative evolution path with momentum."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Track cumulative evolution path (momentum of mean movement)
    if not hasattr(self, 'ps_path'):
        self.ps_path = np.zeros(self.dim)
    
    # Evolution path update with cumulation
    self.ps_path = (1.0 - self.cs) * self.ps_path + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean
    
    # Path length as measure of step-size adequacy
    path_norm = np.linalg.norm(self.ps_path)
    
    # Expected path length under random selection (approx sqrt(dim))
    expected_path = np.sqrt(self.dim) * (1.0 - (1.0 - self.cs) ** (2.0 * self.generation + 1))
    expected_path = max(expected_path, 0.1)
    
    # Success measure from path length ratio
    success_ratio = path_norm / expected_path if expected_path > 1e-15 else 1.0
    success_ratio = np.clip(success_ratio, 0.01, 10.0)
    
    # Logarithmic adaptation for multiplicative step-size control
    adaptation = np.log(success_ratio) / np.log(2.0)
    adaptation = np.clip(adaptation, -0.8, 0.8)
    
    # Adaptive damping based on dimension and path length
    damping_adaptive = self.damping * (0.3 + 0.7 * np.log1p(self.dim) / np.log1p(path_norm + 1e-10))
    damping_adaptive = np.clip(damping_adaptive, 0.05, 200.0)
    
    # Exponential step-size update with momentum
    self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
    self.sigma = np.clip(self.sigma, 1e-12, 20.0)
```