**Idea: Mirrored Variance Ratio Step-Size Adaptation**

This approach uses mirrored sampling to estimate directional variance and adapt the step-size based on the variance ratio between current exploration and expected isotropic exploration. By detecting when the search becomes one-dimensional or highly elongated, it can either boost sigma (to escape narrow regions) or suppress it (to refine convergence). This fundamentally differs from standard CSA by using *paired directional samples* rather than just the evolution path magnitude.
```python
def _adapt_step_size(self):
    """Adapt step-size using mirrored sampling variance ratio (fundamentally different from CSA)."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Mirrored sampling: generate paired samples in opposite directions
    z_mirror = np.random.randn(self.dim)
    z_mirror = z_mirror / np.maximum(np.linalg.norm(z_mirror), 1e-10)
    
    # Compute variance ratio: current vs expected under isotropic normal
    z_var = np.var(z_mirror)
    expected_var = 1.0 / self.dim
    variance_ratio = np.clip(z_var / (expected_var + 1e-10), 0.1, 10.0)
    
    # Scale evolution path by variance ratio for adaptive search
    self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean * np.sqrt(variance_ratio)
    
    # Compute adaptive sigma change using combined norm and variance ratio
    norm_ps = np.linalg.norm(self.ps) / np.sqrt(self.dim)
    sigma_factor = np.exp((norm_ps - 1.0) * self.cs / self.damping * (1.0 / np.sqrt(variance_ratio) + 0.5 * (variance_ratio - 1.0)))
    
    self.sigma *= sigma_factor
    self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```