Looking at the error magnitudes (10^1 to 10^2 range for worst tasks), the optimizer is getting **trapped in poor local optima** due to the covariance matrix becoming highly degenerate. The condition number of C grows too large, making exploration along certain axes extremely difficult.

**Idea: Restart Covariance on Degeneracy**
When the condition number exceeds a dimension-scaled threshold (e.g., 1000×dim), the covariance has become too elongated for effective exploration. This variant detects degeneracy and reinitializes C to isotropic, forcing fresh exploration to escape local optima. This is fundamentally different from previous variants (which all keep C intact) by explicitly breaking the covariance collapse that causes entrapment.

```python
def _sample_trials_batch(self):
    """Sample with covariance restart on degeneracy detection."""
    eigvals, eigvecs = np.linalg.eigh(self.C)
    
    min_eig = np.min(eigvals)
    max_eig = np.max(eigvals)
    cond_num = max_eig / max(min_eig, 1e-30)
    
    # Restart covariance to isotropic if severely degenerate
    if cond_num > 1000.0 * self.dim:
        self.C = self.sigma * np.eye(self.dim)
        eigvals, eigvecs = np.linalg.eigh(self.C)
    
    eigvals = np.maximum(eigvals, 1e-10)
    L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    
    z = np.random.randn(self.NP, self.dim)
    self.trials = self.mean + self.sigma * (z @ L.T)
    self.trials = self._clip_to_bounds(self.trials)
```