**Idea: Stagnation-Triggered Large Jump Escape**

When the optimizer is stuck (stagnation), inject a LARGE directional perturbation to the mean and reset the covariance matrix to escape local optima. This is fundamentally different from scaling learning rates — it actively forces exploration in a new region.

```python
def _adapt_covariance_variant_06(self):
    """Stagnation-triggered escape with large perturbation and covariance reset."""
    
    # Detect stagnation earlier (shorter threshold than default)
    stagnation_threshold = max(15, self.dim)
    is_stagnant = self.stagnation_counter > stagnation_threshold
    
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    self.C = ((1.0 - self.ccov) * self.C + 
              self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
    
    self.C = self._ensure_positive_definite(self.C)
    
    if is_stagnant:
        # Generate random escape direction
        escape_dir = np.random.randn(self.dim)
        escape_dir /= (np.linalg.norm(escape_dir) + 1e-10)
        
        # Compute perturbation magnitude based on step size and covariance scale
        eigvals = np.linalg.eigvalsh(self.C)
        cov_scale = np.sqrt(np.max(eigvals) + 1e-10)
        jump_magnitude = 15.0 * self.sigma * cov_scale
        
        # Apply large perturbation to mean
        self.mean = self.mean + jump_magnitude * escape_dir
        
        # Reset covariance to identity (wider exploration)
        self.C = np.eye(self.dim)
        
        # Reset evolution path
        self.pc = np.zeros(self.dim)
        
        # Increase step size for wider search
        self.sigma *= 2.5
        
        # Reset stagnation counter
        self.stagnation_counter = 0
        
        # Ensure mean stays within bounds
        self.mean = self._clip_to_bounds(self.mean)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```