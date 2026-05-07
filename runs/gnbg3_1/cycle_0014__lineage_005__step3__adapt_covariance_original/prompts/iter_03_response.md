**Idea: Mean-Stagnation Eigenscale Reset**

When the population mean barely moves (indicating the algorithm is stuck in a local optimum or the covariance is misaligned), this variant detects this condition and applies an exponential eigenscale perturbation to the covariance matrix — pushing exploration outward along principal axes to break free.

```python
def _adapt_covariance_original(self):
    """Mean-stagnation eigenscale reset: push exploration outward when stuck."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    # Detect mean stagnation: is the mean barely moving relative to step size?
    mean_displacement = np.linalg.norm(self.mean - self.old_mean)
    stagnation_threshold = 1e-3 * self.sigma
    
    if mean_displacement < stagnation_threshold and mean_displacement > 1e-15:
        # Mean is stagnant — apply exponential eigenscale perturbation to covariance
        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)
            
            # Exponential scaling: push outward along each principal axis
            # Stronger expansion along low-eigenvalue (narrow) directions
            log_eigvals = np.log(eigvals + 1e-10)
            log_scale = log_eigvals - np.min(log_eigvals) + 1.0
            scale_factors = np.power(log_scale, -1.5)
            scale_factors = np.clip(scale_factors, 0.1, 10.0)
            
            # Perturb eigvals: expand narrow directions, contract wide ones
            perturbed_eigvals = eigvals * scale_factors
            perturbed_eigvals = np.maximum(perturbed_eigvals, 1e-10)
            
            # Reconstruct perturbed covariance
            self.C = eigvecs @ np.diag(perturbed_eigvals) @ eigvecs.T
            
        except np.linalg.LinAlgError:
            pass
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    self.C = ((1.0 - self.ccov) * self.C + 
              self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
    
    self.C = self._ensure_positive_definite(self.C)
```