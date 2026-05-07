**Idea: Eigenvalue-Aware Anisotropic Covariance Stretching**

This variant detects when the covariance matrix becomes ill-conditioned (narrowing along certain directions) and actively stretches the covariance along the smallest eigendirections to prevent distribution collapse and escape local optima. Unlike variant_08 which triggers on stagnation counter, this uses eigenvalue spread as the primary detection signal.

```python
def _adapt_covariance_variant_09(self):
    """Eigenvalue-aware anisotropic stretching for escaping ill-conditioned traps."""
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
    
    # Eigenvalue-aware anisotropic stretching
    eigvals, eigvecs = np.linalg.eigh(self.C)
    eigvals = np.maximum(eigvals, 1e-15)
    eig_max = np.max(eigvals)
    eig_min = np.min(eigvals)
    cond = eig_max / eig_min
    
    # Detect ill-conditioned covariance (collapsed along some directions)
    stretch_threshold = max(1e5, self.dim * 100)
    needs_stretch = cond > stretch_threshold
    
    # Also check for eigenvalue starvation (smallest eigenvalue too small)
    starvation_threshold = 1e-10 * eig_max
    needs_stretch = needs_stretch or (eig_min < starvation_threshold and cond > 1e3)
    
    if needs_stretch:
        # Compute stretch factors inversely proportional to eigenvalue magnitude
        # Small eigenvalues get large stretch, large eigenvalues get small stretch
        stretch_factors = np.ones(self.dim)
        for j in range(self.dim):
            stretch_factors[j] = max(1.0, 5.0 * (eig_max / (eigvals[j] + 1e-15)) ** 0.5)
        
        # Apply anisotropic perturbation along each eigenvector
        stretch_perturb = np.zeros((self.dim, self.dim))
        for j in range(self.dim):
            v_j = eigvecs[:, j]
            stretch_perturb += (stretch_factors[j] - 1.0) * np.outer(v_j, v_j)
        
        self.C = self.C + 0.3 * stretch_perturb
        
        # Renormalize to prevent runaway growth
        new_eigvals = np.linalg.eigvalsh(self.C)
        if np.max(new_eigvals) > 10.0 * eig_max:
            self.C *= 0.5
        
        # Reset evolution path to prevent interference
        self.pc *= 0.5
        
        self.C = self._ensure_positive_definite(self.C)
```