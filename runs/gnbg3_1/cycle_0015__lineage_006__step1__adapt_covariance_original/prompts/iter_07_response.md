**Idea: Eigenvalue Floor with Stagnation Detection**
A fundamentally different covariance adaptation that prevents eigenvalue collapse by maintaining a minimum eigenvalue floor and explicitly detecting and boosting exploration along stagnated eigendirections. This directly targets the multi-modal trap causing errors in the 1e+1 range.

```python
def _adapt_covariance_original(self):
    """Eigenvalue floor with stagnation detection for escaping local optima."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(self.C)
    eigvals = np.maximum(eigvals, 1e-12)
    
    # Detect stagnation: mean movement in eigenspace
    y_mean_scaled = y_mean @ eigvecs
    progress_per_dir = np.abs(y_mean_scaled)
    
    # Eigenvalue floor to prevent collapse
    min_eig = 1e-6 * np.max(eigvals)
    eigvals_clamped = np.maximum(eigvals, min_eig)
    
    # Boost stagnated eigendirections
    progress_threshold = 1e-4 * np.linalg.norm(y_mean_scaled) + 1e-10
    stagnation_mask = progress_per_dir < progress_threshold
    
    if np.any(stagnation_mask):
        boost_factor = 5.0
        eigvals_boosted = eigvals_clamped.copy()
        eigvals_boosted[stagnation_mask] *= boost_factor
        eigvals_clamped = eigvals_boosted
    
    # Reconstruct covariance from modified eigenvalues
    self.C = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T
    self.C = 0.5 * (self.C + self.C.T)
    
    # Standard evolution path update
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Blend with standard rank-mu update
    blend_weight = 0.7
    self.C = ((1.0 - blend_weight * self.ccov) * self.C + 
              blend_weight * self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
    
    self.C = self._ensure_positive_definite(self.C)
```