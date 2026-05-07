**Idea: Eigenvalue-Weighted Velocity Projection**
Category B: Uses eigenvalue decomposition of population covariance to modulate velocity along principal axes — particles explore more aggressively along collapsed (low-variance) directions to counteract anisotropy and escape narrow basins.

```python
def _position_update_fitness_rank(self):
    """Spectral / linear-algebraic: eigenvalue-weighted velocity projection (variant_10).
    
    Key insight: project velocity onto principal components of population
    covariance, then scale each component INVERSELY to its eigenvalue.
    High-eigenvalue directions get dampened (already well-explored),
    low-eigenvalue directions get amplified (collapsed/neglected subspace).
    Condition number drives global exploration factor.
    """
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)
        
        cond = eigenvalues[-1] / (eigenvalues[0] + 1e-10)
        cond = np.clip(cond, 1.0, 10000.0)
        
        total_var = np.sum(eigenvalues)
        if total_var > 1e-10:
            sorted_desc = eigenvalues[::-1]
            cumvar = np.cumsum(sorted_desc) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
        else:
            eff_dim = self.dim
        
        spectral_signal = np.clip((cond - 10.0) / 90.0, 0.0, 1.0)
        exploration_scale = 1.0 + 0.7 * spectral_signal
        
        max_eig = eigenvalues[-1] + 1e-10
        inv_eig_weights = max_eig / (eigenvalues + 1e-10)
        inv_eig_weights = inv_eig_weights / (np.max(inv_eig_weights) + 1e-10)
        
        vel_proj = self.velocity @ eigenvectors
        vel_scaled = vel_proj * inv_eig_weights
        
        new_population = self.population + exploration_scale * (vel_scaled @ eigenvectors.T)
        
    except np.linalg.LinAlgError:
        new_population = self.population + 1.2 * self.velocity
    
    self.population = self._clip_to_bounds(new_population)
```