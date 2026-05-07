**Idea: Eigenvalue-Anisotropic Velocity Modulation**

Leverage eigendecomposition of the population covariance to modulate velocity updates anisotropically along principal variance axes, scaling movement inversely to local spread to counteract premature convergence on the worst-performing tasks (5, 17, 16 with errors >1e+03).

```python
def _position_update_batch(self):
    """Update positions with eigenvalue-anisotropic velocity modulation."""
    # Compute population covariance for spectral analysis
    centered = self.population - np.mean(self.population, axis=0)
    cov = np.cov(centered.T)
    
    # Eigendecomposition for spectral scaling
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort by descending eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute condition-number-based scaling (stuck populations have high condition number)
        cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        spectral_scale = np.clip(cond / 100.0, 0.5, 2.0)
        
        # Project velocity onto principal axes and scale
        vel_proj = self.velocity @ eigenvectors
        vel_scaled = vel_proj * spectral_scale
        
        # Transform back to original space
        new_population = self.population + (vel_scaled @ eigenvectors.T)
    except np.linalg.LinAlgError:
        # Fallback to standard update on decomposition failure
        new_population = self.population + self.velocity
    
    self.population = self._clip_to_bounds(new_population)
```