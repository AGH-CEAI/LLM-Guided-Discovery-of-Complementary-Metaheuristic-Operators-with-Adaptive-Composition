**Idea: Eigenvalue-Ratio Spectral Modulation with Condition-Number Trigger**

Category B: Spectral / linear-algebraic approach using eigenvalue ratios to modulate velocity components and condition-number-triggered exploration.

```python
def _position_update_centroid_knn(self):
    """Eigenvalue-ratio spectral modulation with condition-number exploration (Category B).
    
    Uses eigenvalue ratios (relative to geometric mean) as spectral weights.
    High condition number triggers uniform exploration bonus.
    Distinct from SVD whitening (which uses absolute singular values).
    Targets worst tasks 17, 16, 6 with severe anisotropy.
    """
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        cond = np.clip(cond, 1.0, 1e6)
        
        # Eigenvalue ratio relative to geometric mean (different from SVD whitening)
        log_ratios = np.log(eigenvalues / (np.exp(np.mean(np.log(eigenvalues))) + 1e-10) + 1e-10)
        spectral_weights = np.exp(-0.3 * log_ratios)
        spectral_weights = np.clip(spectral_weights, 0.1, 10.0)
        
        # Condition-number-triggered exploration (distinct from eigenvalue ratio)
        if cond > 10.0:
            explore_bonus = 1.0 + 0.5 * np.log1p(cond) / np.log1p(1000.0)
        else:
            explore_bonus = 1.0
        
        vel_proj = self.velocity @ eigenvectors
        vel_scaled = vel_proj * spectral_weights * explore_bonus
        new_population = self.population + vel_scaled @ eigenvectors.T
        
    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity
    
    self.population = self._clip_to_bounds(new_population)
```