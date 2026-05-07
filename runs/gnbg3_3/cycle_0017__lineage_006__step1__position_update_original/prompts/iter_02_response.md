**Idea: Effective-Dimensionality Modulation**
Use the effective dimensionality (count of eigenvalues needed to explain 95% variance) as a continuous modulation signal, combined with per-particle eigenspace projection and inverse-eigenvalue-weighted velocity scaling. This targets the anisotropy of the worst unsolved tasks (17, 5, 16, 6) through a fundamentally different spectral mechanism than the condition-number scalar used by the original.

```python
def _position_update_original(self):
    """Category B: Effective-dimensionality-modulated eigenspace velocity scaling."""
    centered = self.population - np.mean(self.population, axis=0)
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        eigenvalues = np.clip(eigenvalues, 1e-12, None)
        total_var = np.sum(eigenvalues)
        
        # Effective dimensionality: how many components explain 95% of variance
        cumvar = np.cumsum(eigenvalues) / (total_var + 1e-12)
        eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
        eff_dim_ratio = eff_dim / self.dim
        
        # Per-component inverse-variance scaling (key spectral mechanism)
        # Small eigenvalue → collapsed direction → amplify (escape collapse)
        # Large eigenvalue → expanded direction → dampen (avoid overshoot)
        var_normalized = eigenvalues / (eigenvalues[0] + 1e-12)
        inv_var_scale = 1.0 / (var_normalized + 0.05)
        inv_var_scale = inv_var_scale / (np.max(inv_var_scale) + 1e-12)
        
        # Blend: exploration mode when eff_dim is low (swarm collapsed into subspace)
        # exploitation mode when eff_dim is high (swarm spans full space)
        blend_weight = np.clip((1.0 - eff_dim_ratio) * 2.0, 0.0, 1.0)
        
        uniform_scale = np.clip(eff_dim_ratio * 1.5 + 0.5, 0.4, 2.0)
        spectral_scale = uniform_scale * (1.0 - blend_weight) + inv_var_scale * blend_weight
        
        # Project velocity to eigenspace, scale, reconstruct
        vel_proj = self.velocity @ eigenvectors
        vel_scaled = vel_proj * spectral_scale
        
        new_population = self.population + (vel_scaled @ eigenvectors.T)
    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity
    
    self.population = self._clip_to_bounds(new_population)
```