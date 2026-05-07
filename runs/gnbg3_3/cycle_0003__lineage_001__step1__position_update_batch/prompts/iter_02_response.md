**Idea: SVD-Whitened Velocity Decomposition**
Use singular value decomposition to whiten the population space and apply component-wise spectral modulation that specifically counteracts anisotropy in ill-conditioned tasks.

```python
def _position_update_batch(self):
    """Update positions via SVD-based population whitening with anisotropic damping."""
    centered = self.population - np.mean(self.population, axis=0)
    
    try:
        U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # Guard against degenerate population
        singular_values = np.clip(singular_values, 1e-10, None)
        
        # Compute condition number and total variance
        cond = singular_values[0] / (singular_values[-1] + 1e-10)
        total_var = np.sum(singular_values ** 2) + 1e-10
        
        # Normalize singular values to [0, 1] for per-component scaling
        sv_norm = singular_values / (singular_values[0] + 1e-10)
        
        # Compute spectral modulation: directions with LARGE spread get DAMPENED,
        # directions with SMALL spread get AMPLIFIED (counteract anisotropy)
        # Using inverse-square-law for stronger effect on collapsed directions
        spectral_modulation = 1.0 / (sv_norm ** 2 + 0.01)
        
        # Normalize to prevent explosion
        spectral_modulation = spectral_modulation / (np.max(spectral_modulation) + 1e-10)
        
        # Blend with uniform scaling based on condition number
        # High condition number -> more aggressive per-component modulation
        blend_weight = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
        uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
        per_component_scale = uniform_scale * (1.0 - blend_weight) + spectral_modulation * blend_weight
        
        # Project velocity onto principal axes
        vel_proj = self.velocity @ Vt.T
        
        # Apply component-wise spectral scaling
        vel_scaled = vel_proj * per_component_scale
        
        # Reconstruct and add to population
        new_population = self.population + (vel_scaled @ Vt)
        
    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity
    
    self.population = self._clip_to_bounds(new_population)
```