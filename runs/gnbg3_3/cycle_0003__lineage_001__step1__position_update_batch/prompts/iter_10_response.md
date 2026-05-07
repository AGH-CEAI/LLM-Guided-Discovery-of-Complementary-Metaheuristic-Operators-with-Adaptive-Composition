**Idea: SVD Velocity Subspace Anisotropic Scaling**

Use SVD of the velocity matrix to analyze its effective rank and singular value distribution, then apply per-component anisotropic scaling based on singular value ratios and condition number. This differs from the previous Category B approach (eigendecomposition of covariance with single uniform scale) by treating the velocity's own spectral structure and modulating each principal direction differently.

```python
def _position_update_batch(self):
    """Update positions using SVD-based velocity subspace analysis and anisotropic scaling."""
    try:
        centered_pop = self.population - np.mean(self.population, axis=0)
        
        # SVD of velocity to analyze its spectral structure
        U, singular_values, Vt = np.linalg.svd(self.velocity, full_matrices=False)
        
        # Clip singular values for numerical stability
        singular_values = np.clip(singular_values, 1e-10, None)
        
        # Condition number and effective dimensionality of velocity
        if singular_values[-1] > 0:
            cond = singular_values[0] / singular_values[-1]
        else:
            cond = 1.0
        
        # Effective rank: number of singular values needed to explain 95% of velocity variance
        total_var = np.sum(singular_values ** 2) + 1e-10
        cumvar = np.cumsum(singular_values ** 2) / total_var
        effective_rank = np.searchsorted(cumvar, 0.95) + 1
        
        # Log-scaled condition factor for smooth adaptation
        cond_factor = np.log1p(cond) / (np.log1p(cond) + 1.0)
        
        # Per-component scaling: high-order modes get boosted when velocity is ill-conditioned
        # This helps escape stagnation in collapsed subspaces
        n_components = len(singular_values)
        scale_per_component = np.ones(n_components)
        
        if n_components > 1:
            # Ratio of each singular value to the first
            sv_ratios = singular_values / (singular_values[0] + 1e-10)
            
            # Low-ratio components (minor modes) get amplified when condition is high
            # This helps escape narrow/collapsed velocity subspaces
            amplification = 1.0 + cond_factor * (1.0 - sv_ratios) * 2.0
            scale_per_component = np.clip(amplification, 0.5, 3.0)
        
        # Project velocity onto principal components and scale
        vel_proj = self.velocity @ Vt.T  # Project onto right singular vectors
        vel_scaled = vel_proj * scale_per_component
        
        # SVD-based position update with modulated velocity
        new_population = self.population + (vel_scaled @ Vt)
        
        # Also add a small perturbation along minor singular directions when ill-conditioned
        if cond > 100.0 and effective_rank < self.dim * 0.5:
            # Exploit minor singular directions for exploration
            minor_explore = np.random.randn(self.np, self.dim) * 0.1 * cond_factor
            minor_explore = minor_explore @ Vt  # Align exploration to velocity subspace
            new_population = new_population + minor_explore
        
    except np.linalg.LinAlgError:
        # Fallback: simple velocity addition
        new_population = self.population + self.velocity
    
    self.population = self._clip_to_bounds(new_population)
```