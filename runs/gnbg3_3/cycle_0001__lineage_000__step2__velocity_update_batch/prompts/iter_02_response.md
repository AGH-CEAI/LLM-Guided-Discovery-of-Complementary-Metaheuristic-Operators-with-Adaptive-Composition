**Idea: Spectral Velocity Scaling with Condition-Number-Guided Mutation**

Use eigendecomposition of population covariance to scale velocity components along principal axes, and inject mutations aligned with the minor eigenvector when the condition number indicates degeneracy (stagnation trap). This gives a fundamentally different angle on the worst tasks (errors 10^2-10^4) by directly detecting and escaping the collapsed-subspace failure mode.

```python
def _velocity_update_batch(self):
    """Update velocities using spectral analysis of population covariance."""
    cognitive, social = self._adaptive_coefficients()
    
    # Generate random matrices once for efficiency
    r1 = np.random.uniform(0, 1, (self.np, self.dim))
    r2 = np.random.uniform(0, 1, (self.np, self.dim))
    
    # Cognitive component: attraction to personal best
    cognitive_component = cognitive * r1 * (self.personal_best - self.population)
    
    # Social component: attraction to local best (ring topology)
    social_component = social * r2 * (self.local_best - self.population)
    
    # Spectral analysis of population for anisotropic velocity scaling
    centered = self.population - np.mean(self.population, axis=0)
    cov = np.cov(centered.T)
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Condition number for stagnation detection (degenerate subspace = high cond)
        cond = eigenvalues[0] / max(eigenvalues[-1], 1e-10)
        
        # Inject spectral mutation when population is stuck in degenerate subspace
        if cond > 50.0:
            # Mutation along minor eigenvector (least explored direction)
            minor_dir = eigenvectors[:, -1]
            mutation_scale = 2.0 * min(cond / 100.0, 3.0)
            mutation_scale *= (1.0 - self.generation / 5000)  # Decreases over time
            spectral_mutation = mutation_scale * self.v_max * minor_dir
        else:
            spectral_mutation = np.zeros(self.dim)
        
        # Project velocity onto principal axes and scale inversely to eigenvalues
        # (more velocity along less-explored minor axes)
        vel_proj = self.velocity @ eigenvectors
        spectral_scaling = np.sqrt(eigenvalues[-1] / (eigenvalues + 1e-10))
        vel_scaled = vel_proj * spectral_scaling
        
        # Transform back to original space
        spectral_component = vel_scaled @ eigenvectors.T
        
        # Velocity update with inertia, cognitive, social, spectral scaling, and mutation
        new_velocity = (
            self.inertia_weight * spectral_component +
            cognitive_component +
            social_component +
            spectral_mutation
        )
    except np.linalg.LinAlgError:
        # Fallback to standard velocity update on decomposition failure
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component
        )
    
    # Clip velocity to bounds
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```