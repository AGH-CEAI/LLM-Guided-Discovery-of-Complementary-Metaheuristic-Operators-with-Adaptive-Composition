**Idea: Spectral Velocity Anisotropy Modulation**

Spectral / linear-algebraic approach: Compute eigendecomposition of population covariance and modulate velocity *before* position update. High condition number indicates collapsed/anisotropic population → dampen velocity in high-variance directions (already explored) and amplify in low-variance directions (underexplored). This directly counteracts the population collapse causing the worst errors on Tasks 17, 16, 6, 11.

```python
def _velocity_update_base(self):
    """Spectral velocity modulation: dampen high-variance directions, amplify low-variance."""
    cognitive, social = self._adaptive_coefficients()
    
    # --- SPECTRAL ANALYSIS ---
    centered = self.population - np.mean(self.population, axis=0)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        cond = eigenvalues[0] / eigenvalues[-1]
        
        # Per-direction modulation: inverse-variance weighting
        # High variance direction → population already covers it → dampen
        # Low variance direction → potential unexplored region → amplify
        total_var = np.sum(eigenvalues) + 1e-10
        var_ratio = eigenvalues / total_var
        
        # Modulation: sqrt inverse prevents over-amplification of near-zero eigenvalues
        spectral_scale = np.sqrt(var_ratio[-1] / (var_ratio + 1e-10))
        spectral_scale = np.clip(spectral_scale, 0.3, 3.0)
        
        # Condition-based blending: mild modulation for well-conditioned, stronger for ill-conditioned
        blend = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
        effective_scale = 1.0 * (1.0 - blend) + spectral_scale * blend
        
        # Project velocity onto eigenbasis, apply modulation, project back
        vel_proj = self.velocity @ eigenvectors
        vel_modulated = vel_proj * effective_scale
        modulated_velocity = vel_modulated @ eigenvectors.T
    except np.linalg.LinAlgError:
        modulated_velocity = self.velocity.copy()
    
    # --- COGNITIVE + SOCIAL ---
    r1 = np.random.uniform(0, 1, (self.np, self.dim))
    r2 = np.random.uniform(0, 1, (self.np, self.dim))
    cognitive_component = cognitive * r1 * (self.personal_best - self.population)
    social_component = social * r2 * (self.local_best - self.population)
    
    # --- DE MUTATION ---
    mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
    mutation_threshold = 0.1 * (1.0 - self.generation / 5000)
    mutation_active = mutation_mask < mutation_threshold
    
    mutation_vectors = np.zeros((self.np, self.dim))
    for i in range(self.np):
        indices = np.random.choice(
            [j for j in range(self.np) if j != i], 3, replace=False
        )
        mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
            self.population[indices[1]] - self.population[indices[2]]
        )
    
    mutation_component = np.where(
        mutation_active,
        0.3 * (mutation_vectors - self.population),
        0.0
    )
    
    # --- COMBINE WITH SPECTRAL MODULATION ---
    new_velocity = (
        self.inertia_weight * modulated_velocity +
        cognitive_component +
        social_component +
        mutation_component
    )
    
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```