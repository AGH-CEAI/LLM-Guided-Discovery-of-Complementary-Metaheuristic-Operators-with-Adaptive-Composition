**Idea: Eigenvalue-Decomposed Velocity Scaling**

Modulate velocity along principal components using eigendecomposition of population covariance. Per-direction scaling inversely proportional to eigenvalue magnitude (dampen high-variance directions, amplify collapsed directions). Use condition number to adapt velocity bounds and entropy-based damping for exploration/exploitation balance.

```python
def _velocity_update_base(self):
    """Spectral velocity update: eigendecomposition-based per-direction scaling."""
    cognitive, social = self._adaptive_coefficients()
    
    r1 = np.random.uniform(0, 1, (self.np, self.dim))
    r2 = np.random.uniform(0, 1, (self.np, self.dim))
    
    cognitive_component = cognitive * r1 * (self.personal_best - self.population)
    social_component = social * r2 * (self.local_best - self.population)
    
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
    
    # --- SPECTRAL VELOCITY MODULATION ---
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        cond = eigenvalues[0] / eigenvalues[-1]
        total_var = np.sum(eigenvalues) + 1e-10
        eigenvalues_norm = eigenvalues / total_var
        
        cumvar = np.cumsum(eigenvalues_norm)
        eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
        eff_dim_ratio = np.clip(eff_dim / self.dim, 0.0, 1.0)
        
        spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
        max_entropy = np.log(self.dim + 1e-10)
        entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)
        
        sv_scale = np.sqrt(eigenvalues)
        sv_norm = sv_scale / (sv_scale[0] + 1e-10)
        
        per_component_scale = 1.0 / (sv_norm ** 0.5 + 0.1)
        per_component_scale = per_component_scale / (np.max(per_component_scale) + 1e-10)
        
        log_cond = np.log1p(cond) / np.log1p(1000.0)
        log_cond = np.clip(log_cond, 0.0, 1.0)
        
        spectral_damp = 0.5 + 0.3 * (1.0 - entropy_ratio) + 0.2 * log_cond
        per_component_scale = per_component_scale * spectral_damp
        
        vel_proj = self.velocity @ eigenvectors
        vel_scaled = vel_proj * per_component_scale
        spectral_velocity = vel_scaled @ eigenvectors.T
    except np.linalg.LinAlgError:
        spectral_velocity = self.velocity
    
    new_velocity = (
        self.inertia_weight * spectral_velocity +
        cognitive_component +
        social_component +
        mutation_component
    )
    
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```