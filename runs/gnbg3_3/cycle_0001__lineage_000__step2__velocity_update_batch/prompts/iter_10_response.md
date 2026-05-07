**Idea: Spectral-Condition-Guided Anisotropic Mutation**

Uses the condition number of the population's singular value decomposition to adaptively scale and direct the mutation component. When the population collapses into a low-rank subspace (high condition number), the mutation strength is modulated to counteract anisotropy by injecting velocity orthogonal to the principal axes. This is fundamentally different from the current approach which ignores spectral structure in the velocity update.

```python
def _velocity_update_batch(self):
    """Update velocities with spectral-condition-guided anisotropic mutation."""
    cognitive, social = self._adaptive_coefficients()
    
    # Generate random matrices once for efficiency
    r1 = np.random.uniform(0, 1, (self.np, self.dim))
    r2 = np.random.uniform(0, 1, (self.np, self.dim))
    
    # Cognitive component: attraction to personal best
    cognitive_component = cognitive * r1 * (self.personal_best - self.population)
    
    # Social component: attraction to local best (ring topology)
    social_component = social * r2 * (self.local_best - self.population)
    
    # Compute spectral condition number from centered population
    centered = self.population - np.mean(self.population, axis=0)
    
    # Use SVD for numerical stability (avoids covariance matrix issues)
    try:
        _, singular_values, right_sv = np.linalg.svd(centered, full_matrices=False)
        
        # Condition number as ratio of max to min singular value
        # High cond => population in low-rank subspace => needs more exploration
        # Low cond => well-distributed across dimensions
        cond = singular_values[0] / (singular_values[-1] + 1e-10)
        
        # Modulate mutation strength inversely with condition number
        # When cond is high (population collapsed), reduce mutation to avoid over-exploitation
        # When cond is low (good spread), can use stronger mutation
        mutation_strength = np.clip(0.5 / (1.0 + 0.1 * np.log1p(cond)), 0.1, 0.5)
        
        # Determine effective dimensionality from singular values
        total_variance = np.sum(singular_values ** 2) + 1e-10
        cumvar = np.cumsum(singular_values ** 2) / total_variance
        effective_dim = np.searchsorted(cumvar, 0.95) + 1
        
        # When population is in low effective dimension, inject velocity
        # orthogonal to principal axes to break anisotropy
        if effective_dim < self.dim * 0.5:
            # Project population onto principal subspace and reconstruct residual
            principal_axes = right_sv[:effective_dim].T  # (dim, effective_dim)
            
            # Compute component orthogonal to principal subspace
            parallel_component = self.population @ principal_axes @ principal_axes.T
            orthogonal_residual = self.population - parallel_component
            
            # Add scaled orthogonal component as "spectral mutation"
            orthogonal_strength = 0.3 * (1.0 - effective_dim / self.dim)
            mutation_component = orthogonal_strength * orthogonal_residual
        else:
            mutation_component = np.zeros((self.np, self.dim))
        
        # DE mutation with spectral-modulated strength
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = mutation_strength * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold
        
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False
            )
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        de_component = np.where(
            mutation_active,
            mutation_strength * (mutation_vectors - self.population),
            0.0
        )
        
        # Combine spectral mutation and DE mutation
        total_mutation = mutation_component + de_component
        
    except np.linalg.LinAlgError:
        # Fallback to standard DE mutation on decomposition failure
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.3 * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold
        
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False
            )
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        total_mutation = np.where(
            mutation_active,
            0.3 * (mutation_vectors - self.population),
            0.0
        )
    
    # Velocity update with inertia, cognitive, social, and mutation
    new_velocity = (
        self.inertia_weight * self.velocity +
        cognitive_component +
        social_component +
        total_mutation
    )
    
    # Clip velocity to bounds
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```