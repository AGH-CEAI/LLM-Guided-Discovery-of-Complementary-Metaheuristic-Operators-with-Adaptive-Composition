**Idea: Entropy-Modulated Velocity with Distribution Matching**

Use population entropy (from covariance matrix) to modulate velocity magnitude, combined with distribution matching toward a reference Gaussian. Low entropy triggers exploration boost; high entropy enables exploitation. This is distinctly information-theoretic because it treats the swarm as a probability distribution and uses entropy as the control signal.

```python
def _velocity_update_base(self):
    """Information-theoretic velocity update using population entropy and distribution matching.
    
    Category C: Treats population as probability distribution, computes entropy from
    covariance, and uses entropy signal to modulate velocity magnitude.
    
    Key insight: Low population entropy = concentrated/bunched = premature convergence
    → boost exploration. High entropy = diverse spread → can exploit more.
    
    Also uses KL-like distribution matching toward a well-conditioned reference.
    """
    cognitive, social = self._adaptive_coefficients()
    
    # === ENTROPY COMPUTATION FROM POPULATION DISTRIBUTION ===
    centered = self.population - np.mean(self.population, axis=0)
    
    try:
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        
        # Gaussian entropy: H = 0.5 * log((2*pi*e)^d * det(cov))
        # Higher det(cov) = more spread = higher entropy
        dim = self.dim
        var_range = (self.upper_bound - self.lower_bound) ** 2
        det_cov = np.prod(eigenvalues) + 1e-10
        
        entropy = 0.5 * (dim * np.log(2 * np.pi * np.e * var_range / dim) + np.log(det_cov))
        
        # Normalize entropy to [0, 1] using theoretical bounds
        max_entropy = 0.5 * dim * np.log(2 * np.pi * np.e * var_range / dim)
        min_entropy = 0.5 * dim * np.log(2 * np.pi * np.e * 1e-4)
        normalized_entropy = np.clip((entropy - min_entropy) / (max_entropy - min_entropy + 1e-10), 0.0, 1.0)
        
        # Also compute condition number as anisotropy signal
        cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        anisotropy = np.clip(np.log1p(cond) / np.log1p(1e4), 0.0, 1.0)
        
    except:
        normalized_entropy = 0.5
        anisotropy = 0.5
    
    # === ENTROPY-BASED EXPLORATION/EXPLOITATION BALANCE ===
    # Low entropy -> concentrated population -> boost exploration
    # High entropy -> diverse population -> can exploit more
    exploration_boost = 1.0 + 0.6 * (1.0 - normalized_entropy)
    exploitation_damp = 1.0 - 0.3 * normalized_entropy
    
    # === FITNESS ENTROPY (distribution of fitness values) ===
    try:
        # Compute entropy of fitness distribution (histogram-based)
        hist, _ = np.histogram(self.current_fitness, bins=min(10, self.np), density=True)
        hist = hist + 1e-10
        hist = hist / np.sum(hist)
        fitness_entropy = -np.sum(hist * np.log(hist + 1e-10))
        max_fitness_entropy = np.log(min(10, self.np))
        normalized_fitness_entropy = np.clip(fitness_entropy / (max_fitness_entropy + 1e-10), 0.0, 1.0)
    except:
        normalized_fitness_entropy = 0.5
    
    # === COMBINED INFORMATION-THEORETIC SIGNAL ===
    combined_entropy = 0.6 * normalized_entropy + 0.4 * normalized_fitness_entropy
    
    # === KL-LIKE DISTRIBUTION MATCHING ===
    # Compute deviation from well-conditioned reference (identity-scaled covariance)
    # This provides a "pull" toward better-conditioned configurations
    try:
        eigenvalues_normalized = eigenvalues / (eigenvalues[0] + 1e-10)
        kl_like = np.mean(np.log(eigenvalues_normalized + 1e-10))
        kl_modulation = 1.0 + 0.2 * kl_like  # Positive kl_like = elongated -> reduce
        kl_modulation = np.clip(kl_modulation, 0.5, 1.5)
    except:
        kl_modulation = 1.0
    
    # === RANDOM COEFFICIENTS ===
    r1 = np.random.uniform(0, 1, (self.np, self.dim))
    r2 = np.random.uniform(0, 1, (self.np, self.dim))
    
    # === VELOCITY COMPONENTS ===
    cognitive_component = cognitive * r1 * (self.personal_best - self.population)
    social_component = social * r2 * (self.local_best - self.population)
    
    # === DE MUTATION COMPONENT ===
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
    
    # === COMBINE WITH INFORMATION-THEORETIC MODULATION ===
    # Scale cognitive/social by exploration boost and KL modulation
    # Scale mutation by entropy (more entropy = less need for mutation exploration)
    mutation_scale = 1.0 - 0.3 * combined_entropy
    
    new_velocity = (
        self.inertia_weight * self.velocity +
        exploration_boost * cognitive_component +
        exploitation_damp * social_component +
        kl_modulation * mutation_scale * mutation_component
    )
    
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```