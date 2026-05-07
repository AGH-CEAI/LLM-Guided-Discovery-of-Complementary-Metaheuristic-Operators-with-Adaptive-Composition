**Idea: Axis-Aligned Spread Modulation**
Modulate velocity components based on axis-aligned spread and centroid-relative geometry. Dimensions with high spread get dampened exploration, while collapsed dimensions get amplified movement. Particles far from centroid receive stronger social correction toward the swarm center.

```python
def _velocity_update_base(self):
    """Geometry-aware velocity update using axis-aligned spread and centroid geometry."""
    cognitive, social = self._adaptive_coefficients()
    
    # === GEOMETRIC COMPUTATIONS (Category A) ===
    centroid = np.mean(self.population, axis=0)
    
    # Axis-aligned spread (range) per dimension
    dim_min = np.min(self.population, axis=0)
    dim_max = np.max(self.population, axis=0)
    dim_range = dim_max - dim_min + 1e-10
    
    # Normalized spread per dimension: 0 = collapsed, 1 = well-spread
    max_range = np.max(dim_range)
    normalized_spread = np.clip(dim_range / (max_range + 1e-10), 0.01, 1.0)
    
    # Inverse spread: amplify velocity in collapsed dimensions
    spread_modulation = 1.0 / normalized_spread
    spread_modulation = np.clip(spread_modulation, 0.3, 3.0)
    spread_modulation = spread_modulation / (np.max(spread_modulation) + 1e-10)
    
    # Centroid-relative geometry: distance of each particle to centroid
    dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True) + 1e-10
    global_spread = np.mean(dist_to_centroid) + 1e-10
    
    # Particles far from centroid get stronger pull toward center
    centroid_attraction_strength = np.clip(dist_to_centroid / (2.0 * global_spread), 0.0, 1.5)
    
    # Direction toward centroid for each particle
    to_centroid = centroid - self.population
    to_centroid_norm = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
    to_centroid_dir = to_centroid / to_centroid_norm
    
    # === RANDOM COEFFICIENTS ===
    r1 = np.random.uniform(0, 1, (self.np, self.dim))
    r2 = np.random.uniform(0, 1, (self.np, self.dim))
    
    # === MODULATED COGNITIVE COMPONENT ===
    cognitive_component = cognitive * r1 * (self.personal_best - self.population)
    
    # === MODULATED SOCIAL COMPONENT ===
    social_component = social * r2 * (self.local_best - self.population)
    
    # Add centroid correction: particles far from center get pulled inward
    centroid_correction = 0.2 * centroid_attraction_strength * to_centroid_dir
    
    # === GEOMETRY-MODULATED MUTATION COMPONENT ===
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
    
    # Apply spread modulation to mutation: more mutation in collapsed dimensions
    mutation_component = np.where(
        mutation_active,
        0.3 * spread_modulation * (mutation_vectors - self.population),
        0.0
    )
    
    # === COMBINE ALL COMPONENTS ===
    # Apply spread modulation to cognitive/social (dampen high-spread dims, amplify collapsed)
    new_velocity = (
        self.inertia_weight * self.velocity * spread_modulation +
        cognitive_component * spread_modulation +
        social_component * spread_modulation +
        mutation_component +
        centroid_correction
    )
    
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```