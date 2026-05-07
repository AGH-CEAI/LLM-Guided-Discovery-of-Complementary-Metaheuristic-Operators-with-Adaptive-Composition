**Idea: k-NN Density + Axis-Aligned Spread Modulation**
Uses k-nearest-neighbor distances to detect clustering (sparse regions get stronger pulls, dense regions get dampened) and axis-aligned standard deviation to modulate exploration per dimension (collapsed dimensions get amplified velocity). Directly targets the fragmentation/clustering failure mode causing the worst task errors.
```python
def _velocity_update_base(self):
    """Base velocity update with k-NN density + axis-aligned spread modulation (Category A).
    
    Geometric mechanisms:
      1. k-NN density: Particles in sparse regions get AMPLIFIED cognitive/social pulls;
         particles in dense clusters get DAMPENED velocity to prevent over-crowding.
      2. Axis-aligned spread: Per-dimension std modulates velocity inversely —
         collapsed dimensions (low std) get amplified exploration, spread-out
         dimensions (high std) get dampened to avoid redundant exploration.
    
    This is purely geometric (distances, k-NN structure, axis-aligned spread)
    — no fitness values, eigenvalues, or information-theoretic quantities.
    """
    # === MECHANISM 1: k-NN DENSITY ESTIMATION ===
    k = min(5, self.np - 1)
    
    # Compute squared pairwise distances (N x N matrix)
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # k-NN distances: average distance to k nearest neighbors
    knn_dists = np.sort(sq_dists, axis=1)[:, :k]
    mean_knn_dist = np.mean(knn_dists, axis=1) + 1e-10
    
    # Density = inverse of k-NN distance (high distance = sparse = high density factor)
    # Add small epsilon to avoid division issues
    density_factor = mean_knn_dist / (np.mean(mean_knn_dist) + 1e-10)
    density_factor = np.clip(density_factor, 0.3, 3.0)
    
    # === MECHANISM 2: AXIS-ALIGNED SPREAD MODULATION ===
    # Compute per-dimension standard deviation (geometric spread)
    per_dim_std = np.std(self.population, axis=0) + 1e-10
    mean_spread = np.mean(per_dim_std) + 1e-10
    
    # Spread ratio: how spread out is each dimension relative to average?
    spread_ratio = per_dim_std / mean_spread
    
    # Inverse spread modulation: collapsed dimensions (low spread_ratio) get
    # AMPLIFIED velocity, spread-out dimensions get dampened
    spread_modulation = 1.0 / (spread_ratio + 0.5)
    spread_modulation = np.clip(spread_modulation, 0.5, 2.5)
    
    # === COMBINE GEOMETRIC SIGNALS ===
    # Density-based scaling: sparse particles move more, dense particles move less
    density_scale = density_factor[:, np.newaxis]  # (np, 1) broadcast to (np, dim)
    
    # Spread-based scaling: per-dimension, broadcast across particles
    spread_scale = spread_modulation[np.newaxis, :]  # (1, dim) broadcast to (np, dim)
    
    # Combined geometric modulation
    geometric_scale = density_scale * spread_scale
    geometric_scale = np.clip(geometric_scale, 0.2, 4.0)
    
    # === STANDARD COGNITIVE + SOCIAL COMPONENTS ===
    cognitive, social = self._adaptive_coefficients()
    
    r1 = np.random.uniform(0, 1, (self.np, self.dim))
    r2 = np.random.uniform(0, 1, (self.np, self.dim))
    
    cognitive_component = cognitive * r1 * (self.personal_best - self.population)
    social_component = social * r2 * (self.local_best - self.population)
    
    # Apply geometric modulation to cognitive and social components
    cognitive_component = cognitive_component * geometric_scale
    social_component = social_component * geometric_scale
    
    # === DE MUTATION COMPONENT (also modulated geometrically) ===
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
    
    # Apply geometric modulation to mutation component as well
    mutation_component = mutation_component * geometric_scale
    
    # === VELOCITY UPDATE ===
    new_velocity = (
        self.inertia_weight * self.velocity +
        cognitive_component +
        social_component +
        mutation_component
    )
    
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```