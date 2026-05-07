**Idea: Centroid Attraction with k-NN Repulsion and Spread Modulation**
Geometry/spatial velocity update using centroid-relative positioning, k-nearest-neighbor repulsion, and axis-aligned spread-based velocity modulation. No fitness or eigenvalue information.

```python
def _velocity_update_batch(self):
    """Update velocities using geometric layout: centroid, k-NN repulsion, spread."""
    # Compute centroid of population
    centroid = np.mean(self.population, axis=0)
    
    # Compute axis-aligned spread (range) per dimension
    min_pos = np.min(self.population, axis=0)
    max_pos = np.max(self.population, axis=0)
    spread = max_pos - min_pos + 1e-10
    
    # Compute distance from centroid for each particle
    dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)
    
    # Generate random matrices
    r1 = np.random.uniform(0, 1, (self.np, self.dim))
    r2 = np.random.uniform(0, 1, (self.np, self.dim))
    
    # Cognitive component: attraction to personal best
    cognitive = self.cognitive_base * (1.0 + 0.1 * np.exp(-dist_to_centroid / 50.0))
    cognitive_component = cognitive * r1 * (self.personal_best - self.population)
    
    # Social component: attraction to local best
    social = self.social_base * (1.0 + 0.1 * np.exp(-dist_to_centroid / 50.0))
    social_component = social * r2 * (self.local_best - self.population)
    
    # Geometric centroid component: attract toward swarm centroid
    centroid_direction = centroid - self.population
    centroid_strength = np.exp(-dist_to_centroid / 100.0)
    centroid_component = 0.3 * centroid_strength * centroid_direction
    
    # k-NN repulsion from spatial neighbors (purely geometric)
    k = min(5, self.np - 1)
    knn_repulsion = np.zeros((self.np, self.dim))
    for i in range(self.np):
        diffs = self.population - self.population[i]
        dists = np.linalg.norm(diffs, axis=1)
        dists[i] = np.inf
        nn_indices = np.argpartition(dists, k)[:k]
        nn_dists = dists[nn_indices]
        for j, nn_idx in enumerate(nn_indices):
            if nn_dists[j] > 1e-10:
                diff = self.population[i] - self.population[nn_idx]
                knn_repulsion[i] += diff / (nn_dists[j] ** 2 + 1e-10)
    knn_component = 0.1 * knn_repulsion
    
    # Spread-based velocity modulation: boost under-explored dimensions
    spread_modulation = np.mean(spread) / (spread + 1e-10)
    
    # Velocity update
    new_velocity = (
        self.inertia_weight * self.velocity +
        cognitive_component +
        social_component +
        centroid_component +
        knn_component
    )
    new_velocity = new_velocity * spread_modulation
    
    # Clip velocity to bounds
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```