Looking at the unsolved tasks, the worst performers (Tasks 17, 16, 6, 11, 19) have errors 6+ orders of magnitude above target, suggesting severe convergence failures. The current velocity update uses random coefficients that ignore the geometric structure of the swarm.

**Idea: k-NN Isolation-Weighted Velocity Update**

Category A: Geometry / spatial — Use k-nearest neighbor distances to compute per-particle isolation weights. Particles in sparse regions (far from neighbors) get stronger attraction toward bests; particles in dense clusters get moderated attraction. This directly addresses convergence failures by geometrically-informed force scaling.

```python
def _velocity_update_batch(self):
    """Update velocities with k-NN isolation-weighted geometric components."""
    k = min(5, self.np - 1)
    
    # Compute pairwise distances
    dist_matrix = np.linalg.norm(
        self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :], axis=2
    )
    np.fill_diagonal(dist_matrix, np.inf)
    
    # k-NN isolation weight per particle (higher = more isolated)
    knn_distances = np.sort(dist_matrix, axis=1)[:, :k]
    mean_knn_dist = np.mean(knn_distances, axis=1)
    isolation_weight = mean_knn_dist / (mean_knn_dist.mean() + 1e-10)
    isolation_weight = np.clip(isolation_weight, 0.5, 2.0)
    
    # Per-particle random coefficients weighted by isolation
    r1 = np.random.uniform(0, 1, (self.np, self.dim))
    r2 = np.random.uniform(0, 1, (self.np, self.dim))
    
    # Cognitive component: isolation-weighted attraction to personal best
    cognitive_component = (
        self.cognitive_base * r1 * (self.personal_best - self.population) * isolation_weight[:, np.newaxis]
    )
    
    # Social component: inverse-isolation-weighted attraction to local best
    social_component = (
        self.social_base * r2 * (self.local_best - self.population) / isolation_weight[:, np.newaxis]
    )
    
    # DE/rand/1 mutation with isolation-weighted scaling
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
    
    # Mutation scaled by isolation (isolated particles get stronger mutation pull)
    mutation_component = np.where(
        mutation_active,
        0.3 * isolation_weight[:, np.newaxis] * (mutation_vectors - self.population),
        0.0
    )
    
    # Velocity update with inertia and geometric components
    new_velocity = (
        self.inertia_weight * self.velocity +
        cognitive_component +
        social_component +
        mutation_component
    )
    
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```