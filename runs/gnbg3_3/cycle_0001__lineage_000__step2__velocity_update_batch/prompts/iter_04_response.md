**Idea: Fitness-Rank Correlation Adaptive Velocity**

Uses Spearman rank correlation between fitness and distance-to-global-best to modulate social attraction strength. When correlation is high (fitness correlates with proximity to global best), trust the social component more; when low (rugged landscape), reduce social attraction and boost mutation. This is purely fitness-signal reasoning—only rank order matters, not raw distances.

```python
def _velocity_update_batch(self):
    """Update velocities with fitness-rank-adaptive components."""
    # Compute Spearman rank correlation between fitness and distance to global best
    if self.global_best is not None:
        # Compute distances from each particle to global best
        distances = np.linalg.norm(self.population - self.global_best, axis=1)
        
        # Compute Spearman rank correlation between fitness and distance
        # (negative correlation = better fitness is closer to global best)
        if np.std(distances) > 1e-10 and np.std(self.current_fitness) > 1e-10:
            corr = np.corrcoef(
                np.argsort(np.argsort(self.current_fitness)),
                np.argsort(np.argsort(distances))
            )[0, 1]
        else:
            corr = 0.0
    else:
        corr = 0.0
    
    # Adaptive social coefficient based on rank correlation
    # High correlation -> trust social component more; low -> rely on exploration
    social = self.social_base * np.clip(1.0 + corr, 0.2, 1.5)
    cognitive = self.cognitive_base
    
    # Generate random matrices once for efficiency
    r1 = np.random.uniform(0, 1, (self.np, self.dim))
    r2 = np.random.uniform(0, 1, (self.np, self.dim))
    
    # Cognitive component: attraction to personal best
    cognitive_component = cognitive * r1 * (self.personal_best - self.population)
    
    # Social component: attraction to local best (ring topology)
    social_component = social * r2 * (self.local_best - self.population)
    
    # Mutation injection strength inversely proportional to correlation
    # Low correlation = rugged landscape = more exploration needed
    mutation_strength = 0.3 * np.clip(1.0 - corr, 0.1, 1.0)
    
    mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
    mutation_threshold = 0.1 * (1.0 - self.generation / 5000)
    mutation_active = mutation_mask < mutation_threshold
    
    # Generate mutation vectors from random distinct indices
    mutation_vectors = np.zeros((self.np, self.dim))
    for i in range(self.np):
        indices = np.random.choice(
            [j for j in range(self.np) if j != i], 3, replace=False
        )
        mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
            self.population[indices[1]] - self.population[indices[2]]
        )
    
    # Apply mutation where active
    mutation_component = np.where(
        mutation_active,
        mutation_strength * (mutation_vectors - self.population),
        0.0
    )
    
    # Velocity update with inertia, cognitive, social, and mutation
    new_velocity = (
        self.inertia_weight * self.velocity +
        cognitive_component +
        social_component +
        mutation_component
    )
    
    # Clip velocity to bounds
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```