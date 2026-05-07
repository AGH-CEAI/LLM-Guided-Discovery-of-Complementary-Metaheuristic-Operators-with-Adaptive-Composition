Looking at the worst unsolved tasks (errors 1e+01 to 5e+04), these are likely multi-modal landscapes where the population gets trapped in disconnected basins. The key insight for Category E: use graph topology to detect isolated particles and modulate their velocity toward connectivity.

**Idea: k-NN Graph Centrality Velocity Modulation**

Build a k-NN graph over the population and compute each particle's graph centrality (inverse average distance to neighbors). Particles with low centrality (isolated, on the periphery of the swarm) get a stronger velocity boost toward the global best, breaking them out of disconnected traps. This is fundamentally different from spectral (B), fitness (D), or temporal (F) approaches because it uses the *graph structure itself* to identify and correct isolated particles.
```python
def _velocity_update_base(self):
    """Topology-aware velocity update using k-NN graph centrality."""
    cognitive, social = self._adaptive_coefficients()
    
    r1 = np.random.uniform(0, 1, (self.np, self.dim))
    r2 = np.random.uniform(0, 1, (self.np, self.dim))
    
    cognitive_component = cognitive * r1 * (self.personal_best - self.population)
    social_component = social * r2 * (self.local_best - self.population)
    
    # === CATEGORY E: k-NN Graph Centrality ===
    k = min(5, self.np - 1)
    
    # Build k-NN distance matrix
    sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
    
    # Compute average distance to k nearest neighbors (inverse centrality)
    knn_dists = np.sqrt(np.sort(sq_dists, axis=1)[:, :k])
    avg_knn_dist = np.mean(knn_dists, axis=1) + 1e-10
    
    # Graph centrality: inverse of average neighbor distance
    # Higher centrality = well-connected = less correction needed
    global_avg_dist = np.mean(avg_knn_dist) + 1e-10
    centrality = global_avg_dist / avg_knn_dist
    
    # Normalize centrality to [0.5, 2.0] range
    centrality = np.clip(centrality / (np.max(centrality) + 1e-10) * 1.5, 0.5, 2.0)
    
    # Compute connected components to detect fragmentation
    visited = np.zeros(self.np, dtype=bool)
    components = []
    for start in range(self.np):
        if visited[start]:
            continue
        component = []
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            for neighbor in knn_indices[node]:
                if not visited[neighbor]:
                    stack.append(neighbor)
        components.append(component)
    
    n_components = len(components)
    component_sizes = np.array([len(c) for c in components])
    
    # Per-particle component size factor
    particle_component_size = np.array([
        component_sizes[next((i for i, c in enumerate(components) if p in c), 0)]
        for p in range(self.np)
    ])
    
    # Fragmentation correction: particles in small components get stronger boost
    size_factor = np.clip(particle_component_size / max(1, self.np), 0.1, 1.0)
    
    # Combined topology correction: low centrality + small component = strong pull
    topology_correction_strength = (2.0 - centrality) * (2.0 - size_factor)
    topology_correction_strength = np.clip(topology_correction_strength, 0.0, 2.0)
    
    # Apply topology-corrected velocity toward global best
    if self.global_best is not None:
        to_global = self.global_best - self.population
        to_global_dist = np.linalg.norm(to_global, axis=1, keepdims=True) + 1e-10
        to_global_dir = to_global / to_global_dist
        topology_correction = topology_correction_strength[:, np.newaxis] * to_global_dir
    else:
        # Fallback: random perturbation scaled by isolation
        topology_correction = topology_correction_strength[:, np.newaxis] * np.random.uniform(-1, 1, (self.np, self.dim))
    
    # === DE Mutation Component ===
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
    
    # === Combine all components ===
    new_velocity = (
        self.inertia_weight * self.velocity +
        cognitive_component +
        social_component +
        0.5 * topology_correction +
        mutation_component
    )
    
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```