Looking at the worst unsolved tasks (17 with error 6.76e+04, 16 with 4.52e+03, etc.), the optimizer is getting trapped in local optima. The current ring topology is too rigid and doesn't adapt to the actual spatial structure of the population. Category E gives me a unique angle: I can build a k-NN graph based on actual particle positions, compute graph connectivity metrics, and use these to dynamically route social influence through the swarm — effectively creating information highways to propagate best solutions across fragmented clusters.

**Idea: k-NN Graph Connectivity with Betweenness-Weighted Social Influence**
One-line description: Build position-based k-NN graph, compute betweenness centrality to identify bridge particles, use Fiedler eigenvalue to detect fragmentation, route social influence through graph topology (Category E).

```python
def _velocity_update_batch(self):
    """Update velocities using k-NN graph topology and betweenness-based social routing."""
    cognitive, social = self._adaptive_coefficients()
    
    r1 = np.random.uniform(0, 1, (self.np, self.dim))
    r2 = np.random.uniform(0, 1, (self.np, self.dim))
    
    # Cognitive component: attraction to personal best
    cognitive_component = cognitive * r1 * (self.personal_best - self.population)
    
    # Build k-NN graph based on Euclidean distance in position space
    k = min(max(3, self.neighborhood_size), self.np - 1)
    
    # Compute pairwise squared distances (vectorized)
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)  # Exclude self
    
    # Find k-nearest neighbors for each particle
    knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
    
    # Build adjacency matrix
    adjacency = np.zeros((self.np, self.np), dtype=np.float64)
    for i in range(self.np):
        adjacency[i, knn_indices[i]] = 1.0
        adjacency[knn_indices[i], i] = 1.0
    
    # Compute degree centrality
    degree = np.sum(adjacency, axis=1)
    degree_norm = degree / (degree.max() + 1e-10)
    
    # Compute Fiedler eigenvalue (algebraic connectivity) via power iteration approximation
    # Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
    d_sqrt_inv = np.diag(1.0 / (np.sqrt(degree) + 1e-10))
    laplacian = np.eye(self.np) - d_sqrt_inv @ adjacency @ d_sqrt_inv
    
    # Power iteration to find smallest non-zero eigenvalue (Fiedler)
    np.random.seed(42)  # Reproducible but still varying per call
    v = np.random.randn(self.np)
    v = v / (np.linalg.norm(v) + 1e-10)
    for _ in range(20):
        v = laplacian @ v
        v = v / (np.linalg.norm(v) + 1e-10)
    
    # Rayleigh quotient for Fiedler value estimate
    fiedler = np.abs(v @ laplacian @ v) / (np.dot(v, v) + 1e-10)
    connectivity_strength = np.clip(fiedler * 5.0, 0.1, 2.0)  # Low = fragmented
    
    # Compute betweenness centrality via approximate method (sample paths)
    betweenness = np.zeros(self.np)
    n_samples = min(50, self.np)
    sample_nodes = np.random.choice(self.np, n_samples, replace=False)
    
    for src in sample_nodes:
        # BFS from source
        dist = np.full(self.np, np.inf)
        pred = [[] for _ in range(self.np)]
        dist[src] = 0
        queue = [src]
        
        while queue:
            curr = queue.pop(0)
            for nb in knn_indices[curr]:
                if dist[nb] == np.inf:
                    dist[nb] = dist[curr] + 1
                    queue.append(nb)
                if dist[nb] == dist[curr] + 1:
                    pred[nb].append(curr)
        
        # Count shortest paths through each node
        sigma = np.zeros(self.np)
        sigma[src] = 1
        for d in range(int(dist.max()) + 1) if dist.max() < np.inf else []:
            for node in np.where(dist == d)[0]:
                for p in pred[node]:
                    sigma[node] += sigma[p]
        
        # Accumulate betweenness
        for node in range(self.np):
            if node != src and dist[node] < np.inf:
                for p in pred[node]:
                    betweenness[node] += sigma[p] / (sigma[node] + 1e-10)
    
    betweenness_norm = betweenness / (betweenness.max() + 1e-10)
    
    # Graph-based social component: route influence through topology
    social_component = np.zeros((self.np, self.dim))
    
    for i in range(self.np):
        # Get neighborhood best from k-NN graph
        neighbor_best_idx = knn_indices[i][np.argmin(self.personal_best_fitness[knn_indices[i]])]
        neighbor_best_pos = self.personal_best[neighbor_best_idx]
        
        # Weight by degree and betweenness (bridge particles have more influence)
        influence_weight = 0.5 * (1.0 + degree_norm[i]) * (1.0 + betweenness_norm[i])
        
        # Scale social coefficient by connectivity
        social_scaled = social * connectivity_strength
        
        social_component[i] = social_scaled * r2[i] * influence_weight * (neighbor_best_pos - self.population[i])
    
    # Fragmentation rescue: if Fiedler is very low, pull toward global best
    if fiedler < 0.2:
        rescue_strength = np.clip((0.2 - fiedler) * 3.0, 0.0, 0.8)
        if self.global_best is not None:
            rescue = rescue_strength * (self.global_best - self.population)
        else:
            # Use centroid of best particles
            best_third = np.argsort(self.personal_best_fitness)[:max(1, self.np // 3)]
            centroid = np.mean(self.population[best_third], axis=0)
            rescue = rescue_strength * (centroid - self.population)
        social_component += rescue
    
    # DE/rand/1 mutation (reduced to avoid overwhelming graph-based dynamics)
    mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
    mutation_threshold = 0.05 * (1.0 - self.generation / 5000)
    mutation_active = mutation_mask < mutation_threshold
    
    mutation_vectors = np.zeros((self.np, self.dim))
    for i in range(self.np):
        indices = np.random.choice([j for j in range(self.np) if j != i], 3, replace=False)
        mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
            self.population[indices[1]] - self.population[indices[2]]
        )
    
    mutation_component = np.where(mutation_active, 0.2 * (mutation_vectors - self.population), 0.0)
    
    # Velocity update
    new_velocity = (
        self.inertia_weight * self.velocity +
        cognitive_component +
        social_component +
        mutation_component
    )
    
    self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
```