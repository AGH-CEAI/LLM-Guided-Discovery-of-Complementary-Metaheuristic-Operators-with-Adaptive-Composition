Looking at the worst unsolved tasks (17, 16, 6, 11, 19, 9, 23, 12) with errors ranging from 10^1 to 10^4, the algorithm is clearly getting trapped — likely due to population fragmentation or loss of connectivity. The current inertia adaptation uses spectral (eigenvalue) analysis which doesn't directly capture graph connectivity. Category E gives me a fundamentally different angle: detecting fragmentation and isolation through graph structure rather than covariance shape.

**Idea: Graph Laplacian Connectivity-Driven Inertia**
Build k-NN graph over population, compute normalized Laplacian eigenvalues (spectral gap), count connected components, and measure average shortest-path length. Fragmented/isolated populations → higher inertia for exploration; well-connected populations → lower inertia for exploitation.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using k-NN graph connectivity and clustering (Category E).
    
    Key topology signals:
    - Spectral gap (2nd smallest Laplacian eigenvalue): small = fragmented population
    - Connected component count: >1 = population split into isolated clusters
    - Average path length: long = poor global connectivity
    - Clustering coefficient: high = particles clustered in local groups
    
    High fragmentation → MORE exploration (higher inertia)
    Well-connected → MORE exploitation (lower inertia)
    """
    try:
        np = self.np
        
        # Build k-NN graph over population
        k = min(5, np - 1)
        sq_dists = np.sum(
            (self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2,
            axis=2
        )
        np.fill_diagonal(sq_dists, np.inf)
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
        
        # --- Connected Component Analysis ---
        visited = np.zeros(np, dtype=bool)
        components = []
        for start in range(np):
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
        
        # Fragmentation signal: how many isolated clusters?
        # 1 component = fully connected, >1 = fragmented
        fragmentation_ratio = max(0.0, (n_components - 1) / max(np, 1))
        
        # --- Graph Laplacian Spectral Gap ---
        # Compute normalized Laplacian eigenvalues
        row_indices = np.repeat(np.arange(np), k)
        col_indices = knn_indices.ravel()
        data = np.ones(len(row_indices))
        
        from scipy.sparse import csr_matrix
        adj = csr_matrix((data, (row_indices, col_indices)), shape=(np, np))
        adj = adj + adj.T
        adj.data[:] = 1.0
        
        degrees = np.array(adj.sum(axis=1)).ravel() + 1e-10
        d_inv_sqrt = 1.0 / np.sqrt(degrees)
        d_inv_sqrt_diag = csr_matrix(
            (d_inv_sqrt, (np.arange(np), np.arange(np))), shape=(np, np)
        )
        
        # Normalized Laplacian: I - D^{-1/2} * A * D^{-1/2}
        import scipy.sparse as sp
        identity = sp.identity(np, format='csr')
        lap = identity - d_inv_sqrt_diag @ adj @ d_inv_sqrt_diag
        
        from scipy.sparse.linalg import eigsh
        n_eigs = min(5, np - 1)
        eigenvalues = eigsh(lap, k=n_eigs, return_eigenvectors=False)
        eigenvalues = np.sort(eigenvalues)
        
        # Spectral gap: second-smallest eigenvalue
        # Small gap = population fragmented into clusters
        # Large gap (~2.0 for k-NN) = well-connected population
        spectral_gap = float(eigenvalues[1]) if len(eigenvalues) > 1 else 2.0
        spectral_gap = np.clip(spectral_gap, 0.0, 2.0)
        
        # Normalize spectral gap to [0, 1]: 0 = fragmented, 1 = well-connected
        connectivity_signal = spectral_gap / 2.0
        
        # --- Average Path Length (Floyd-Warshall on k-NN graph) ---
        # Approximate via component analysis: isolated particles have infinite path
        max_component_size = np.max(component_sizes)
        largest_component_ratio = max_component_size / np
        
        # Particles in small components are "isolated" → need exploration
        particle_to_component = np.zeros(np, dtype=int)
        for idx, comp in enumerate(components):
            for p_idx in comp:
                particle_to_component[p_idx] = idx
        
        isolation_penalty = np.mean([
            1.0 - (component_sizes[particle_to_component[i]] / max_component_size)
            for i in range(np)
        ])
        
        # --- Clustering Coefficient ---
        # Local clustering: triangles / possible triangles for each node
        clustering = np.zeros(np)
        for i in range(np):
            neighbors_i = set(knn_indices[i])
            if len(neighbors_i) < 2:
                clustering[i] = 0.0
                continue
            triangles = 0
            for j in neighbors_i:
                neighbors_j = set(knn_indices[j])
                triangles += len(neighbors_i & neighbors_j)
            max_triangles = len(neighbors_i) * (len(neighbors_i) - 1) / 2
            clustering[i] = triangles / max_triangles if max_triangles > 0 else 0.0
        avg_clustering = np.mean(clustering)
        
        # High clustering = particles forming tight local groups → need to break out
        clustering_signal = avg_clustering
        
        # --- Combine Topology Signals ---
        # Fragmented/isolated/highly-clustered → more exploration (higher inertia)
        # Well-connected → more exploitation (lower inertia)
        
        # Composite exploration signal: [0, 1] where 1 = need exploration
        exploration_signal = (
            fragmentation_ratio * 0.3 +                    # More components = more exploration
            (1.0 - connectivity_signal) * 0.4 +            # Low spectral gap = fragmented
            isolation_penalty * 0.2 +                       # Isolated particles = need exploration
            clustering_signal * 0.1                         # High clustering = local trapping
        )
        exploration_signal = np.clip(exploration_signal, 0.0, 1.0)
        
        # Map exploration signal to inertia range [0.4, 0.95]
        # High exploration signal → high inertia (0.9-0.95)
        # Low exploration signal → low inertia (0.4-0.5)
        target_inertia = 0.4 + 0.55 * exploration_signal
        
        # Blend with time-based decay for smooth transition
        decay = 0.729 - 0.15 * (self.generation / 1000)
        target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)
        
        # EMA smoothing to avoid jitter
        self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
        
    except Exception:
        # Fallback: time-based decay
        decay = 0.729 - 0.15 * (self.generation / 1000)
        self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```