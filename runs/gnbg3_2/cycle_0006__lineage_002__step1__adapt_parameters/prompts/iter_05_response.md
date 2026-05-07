**Idea: k-NN Graph Connectivity Adaptation**

One-line description: Build a k-nearest-neighbor graph over the population each generation and use component fragmentation, clustering coefficient, and average shortest-path length to detect exploration/exploitation regimes and adapt F/Cr accordingly (Category E: Topology / graph-based).

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using k-NN graph topology properties."""
    # Build k-NN graph over population (using current population from optimizer)
    k = max(3, min(7, self.np // 10))
    pop = self.population_buffer if hasattr(self, 'population_buffer') else None
    
    if pop is not None and len(pop) >= k + 1:
        # Compute pairwise distances
        dists = np.linalg.norm(pop[:, np.newaxis] - pop[np.newaxis, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # k-NN adjacency: each node connects to its k nearest neighbors
        knn_indices = np.argsort(dists, axis=1)[:, :k]
        row_idx = np.arange(len(pop)).repeat(k)
        col_idx = knn_indices.ravel()
        
        # Build sparse adjacency (undirected)
        n = len(pop)
        adj = np.zeros((n, n), dtype=bool)
        adj[row_idx, col_idx] = True
        adj = adj | adj.T
        
        # Component analysis: count connected components
        visited = np.zeros(n, dtype=bool)
        components = []
        for start in range(n):
            if not visited[start]:
                comp = []
                stack = [start]
                while stack:
                    node = stack.pop()
                    if not visited[node]:
                        visited[node] = True
                        comp.append(node)
                        neighbors = np.where(adj[node])[0]
                        for nb in neighbors:
                            if not visited[nb]:
                                stack.append(nb)
                components.append(comp)
        
        component_sizes = np.array([len(c) for c in components])
        n_components = len(components)
        main_component_frac = np.max(component_sizes) / n if n_components > 0 else 1.0
        
        # Clustering coefficient: for each node, fraction of neighbor pairs that are connected
        clustering_sum = 0.0
        for i in range(n):
            neighbors = np.where(adj[i])[0]
            deg = len(neighbors)
            if deg >= 2:
                subgraph = adj[np.ix_(neighbors, neighbors)]
                edges = np.sum(subgraph) / 2
                possible = deg * (deg - 1) / 2
                clustering_sum += edges / possible
        avg_clustering = clustering_sum / n if n > 0 else 0.0
        
        # Average shortest path in main component (approximate via sampling for efficiency)
        main_comp_nodes = np.where(component_sizes == np.max(component_sizes))[0]
        if len(main_comp_nodes) > 1:
            sample_size = min(20, len(main_comp_nodes))
            sample_nodes = np.random.choice(main_comp_nodes, sample_size, replace=False)
            path_lengths = []
            for src in sample_nodes:
                dists_src = np.full(n, np.inf)
                dists_src[src] = 0
                queue = [src]
                head = 0
                while head < len(queue):
                    node = queue[head]
                    head += 1
                    for nb in np.where(adj[node])[0]:
                        if dists_src[nb] == np.inf:
                            dists_src[nb] = dists_src[node] + 1
                            queue.append(nb)
                path_lengths.extend([d for d in dists_src[main_comp_nodes] if d < np.inf and d > 0])
            avg_path = np.mean(path_lengths) if path_lengths else 1.0
        else:
            avg_path = 1.0
        
        # Graph-based regime detection
        graph_fragmented = n_components > n / 3 or main_component_frac < 0.6
        graph_well_connected = n_components <= 2 and main_component_frac > 0.85
        graph_isolated_nodes = np.sum(component_sizes == 1) > n * 0.1
        
    else:
        n_components = 1
        main_component_frac = 1.0
        avg_clustering = 0.5
        avg_path = 1.0
        graph_fragmented = False
        graph_well_connected = True
        graph_isolated_nodes = False
    
    # Initialize graph history on first call
    if not hasattr(self, 'graph_n_components_history'):
        self.graph_n_components_history = []
        self.graph_clustering_history = []
        self.graph_path_history = []
    
    # Track graph properties over time
    self.graph_n_components_history.append(n_components)
    self.graph_clustering_history.append(avg_clustering)
    self.graph_path_history.append(avg_path)
    if len(self.graph_n_components_history) > 10:
        self.graph_n_components_history.pop(0)
        self.graph_clustering_history.pop(0)
        self.graph_path_history.pop(0)
    
    # Compute trend in graph connectivity
    if len(self.graph_n_components_history) >= 3:
        recent_frac_change = (self.graph_n_components_history[-1] - 
                              np.mean(self.graph_n_components_history[-3:])) / max(1, np.mean(self.graph_n_components_history[-3:]))
    else:
        recent_frac_change = 0.0
    
    # Compute graph-based adjustment factor
    if graph_fragmented:
        # Fragmented graph: need more exploration
        component_entropy = -np.sum((component_sizes / n) * np.log(component_sizes / n + 1e-10))
        max_entropy = np.log(n_components + 1e-10)
        norm_entropy = component_entropy / (max_entropy + 1e-10)
        F_adjustment = 1.0 + 0.25 * norm_entropy + 0.1 * recent_frac_change
        Cr_adjustment = 0.85
        self.stagnation_counter = getattr(self, 'stagnation_counter', 0) + 1
    elif graph_isolated_nodes:
        # Many isolated nodes: need diversity injection
        F_adjustment = 1.15
        Cr_adjustment = 0.80
        self.stagnation_counter = getattr(self, 'stagnation_counter', 0) + 1
    elif graph_well_connected:
        # Well-connected graph: can exploit more
        F_adjustment = 0.90 - 0.1 * avg_clustering
        Cr_adjustment = 1.05 + 0.05 * (1.0 - avg_clustering)
        self.stagnation_counter = max(0, getattr(self, 'stagnation_counter', 0) - 1)
    else:
        # Moderate connectivity: balanced adjustment
        delta_comp = (np.mean(self.graph_n_components_history[-3:]) - 
                      np.mean(self.graph_n_components_history[:3])) if len(self.graph_n_components_history) >= 6 else 0.0
        F_adjustment = 1.0 + 0.05 * delta_comp - 0.05 * recent_frac_change
        Cr_adjustment = 1.0 + 0.02 * (0.5 - avg_clustering)
    
    # Apply adjustments with bounds
    F_adjustment = np.clip(F_adjustment, 0.7, 1.4)
    Cr_adjustment = np.clip(Cr_adjustment, 0.8, 1.2)
    
    self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
    self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
```