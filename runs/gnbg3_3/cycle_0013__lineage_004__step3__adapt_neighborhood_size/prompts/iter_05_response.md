**Idea: k-NN Graph Connectivity Adaptation**

Adapt ring neighborhood size using k-NN graph properties: connected component count (fragmentation), graph Laplacian spectral gap (mixing quality), and clustering coefficient. When the population graph is fragmented (multiple components, low spectral gap), increase neighborhood to improve information flow across disconnected subpopulations. When well-connected and converging, decrease neighborhood for faster exploitation.

```python
def _adapt_neighborhood_size(self):
    """Adapt ring topology neighborhood size based on k-NN graph connectivity.
    
    Category E: Build k-NN graph over population; use connected component count,
    spectral gap, and graph density to detect fragmentation vs convergence.
    """
    n = self.np
    k = min(5, n - 1)
    
    # Build k-NN graph
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
    
    # Connected component analysis via BFS
    visited = np.zeros(n, dtype=bool)
    components = []
    for start in range(n):
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
    max_component_size = np.max(component_sizes)
    
    # Fragmentation ratio: how unevenly distributed is the population?
    fragmentation_ratio = 1.0 - (max_component_size / (n + 1e-10))
    
    # Compute graph density
    max_edges = n * (n - 1) / 2
    actual_edges = n * k / 2
    graph_density = actual_edges / (max_edges + 1e-10)
    
    # Spectral gap from normalized graph Laplacian
    try:
        row_indices = np.repeat(np.arange(n), k)
        col_indices = knn_indices.ravel()
        data = np.ones(len(row_indices))
        from scipy.sparse import csr_matrix
        adj = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        adj = adj + adj.T
        adj.data[:] = 1.0
        
        degrees = np.array(adj.sum(axis=1)).ravel() + 1e-10
        d_inv_sqrt = 1.0 / np.sqrt(degrees)
        d_inv_sqrt_diag = csr_matrix((d_inv_sqrt, (np.arange(n), np.arange(n))), shape=(n, n))
        lap = dspy.identity(n) - d_inv_sqrt_diag @ adj @ d_inv_sqrt_diag
        
        from scipy.sparse.linalg import eigsh
        eigenvalues = eigsh(lap, k=min(5, n - 1), return_eigenvectors=False)
        eigenvalues = np.sort(eigenvalues)
        
        spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 1.0
        spectral_gap = np.clip(spectral_gap, 0.0, 1.0)
    except:
        spectral_gap = 1.0
    
    # Decision logic based on graph topology
    # Fragmented population (multiple components or high fragmentation) → increase neighborhood
    is_fragmented = (n_components > 1) or (fragmentation_ratio > 0.3)
    # Poor mixing (low spectral gap) → increase neighborhood
    poor_mixing = spectral_gap < 0.3
    # Well-connected and converging → decrease neighborhood
    is_well_connected = (n_components == 1) and (spectral_gap > 0.6) and (graph_density > 0.3)
    
    if is_fragmented or poor_mixing:
        # Population graph is fragmented → expand neighborhood for better information flow
        self.neighborhood_size = min(n // 2, self.neighborhood_size + 1)
    elif is_well_connected:
        # Population is well-mixed → shrink neighborhood for faster exploitation
        self.neighborhood_size = max(1, self.neighborhood_size - 1)
```