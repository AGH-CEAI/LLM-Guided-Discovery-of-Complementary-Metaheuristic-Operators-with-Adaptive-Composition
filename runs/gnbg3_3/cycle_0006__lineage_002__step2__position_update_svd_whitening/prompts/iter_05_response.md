**Idea: Graph Laplacian Spectral Modulation**
Topology-based velocity scaling using k-NN graph Laplacian eigenvalues to detect clustering and modulate exploration/exploitation without fitness information.

```python
def _position_update_svd_whitening(self):
    """Graph topology-based velocity modulation (Category E).
    
    Key insight: Build k-NN graph over population; use Laplacian spectral gap
    to detect clustering. Small gap = clustered → more exploration. Use
    graph centrality and clustering coefficient to modulate velocity per
    particle. Orthogonal to spectral-matrix (B) and fitness-rank (D) approaches.
    Targets worst tasks (17, 16, 6) by preventing premature convergence.
    """
    # Build k-NN graph
    k = min(5, self.np - 1)
    sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
    knn_sq_dists = np.take_along_axis(sq_dists, knn_indices, axis=1)
    
    # Build symmetric adjacency matrix (only k-NN connections)
    adj = np.zeros((self.np, self.np))
    row_idx = np.repeat(np.arange(self.np), k)
    col_idx = knn_indices.ravel()
    np.add.at(adj, (row_idx, col_idx), 1.0)
    adj = np.minimum(adj, 1.0)
    adj = (adj + adj.T) / 2.0
    
    # Degree and normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    degrees = np.sum(adj, axis=1)
    degrees = np.clip(degrees, 1e-10, None)
    d_inv_sqrt = 1.0 / np.sqrt(degrees)
    d_matrix = np.diag(d_inv_sqrt)
    lap = np.eye(self.np) - d_matrix @ adj @ d_matrix
    
    # Compute spectral gap (2nd smallest eigenvalue of Laplacian)
    try:
        eigenvalues = np.linalg.eigvalsh(lap)
        eigenvalues = np.sort(eigenvalues)
        spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 1.0
        spectral_gap = np.clip(spectral_gap, 0.0, 1.0)
    except np.linalg.LinAlgError:
        spectral_gap = 1.0
    
    # Local clustering coefficient (triangle count / possible triangles)
    tri_count = np.zeros(self.np)
    for i in range(self.np):
        neighbors = np.where(adj[i] > 0.5)[0]
        n_nbrs = len(neighbors)
        if n_nbrs >= 2:
            nbr_adj = adj[np.ix_(neighbors, neighbors)]
            tri_count[i] = np.sum(nbr_adj) / 2.0
            possible = n_nbrs * (n_nbrs - 1) / 2.0
            if possible > 0:
                tri_count[i] /= possible
    
    # Boundary detection: particles with below-median degree (likely on frontier)
    degree_median = np.median(degrees)
    is_boundary = degrees < degree_median
    
    # Local density from average k-NN distance
    avg_knn_dist = np.mean(np.sqrt(knn_sq_dists), axis=1) + 1e-10
    global_avg_dist = np.mean(avg_knn_dist)
    density_ratio = avg_knn_dist / global_avg_dist
    
    # Spectral-based global modulation
    # Small spectral_gap = clustered population → increase exploration
    # Large spectral_gap = well-dispersed population → can exploit more
    exploration_factor = 1.0 + 0.6 * (1.0 - spectral_gap)
    exploitation_factor = 1.0 + 0.4 * spectral_gap
    
    # Per-particle velocity scaling from topology signals
    # Boundary particles (low degree) → more exploration
    # High clustering coefficient → more exploration (stuck in tight cluster)
    # Low local density (far from neighbors) → more exploration
    boundary_boost = np.where(is_boundary, 1.3, 1.0)
    clustering_boost = 1.0 + 0.4 * tri_count
    density_modulation = np.clip(density_ratio, 0.5, 2.0)
    
    vel_scale = exploration_factor * boundary_boost * clustering_boost / density_modulation
    vel_scale = np.clip(vel_scale, 0.3, 2.5)
    
    # Directional nudge toward less-explored regions using graph centrality
    # Low-degree particles get pulled toward centroid of high-degree particles
    centrality = degrees / (np.sum(degrees) + 1e-10)
    centroid = np.mean(self.population, axis=0)
    
    # Per-particle: compute "frontier direction" as vector away from local density center
    frontier_direction = np.zeros((self.np, self.dim))
    for i in range(self.np):
        neighbors = knn_indices[i]
        nbr_center = np.mean(self.population[neighbors], axis=0)
        frontier_direction[i] = self.population[i] - nbr_center
    
    frontier_magnitude = np.linalg.norm(frontier_direction, axis=1, keepdims=True) + 1e-10
    frontier_direction = frontier_direction / frontier_magnitude
    
    # Boundary particles get stronger frontier push
    frontier_push = frontier_direction * np.where(is_boundary, 0.3, 0.1) * vel_scale[:, np.newaxis]
    
    # Apply topology-modulated velocity update
    new_population = self.population + vel_scale[:, np.newaxis] * self.velocity + frontier_push
    
    self.population = self._clip_to_bounds(new_population)
```