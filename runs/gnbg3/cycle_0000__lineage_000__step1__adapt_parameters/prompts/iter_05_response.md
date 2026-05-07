**Idea: k-NN Graph Topology Parameter Adaptation**
Use k-nearest neighbors graph properties (clustering coefficient, average edge length) to detect population structure and adapt F/CR accordingly — high clustering triggers exploration boost, sparse graph triggers exploitation boost.

```python
def _adapt_parameters(self, improvement_rate):
    """Adapt F and CR based on k-NN graph topology of the population."""
    # Store population reference for graph analysis
    if not hasattr(self, '_graph_population'):
        self._graph_population = None
        self._graph_metrics_history = []
    
    # Get current population from the optimizer's state
    # The population is accessible via the caller's frame or stored attribute
    if not hasattr(self, '_current_population'):
        return  # No population available yet
    
    pop = self._current_population
    if pop is None or len(pop) < 5:
        return
    
    NP, dim = pop.shape
    
    # Build k-NN graph (k based on population size)
    k = max(2, min(5, NP // 10))
    
    # Compute pairwise Euclidean distances (vectorized)
    # Shape: (NP, NP)
    diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    
    # Find k nearest neighbors for each point (exclude self)
    np.fill_diagonal(sq_dists, np.inf)
    nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
    nearest_sq_dists = sq_dists[np.arange(NP)[:, np.newaxis], nearest_indices]
    
    # Compute average edge length of k-NN graph
    avg_edge_length = np.mean(np.sqrt(nearest_sq_dists))
    
    # Compute local clustering coefficient for each node
    clustering_coeffs = np.zeros(NP)
    for i in range(NP):
        neighbors = set(nearest_indices[i])
        if len(neighbors) < 2:
            clustering_coeffs[i] = 0.0
            continue
        # Count edges among neighbors
        edges_among_neighbors = 0
        possible_edges = 0
        for ni in neighbors:
            for nj in neighbors:
                if ni < nj:
                    possible_edges += 1
                    if nj in set(nearest_indices[ni]):
                        edges_among_neighbors += 1
        clustering_coeffs[i] = edges_among_neighbors / max(possible_edges, 1)
    
    avg_clustering = np.mean(clustering_coeffs)
    
    # Compute graph diameter proxy (max distance to k-th nearest neighbor)
    graph_diameter = np.max(np.sqrt(nearest_sq_dists))
    
    # Store current metrics
    current_metrics = {
        'avg_edge_length': avg_edge_length,
        'avg_clustering': avg_clustering,
        'graph_diameter': graph_diameter
    }
    self._graph_metrics_history.append(current_metrics)
    
    # Keep history bounded
    if len(self._graph_metrics_history) > 20:
        self._graph_metrics_history.pop(0)
    
    # Determine trend if we have history
    if len(self._graph_metrics_history) >= 3:
        recent_edge_lengths = [m['avg_edge_length'] for m in self._graph_metrics_history[-3:]]
        recent_clustering = [m['avg_clustering'] for m in self._graph_metrics_history[-3:]]
        
        edge_length_trend = recent_edge_lengths[-1] - recent_edge_lengths[0]
        clustering_trend = recent_clustering[-1] - recent_clustering[0]
    else:
        edge_length_trend = 0.0
        clustering_trend = 0.0
    
    # Adapt F based on graph topology
    # High clustering = population fragmented into niches -> increase F for exploration
    # Decreasing edge length = population converging -> increase F to escape
    clustering_factor = 1.0 + 0.15 * (avg_clustering - 0.3) + 0.1 * (edge_length_trend < -0.01)
    clustering_factor += 0.08 * (clustering_trend > 0.05)  # Increasing clustering -> niche formation
    self.F = np.clip(self.F * clustering_factor, 0.3, 2.0)
    
    # Adapt CR based on graph topology
    # Sparse graph (large edge lengths) -> increase CR for more exploitation of good regions
    # Dense graph (small edge lengths) -> decrease CR to maintain diversity
    median_edge = np.median(np.sqrt(sq_dists[sq_dists != np.inf]))
    if avg_edge_length > median_edge * 0.5:
        # Population well-spread -> encourage exploitation
        self.CR = np.clip(self.CR * 1.08, 0.1, 0.95)
    else:
        # Population clustered -> encourage exploration via crossover
        self.CR = np.clip(self.CR * 0.93, 0.1, 0.95)
    
    # Additional adaptation based on improvement rate (blended with topology)
    if improvement_rate > 0.15:
        self.F = np.clip(self.F * 1.05, 0.3, 2.0)
    elif improvement_rate < 0.03:
        self.F = np.clip(self.F * 1.12, 0.3, 2.0)  # Boost exploration on stagnation
    
    # Record history
    self.F_history.append(self.F)
    self.CR_history.append(self.CR)
```