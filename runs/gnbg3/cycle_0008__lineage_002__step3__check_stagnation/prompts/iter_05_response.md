**Idea: MST Weight & Component Fragmentation Stagnation Detector**

One-line description: Detect stagnation by tracking connected component count and minimum spanning tree total weight — when the population graph becomes too fragmented or the MST weight shrinks consistently, trigger restart (Category E: Topology/graph-based, targeting task 23).

```python
def _check_stagnation(self, best_fitness):
    """Detect stagnation via graph-theoretic properties: connected components and MST weight."""
    # Get current population from the optimizer's state
    if not hasattr(self, '_current_population') or self._current_population is None:
        return False
    
    pop = self._current_population
    NP, dim = pop.shape
    
    if NP < 4:
        return False
    
    # Initialize history for temporal tracking (F: temporal/dynamical component)
    if not hasattr(self, '_topo_stagnation_history'):
        self._topo_stagnation_history = []
    
    # Build k-NN graph (k adapts to population size)
    k = max(2, min(5, NP // 8))
    
    # Compute pairwise Euclidean distances (E: topology via graph structure)
    diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Find k nearest neighbors for each point
    nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
    
    # === Compute connected components using Union-Find ===
    parent = np.arange(NP)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # Path compression
            x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Build undirected k-NN graph and compute components
    for i in range(NP):
        for j in nearest_indices[i]:
            union(i, j)
    
    # Count unique components
    components = np.unique([find(i) for i in range(NP)])
    n_components = len(components)
    
    # === Compute MST total weight using Prim's algorithm ===
    in_mst = np.zeros(NP, dtype=bool)
    in_mst[0] = True
    mst_edges = []
    mst_total_weight = 0.0
    
    for _ in range(NP - 1):
        min_weight = np.inf
        min_edge = None
        for i in range(NP):
            if not in_mst[i]:
                continue
            # Find minimum edge from MST to unvisited vertex
            for j in nearest_indices[i]:
                if not in_mst[j] and sq_dists[i, j] < min_weight:
                    min_weight = sq_dists[i, j]
                    min_edge = (i, j)
        if min_edge is not None:
            mst_edges.append(min_edge)
            mst_total_weight += np.sqrt(min_weight)
            in_mst[min_edge[1]] = True
        else:
            break
    
    # Normalize MST weight by population size for scale-invariance
    avg_mst_edge_weight = mst_total_weight / max(len(mst_edges), 1)
    
    # === Compute average edge length in k-NN graph ===
    edge_lengths = []
    for i in range(NP):
        for j in nearest_indices[i]:
            if i < j:  # Count each edge once
                edge_lengths.append(np.sqrt(sq_dists[i, j]))
    
    avg_knn_edge = np.mean(edge_lengths) if edge_lengths else 0.0
    
    # === Compute component size statistics ===
    component_sizes = []
    for c in components:
        size = sum(1 for i in range(NP) if find(i) == c)
        component_sizes.append(size)
    
    largest_component_ratio = max(component_sizes) / NP if component_sizes else 1.0
    
    # Store current metrics for temporal analysis
    current_metrics = {
        'n_components': n_components,
        'avg_mst_edge': avg_mst_edge_weight,
        'largest_component_ratio': largest_component_ratio,
        'avg_knn_edge': avg_knn_edge
    }
    self._topo_stagnation_history.append(current_metrics)
    
    # Keep history bounded
    if len(self._topo_stagnation_history) > 12:
        self._topo_stagnation_history.pop(0)
    
    # === Stagnation detection via topology metrics ===
    # Need sufficient history for temporal analysis
    if len(self._topo_stagnation_history) < 4:
        return False
    
    # Metric 1: Component fragmentation (E: connectivity-based)
    # Too many small components = population fragmented into isolated niches
    fragmentation_threshold = NP / 3
    is_fragmented = n_components > fragmentation_threshold
    
    # Metric 2: MST weight convergence (E: path-length based)
    # Consistently decreasing MST weight = population converging spatially
    recent_mst = [m['avg_mst_edge'] for m in self._topo_stagnation_history[-4:]]
    mst_decreasing = all(recent_mst[i] >= recent_mst[i+1] * 0.98 for i in range(len(recent_mst)-1))
    mst_shrinking_rate = (recent_mst[0] - recent_mst[-1]) / max(recent_mst[0], 1e-10)
    
    # Metric 3: Largest component shrinkage
    # If the biggest connected component is shrinking, population is fragmenting
    recent_lcr = [m['largest_component_ratio'] for m in self._topo_stagnation_history[-4:]]
    lcr_decreasing = recent_lcr[-1] < recent_lcr[0] * 0.85
    
    # Metric 4: k-NN edge length shrinkage (E: neighborhood-based)
    recent_knn = [m['avg_knn_edge'] for m in self._topo_stagnation_history[-4:]]
    knn_decreasing = all(recent_knn[i] >= recent_knn[i+1] * 0.97 for i in range(len(recent_knn)-1))
    
    # === Combined stagnation decision ===
    stagnation_score = 0
    
    if is_fragmented:
        stagnation_score += 3
    elif n_components > 2:
        stagnation_score += 1
    
    if mst_decreasing and mst_shrinking_rate > 0.05:
        stagnation_score += 2
    
    if lcr_decreasing:
        stagnation_score += 2
    
    if knn_decreasing:
        stagnation_score += 1
    
    # Stagnation triggers if score >= 4 (multiple signals align)
    is_stagnated = stagnation_score >= 4
    
    if is_stagnated:
        self.stagnation_counter += 1
    else:
        # Decay counter when not stagnated (reset slowly)
        self.stagnation_counter = max(0, self.stagnation_counter - 1)
    
    # Also trigger on extreme single signals
    extreme_mst_shrink = mst_shrinking_rate > 0.15
    extreme_fragmentation = n_components > NP / 2
    
    return is_stagnated or extreme_mst_shrink or extreme_fragmentation
```