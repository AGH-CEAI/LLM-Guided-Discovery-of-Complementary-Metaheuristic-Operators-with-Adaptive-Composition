**Idea: k-NN Graph Connectivity and MST Path-Length Inertia**

Adapt inertia weight using graph-theoretic properties of the population: k-nearest-neighbor graph connectivity ratio (detects fragmentation/multimodality) and minimum spanning tree average edge weight (captures true geodesic spread). High connectivity → increase inertia to maintain exploration; fragmented/sparse → decrease inertia to accelerate convergence. Category E.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using k-NN graph connectivity and MST path-length.
    
    Category E (Topology / graph-based):
      - Build k-NN graph over population
      - Compute connectivity ratio (fraction of particles with ≥k/2 neighbors)
      - Compute MST average edge weight as true geodesic spread
      - High connectivity + low spread → increase inertia (maintain exploration)
      - Fragmented graph or high spread → decrease inertia (converge faster)
    """
    npops = self.np
    k = min(5, npops - 1)
    
    # --- Build k-NN graph and compute pairwise distances ---
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    
    # k-NN adjacency: k nearest neighbors per particle (excluding self)
    knn_indices = np.argsort(sq_dists, axis=1)[:, 1:k+1]
    knn_dists = np.take_along_axis(sq_dists, knn_indices, axis=1)
    
    # --- Graph Property 1: Connectivity ratio ---
    # Count how many neighbors each particle has within its k-th neighbor distance
    threshold = knn_dists[:, -1:]
    connection_counts = np.sum(sq_dists[:, 1:] <= threshold, axis=1)
    connectivity_ratio = np.mean(connection_counts >= (k // 2 + 1))
    
    # --- Graph Property 2: MST average edge weight (true geodesic spread) ---
    # Use k-NN graph edges as candidate MST edges (sparse approximation)
    # This captures actual path distances, not axis-aligned bounds
    candidate_edges = []
    candidate_weights = []
    for i in range(npops):
        for j in knn_indices[i]:
            if i < j:
                candidate_edges.append((i, j))
                candidate_weights.append(np.sqrt(sq_dists[i, j]))
    
    if len(candidate_weights) >= npops - 1:
        # Kruskal-like: sort and greedily build MST
        sorted_edge_indices = np.argsort(candidate_weights)
        parent = np.arange(npops)
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        mst_edges = []
        mst_total_weight = 0.0
        edges_added = 0
        for idx in sorted_edge_indices:
            if edges_added >= npops - 1:
                break
            i, j = candidate_edges[idx]
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
                mst_edges.append((i, j))
                mst_total_weight += candidate_weights[idx]
                edges_added += 1
        
        mst_avg_edge = mst_total_weight / max(edges_added, 1)
    else:
        # Fallback: use average k-NN distance
        mst_avg_edge = np.mean(knn_dists)
    
    # Normalize MST edge weight using population scale estimate
    pop_scale = np.mean(np.std(self.population, axis=0)) + 1e-10
    mst_spread_normalized = np.clip(mst_avg_edge / (pop_scale * 3.0), 0.0, 1.0)
    
    # --- Combine graph signals into target inertia ---
    # Connectivity: high → population is cohesive → higher inertia (explore more)
    # Spread: high → particles far apart → lower inertia (converge faster)
    connectivity_component = 0.4 + 0.4 * connectivity_ratio
    spread_component = 0.95 - 0.4 * mst_spread_normalized
    
    target_inertia = connectivity_component * 0.6 + spread_component * 0.4
    target_inertia = np.clip(target_inertia, 0.4, 0.95)
    
    # Smooth EMA update
    self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
    self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```