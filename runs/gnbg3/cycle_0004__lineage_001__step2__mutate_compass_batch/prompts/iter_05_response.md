**Idea: k-NN Graph Betweenness and MST Edge Mutation**
Uses k-NN graph centrality to identify hub individuals, extracts a minimum spanning tree to find structural connections, and generates mutation vectors from different graph regions with scores based on neighborhood distinctiveness.

```python
def _mutate_compass_batch(self, population, fitness):
    """Graph topology-based compass mutation using k-NN centrality and MST edges."""
    NP, dim = population.shape
    
    # Build k-NN graph (k based on population size)
    k = max(2, min(5, NP // 10))
    
    # Compute pairwise distances
    diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Find k nearest neighbors for each individual
    nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
    
    # Compute degree centrality (number of times each individual appears in others' k-NN)
    centrality = np.zeros(NP)
    for neighbors in nearest_indices:
        centrality[neighbors] += 1
    centrality /= (NP - 1)  # Normalize
    
    # Build MST using Kruskal's algorithm to find structural connections
    # Flatten distance matrix for edge list
    edge_indices = np.triu_indices(NP, k=1)
    edge_dists = sq_dists[edge_indices]
    sorted_edge_order = np.argsort(edge_dists)
    
    # Union-Find for Kruskal
    parent = np.arange(NP)
    rank = np.zeros(NP)
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True
    
    mst_edges = []
    for idx in sorted_edge_order:
        i, j = edge_indices[0][idx], edge_indices[1][idx]
        if union(i, j):
            mst_edges.append((i, j, edge_dists[idx]))
            if len(mst_edges) >= NP - 1:
                break
    
    # Select mutation sources based on graph topology
    # High-centrality nodes: well-connected hub individuals
    hub_indices = np.argsort(centrality)[-max(3, NP // 5):]
    hub = hub_indices[np.random.randint(len(hub_indices))]
    
    # Boundary nodes: low centrality (potentially isolated or in sparse regions)
    boundary_indices = np.argsort(centrality)[:max(3, NP // 5)]
    boundary = boundary_indices[np.random.randint(len(boundary_indices))]
    
    # Generate candidate mutation vectors from different graph regions
    candidates = []
    
    # Candidate 1: Hub to boundary direction (exploit hub influence, explore sparse regions)
    if np.random.rand() < 0.3:
        d1 = population[hub] + self.F * (population[hub] - population[boundary])
    else:
        d1 = population[hub] + self.F * (population[hub] - population[boundary])
    score1 = centrality[hub] * 0.5  # Reward using high-centrality source
    
    # Candidate 2: Use MST edge (structurally connected pair)
    if len(mst_edges) >= 2:
        edge1_idx = np.random.randint(len(mst_edges))
        edge2_idx = (edge1_idx + 1) % len(mst_edges)
        
        e1_i, e1_j, _ = mst_edges[edge1_idx]
        e2_i, e2_j, _ = mst_edges[edge2_idx]
        
        # Direction along MST edge 1, weighted by MST edge 2
        d2 = population[e1_i] + self.F * (population[e1_j] - population[e1_i]) + \
             0.5 * self.F * (population[e2_i] - population[e2_j])
        score2 = 0.4  # Moderate score for MST-based mutation
        
        # Candidate 3: Between MST nodes (structural exploration)
        mst_nodes = list(set([e[0] for e in mst_edges] + [e[1] for e in mst_edges]))
        if len(mst_nodes) >= 2:
            n1, n2 = np.random.choice(mst_nodes, 2, replace=False)
            d3 = population[n1] + self.F * (population[n2] - population[n1])
            
            # Score by neighborhood distinctiveness (Jaccard distance of k-NN neighborhoods)
            nn1 = set(nearest_indices[n1])
            nn2 = set(nearest_indices[n2])
            jaccard = len(nn1 & nn2) / max(len(nn1 | nn2), 1)
            score3 = 1.0 - jaccard  # Higher score for distinct neighborhoods
        else:
            d3 = population[hub] + self.F * (population[boundary] - population[hub])
            score3 = 0.3
        
        candidates = [(d1, score1), (d2, score2), (d3, score3)]
    else:
        # Fallback if MST construction fails
        d2 = population[hub] + self.F * (population[np.random.randint(NP)] - population[boundary])
        score2 = 0.3
        candidates = [(d1, score1), (d2, score2)]
    
    # Select best candidate based on score
    best_dir, _ = max(candidates, key=lambda x: x[1])
    
    # Create mutants by applying the selected direction to all individuals
    # Each individual gets the same topology-guided direction (batch operation)
    mutants = population + np.random.uniform(0.5, 1.5, (NP, 1)) * self.F * (best_dir - population)
    
    return np.clip(mutants, -100.0, 100.0)
```