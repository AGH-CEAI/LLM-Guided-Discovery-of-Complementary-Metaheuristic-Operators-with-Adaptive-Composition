**Idea: MST-Backbone Bridge Detection**
One-line description (Category E: Topology/graph-based). Build a Minimum Spanning Tree over the population to identify bridge particles connecting subpopulations; particles at network bottlenecks receive targeted perturbations to escape local optima.

```python
def _position_update_temporal_drift(self):
    """MST-backbone graph topology with bridge detection (Category E).
    
    Key insight: Use Minimum Spanning Tree (MST) to find the skeletal
    backbone structure of the population. Bridge edges (whose removal
    disconnects the MST) mark critical bottlenecks where particles may
    be trapped between subpopulations. Particles at bridges or in
    peripheral branches get stronger exploration nudges toward the
    global best. This is fundamentally different from k-NN (local
    neighborhoods) or Laplacian (spectral clustering) approaches.
    """
    np = self.np
    dim = self.dim
    
    # ---- Step 1: Build complete distance matrix ----
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # ---- Step 2: Kruskal's MST using union-find ----
    parent = np.arange(np)
    rank = np.zeros(np, dtype=int)
    
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
    
    edge_list = []
    for i in range(np):
        for j in range(i + 1, np):
            edge_list.append((sq_dists[i, j], i, j))
    edge_list.sort()
    
    mst_edges = []
    for dist, i, j in edge_list:
        if union(i, j):
            mst_edges.append((i, j, dist))
        if len(mst_edges) == np - 1:
            break
    
    # Fallback if MST construction failed
    if len(mst_edges) < np - 1:
        new_pop = self.population + self.velocity
        self.population = self._clip_to_bounds(new_pop)
        return
    
    # ---- Step 3: Find bridge edges in MST (2-edge-connected components) ----
    # Build adjacency list from MST edges
    adj = [[] for _ in range(np)]
    for i, j, d in mst_edges:
        adj[i].append((j, d))
        adj[j].append((i, d))
    
    # DFS to find bridges using Tarjan's algorithm
    disc = np.full(np, -1, dtype=int)
    low = np.full(np, -1, dtype=int)
    time = [0]
    bridges = set()
    
    def dfs_bridge(u, parent_edge_idx):
        disc[u] = low[u] = time[0]
        time[0] += 1
        for v, d in adj[u]:
            if disc[v] == -1:
                dfs_bridge(v, (min(u, v), max(u, v)))
                low[u] = min(low[u], low[v])
                # Bridge condition: low[v] > disc[u]
                if low[v] > disc[u]:
                    bridges.add((min(u, v), max(u, v)))
            elif (min(u, v), max(u, v)) != parent_edge_idx:
                low[u] = min(low[u], disc[v])
    
    for i in range(np):
        if disc[i] == -1:
            dfs_bridge(i, (-1, -1))
    
    # ---- Step 4: Compute MST-based centrality (distance from each node to MST centroid) ----
    # Approximate MST centroid as the node with minimum total path distance
    # Use all-pairs shortest paths on MST (tree DP)
    mst_dist = np.full((np, np), np.inf)
    for i in range(np):
        mst_dist[i, i] = 0.0
    
    # BFS from each node to compute tree distances
    for start in range(np):
        visited = np.zeros(np, dtype=bool)
        queue = [start]
        visited[start] = True
        dist_map = {start: 0}
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v, d in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
                    dist_map[v] = dist_map[u] + d
    
    for i, j in dist_map.items():
        mst_dist[start, i] = j
    
    # Node centrality = mean distance to all other nodes in MST
    centrality = np.mean(mst_dist, axis=1)
    # High centrality = near MST center; low = peripheral
    centrality_norm = centrality / (np.max(centrality) + 1e-10)
    
    # ---- Step 5: Identify bridge endpoint particles ----
    bridge_endpoints = np.zeros(np, dtype=bool)
    for i, j in bridges:
        bridge_endpoints[i] = True
        bridge_endpoints[j] = True
    
    # ---- Step 6: Compute velocity modulation factors ----
    # Particles at bridges (bottlenecks) get exploration boost
    bridge_boost = np.where(bridge_endpoints, 1.5, 1.0)
    # Peripheral particles (low MST centrality) get exploration boost
    peripheral_boost = 1.0 + 0.3 * (1.0 - centrality_norm)
    # Isolated particles (small MST degree) get extra boost
    mst_degree = np.array([len(adj[i]) for i in range(np)])
    isolated_boost = np.where(mst_degree == 1, 1.3, 1.0)
    
    vel_scale = bridge_boost * peripheral_boost * isolated_boost
    vel_scale = np.clip(vel_scale, 0.5, 2.5)
    
    # ---- Step 7: Apply MST-aware directional perturbations ----
    # Pull peripheral and bridge particles toward global best
    if self.global_best is not None:
        to_best = self.global_best - self.population
        to_best_dist = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
        to_best_dir = to_best / to_best_dist
        
        # Stronger pull for peripheral/bridge particles
        pull_strength = 0.3 * (1.0 - centrality_norm) * np.where(bridge_endpoints, 1.5, 1.0)
        directional = pull_strength[:, np.newaxis] * to_best_dir
    else:
        directional = 0.1 * np.random.uniform(-1, 1, (np, dim))
    
    # ---- Step 8: Update positions ----
    new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + directional
    self.population = self._clip_to_bounds(new_population)
```