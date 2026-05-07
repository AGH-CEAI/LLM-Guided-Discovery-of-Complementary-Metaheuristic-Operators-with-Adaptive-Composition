**Idea: k-NN Component & MST Diameter Restart**

Use graph topology to detect population fragmentation (via connected components in k-NN graph) and convergence (via MST diameter on fitness landscape). Restart when topology signals entrapment, preserving elite positions guided by MST spacing.
```python
def _restart_if_needed(self, population, fitness):
    """Restart based on k-NN graph connectivity and MST diameter analysis."""
    NP, dim = population.shape
    
    # Build k-NN graph and compute topology metrics
    k = max(2, NP // 10)
    
    # Compute pairwise Euclidean distances
    diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    
    # Find k nearest neighbors
    nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
    
    # Build k-NN adjacency matrix
    knn_adj = np.zeros((NP, NP), dtype=bool)
    for i in range(NP):
        knn_adj[i, nearest_indices[i]] = True
        knn_adj[nearest_indices[i], i] = True
    
    # Compute connected components via BFS (no scipy needed)
    visited = np.zeros(NP, dtype=bool)
    components = []
    for start in range(NP):
        if visited[start]:
            continue
        component = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            component.append(node)
            for neighbor in np.where(knn_adj[node])[0]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(component)
    
    n_components = len(components)
    
    # Compute clustering coefficient from k-NN graph
    clustering = np.zeros(NP)
    for i in range(NP):
        neighbors = np.where(knn_adj[i])[0]
        if len(neighbors) < 2:
            continue
        edges_among = np.sum(knn_adj[np.ix_(neighbors, neighbors)]) / 2
        possible = len(neighbors) * (len(neighbors) - 1) / 2
        clustering[i] = edges_among / possible if possible > 0 else 0
    avg_clustering = np.mean(clustering)
    
    # Compute MST diameter on fitness landscape (fitness-distance MST)
    # Use Prim's algorithm for MST
    def prim_mst(weights):
        n = len(weights)
        in_mst = np.zeros(n, dtype=bool)
        in_mst[0] = True
        mst_edges = []
        edges = [(weights[0, j], 0, j) for j in range(n) if weights[0, j] < np.inf]
        heapq.heapify(edges)
        
        while edges and len(mst_edges) < n - 1:
            w, u, v = heapq.heappop(edges)
            if in_mst[v]:
                continue
            in_mst[v] = True
            mst_edges.append((u, v, w))
            for j in range(n):
                if not in_mst[j] and weights[v, j] < np.inf:
                    heapq.heappush(edges, (weights[v, j], v, j))
        
        return mst_edges
    
    # Fitness distance matrix
    fitness_dists = np.abs(fitness[:, np.newaxis] - fitness[np.newaxis, :])
    np.fill_diagonal(fitness_dists, 0)
    
    import heapq
    mst_edges = prim_mst(fitness_dists)
    
    if mst_edges:
        mst_diameter = max(edge[2] for edge in mst_edges) if mst_edges else 0
    else:
        mst_diameter = 0
    
    fitness_range = np.max(fitness) - np.min(fitness) + 1e-8
    
    # Decide restart: graph topology signals entrapment
    should_restart = False
    restart_reason = ""
    
    if n_components > NP // 3:
        should_restart = True
        restart_reason = f"fragmented_{n_components}_components"
    elif avg_clustering > 0.6 and n_components > 1:
        should_restart = True
        restart_reason = f"clustered_high_{avg_clustering:.2f}"
    elif mst_diameter < 0.1 * fitness_range:
        should_restart = True
        restart_reason = f"mst_narrow_{mst_diameter:.4f}"
    
    if not should_restart:
        return (None, None, None)
    
    # Restart: preserve elite individuals, regenerate rest using graph structure
    n_elites = max(3, NP // 5)
    elite_indices = np.argsort(fitness)[:n_elites]
    elites = population[elite_indices].copy()
    
    # Build MST on elite positions to determine spacing
    if len(elites) > 1:
        elite_diffs = elites[:, np.newaxis, :] - elites[np.newaxis, :, :]
        elite_sq_dists = np.sum(elite_diffs ** 2, axis=2)
        np.fill_diagonal(elite_sq_dists, 0)
        elite_mst_edges = prim_mst(elite_sq_dists)
        if elite_mst_edges:
            avg_spacing = np.mean([e[2] for e in elite_mst_edges])
            avg_spacing = max(avg_spacing, 1.0)
        else:
            avg_spacing = 20.0
    else:
        avg_spacing = 20.0
    
    # Regenerate population using graph-guided diversity
    new_population = np.empty((NP, dim))
    new_population[:n_elites] = elites
    
    for i in range(n_elites, NP):
        if np.random.rand() < 0.4 and len(elites) > 0:
            # Expand along directions away from elite MST
            base_elite = elites[np.random.randint(len(elites))]
            direction = np.random.randn(dim)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            distance = np.random.uniform(1.5 * avg_spacing, 3.0 * avg_spacing)
            new_population[i] = base_elite + direction * distance
        else:
            # Random with spread based on current population extent
            new_population[i] = np.random.uniform(-100, 100, dim)
    
    new_population = np.clip(new_population, -100.0, 100.0)
    
    return (new_population, None, None)
```