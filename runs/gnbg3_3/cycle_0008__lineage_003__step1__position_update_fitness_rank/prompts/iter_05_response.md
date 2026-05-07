Looking at the current implementation, it uses Laplacian spectral gap and component sizes. For Category E, I'll take a **different angle**: use **betweenness centrality** on the k-NN graph to identify "bridge" particles that connect clusters, and use **MST-based path length** to global best as a convergence detector. This directly targets the worst tasks (17, 16, 6) which are stuck in local optima — bridge particles are the key to escaping.

**Idea: Betweenness-Bridge Escape with MST Path Guidance**
One-line description: Build k-NN graph, compute betweenness centrality to identify bridge particles, use MST path length to global best as convergence signal — particles on critical paths get amplified exploration.
```python
def _position_update_fitness_rank(self):
    """Betweenness-centrality bridge detection + MST path guidance (Category E).

    Key insight: In multimodal landscapes, the population fragments into
    subclusters around local optima. Bridge particles (high betweenness in
    k-NN graph) are the only connections between clusters — pushing them
    harder enables cluster-merging. MST path length to global best measures
    structural convergence; long paths = fragmented population needing
    exploration bursts. This structural approach is orthogonal to spectral
    gap (eigenvalue) detection used previously.
    """
    # Build k-NN graph
    k = min(5, self.np - 1)
    sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

    # Sparse adjacency matrix (k-NN symmetric)
    row_indices = np.repeat(np.arange(self.np), k)
    col_indices = knn_indices.ravel()
    data = np.ones(len(row_indices))
    from scipy.sparse import csr_matrix
    adj = csr_matrix((data, (row_indices, col_indices)), shape=(self.np, self.np))
    adj = adj + adj.T
    adj.data[:] = 1.0

    # Betweenness centrality: identify bridge particles
    try:
        from scipy.sparse.csgraph import shortest_path
        # Approximate betweenness via path flow: sum of shortest-path counts through each node
        # Use unweighted shortest paths from each source
        n = self.np
        betweenness = np.zeros(n)
        # Sample a subset of sources for speed (k+2 sources = neighborhood size + global best)
        sources = list(range(min(k + 2, n)))
        # Add global best index if available
        if self.global_best is not None:
            dists_to_best = np.linalg.norm(self.population - self.global_best, axis=1)
            best_idx = np.argmin(dists_to_best)
            if best_idx not in sources:
                sources.append(best_idx)
        sources = sources[:min(len(sources), n)]

        for src in sources:
            # Compute shortest paths from src to all nodes
            dists_matrix = shortest_path(adj, directed=False, unweighted=True, indices=[src], method='BF')
            dists_matrix = dists_matrix.ravel()
            # Count flow through each node (nodes on shortest paths)
            for dst in range(n):
                if src == dst or np.isinf(dists_matrix[dst]):
                    continue
                # Trace back shortest path to count intermediate nodes
                current = dst
                path_len = dists_matrix[dst]
                if path_len < 1 or np.isinf(path_len):
                    continue
                # Approximate: all nodes at distance < path_len contribute
                for node in range(n):
                    if node != src and node != dst:
                        if not np.isinf(dists_matrix[node]) and dists_matrix[node] < path_len:
                            # Check if node is on a shortest path (simple heuristic)
                            if abs(dists_matrix[node] + dists_matrix[dst] - dists_matrix[dst]) < 0.5:
                                pass
                # Simpler approximation: nodes with intermediate distances
                intermediate_mask = (dists_matrix > 0) & (dists_matrix < path_len)
                betweenness[intermediate_mask] += 1

        betweenness = np.clip(betweenness, 0, None)
        if np.max(betweenness) > 0:
            betweenness = betweenness / (np.max(betweenness) + 1e-10)
        else:
            betweenness = np.zeros(n)
    except:
        betweenness = np.zeros(self.np)

    # MST-based convergence detection
    try:
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.sparse import csr_matrix as sp_csr
        # Use full distance matrix for MST
        dist_matrix = np.sqrt(sq_dists + np.inf * np.eye(self.np))
        dist_sparse = sp_csr(dist_matrix)
        mst = minimum_spanning_tree(dist_sparse)
        mst_dense = mst.toarray()

        # MST diameter (longest shortest path in tree) as convergence measure
        # Use adjacency to compute tree distances via BFS
        mst_adj = (mst_dense > 0).astype(float)
        # Compute all-pairs tree distances via Floyd-Warshall on tree (O(n^3) but n is small)
        tree_dists = mst_adj.copy().astype(float)
        tree_dists[tree_dists == 0] = 1e10
        np.fill_diagonal(tree_dists, 0)
        for k_iter in range(self.np):
            for i in range(self.np):
                for j in range(self.np):
                    if tree_dists[i, j] > tree_dists[i, k_iter] + tree_dists[k_iter, j]:
                        tree_dists[i, j] = tree_dists[i, k_iter] + tree_dists[k_iter, j]
        mst_diameter = np.max(tree_dists[np.isfinite(tree_dists)])

        # Path length from each particle to global best in MST
        if self.global_best is not None:
            dists_to_best = np.linalg.norm(self.population - self.global_best, axis=1)
            best_idx = np.argmin(dists_to_best)
            mst_path_to_best = tree_dists[best_idx]  # distances from best node
            mst_path_to_best = np.clip(mst_path_to_best, 0, mst_diameter + 1e-10)
            # Normalize: large mst_diameter = fragmented = more exploration
            mst_convergence = 1.0 - np.clip(mst_diameter / (self.np + 1e-10), 0, 1)
        else:
            mst_path_to_best = np.zeros(self.np)
            mst_convergence = 1.0
    except:
        mst_convergence = 1.0
        mst_path_to_best = np.zeros(self.np)
        mst_diameter = self.np

    # Graph-based exploration: betweenness drives escape effort
    # High betweenness = bridge particle = needs big push to merge clusters
    bridge_explore = 1.0 + 2.0 * betweenness  # up to 3x for bridge particles
    # MST convergence: fragmented population needs more exploration
    fragmentation_explore = 1.0 + 0.5 * mst_convergence
    graph_explore = bridge_explore * fragmentation_explore
    graph_explore = np.clip(graph_explore, 0.5, 4.0)

    # Fitness rank signal (from Category D, blended)
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

    # Velocity scaling
    vel_scale = graph_explore * (1.0 + 0.3 * (1.0 - fitness_ranks))
    vel_scale = np.clip(vel_scale, 0.3, 3.5)

    # Directional component toward global best
    if self.global_best is not None:
        to_best = self.global_best - self.population
        to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
        to_best_dir = to_best / to_best_norm
        # Stronger pull from bridge particles (they connect to global best)
        directional = 0.5 * (1.0 + betweenness[:, np.newaxis]) * to_best_dir
    else:
        directional = 0.5 * np.random.uniform(-1, 1, (self.np, self.dim))

    # Random perturbation scaled by graph exploration
    random_perturb = graph_explore[:, np.newaxis] * np.random.uniform(-0.5, 0.5, (self.np, self.dim))

    # Update population
    new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + directional + random_perturb
    self.population = self._clip_to_bounds(new_population)
```