**Idea: MST-Betweenness Explorer**
Category E: Uses Minimum Spanning Tree (MST) edge weights and betweenness-like analysis to detect graph bottlenecks and drive particles from sparse MST regions toward the global optimum.

```python
def _position_update_centroid_knn(self):
    """MST edge-weight analysis for topology-aware exploration (Category E).
    
    Key insight: Build MST over population; particles in sparse MST regions
    (long edges, low betweenness) get strong corrective pulls toward the
    global best. This is orthogonal to centroid-gravity (A) and density-knn (A)
    approaches — it uses GRAPH STRUCTURE (edge weights, connectivity) to
    identify underexplored regions.
    """
    # Build MST over population using Prim's algorithm (vectorized)
    n = self.np
    dists = np.linalg.norm(self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :], axis=2)
    
    in_tree = np.zeros(n, dtype=bool)
    parent = np.full(n, -1, dtype=int)
    min_edge = np.full(n, np.inf)
    min_edge[0] = 0.0
    in_tree[0] = True
    
    for _ in range(n - 1):
        best_idx = -1
        best_val = np.inf
        for i in range(n):
            if not in_tree[i]:
                if dists[i, np.where(in_tree)[0]].min() < best_val:
                    best_val = dists[i, np.where(in_tree)[0]].min()
                    best_idx = i
        if best_idx >= 0:
            in_tree[best_idx] = True
            j = np.argmin(dists[best_idx, np.where(in_tree)[0][:-1]]) if np.sum(in_tree) > 1 else 0
            parent[best_idx] = np.where(in_tree)[0][j if np.sum(in_tree) > 1 else 0]
            min_edge[best_idx] = dists[best_idx, parent[best_idx]]
    
    # Compute MST-based signals
    valid_edges = min_edge[min_edge < np.inf]
    if len(valid_edges) == 0:
        correction = np.zeros((n, self.dim))
    else:
        edge_mean = np.mean(valid_edges)
        edge_std = np.std(valid_edges) + 1e-10
        
        # Betweenness approximation: particles on long MST edges (sparse regions)
        # are "bridges" that need to move toward the global best
        particle_betweenness = np.zeros(n)
        for i in range(n):
            if parent[i] >= 0:
                edge_len = min_edge[i]
                # Higher betweenness for particles on longer-than-average edges
                particle_betweenness[i] = max(0, (edge_len - edge_mean) / edge_std)
            if parent[i] >= 0:
                edge_len = min_edge[parent[i]]
                particle_betweenness[parent[i]] += max(0, (edge_len - edge_mean) / edge_std)
        
        # Sparse regions get strong pull toward global best
        betweenness_factor = np.clip(1.0 + 0.4 * particle_betweenness, 1.0, 3.0)
        
        # Also use MST edge weight as direct sparsity signal
        mst_sparsity = np.clip(min_edge / (edge_mean + 1e-10), 0.5, 2.0)
        mst_sparsity = np.where(np.isinf(mst_sparsity) | (min_edge == 0), 1.0, mst_sparsity)
        
        # Combined topology signal: sparse MST regions need stronger correction
        topology_modulation = betweenness_factor * mst_sparsity
        
        # Apply correction toward global best
        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_dist = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_dist
            
            # Stronger correction for sparse/topologically-isolated particles
            correction_strength = 0.3 * topology_modulation[:, np.newaxis]
            correction = correction_strength * to_best_dir
        else:
            # Fallback: move toward centroid
            centroid = np.mean(self.population, axis=0)
            to_centroid = centroid - self.population
            to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
            to_centroid_dir = to_centroid / to_centroid_dist
            correction = 0.3 * topology_modulation[:, np.newaxis] * to_centroid_dir
    
    # Velocity damping in sparse regions (more exploration)
    sparsity_scale = np.clip(1.0 / (np.mean(valid_edges) / (valid_edges + 1e-10) + 1e-10), 0.5, 1.5)
    damped_velocity = self.velocity * sparsity_scale[:, np.newaxis]
    
    new_population = self.population + self.inertia_weight * damped_velocity + 0.4 * correction
    self.population = self._clip_to_bounds(new_population)
```