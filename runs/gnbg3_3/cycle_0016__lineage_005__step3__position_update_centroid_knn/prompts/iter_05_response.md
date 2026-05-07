Looking at the current implementation, it uses k-NN distances but doesn't fully leverage the **graph structure** of the population. For Category E, I should build a proper graph (MST or k-NN graph) and derive properties like centrality, betweenness, and connectivity components.

**Idea: MST Betweenness-Corrected Centroid Pull**

Key insight: The current centroid_knn treats all particles equally. By building a **Minimum Spanning Tree (MST)** over the population, I can identify:
1. **Leaf nodes** (degree-1 in MST) — peripheral particles that need stronger centripetal correction
2. **Internal nodes** — core particles that are already well-connected
3. **MST edge weights** — measure local "tension" in the topology; heavy edges indicate sparse regions
4. **Betweenness centrality** — particles on many shortest paths are critical bridges

This is fundamentally different from distance-based approaches (Category A) because it captures **relational structure** rather than absolute positions.

```python
def _position_update_centroid_knn(self):
    """MST betweenness-corrected centroid pull (Category E: Topology/graph-based).
    
    Builds Minimum Spanning Tree over population; uses graph centrality
    and MST edge weights to modulate per-particle centroid attraction.
    Peripheral/leaves get stronger pull; internal nodes get gentler nudge.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform
    
    centroid = np.mean(self.population, axis=0)
    dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)
    
    # === STEP 1: Build distance matrix and MST ===
    if self.np > 2:
        # Compute pairwise distances
        try:
            pairwise_dists = squareform(pdist(self.population, metric='euclidean'))
            np.fill_diagonal(pairwise_dists, 1e-10)
            
            # Compute MST from distance matrix
            mst = minimum_spanning_tree(pairwise_dists)
            mst_dense = np.array(mst.todense())
            
            # Get MST edge weights
            mst_edges = np.triu(mst_dense)
            mst_edge_weights = mst_edges[mst_edges > 0]
            
            # Global MST tension (mean edge weight = typical inter-particle distance)
            global_mst_tension = np.mean(mst_edge_weights) + 1e-10
            max_mst_tension = np.max(mst_edge_weights) + 1e-10
            
            # === STEP 2: Compute per-particle MST centrality ===
            # Degree-based centrality: count connections in MST
            mst_degrees = np.array(np.sum(mst_dense > 0, axis=1)).ravel()
            
            # Betweenness approximation: particles with degree-1 (leaves) are on tree periphery
            # Higher degree = more central in MST topology
            leaf_mask = mst_degrees == 1
            
            # === STEP 3: MST edge weight per particle ===
            # For each particle, find its heaviest incident MST edge
            particle_mst_weight = np.zeros(self.np)
            for i in range(self.np):
                incident_edges = mst_dense[i, :]
                if np.any(incident_edges > 0):
                    particle_mst_weight[i] = np.max(incident_edges)
                else:
                    particle_mst_weight[i] = global_mst_tension
            
            # Normalize: heavy edge = sparse region = needs more exploration
            edge_weight_modulation = np.clip(particle_mst_weight / global_mst_tension, 0.3, 3.0)
            
            # === STEP 4: Combine signals for correction strength ===
            # Leaf particles (degree=1): more centripetal pull needed
            # Internal particles (degree>1): gentler correction
            degree_modulation = np.where(leaf_mask, 1.5, 0.7)
            
            # Combined: edge weight × degree modulation
            mst_modulation = degree_modulation * edge_weight_modulation
            
        except Exception:
            # Fallback: uniform modulation
            mst_modulation = np.ones(self.np)
            global_mst_tension = 1.0
    else:
        mst_modulation = np.ones(self.np)
        global_mst_tension = 1.0
    
    # === STEP 5: k-NN density (for comparison/fusion) ===
    k = min(5, self.np - 1)
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    knn_dists = np.sort(sq_dists, axis=1)[:, :k]
    knn_avg_dist = np.mean(np.sqrt(knn_dists), axis=1, keepdims=True) + 1e-10
    global_spread = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10
    
    # Density from k-NN
    density_modulation = np.clip(knn_avg_dist / global_spread, 0.2, 2.0)
    
    # === STEP 6: Centroid attraction with MST + density fusion ===
    centroid_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))
    
    to_centroid_dir = centroid - self.population
    to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
    to_centroid_dir = to_centroid_dir / to_centroid_dist
    
    # Fuse MST modulation with density modulation
    combined_modulation = mst_modulation[:, np.newaxis] * density_modulation
    combined_modulation = np.clip(combined_modulation, 0.2, 3.0)
    
    correction = centroid_attraction * to_centroid_dir * combined_modulation
    
    new_population = self.population + self.inertia_weight * self.velocity + 0.3 * correction
    self.population = self._clip_to_bounds(new_population)
```