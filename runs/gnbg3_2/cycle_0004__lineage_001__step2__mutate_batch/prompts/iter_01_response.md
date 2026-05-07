**Idea: k-NN Local Structure with Convex Hull Boundary Detection**

Category A: Geometry / spatial. Uses pairwise distances to find k-nearest neighbors for computing local direction vectors, identifies convex hull boundary points, and scales mutation strength inversely with centroid distance for geometric exploration.

```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Generate mutation vectors using geometric structure:
        - k-NN local direction vectors for each individual
        - Convex hull boundary detection for adaptive scaling
        - Axis-aligned spread normalization
        """
        np_pop, dim = population.shape
        
        # Compute pairwise Euclidean distance matrix
        expanded = population[:, np.newaxis, :]
        diff_matrix = expanded - population[np.newaxis, :, :]
        pairwise_dist = np.linalg.norm(diff_matrix, axis=2)
        
        # Replace zero diagonal with inf to exclude self
        np.fill_diagonal(pairwise_dist, np.inf)
        
        # k-NN structure: find k nearest neighbors
        k = min(5, np_pop - 1)
        knn_indices = np.argsort(pairwise_dist, axis=1)[:, :k]
        
        # Compute local direction vectors from k-NN
        # For each individual, direction = weighted mean of neighbor offsets
        neighbor_weights = 1.0 / (pairwise_dist[np.arange(np_pop)[:, None], knn_indices] + 1e-10)
        neighbor_weights = neighbor_weights / (neighbor_weights.sum(axis=1, keepdims=True) + 1e-10)
        
        local_directions = np.zeros((np_pop, dim))
        for i in range(np_pop):
            neighbors = population[knn_indices[i]]
            offsets = neighbors - population[i]
            local_directions[i] = np.sum(offsets * neighbor_weights[i, :, np.newaxis], axis=0)
        
        # Compute population centroid and centroid distances (geometric only)
        centroid = np.mean(population, axis=0)
        centroid_distances = np.linalg.norm(population - centroid, axis=1)
        max_centroid_dist = np.max(centroid_distances) + 1e-10
        
        # Normalize centroid distance to [0.1, 1.0] range
        norm_centroid_dist = 0.1 + 0.9 * (centroid_distances / max_centroid_dist)
        
        # Identify convex hull boundary points via Graham scan approach
        # Use axis-aligned bounding box corners as reference
        mins = np.min(population, axis=0)
        maxs = np.max(population, axis=0)
        
        # Compute boundary score: higher = closer to boundary
        boundary_score = np.zeros(np_pop)
        for d in range(dim):
            boundary_score += (population[:, d] - mins[d]) / (maxs[d] - mins[d] + 1e-10)
            boundary_score += (maxs[d] - population[:, d]) / (maxs[d] - mins[d] + 1e-10)
        boundary_score = boundary_score / (2 * dim)
        
        # Convex hull approximation: extreme points have boundary_score near 0 or 2
        is_hull_point = np.any([
            np.allclose(population, mins, atol=1e-5),
            np.allclose(population, maxs, atol=1e-5)
        ], axis=0)
        
        # Ring-based parents for baseline differential component
        r1 = ring_prev
        r2 = ring_next
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        # Adaptive F with geometric scaling
        current_F = self.F * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)
        
        # Base differential mutation
        base_mutation = rand1_vectors + current_F * (rand2_vectors - population)
        
        # Geometric mutation strength: inverse scaling from centroid distance
        # Points near centroid (low norm_dist) get stronger exploration
        exploration_factor = 1.0 / (norm_centroid_dist + 0.2)
        
        # Hull points get additional outward bias
        hull_boost = np.where(is_hull_point, 1.3, 1.0)
        total_scale = exploration_factor * hull_boost
        total_scale = np.clip(total_scale, 0.5, 2.5)
        
        # Geometric component: centroid + local k-NN direction
        geometric_component = centroid + total_scale[:, np.newaxis] * local_directions
        
        # Blend geometric and base mutations
        # Boundary points favor geometric (exploration), interior favor base (exploitation)
        blend_weight = 0.15 + 0.35 * boundary_score
        blend_weight = np.clip(blend_weight, 0.1, 0.5)
        
        trials = (1 - blend_weight[:, np.newaxis]) * base_mutation + blend_weight[:, np.newaxis] * geometric_component
        
        return trials, current_F
```