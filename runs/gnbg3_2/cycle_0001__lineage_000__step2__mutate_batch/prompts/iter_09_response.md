**Idea: Convex Hull Boundary-Driven Mutation**

Category A (Geometry / spatial) mutation using convex hull vertices and pairwise distances between extreme points to drive mutation toward population boundaries. This is geometrically distinct from the centroid-based approach by focusing on the convex hull structure rather than weighted centroids.

```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Generate mutation vectors using convex hull geometry:
        - Compute convex hull of population
        - Select hull vertices as extreme boundary points
        - Use pairwise distances between hull points to weight mutations
        - Drive mutations toward unexplored boundary regions
        """
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        # Compute convex hull of population to identify boundary points
        from scipy.spatial import ConvexHull
        
        try:
            hull = ConvexHull(population)
            hull_indices = hull.vertices
        except Exception:
            # Fallback: use centroid-based mutation if hull fails
            hull_indices = np.arange(np_pop)
        
        hull_points = population[hull_indices]
        n_hull = len(hull_points)
        
        if n_hull >= 2:
            # Compute pairwise distances between all hull points
            hull_dist = np.zeros((n_hull, n_hull))
            for i in range(n_hull):
                hull_dist[i] = np.linalg.norm(hull_points - hull_points[i], axis=1)
            
            # Find the pair of hull points with maximum distance
            max_dist_idx = np.unravel_index(np.argmax(hull_dist), hull_dist.shape)
            extreme_a = hull_points[max_dist_idx[0]]
            extreme_b = hull_points[max_dist_idx[1]]
            
            # Direction from extreme_a to extreme_b (boundary exploration direction)
            boundary_direction = extreme_b - extreme_a
            
            # Compute average distance to normalize
            avg_hull_dist = np.mean(hull_dist[hull_dist > 0]) + 1e-10
        else:
            # Fallback: use centroid difference
            centroid = np.mean(population, axis=0)
            boundary_direction = np.zeros(dim)
            avg_hull_dist = 1.0
        
        # Ring-based parents for rand/1 component
        r1 = ring_prev
        r2 = ring_next
        
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        # Adaptive F with geometric scaling
        current_F = self.F * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)
        
        # Base mutation: ring-based rand/1
        base_mutation = rand1_vectors + current_F * (rand2_vectors - population)
        
        # Boundary-driven component: push toward hull extremes
        boundary_component = np.tile(boundary_direction, (np_pop, 1))
        
        # Scale boundary component inversely with average hull distance
        boundary_strength = 0.2 * (1.0 / (avg_hull_dist + 1.0))
        boundary_component = boundary_strength * boundary_component
        
        # Blend: geometric component + rand/1
        blend_weight = 0.15 + 0.25 * (1.0 - self.Cr)
        trials = (1 - blend_weight) * base_mutation + blend_weight * (population + boundary_component)
        
        return trials, current_F
```