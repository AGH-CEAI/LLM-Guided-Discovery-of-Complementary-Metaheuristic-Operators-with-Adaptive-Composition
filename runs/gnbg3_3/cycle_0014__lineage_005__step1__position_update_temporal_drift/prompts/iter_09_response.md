**Idea: Convex Hull Expansion with Axis-Aligned Spread Monitoring**

Geometry/Spatial mechanism using convex hull volume detection, axis-aligned bounding box spread analysis, pairwise distance distribution, and k-NN density isolation. When geometric signals indicate population collapse, applies hull-based repulsion, centroid expansion forces, dimension-wise expansion for collapsed axes, and k-NN isolation correction to escape geometric traps.

```python
def _position_update_temporal_drift(self):
    """Geometry/Spatial: Convex hull expansion + axis-aligned spread monitoring (Category A).
    
    Operates purely on literal geometric layout:
    1. Convex hull volume to detect population collapse
    2. Axis-aligned bounding box spread per dimension
    3. Pairwise distance distribution statistics
    4. k-NN neighborhood density to detect isolated particles
    
    When collapse detected, applies geometric correction:
    - Hull repulsion: push particles away from hull vertices
    - Centroid expansion: push particles outward from centroid
    - Dimension-wise expansion: expand collapsed dimensions
    - k-NN isolation correction: route isolated particles toward denser regions
    
    Targets worst tasks (17, 16, 6, 5) where population geometric collapse
    causes the swarm to get trapped in disconnected local optima.
    """
    try:
        from scipy.spatial import ConvexHull
        
        # === GEOMETRIC SIGNAL EXTRACTION ===
        
        # 1. Convex Hull Volume (pure geometric property)
        try:
            hull = ConvexHull(self.population)
            hull_volume = hull.volume if hasattr(hull, 'volume') else hull.area
            hull_vertices = hull.vertices
            n_hull_vertices = len(hull_vertices)
            
            # Expected volume for random points in dim-ball (rough approximation)
            expected_volume = np.pi ** (self.dim / 2) / np.math.gamma(self.dim / 2 + 1) * (self.upper_bound - self.lower_bound) ** self.dim
            volume_ratio = hull_volume / (expected_volume + 1e-10)
        except:
            hull_volume = 0.0
            volume_ratio = 0.0
            n_hull_vertices = 0
            hull_vertices = []
        
        # 2. Axis-Aligned Spread per dimension
        pop_min = np.min(self.population, axis=0)
        pop_max = np.max(self.population, axis=0)
        pop_range = pop_max - pop_min
        total_range = self.upper_bound - self.lower_bound
        
        spread_ratio = pop_range / (total_range + 1e-10)
        collapsed_dims = np.where(spread_ratio < 0.05)[0]
        n_collapsed_dims = len(collapsed_dims)
        
        # 3. Pairwise Distance Distribution
        sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        valid_dists = np.sqrt(sq_dists[sq_dists < np.inf])
        
        if len(valid_dists) > 0:
            mean_pairwise_dist = np.mean(valid_dists)
            std_pairwise_dist = np.std(valid_dists)
        else:
            mean_pairwise_dist = 0.0
            std_pairwise_dist = 0.0
        
        # Expected mean pairwise distance for uniform distribution
        if self.dim == 2:
            expected_mean_dist = 0.5 * (self.upper_bound - self.lower_bound)
        elif self.dim == 3:
            expected_mean_dist = (11.0 / 32.0) * (self.upper_bound - self.lower_bound)
        else:
            expected_mean_dist = 0.3 * (self.upper_bound - self.lower_bound)
        
        pairwise_ratio = mean_pairwise_dist / (expected_mean_dist + 1e-10)
        
        # 4. k-NN Density Analysis
        k = min(5, self.np - 1)
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
        knn_dists = np.sqrt(np.sort(sq_dists, axis=1)[:, :k])
        knn_mean_dist = np.mean(knn_dists) + 1e-10
        
        # Centroid and spread computation
        centroid = np.mean(self.population, axis=0)
        centroid_to_bounds_dist = np.minimum(centroid - self.lower_bound, self.upper_bound - centroid)
        min_boundary_dist = np.min(centroid_to_bounds_dist)
        
        # === COLLAPSE DETECTION (pure geometric signals) ===
        is_volume_collapsed = volume_ratio < 0.01
        is_spread_collapsed = n_collapsed_dims > self.dim // 3
        is_pairwise_collapsed = pairwise_ratio < 0.1
        is_near_boundary = min_boundary_dist < 0.1 * (self.upper_bound - self.lower_bound)
        
        # Combined collapse score
        collapse_score = (0.3 * float(is_volume_collapsed) + 
                         0.3 * float(is_spread_collapsed) + 
                         0.2 * float(is_pairwise_collapsed) + 
                         0.2 * float(is_near_boundary))
        
        # === GEOMETRIC CORRECTION MECHANISMS ===
        
        # Direction from centroid to each particle
        to_centroid = centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid / to_centroid_dist
        
        # Mechanism 1: Hull Vertex Repulsion (particles near hull get pushed inward)
        hull_repulsion = np.zeros((self.np, self.dim))
        if n_hull_vertices > 0:
            hull_particles = self.population[hull_vertices]
            for i in range(self.np):
                dists_to_hull = np.linalg.norm(self.population[i] - hull_particles, axis=1)
                nearest_idx = np.argmin(dists_to_hull)
                nearest_point = hull_particles[nearest_idx]
                
                # Push away from nearest hull vertex
                direction = self.population[i] - nearest_point
                direction_norm = np.linalg.norm(direction) + 1e-10
                hull_repulsion[i] = 0.2 * (direction / direction_norm)
        
        # Mechanism 2: Centroid Expansion Force (push outward when collapsed)
        expansion_strength = 0.5 * collapse_score
        expansion_force = expansion_strength * to_centroid_dir
        
        # Mechanism 3: Dimension-wise Expansion for collapsed dimensions
        dim_expansion = np.zeros((self.np, self.dim))
        if n_collapsed_dims > 0:
            dim_expansion_strength = 0.3
            dim_center = (self.lower_bound + self.upper_bound) / 2
            for dim_idx in collapsed_dims:
                dim_dist = self.population[:, dim_idx] - dim_center
                dim_dir = np.sign(dim_dist)
                if np.all(dim_dir == 0):
                    dim_dir = np.random.choice([-1, 1], self.np)
                dim_expansion[:, dim_idx] = dim_expansion_strength * dim_dir
        
        # Mechanism 4: k-NN Isolation Correction (push isolated particles toward denser regions)
        knn_isolation = np.zeros((self.np, self.dim))
        for i in range(self.np):
            neighbors = knn_indices[i]
            neighbor_centroid = np.mean(self.population[neighbors], axis=0)
            direction = neighbor_centroid - self.population[i]
            direction_norm = np.linalg.norm(direction) + 1e-10
            knn_isolation[i] = 0.1 * (direction / direction_norm)
        
        # Combine all geometric forces
        total_correction = hull_repulsion + expansion_force + dim_expansion + knn_isolation
        
        # Scale correction based on collapse severity
        correction_scale = 0.3 + 0.7 * collapse_score
        correction_scale = np.clip(correction_scale, 0.3, 1.0)
        
        # === FINAL POSITION UPDATE ===
        new_population = self.population + self.inertia_weight * self.velocity + correction_scale * total_correction
        
    except (np.linalg.LinAlgError, ImportError):
        new_population = self.population + self.velocity
    
    self.population = self._clip_to_bounds(new_population)
```