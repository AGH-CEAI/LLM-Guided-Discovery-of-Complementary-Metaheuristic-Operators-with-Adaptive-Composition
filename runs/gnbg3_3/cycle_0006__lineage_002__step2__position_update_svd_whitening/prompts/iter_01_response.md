**Idea: Pairwise Distance Distribution Modulation**

Category A geometry variant using literal pairwise distance statistics and axis-aligned spread instead of SVD/spectral analysis. Operates on the raw spatial layout without matrix decomposition.
```python
def _position_update_svd_whitening(self):
    """Geometric pairwise distance + axis-aligned spread modulation (Category A).
    
    Operates on literal geometric layout: pairwise distance distribution,
    axis-aligned bounding box spread, and centroid distances. No spectral
    decomposition - purely geometric statistics of point configuration.
    """
    # Axis-aligned bounding box per dimension
    pop_min = np.min(self.population, axis=0)
    pop_max = np.max(self.population, axis=0)
    axis_spread = pop_max - pop_min + 1e-10
    
    # Relative spread per dimension (normalized by search bounds)
    bounds_range = self.upper_bound - self.lower_bound + 1e-10
    rel_spread = axis_spread / bounds_range
    
    # Pairwise distance distribution statistics (geometric feature)
    # Sample pairs for efficiency on large populations
    n = self.np
    if n > 2:
        max_samples = min(500, n * (n - 1) // 2)
        sample_size = min(max_samples, 200)
        
        # Sample random pairs
        all_pairs = []
        for _ in range(sample_size):
            i, j = np.random.choice(n, 2, replace=False)
            all_pairs.append((i, j))
        
        distances = np.array([
            np.linalg.norm(self.population[i] - self.population[j])
            for i, j in all_pairs
        ])
        mean_pair_dist = np.mean(distances)
        std_pair_dist = np.std(distances)
    else:
        mean_pair_dist = bounds_range[0]
        std_pair_dist = 0.0
    
    # Per-particle centroid distance (geometric position)
    centroid = np.mean(self.population, axis=0)
    dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True) + 1e-10
    
    # Geometric modulation: axis-aligned spread per dimension
    # Low spread dimension = potentially collapsed = amplify velocity
    # High spread dimension = already explored = dampen velocity
    spread_modulation = np.clip(1.0 / (rel_spread + 0.05), 0.3, 3.0)
    
    # Geometric modulation: per-particle centroid proximity
    # Particles far from centroid in sparse regions get exploration boost
    # Particles near centroid in dense regions get dampening
    centroid_scale = np.clip(dist_to_centroid / mean_pair_dist, 0.2, 2.5)
    
    # Detect collapsed population (all points very close together)
    collapse_threshold = 0.1 * np.mean(bounds_range)
    is_collapsed = mean_pair_dist < collapse_threshold
    
    # Combine geometric signals
    per_component_scale = spread_modulation * centroid_scale
    
    if is_collapsed:
        # Isotropic expansion for collapsed population
        per_component_scale = np.clip(per_component_scale, 1.5, 3.0)
    else:
        per_component_scale = np.clip(per_component_scale, 0.3, 2.0)
    
    # Apply geometric modulation to velocity
    vel_scaled = self.velocity * per_component_scale
    new_population = self.population + vel_scaled
    
    self.population = self._clip_to_bounds(new_population)
```