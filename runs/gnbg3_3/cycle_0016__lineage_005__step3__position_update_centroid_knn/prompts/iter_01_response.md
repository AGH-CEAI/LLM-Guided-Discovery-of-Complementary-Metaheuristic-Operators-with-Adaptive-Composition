**Idea: Convex Hull Boundary + Axis-Aligned Spread**
Uses convex hull to identify boundary particles and applies inward correction; uses axis-aligned bounding box to modulate per-dimension movement based on how compressed the population is along each axis. Different mechanism from k-NN density modulation.
```python
def _position_update_centroid_knn(self):
        """Convex hull boundary + axis-aligned spread modulation (Category A).
        
        Key insight: boundary particles (on convex hull) are pushed inward;
        interior particles get exploration modulated by axis-aligned spread
        per dimension. Orthogonal to k-NN density: uses global hull geometry
        and per-axis spread instead of local nearest-neighbor distances.
        """
        # --- Axis-aligned bounding box ---
        pop_min = np.min(self.population, axis=0)
        pop_max = np.max(self.population, axis=0)
        pop_range = pop_max - pop_min + 1e-10
        
        # Per-dimension spread ratio (0 = collapsed, 1 = full range)
        global_range = self.upper_bound - self.lower_bound
        spread_ratio = pop_range / (global_range + 1e-10)
        spread_ratio = np.clip(spread_ratio, 0.01, 1.0)
        
        # --- Convex hull boundary detection ---
        # Use QuickHull-style boundary check via extreme point analysis
        # A particle is on the boundary if it lies near the hull in any direction
        centroid = np.mean(self.population, axis=0)
        
        # Compute direction from centroid to each particle
        to_particle = self.population - centroid
        to_particle_dist = np.linalg.norm(to_particle, axis=1, keepdims=True) + 1e-10
        to_particle_dir = to_particle / to_particle_dist
        
        # Project all directions onto principal directions to find extreme rays
        # Use axis-aligned directions as proxies for hull vertices
        axis_directions = np.eye(self.dim)
        axis_directions = np.vstack([axis_directions, -axis_directions])  # 2*dim directions
        
        # For each particle, find how extreme it is along each axis direction
        projection_scores = np.zeros((self.np, 2 * self.dim))
        for d in range(self.dim):
            # Positive axis direction
            axis_vec = np.zeros(self.dim)
            axis_vec[d] = 1.0
            projection_scores[:, d] = self.population[:, d] @ axis_vec
            # Negative axis direction
            projection_scores[:, self.dim + d] = -self.population[:, d] @ axis_vec
        
        # Normalize scores per direction
        for d in range(2 * self.dim):
            col = projection_scores[:, d]
            col_min, col_max = np.min(col), np.max(col)
            col_range = col_max - col_min + 1e-10
            projection_scores[:, d] = (col - col_min) / col_range
        
        # Boundary score: how many extreme-direction slots a particle occupies
        # High score = particle is at an extreme along multiple axes = on hull boundary
        threshold_high = 0.95
        threshold_low = 0.05
        is_extreme_pos = projection_scores >= threshold_high
        is_extreme_neg = projection_scores <= threshold_low
        
        # Count how many "boundary slots" each particle fills
        boundary_score = np.sum(is_extreme_pos | is_extreme_neg, axis=1, keepdims=True)
        max_possible_boundary = 2 * self.dim
        boundary_fraction = boundary_score / (max_possible_boundary + 1e-10)
        
        # Boundary particles get inward correction; interior particles get exploration
        is_boundary = boundary_fraction > 0.3
        
        # --- Centroid-based correction ---
        to_centroid_dir = centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
        to_centroid_unit = to_centroid_dir / to_centroid_dist
        
        # Distance from centroid normalized by global spread
        global_spread = np.mean(pop_range) + 1e-10
        dist_normalized = to_centroid_dist / (global_spread + 1e-10)
        
        # Attraction strength: stronger for particles far from centroid
        centroid_attraction = 0.5 * dist_normalized / (dist_normalized + 1.0)
        
        # --- Per-axis modulation from spread ratio ---
        # Compressed axes (low spread_ratio) need more exploration
        # Expanded axes (high spread_ratio) are well-explored, reduce correction
        axis_modulation = 1.0 / (spread_ratio + 0.1)
        axis_modulation = axis_modulation / (np.max(axis_modulation) + 1e-10)
        axis_modulation = np.clip(axis_modulation, 0.3, 2.5)
        
        # --- Combine corrections ---
        # Boundary particles: strong inward pull
        inward_correction = np.where(
            is_boundary,
            0.8 * centroid_attraction * to_centroid_unit,
            0.3 * centroid_attraction * to_centroid_unit
        )
        
        # Apply per-axis modulation
        inward_correction = inward_correction * axis_modulation[np.newaxis, :]
        
        # --- Spread correction: push particles toward uncovered regions ---
        # Compute current centroid position in [0,1] normalized space
        normalized_pos = (self.population - self.lower_bound) / (global_range + 1e-10)
        normalized_centroid = (centroid - self.lower_bound) / (global_range + 1e-10)
        
        # Target: move centroid toward center of search space
        target_normalized = np.ones(self.dim) * 0.5
        spread_correction_dir = target_normalized - normalized_centroid
        
        # Apply spread correction weighted by how compressed the population is
        spread_weight = 1.0 - np.mean(spread_ratio)
        spread_correction = spread_weight * 0.2 * spread_correction_dir
        
        # --- Combine all corrections ---
        correction = inward_correction + spread_correction[np.newaxis, :]
        
        new_population = self.population + self.inertia_weight * self.velocity + 0.3 * correction
        
        self.population = self._clip_to_bounds(new_population)
```