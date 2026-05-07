**Idea: Axis-Aligned Bounding Box Centroid Distance Weighting**

Category A: Geometry / spatial. Operates on axis-aligned spread (bounding box per dimension) and centroid distances — a fundamentally different geometric lens than the SVD-based singular value approach.

```python
def _position_update_svd_whitening(self):
    """Axis-aligned bounding box normalization + centroid-distance-weighted velocity.
    
    Category A geometry: uses literal spatial layout — per-dimension bounding box
    to detect anisotropy, then centroid distance to weight exploration. Different
    from SVD-based whitening which reasons about covariance/principal components.
    """
    # Compute axis-aligned bounding box per dimension
    pop_min = np.min(self.population, axis=0)
    pop_max = np.max(self.population, axis=0)
    bb_range = pop_max - pop_min
    bb_range = np.clip(bb_range, 1e-10, None)
    
    # Condition from bounding box: ratio of max to min range
    bb_cond = np.max(bb_range) / (np.min(bb_range) + 1e-10)
    bb_cond = np.clip(bb_cond, 1.0, 1000.0)
    
    # Normalize each dimension by its bounding box range
    centered = self.population - np.mean(self.population, axis=0)
    bb_normalized = centered / bb_range
    
    # Compute centroid distances for each particle
    centroid = np.mean(self.population, axis=0)
    dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1)
    max_dist = np.max(dist_to_centroid) + 1e-10
    norm_dist = dist_to_centroid / max_dist
    
    # Particles far from centroid: amplify velocity (more exploration needed)
    # Particles near centroid: dampen velocity (exploit current region)
    # Use inverse of normalized distance so far particles get higher weights
    centroid_weights = 1.0 / (norm_dist + 0.1)
    centroid_weights = centroid_weights / (np.max(centroid_weights) + 1e-10)
    centroid_weights = np.clip(centroid_weights, 0.3, 2.5)
    
    # Compute axis-aligned spread factor per particle
    # High spread in a dimension → dampen velocity in that direction
    # Low spread → amplify
    per_dim_spread = bb_normalized / (np.std(bb_normalized, axis=0) + 1e-10)
    per_dim_spread = np.clip(np.abs(per_dim_spread), 0.0, 3.0)
    
    # Blend: per-particle centroid weights with per-dimension spread adjustment
    blend_weight = np.clip((bb_cond - 10.0) / 100.0, 0.0, 0.5)
    
    # Velocity scaling: combine centroid-distance weights with axis-aligned spread
    vel_scale = centroid_weights[:, np.newaxis] * (1.0 - blend_weight * per_dim_spread)
    vel_scale = np.clip(vel_scale, 0.3, 2.5)
    
    # Apply scaling to velocity
    new_population = self.population + vel_scale * self.velocity
    
    self.population = self._clip_to_bounds(new_population)
```