**Idea: Convex-Hull Centroid Distance + Axis-Aligned Spread Modulation**

One-line description: Uses literal geometric layout (convex hull volume, axis-aligned bounding box spread, centroid distances) to detect population collapse and modulate exploration — purely geometric signals without fitness or graph topology.

```python
def _position_update_fitness_rank(self):
    """Convex-hull centroid distance + axis-aligned spread velocity modulation (Category A).

    Key insight: Use literal geometric layout of the population — convex hull volume,
    axis-aligned bounding box spread, centroid distances — to detect convergence
    and modulate exploration vs exploitation. Purely geometric, orthogonal to
    graph-based (E) and spectral-matrix (B) approaches.
    """
    # Compute population centroid
    centroid = np.mean(self.population, axis=0)

    # Axis-aligned bounding box diagonal (maximum possible spread)
    pop_min = np.min(self.population, axis=0)
    pop_max = np.max(self.population, axis=0)
    bounding_box_size = pop_max - pop_min
    max_bbox_diagonal = np.linalg.norm(bounding_box_size) + 1e-10

    # Current spread: average distance from centroid
    dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1)
    current_spread = np.mean(dist_to_centroid) + 1e-10

    # Spread ratio (0 = collapsed, 1 = well-spread)
    spread_ratio = current_spread / max_bbox_diagonal
    spread_ratio = np.clip(spread_ratio, 0.01, 1.0)

    # Per-particle spread contribution (relative to mean)
    particle_spread_contrib = dist_to_centroid / (current_spread + 1e-10)

    # Convex hull volume (2D area or higher-D pairwise distance std proxy)
    try:
        from scipy.spatial import ConvexHull
        from scipy.spatial.distance import pdist
        if self.dim <= 3 and self.np >= self.dim + 1:
            hull = ConvexHull(self.population)
            hull_volume = hull.volume if hasattr(hull, 'volume') else hull.area
        else:
            pairwise_dists = pdist(self.population)
            hull_volume = np.std(pairwise_dists)
    except:
        hull_volume = current_spread

    # Normalize hull volume by expected max (bounding box volume proxy)
    max_expected = (max_bbox_diagonal ** self.dim) / max(1, np.math.factorial(self.dim))
    hull_volume_ratio = hull_volume / (max_expected + 1e-10)
    hull_volume_ratio = np.clip(hull_volume_ratio, 0.0, 1.0)

    # Geometric exploration signal: small spread or hull volume → more exploration
    geometric_explore = 1.0 + 1.5 * (1.0 - spread_ratio) + 0.5 * (1.0 - hull_volume_ratio)
    geometric_explore = np.clip(geometric_explore, 0.5, 3.0)

    # Per-particle geometric factor
    particle_geo_factor = np.clip(particle_spread_contrib, 0.5, 2.0)

    # Velocity scaling from geometric signals
    vel_scale = geometric_explore * particle_geo_factor
    vel_scale = np.clip(vel_scale, 0.3, 3.0)

    # Direction toward centroid (counteract collapse)
    to_centroid_dir = centroid - self.population
    to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
    to_centroid_dir_norm = to_centroid_dir / to_centroid_dist

    # Centroid attraction when spread is very small
    centroid_attraction = 0.3 * (1.0 - spread_ratio) * to_centroid_dir_norm

    # Random perturbation scaled by geometric factor
    random_perturb = 0.2 * geometric_explore * np.random.uniform(-1, 1, (self.np, self.dim))

    # Update population
    new_population = self.population + vel_scale[:, np.newaxis] * self.velocity + centroid_attraction + random_perturb
    self.population = self._clip_to_bounds(new_population)
```