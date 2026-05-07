**Idea: Convex Hull + k-NN Density + Axis-Aligned Spread Modulation**

Pure Category A geometry: Uses convex hull volume for global spread, k-NN local density per particle, axis-aligned bounding box aspect ratio, and centroid-hull centroid offset. No temporal, spectral, fitness, or graph-based signals.

```python
def _position_update_temporal_drift(self):
    """Geometry/Spatial: Convex hull + k-NN density + axis-aligned spread (Category A).

    Operates purely on literal geometric layout of points:
      - Convex hull volume for global spread detection
      - k-NN local density per particle for sparse/dense region identification
      - Axis-aligned bounding box aspect ratio for anisotropic spread detection
      - Centroid-hull centroid offset for detecting off-center concentration

    Pure geometric reasoning — NO temporal, spectral, fitness, or graph signals.
    """
    try:
        # === MECHANISM 1: k-NN LOCAL DENSITY ===
        k = min(5, self.np - 1)
        sq_dists = np.sum(
            (self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2,
            axis=2
        )
        np.fill_diagonal(sq_dists, np.inf)
        knn_distances = np.sqrt(np.sort(sq_dists, axis=1)[:, :k])
        knn_mean_dist = np.mean(knn_distances, axis=1) + 1e-10

        # Density = inverse mean k-NN distance; sparse particles get higher values
        density = 1.0 / knn_mean_dist
        global_mean_density = np.mean(density) + 1e-10
        density_ratio = density / global_mean_density

        # Per-particle exploration factor: sparse = more exploration, dense = less
        explore_factor = np.clip(1.0 + 0.5 * (1.0 - np.clip(density_ratio, 0.5, 2.0)), 0.5, 2.0)

        # === MECHANISM 2: AXIS-ALIGNED BOUNDING BOX ===
        min_coords = np.min(self.population, axis=0)
        max_coords = np.max(self.population, axis=0)
        bbox_sizes = max_coords - min_coords
        bbox_sizes = np.clip(bbox_sizes, 1e-10, None)

        # Aspect ratio: max dimension / min dimension
        aspect_ratio = np.max(bbox_sizes) / (np.min(bbox_sizes) + 1e-10)
        aspect_signal = np.clip(aspect_ratio / 10.0, 0.0, 1.0)

        # === MECHANISM 3: CONVEX HULL VOLUME ===
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(self.population)
            hull_volume = hull.volume if hasattr(hull, 'volume') else hull.area

            # Expected volume for uniform distribution in [-100, 100]^dim
            expected_volume = (200.0 ** self.dim) * (np.math.factorial(self.dim) /
                               (np.math.factorial(self.dim // 2) * np.pi ** (self.dim // 2))) ** (1.0 / self.dim)
            expected_volume = max(expected_volume, 1e-10)
            volume_ratio = np.clip(hull_volume / expected_volume, 0.0, 1.0)
        except:
            volume_ratio = 0.5

        # === MECHANISM 4: CENTROID-HULL CENTROID OFFSET ===
        centroid = np.mean(self.population, axis=0)
        hull_vertices = self.population[hull.vertices] if hasattr(hull, 'vertices') else self.population
        hull_centroid = np.mean(hull_vertices, axis=0)
        centroid_offset = np.linalg.norm(centroid - hull_centroid)

        # Normalize offset by population spread
        population_spread = np.std(self.population) + 1e-10
        normalized_offset = centroid_offset / population_spread
        offset_signal = np.clip(normalized_offset / 2.0, 0.0, 1.0)

        # === COMBINED GEOMETRIC SCALING ===
        # High aspect ratio + small hull volume + off-center = need exploration
        global_explore = aspect_signal * (1.0 - volume_ratio) * (1.0 + offset_signal)
        global_explore = np.clip(global_explore, 0.0, 1.0)

        # Base scale: exploration mode
        base_scale = 1.0 + 0.4 * global_explore

        # Per-particle scale: local density-based
        per_particle_scale = explore_factor

        # Combined scale for velocity
        combined_scale = base_scale * per_particle_scale
        combined_scale = np.clip(combined_scale, 0.3, 2.5)

        # === POSITION UPDATE ===
        new_population = self.population + combined_scale[:, np.newaxis] * self.velocity

    except Exception:
        new_population = self.population + self.velocity

    self.population = self._clip_to_bounds(new_population)
```