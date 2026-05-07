**Idea: Convex Hull Volume + k-NN Density Geometry**
Category A geometry: uses convex hull volume for collapse detection and k-NN nearest-neighbor distances for density-based velocity modulation — entirely distinct from prior spectral/eigenvalue approaches.
```python
def _position_update_temporal_drift(self):
    """Convex hull + k-NN geometry-based velocity modulation (Category A).

    Operates on literal spatial layout: convex hull volume for collapse
    detection, k-NN structure for local density, axis-aligned bounding box
    for spread anisotropy. No eigenvalues, entropy, or condition numbers.
    Targets worst tasks (17, 16, 6, 11) where population collapses to
    low-dimensional subspaces.
    """
    try:
        # Geometric signal 1: axis-aligned bounding box
        mins = np.min(self.population, axis=0)
        maxs = np.max(self.population, axis=0)
        ranges = maxs - mins + 1e-10
        aspect_ratio = np.max(ranges) / (np.min(ranges) + 1e-10)
        aspect_signal = np.clip(aspect_ratio / (self.dim + 1.0), 0.0, 1.0)

        # Geometric signal 2: convex hull volume/area
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(self.population)
            if self.dim <= 2:
                hull_volume = hull.volume  # area in 2D
            else:
                hull_volume = hull.volume if hull.volume > 0 else np.prod(ranges)
        except:
            hull_volume = np.prod(ranges)

        # Expected volume of hypercube with mean range
        mean_range = np.mean(ranges)
        expected_vol = mean_range ** self.dim
        expected_vol = np.clip(expected_vol, 1e-100, None)
        hull_normalized = np.clip(hull_volume / (expected_vol + 1e-10), 0.0, 1.0)

        # Geometric signal 3: k-NN density structure
        k = min(5, self.np - 1)
        sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
        knn_dists = np.sqrt(np.take_along_axis(sq_dists, knn_indices, axis=1))
        knn_avg = np.mean(knn_dists, axis=1) + 1e-10

        global_spread = np.mean(knn_avg)
        density_signal = np.clip(global_spread / (knn_avg + 1e-10), 0.2, 5.0)
        density_signal = density_signal / (np.max(density_signal) + 1e-10)

        # Combined geometric signal
        geo_signal = hull_normalized * 0.4 + density_signal * 0.3 + (1.0 - aspect_signal) * 0.3

        # Temporal improvement tracking (EMA)
        if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
            improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_best_fitness = self.global_best_fitness

        if not hasattr(self, '_geo_ema_improvement'):
            self._geo_ema_improvement = 0.0
        self._geo_ema_improvement = 0.3 * improvement + 0.7 * self._geo_ema_improvement

        if self._geo_ema_improvement > 1e-6:
            base_scale = 1.2
        elif self._geo_ema_improvement > 1e-10:
            base_scale = 1.0
        else:
            base_scale = 0.7

        # Velocity scaling driven by geometric signal
        temporal_scale = base_scale * (1.0 + 0.4 * geo_signal)

        # Geometric corrections
        centroid = np.mean(self.population, axis=0)
        to_centroid = centroid - self.population
        dist_to_centroid = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid / dist_to_centroid

        # Centroid correction: sparse particles pulled more
        centroid_correction = 0.2 * to_centroid_dir * (dist_to_centroid / (global_spread + 1e-10))

        # k-NN density correction: push from dense neighborhoods
        knn_center = np.mean(self.population[knn_indices], axis=1, keepdims=True)
        from_center = self.population - knn_center
        from_center_norm = np.linalg.norm(from_center, axis=1, keepdims=True) + 1e-10
        knn_correction = 0.1 * (from_center / from_center_norm) * (1.0 / (density_signal[:, np.newaxis] + 0.1))

        correction = centroid_correction + knn_correction
        correction = np.clip(correction, -2.0, 2.0)

        new_population = self.population + temporal_scale * self.velocity + correction
    except:
        new_population = self.population + self.velocity

    self.population = self._clip_to_bounds(new_population)
```