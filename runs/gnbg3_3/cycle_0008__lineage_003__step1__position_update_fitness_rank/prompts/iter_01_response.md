**Idea: Convex Hull Aspect Ratio Modulation**

Use geometric properties only: convex hull volume for convergence detection, axis-aligned bounding box aspect ratio for anisotropy, particle-to-centroid distances for outlier identification, and k-NN average distances for local density. No graph topology, no fitness signals.

```python
def _position_update_fitness_rank(self):
    """Geometry-only velocity modulation using convex hull and spatial spread (Category A).

    Key geometric signals:
    - Convex hull volume: detects overall population spread/convergence
    - Axis-aligned bounding box aspect ratio: detects dimensional anisotropy
    - Particle-to-centroid distances: identifies outliers vs core cluster
    - k-NN average distance: per-particle local density

    No fitness-based, topology-based, or information-theoretic reasoning.
    """
    # Compute centroid
    centroid = np.mean(self.population, axis=0)

    # --- Signal 1: Convex hull volume (population spread) ---
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(self.population)
        hull_volume = hull.volume if hasattr(hull, 'volume') and hull.volume > 0 else hull.area
        # Normalize by hypercube volume of bounding box
        mins = np.min(self.population, axis=0)
        maxs = np.max(self.population, axis=0)
        bb_volume = np.prod(maxs - mins + 1e-10)
        hull_ratio = hull_volume / (bb_volume + 1e-10)
        hull_ratio = np.clip(hull_ratio, 1e-6, 1.0)
    except:
        hull_ratio = 0.5
        mins = np.min(self.population, axis=0)
        maxs = np.max(self.population, axis=0)

    # --- Signal 2: Axis-aligned bounding box aspect ratio (anisotropy) ---
    ranges = maxs - mins + 1e-10
    aspect_ratio = np.max(ranges) / np.min(ranges)
    aspect_ratio = np.clip(aspect_ratio, 1.0, 1000.0)
    # High aspect ratio = collapsed into subspace → more exploration
    anisotropy_scale = np.clip(aspect_ratio / 50.0, 0.5, 2.0)

    # --- Signal 3: Particle-to-centroid distances (outlier detection) ---
    dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True) + 1e-10
    median_dist = np.median(dist_to_centroid)
    outlier_factor = np.clip(dist_to_centroid / (median_dist + 1e-10), 0.5, 3.0)

    # --- Signal 4: k-NN average distance (local density) ---
    k = min(5, self.np - 1)
    sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    knn_dists = np.sort(sq_dists, axis=1)[:, :k]
    knn_avg = np.mean(np.sqrt(knn_dists), axis=1, keepdims=True) + 1e-10
    global_avg = np.mean(knn_avg) + 1e-10
    # Sparse particles (high knn_avg) need more exploration; dense (low knn_avg) need less
    density_scale = np.clip(knn_avg / global_avg, 0.5, 2.0)

    # --- Combine geometric signals into velocity scaling ---
    # Small hull_ratio = converged population → boost exploration
    convergence_explore = 1.0 + 1.0 * (1.0 - np.log1p(hull_ratio) / np.log1p(1.0))

    # Velocity modulation per particle
    vel_modulation = convergence_explore * anisotropy_scale * density_scale * outlier_factor
    vel_modulation = np.clip(vel_modulation, 0.3, 3.0)

    # --- Directional correction toward centroid for outliers ---
    to_centroid = centroid - self.population
    to_centroid_norm = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
    to_centroid_dir = to_centroid / to_centroid_norm
    # Stronger pull for outliers (far from centroid)
    centroid_pull = 0.3 * ((outlier_factor - 1.0) / 2.0)[:, np.newaxis] * to_centroid_dir

    # --- Random geometric perturbation scaled by sparsity ---
    random_perturb = density_scale * np.random.uniform(-0.3, 0.3, (self.np, self.dim))

    # --- Update population ---
    new_population = self.population + self.velocity * vel_modulation + centroid_pull + random_perturb
    self.population = self._clip_to_bounds(new_population)
```