Looking at the current `_position_update_fitness_rank`, I notice it actually implements Category E (Topology/graph-based) with k-NN graphs and Laplacian eigenvalues. The previous Category A variant (variant_01) FAILED entirely. I need a genuinely different geometric mechanism.

**Analysis of worst tasks (17, 16, 6, 5) with errors ~10-12 decades:**
- These extreme errors suggest the swarm is collapsing to wrong regions or failing to explore basins
- Geometric detection of population collapse and boundary effects could help
- I should use purely spatial metrics: axis-aligned spread, convex hull volume, pairwise distance distribution

**New Category A mechanism:**
- Compute axis-aligned bounding box coverage per particle
- Measure pairwise distance distribution statistics (detect clustering vs over-dispersion)
- Compute convex hull volume (detect population collapse)
- Use these to modulate exploration vs exploitation purely through geometric signals

```python
def _position_update_fitness_rank(self):
    """Axis-aligned bounding box + convex hull + pairwise distance modulation (Category A).

    Key insight: Use ONLY geometric properties of particle positions:
    - Axis-aligned spread per dimension to detect anisotropy
    - Convex hull volume to detect population collapse
    - Pairwise distance distribution to detect clustering

    Purely spatial — no graph topology, no fitness signals, no temporal tracking.
    Targets worst tasks (17, 16, 6, 5) where swarm collapses to wrong regions.
    """
    pop = self.population
    np_particles, dim = pop.shape

    # === AXIS-ALIGNED BOUNDING BOX ANALYSIS ===
    pop_min = np.min(pop, axis=0)
    pop_max = np.max(pop, axis=0)
    pop_range = pop_max - pop_min + 1e-10
    global_range = self.upper_bound - self.lower_bound

    # Per-particle coverage: how much of the search space each particle occupies
    centroid = np.mean(pop, axis=0)
    particle_to_centroid = pop - centroid
    # Signed position within bounding box [-1, 1]
    normalized_pos = 2.0 * (pop - pop_min) / pop_range - 1.0

    # Detect boundary crowding: particles near box edges
    edge_margin = 0.1
    near_edge = np.max(np.abs(normalized_pos), axis=1) > (1.0 - edge_margin)
    edge_pressure = np.sum(near_edge) / np_particles

    # Per-dimension spread ratio (detect anisotropy)
    spread_ratios = pop_range / (global_range + 1e-10)
    min_spread = np.min(spread_ratios)
    max_spread = np.max(spread_ratios)
    anisotropy = max_spread / (min_spread + 1e-10) if min_spread > 1e-10 else 1.0

    # === CONVEX HULL VOLUME (2D/3D approximation or high-D estimate) ===
    try:
        if dim <= 10 and np_particles >= dim + 1:
            # Use QHull via scipy for convex hull
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pop, qhull_options='Qt')
            hull_volume = hull.volume  # In QHull, 'volume' is (n-1)-simplex measure
        else:
            # High-dimensional approximation: product of spreads
            hull_volume = np.prod(pop_range + 1e-10)
    except:
        hull_volume = np.prod(pop_range + 1e-10)

    # Normalize by maximum possible volume
    max_volume = (global_range ** dim)
    normalized_volume = hull_volume / (max_volume + 1e-10)
    normalized_volume = np.clip(normalized_volume, 1e-10, 1.0)

    # === PAIRWISE DISTANCE DISTRIBUTION ===
    sq_dists = np.sum((pop[:, np.newaxis, :] - pop[np.newaxis, :, :]) ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    all_dists = np.sqrt(sq_dists[sq_dists < np.inf])

    if len(all_dists) > 0:
        mean_dist = np.mean(all_dists)
        std_dist = np.std(all_dists)
        min_dist = np.min(all_dists)

        # Clustering ratio: mean distance vs expected uniform distance
        expected_uniform_dist = np.mean(pop_range) * 0.5
        clustering_ratio = mean_dist / (expected_uniform_dist + 1e-10)
    else:
        mean_dist = 1.0
        std_dist = 1.0
        min_dist = 1.0
        clustering_ratio = 1.0

    # === GEOMETRIC EXPLORATION SIGNALS ===
    # Small hull volume = collapsed population → boost exploration
    collapse_explore = 1.0 + 1.5 * (1.0 - normalized_volume)
    collapse_explore = np.clip(collapse_explore, 0.5, 3.0)

    # High anisotropy = elongated cluster → directional spreading
    anisotropy_explore = 1.0 + 0.3 * np.log1p(anisotropy) / np.log1p(100.0)
    anisotropy_explore = np.clip(anisotropy_explore, 0.8, 2.0)

    # Edge crowding → push particles inward
    inward_pressure = edge_pressure * 0.5

    # Small std_dist = over-clustered → add spread
    spread_explore = 1.0 + 0.5 * (1.0 - np.clip(std_dist / (mean_dist + 1e-10), 0.0, 1.0))
    spread_explore = np.clip(spread_explore, 0.5, 2.5)

    # Combined geometric exploration factor
    geo_explore = collapse_explore * anisotropy_explore * spread_explore
    geo_explore = np.clip(geo_explore, 0.3, 4.0)

    # === PER-PARTICLE GEOMETRIC MODULATION ===
    # Particles near edges get inward nudge
    edge_dist = 1.0 - np.max(np.abs(normalized_pos), axis=1)
    inward_strength = inward_pressure * (1.0 - edge_dist)[:, np.newaxis]
    to_centroid = centroid - pop
    to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
    inward_nudge = inward_strength * (to_centroid / to_centroid_dist)

    # === VELOCITY MODULATION FROM GEOMETRY ===
    # Anisotropic scaling: extend along smallest-spread dimensions
    try:
        centered = pop - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)

        # Inverse spread weighting: boost along collapsed directions
        inv_spread = 1.0 / (eigenvalues / (eigenvalues[0] + 1e-10) + 0.01)
        inv_spread = inv_spread / (np.max(inv_spread) + 1e-10)

        vel_proj = self.velocity @ eigenvectors
        vel_corrected = vel_proj * inv_spread
        anisotropic_correction = (vel_corrected @ eigenvectors.T) - self.velocity
    except:
        anisotropic_correction = np.zeros_like(self.velocity)

    # === DIRECTIONAL PERTURBATION FROM GEOMETRY ===
    # Random perturbation scaled by geometric exploration
    random_perturb = geo_explore * np.random.uniform(-0.5, 0.5, (np_particles, dim))

    # === COMBINE ===
    new_population = pop + self.velocity * geo_explore + anisotropic_correction + inward_nudge + random_perturb
    self.population = self._clip_to_bounds(new_population)
```