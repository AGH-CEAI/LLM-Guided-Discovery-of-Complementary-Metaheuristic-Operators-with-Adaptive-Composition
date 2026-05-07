**Idea: Geometric Diameter and k-NN Density Adaptation**

Adapt inertia weight based on population geometric layout: diameter (max pairwise distance approximation), average centroid distance, axis-aligned spread ratio, and k-NN density. Geometric signals only—no spectral/eigenvalue analysis.
```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using geometric layout of the population.
    
    Category A: Operates on literal geometric layout (pairwise distances,
    centroid distances, axis-aligned spread, k-NN density). No eigenvalues,
    condition numbers, or covariance-based spectral analysis.
    """
    try:
        # 1. Population centroid and mean distance to centroid
        centroid = np.mean(self.population, axis=0)
        dists_to_centroid = np.linalg.norm(self.population - centroid, axis=1)
        mean_centroid_dist = np.mean(dists_to_centroid)
        
        # 2. Axis-aligned bounding box spread per dimension
        mins = np.min(self.population, axis=0)
        maxs = np.max(self.population, axis=0)
        ranges = maxs - mins
        mean_range = np.mean(ranges)
        max_range = np.max(ranges)
        
        # 3. Population diameter: use centroid + max range as fast approximation
        diameter = mean_centroid_dist * 2.0 + max_range
        
        # 4. k-NN density: average distance to k nearest neighbors
        k = min(5, self.np - 1)
        sq_dists = np.sum(
            (self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2,
            axis=2
        )
        np.fill_diagonal(sq_dists, np.inf)
        sorted_sq_dists = np.sort(sq_dists, axis=1)
        knn_sq_dists = sorted_sq_dists[:, :k]
        mean_knn_dist = np.mean(np.sqrt(knn_sq_dists)) + 1e-10
        
        # 5. Spread anisotropy ratio (axis-aligned)
        min_range = np.min(ranges) + 1e-10
        spread_ratio = np.clip(max_range / min_range, 1.0, 100.0)
        
        # Normalize diameter signal: map to [0, 1] based on expected bounds
        # Search space diameter ≈ 200 * sqrt(dim) at bounds ±100
        expected_diameter = 200.0 * np.sqrt(self.dim)
        diameter_signal = np.clip(diameter / expected_diameter, 0.0, 1.0)
        
        # Normalize centroid distance signal
        centroid_signal = np.clip(mean_centroid_dist / expected_diameter, 0.0, 1.0)
        
        # Normalize spread ratio: high anisotropy → more exploration signal
        spread_signal = np.clip(np.log1p(spread_ratio) / np.log1p(100.0), 0.0, 1.0)
        
        # Normalize k-NN density: sparse → more exploration signal
        knn_signal = np.clip(1.0 / (mean_knn_dist + 1.0), 0.0, 1.0)
        
        # Composite geometric signal
        # Higher spread/diameter = more exploration (higher inertia)
        # Higher density = more exploitation (lower inertia) → subtract density
        geometric_signal = (
            0.30 * diameter_signal +
            0.25 * centroid_signal +
            0.25 * spread_signal +
            0.20 * knn_signal
        )
        geometric_signal = np.clip(geometric_signal, 0.0, 1.0)
        
        # Map to inertia range [0.4, 0.95]
        # High geometric signal (spread out) → high inertia for exploration
        # Low geometric signal (clustered) → low inertia for exploitation
        target_inertia = 0.4 + 0.55 * geometric_signal
        
        # Blend with time-based baseline
        decay = 0.729 - 0.15 * (self.generation / 1000)
        target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)
        
        # Exponential moving average for smooth adaptation
        self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
        
    except Exception:
        decay = 0.729 - 0.15 * (self.generation / 1000)
        self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```