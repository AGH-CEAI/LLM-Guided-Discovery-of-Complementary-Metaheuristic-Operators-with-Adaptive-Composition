**Idea: Axis-Aligned Bounding Box + k-NN Density Modulation**
Adapt inertia using geometric spread (axis-aligned bounding box ratio relative to search space) and k-NN local density — a fundamentally different geometric mechanism than covariance eigenvalues.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using geometric spread and k-NN density (Category A).
    
    Operates on literal spatial layout:
    - Axis-aligned bounding box ratio per dimension (normalized by search space)
    - k-NN average distance as local density estimate
    - Particle-wise inertia: sparse regions get higher inertia (exploration),
      dense regions get lower inertia (exploitation)
    """
    try:
        # === GEOMETRIC SPREAD: Axis-aligned bounding box ===
        # How much of the search space is actually occupied?
        min_coords = np.min(self.population, axis=0)
        max_coords = np.max(self.population, axis=0)
        search_range = self.upper_bound - self.lower_bound + 1e-10
        
        axis_spread = (max_coords - min_coords) / search_range
        avg_axis_spread = np.mean(axis_spread)
        min_axis_spread = np.min(axis_spread)
        
        # Spread signal: well-spread population -> lower inertia
        # Clamp to handle both over-dispersed (>1.0) and collapsed (<0.1) cases
        spread_signal = np.clip(avg_axis_spread, 0.0, 1.0)
        
        # === k-NN LOCAL DENSITY ===
        # Compute average distance to k nearest neighbors for each particle
        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        sorted_dists = np.sqrt(np.sort(sq_dists, axis=1))
        knn_avg = np.mean(sorted_dists[:, :k], axis=1) + 1e-10
        
        # Global density baseline
        global_avg_knn = np.mean(knn_avg)
        
        # Per-particle density relative to global: sparse particles get higher inertia
        density_ratio = knn_avg / (global_avg_knn + 1e-10)
        
        # Normalize density signal: sparse (>1) -> high inertia, dense (<1) -> low inertia
        # Map to [0.5, 1.5] then shift to influence inertia direction
        density_signal = np.clip(density_ratio, 0.5, 2.0)
        
        # === COMBINED GEOMETRIC SIGNAL ===
        # Primary: axis-aligned spread (global population geometry)
        # Secondary: local density (per-particle adjustment)
        
        # Base inertia from spread: well-spread -> exploit (lower), collapsed -> explore (higher)
        base_inertia = 0.95 - 0.55 * spread_signal
        
        # Density modulation: sparse regions push toward exploration
        density_modulation = 1.0 + 0.2 * (density_signal - 1.0)
        density_modulation = np.clip(density_modulation, 0.85, 1.15)
        
        target_inertia = base_inertia * density_modulation
        
        # Time-based decay: gradual shift toward exploitation over generations
        decay = 0.729 - 0.15 * (self.generation / 1000)
        target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)
        
        # EMA smoothing
        self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    except Exception:
        # Fallback: time-based decay
        decay = 0.729 - 0.15 * (self.generation / 1000)
        self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```