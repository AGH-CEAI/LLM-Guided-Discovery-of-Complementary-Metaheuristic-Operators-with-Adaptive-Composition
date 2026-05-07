**Idea: Geometric Spread Adaptation**
Adapt F and Cr based on population geometric spread metrics (centroid distance, axis-aligned range, k-NN connectivity) to modulate exploration vs exploitation.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using geometric/spatial population characteristics."""
    population = self.population if hasattr(self, 'population') else None
    
    if not hasattr(self, 'geo_centroid_history'):
        self.geo_centroid_history = []
        self.geo_spread_history = []
    
    if population is not None and len(population) > 0:
        # Compute population centroid
        centroid = np.mean(population, axis=0)
        
        # Track centroid drift (temporal, but only as geometric signal)
        self.geo_centroid_history.append(centroid.copy())
        if len(self.geo_centroid_history) > 10:
            self.geo_centroid_history.pop(0)
        
        centroid_drift = 0.0
        if len(self.geo_centroid_history) >= 2:
            centroid_drift = np.linalg.norm(self.geo_centroid_history[-1] - self.geo_centroid_history[0])
        
        # Geometric spread: average distance to centroid
        dist_to_centroid = np.linalg.norm(population - centroid, axis=1)
        mean_dist = np.mean(dist_to_centroid)
        std_dist = np.std(dist_to_centroid)
        
        # Normalize spread relative to search space
        search_range = self.upper - self.lower
        normalized_spread = mean_dist / (search_range + 1e-10)
        
        # Axis-aligned bounding box volume (log scale)
        pop_min = np.min(population, axis=0)
        pop_max = np.max(population, axis=0)
        bbox_sizes = pop_max - pop_min
        bbox_volume = np.prod(bbox_sizes + 1e-10)
        log_bbox_volume = np.log(bbox_volume + 1e-30)
        max_log_volume = np.log((search_range + 1e-10) ** self.dim)
        normalized_bbox = np.clip(log_bbox_volume / (max_log_volume + 1e-10), 0.0, 1.0)
        
        # k-NN connectivity: average distance to 5 nearest neighbors
        k = min(5, len(population) - 1)
        knn_distances = []
        for i in range(len(population)):
            distances = np.linalg.norm(population - population[i], axis=1)
            distances[i] = np.inf
            knn_distances.append(np.mean(np.partition(distances, k)[:k]))
        mean_knn_dist = np.mean(knn_distances)
        normalized_knn = np.clip(mean_knn_dist / (search_range * 0.5 + 1e-10), 0.0, 1.0)
        
        # Store spread for history
        self.geo_spread_history.append(normalized_spread)
        if len(self.geo_spread_history) > 15:
            self.geo_spread_history.pop(0)
        
        # Compute spread trend (increasing = exploring, decreasing = converging)
        spread_trend = 0.0
        if len(self.geo_spread_history) >= 3:
            recent = np.mean(self.geo_spread_history[-2:])
            older = np.mean(self.geo_spread_history[:2])
            spread_trend = (recent - older) / (older + 1e-10)
        
        # Combine geometric signals into adaptation factors
        # Low spread + low knn = clustered population → need more exploration
        # High spread + high knn = well-distributed → can exploit more
        
        exploration_pressure = 1.0 - normalized_spread  # Higher when clustered
        exploration_pressure += (1.0 - normalized_knn) * 0.5
        exploration_pressure = np.clip(exploration_pressure, 0.0, 2.0)
        
        # Adjustments based on geometry
        if exploration_pressure > 1.2:
            # Population is clustered - increase exploration
            F_adjustment = 1.0 + 0.15 * exploration_pressure
            Cr_adjustment = 0.90
        elif exploration_pressure < 0.5:
            # Population is well spread - increase exploitation
            F_adjustment = 0.85
            Cr_adjustment = 1.0 + 0.1 * (0.5 - exploration_pressure)
        else:
            # Balanced regime - use spread trend
            F_adjustment = 1.0 + 0.05 * spread_trend
            Cr_adjustment = 1.0 - 0.03 * spread_trend
        
        # Factor in centroid drift (high drift = population moving = exploration needed)
        if centroid_drift > 0.1 * search_range:
            F_adjustment *= 1.1
            Cr_adjustment *= 0.95
        
        # Apply adjustments
        self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
    else:
        # Fallback: gentle reversion if no population available
        self.F = np.clip(self.F * 0.98 + self.F_base * 0.02, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * 0.98 + self.Cr_base * 0.02, 0.1, 0.9)
```