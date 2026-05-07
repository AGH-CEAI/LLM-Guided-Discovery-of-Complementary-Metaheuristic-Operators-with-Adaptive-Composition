**Idea: Axis-Aligned Spread Correction**
Category A variant using per-dimension bounding box analysis. Instead of k-NN density, compute normalized deviation from population's axis-aligned bounding box to apply directional corrections toward the geometric center.

```python
def _position_update_centroid_knn(self):
    """Axis-aligned bounding box spread correction (Category A: Geometry).
    
    Uses per-dimension min/max to compute normalized deviation from the
    population's axis-aligned bounding box. Particles far outside the box
    get weak correction (already explored); particles near the center
    get stronger correction (promote convergence). Different mechanism
    from k-NN density while staying purely geometric.
    """
    pop_mean = np.mean(self.population, axis=0)
    
    dim_min = np.min(self.population, axis=0)
    dim_max = np.max(self.population, axis=0)
    dim_range = dim_max - dim_min + 1e-10
    
    deviation = self.population - pop_mean
    normalized_dev = deviation / dim_range
    
    max_dev = np.max(np.abs(normalized_dev), axis=1, keepdims=True) + 1e-10
    
    to_center_dir = pop_mean - self.population
    dist_to_center = np.linalg.norm(to_center_dir, axis=1, keepdims=True) + 1e-10
    to_center_dir = to_center_dir / dist_to_center
    
    correction_scale = np.clip(0.3 / max_dev, 0.05, 1.0)
    
    correction = correction_scale * to_center_dir
    
    new_population = self.population + self.inertia_weight * self.velocity + correction
    
    self.population = self._clip_to_bounds(new_population)
```