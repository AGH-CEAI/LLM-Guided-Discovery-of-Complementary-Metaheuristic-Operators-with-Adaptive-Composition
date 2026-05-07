Looking at the worst unsolved tasks (17, 16, 6, 19 with errors 2.6e+04, 3.4e+03, 9e+02, 1.9e+02), these failures suggest the swarm is collapsing into local optima or losing geometric diversity. The current inertia adaptation uses only average distance-to-centroid, which misses anisotropic collapse (swarm collapsing into a line or flat plane).

**Idea: Axis-Aligned Spread Ratio for Dimensional Collapse Detection**

Category A: Geometry / spatial — using axis-aligned bounding box to detect when the swarm collapses non-uniformly across dimensions (dimensional degeneracy), which causes the swarm to miss optima in under-explored dimensions.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight based on axis-aligned spread ratio (Category A: Geometry/Spatial).
    
    Detects dimensional collapse: when the swarm collapses non-uniformly (e.g., spread
    along one axis >> others), the swarm may miss optima in under-explored dimensions.
    The axis-aligned spread ratio measures this anisotropy geometrically.
    """
    # Compute axis-aligned bounding box
    min_pos = np.min(self.population, axis=0)
    max_pos = np.max(self.population, axis=0)
    search_width = self.upper_bound - self.lower_bound + 1e-10
    
    # Per-dimension spread (geometric: how much of each axis is covered)
    spreads = max_pos - min_pos
    normalized_spreads = spreads / search_width
    
    max_spread = np.max(normalized_spreads)
    min_spread = np.min(normalized_spreads)
    
    # Ratio of max to min spread: high ratio = anisotropic collapse
    axis_ratio = max_spread / (min_spread + 1e-10)
    
    # Log-scaled response to ratio (geometric, not spectral)
    log_ratio = np.log1p(axis_ratio)
    
    # Inertia weight: higher when collapsed (more exploration needed)
    if log_ratio > 5.0:  # extreme collapse (ratio > 148)
        self.inertia_weight = 0.92
    elif log_ratio > 3.5:  # severe collapse (ratio > 33)
        self.inertia_weight = 0.85
    elif log_ratio > 2.0:  # moderate collapse (ratio > 7.4)
        self.inertia_weight = 0.78
    elif log_ratio > 1.0:  # mild anisotropy (ratio > 2.7)
        self.inertia_weight = 0.72
    else:  # roughly isotropic spread
        # Gradual convergence schedule
        self.inertia_weight = 0.729 - 0.08 * (self.generation / 1000)
    
    self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```