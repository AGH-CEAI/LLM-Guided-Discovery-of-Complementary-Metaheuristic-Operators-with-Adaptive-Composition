Looking at the current `_position_update_centroid_knn` (which uses centroid gravity + k-NN density), I need to take a **different geometric angle** within Category A.

**Analysis of worst tasks (17, 16, 6, 5):**
- Errors ~1e+02 to 1e+04 suggest particles are trapped in local optima basins
- The current k-NN density approach modulates strength but doesn't distinguish *why* particles are stuck
- Convex hull geometry can identify particles that are "trapped inside" vs "on the exploration boundary"

**My different geometric mechanism:**
- Use **convex hull boundary detection** (not used in current variant)
- Particles **on the hull** (boundary) → mild attraction toward centroid to stabilize exploration
- Particles **inside the hull** (interior/trapped) → stronger repulsion away from centroid to escape local optima
- This is fundamentally different from density-based modulation because it uses topological boundary information, not local neighborhood density

**Idea: Convex Hull Boundary Escape**

```python
def _position_update_centroid_knn(self):
    """Convex hull boundary detection + anisotropic centroid correction.
    
    Category A (Geometry/spatial). Different from k-NN density by using
    TOPOLOGICAL boundary info: hull vertices = exploration frontier,
    interior points = potentially trapped. Interior particles get
    stronger outward push to escape local optima on worst tasks.
    """
    from scipy.spatial import ConvexHull
    
    centroid = np.mean(self.population, axis=0)
    to_centroid = centroid - self.population
    to_centroid_norm = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
    to_centroid_dir = to_centroid / to_centroid_norm
    
    dists_to_centroid = np.linalg.norm(self.population - centroid, axis=1)
    max_dist = np.max(dists_to_centroid) + 1e-10
    normalized_dists = dists_to_centroid / max_dist
    
    # Geometric boundary detection via convex hull
    boundary_mask = np.zeros(self.np, dtype=bool)
    try:
        if self.np >= self.dim + 2 and self.dim <= 15:
            hull = ConvexHull(self.population)
            boundary_mask[hull.vertices] = True
        else:
            boundary_mask = normalized_dists > 0.75
    except:
        boundary_mask = normalized_dists > 0.75
    
    # Interior particles (trapped): stronger outward push to escape
    # Boundary particles (exploring): stabilize with mild centroid pull
    interior_strength = 0.8 * (1.0 - normalized_dists[:, np.newaxis])
    boundary_strength = 0.25 * normalized_dists[:, np.newaxis]
    
    correction = np.where(
        boundary_mask[:, np.newaxis],
        boundary_strength * to_centroid_dir,
        interior_strength * to_centroid_dir
    )
    
    new_population = self.population + self.inertia_weight * self.velocity + 0.5 * correction
    self.population = self._clip_to_bounds(new_population)
```