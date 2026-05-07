**Idea: Geometric Diversity-Driven Adaptation**
Adapt F and Cr based on spatial population structure: axis-aligned spread, pairwise distance statistics, and centroid-relative geometry. When population is geometrically diverse (large spread), reduce F for exploitation; when clustered, increase F for exploration. Aspect ratio of the bounding box modulates Cr.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using population geometry and spatial structure."""
    # Store population geometry for adaptation
    if not hasattr(self, 'pop_geometry_history'):
        self.pop_geometry_history = {
            'pairwise_means': [],
            'pairwise_stds': [],
            'axis_spreads': [],
            'aspect_ratios': [],
            'centroid_dists': []
        }
    
    # Get current population from instance attribute (set by _optimize)
    pop = getattr(self, '_current_population', None)
    if pop is None:
        # Fallback: gentle correction if no population available
        self.F = np.clip(self.F * 1.0, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * 1.0, 0.1, 0.9)
        return
    
    # === Compute geometric features ===
    # 1. Axis-aligned spread (range per dimension)
    pop_min = np.min(pop, axis=0)
    pop_max = np.max(pop, axis=0)
    axis_spreads = pop_max - pop_min
    max_spread = np.max(axis_spreads)
    min_spread = np.min(axis_spreads) + 1e-10
    aspect_ratio = max_spread / min_spread  # Elongation measure
    
    # 2. Pairwise distance statistics (subsample for efficiency)
    n_geo = min(20, len(pop))
    indices = np.random.choice(len(pop), n_geo, replace=False)
    subpop = pop[indices]
    diffs = subpop[:, np.newaxis, :] - subpop[np.newaxis, :, :]
    pairwise_dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    triu_indices = np.triu_indices(n_geo, k=1)
    pairwise_mean = np.mean(pairwise_dists[triu_indices])
    pairwise_std = np.std(pairwise_dists[triu_indices])
    
    # 3. Centroid-relative geometry
    centroid = np.mean(pop, axis=0)
    centroid_dists = np.linalg.norm(pop - centroid, axis=1)
    mean_centroid_dist = np.mean(centroid_dists)
    
    # 4. Store history (window of 10)
    history = self.pop_geometry_history
    history['pairwise_means'].append(pairwise_mean)
    history['pairwise_stds'].append(pairwise_std)
    history['axis_spreads'].append(max_spread)
    history['aspect_ratios'].append(aspect_ratio)
    history['centroid_dists'].append(mean_centroid_dist)
    
    for key in history:
        if len(history[key]) > 10:
            history[key].pop(0)
    
    # === Compute geometric regime indicators ===
    # Compare current geometry to historical average
    hist_pm = np.mean(history['pairwise_means'])
    hist_as = np.mean(history['axis_spreads'])
    
    diversity_ratio = pairwise_mean / (hist_pm + 1e-10)
    spread_ratio = max_spread / (hist_as + 1e-10)
    
    # Detect geometric states
    is_expanded = diversity_ratio > 1.1 or spread_ratio > 1.15
    is_contracted = diversity_ratio < 0.9 or spread_ratio < 0.85
    is_highly_elongated = aspect_ratio > 5.0
    is_near_spherical = aspect_ratio < 2.0
    
    # === Compute success rate for weighting ===
    success_rate = np.mean(improved_mask)
    
    # === Adapt F based on geometric diversity ===
    # High diversity → reduce F (focused exploitation)
    # Low diversity → increase F (wider exploration)
    if is_expanded:
        F_adjustment = 0.88  # Contract: reduce step size
    elif is_contracted:
        F_adjustment = 1.12  # Expand: increase step size
    else:
        # Gentle correction based on diversity trend
        delta = diversity_ratio - 1.0
        F_adjustment = 1.0 - 0.15 * delta
    
    # Success-rate weighting: successful adaptation should stabilize
    if success_rate > 0.3:
        F_adjustment = F_adjustment * 0.95 + 1.0 * 0.05  # Stabilize on success
    elif success_rate < 0.1:
        F_adjustment *= 1.08  # Boost adjustment on failure
    
    # === Adapt Cr based on geometric shape ===
    # Elongated population → higher Cr (recombine across axes)
    # Spherical population → lower Cr (preserve structure)
    if is_highly_elongated:
        Cr_adjustment = 1.10  # Increase recombination
    elif is_near_spherical:
        Cr_adjustment = 0.92  # Decrease recombination
    else:
        # Modulate by aspect ratio deviation from spherical
        aspect_deviation = (aspect_ratio - 2.0) / 8.0  # Normalize: 2→0, 10→1
        aspect_deviation = np.clip(aspect_deviation, -0.2, 0.2)
        Cr_adjustment = 1.0 + 0.15 * aspect_deviation
    
    # Success-rate weighting for Cr
    if success_rate > 0.35:
        Cr_adjustment *= 0.97  # Reduce on high success
    elif success_rate < 0.15:
        Cr_adjustment *= 1.05  # Increase exploration on low success
    
    # === Apply adjustments with clipping ===
    self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
    self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
```