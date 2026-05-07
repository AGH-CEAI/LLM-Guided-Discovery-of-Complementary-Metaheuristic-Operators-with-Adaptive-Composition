Looking at the context carefully:
- I must use **Category A: Geometry / spatial** reasoning only
- The current implementation uses temporal EMA dynamics (Category F) - I must replace it
- This is a **replacement** for `_adapt_parameters`, so I need to completely rewrite it
- I must NOT repeat the previous Category A idea (variant_01_catA_idea_0)

**Analysis of failure modes:**
- Worst tasks (17, 16, 11, 6, 5) have errors in the 10¹-10⁵ range, suggesting severe multimodality, narrow basins, or deceptive landscapes
- The current temporal/fitness-based adaptation isn't detecting when the population is geometrically trapped or dispersed
- **Geometry-based insight**: Population *spatial spread* directly indicates convergence state - tight clustering → exploit, wide spread → explore. Centroid position relative to bounds reveals boundary entrapment. Pairwise distances reveal clustering vs dispersion.

**Idea: Spatial Regime Detection via Population Geometry**
Use population centroid position relative to bounds and axis-aligned spread to classify the optimization regime geometrically. When centroid is near boundaries AND spread is low → trapped in corner → increase F aggressively. When spread is high → population dispersed → reduce F. Use mean pairwise distance to centroid to modulate Cr.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using geometric properties of population layout."""
    # Access stored population from optimization loop
    if not hasattr(self, 'stored_population') or self.stored_population is None:
        self.F = np.clip(self.F * 1.0, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * 1.0, 0.1, 0.9)
        return
    
    pop = self.stored_population
    if len(pop) < 2:
        return
    
    # --- Geometric Feature Computation ---
    # 1. Axis-aligned spread per dimension (geometric range)
    dim_min = np.min(pop, axis=0)
    dim_max = np.max(pop, axis=0)
    dim_spread = dim_max - dim_min
    total_spread = np.sum(dim_spread)
    range_normalized_spread = total_spread / (self.dim * (self.upper - self.lower) + 1e-10)
    
    # 2. Centroid position relative to bounds (geometric centrality)
    centroid = np.mean(pop, axis=0)
    centroid_normalized = (centroid - self.lower) / (self.upper - self.lower + 1e-10)
    corner_proximity = np.min(centroid_normalized)  # Low = near corner/boundary
    edge_bias = np.max(np.abs(centroid_normalized - 0.5))  # High = off-center
    
    # 3. Mean distance to centroid (geometric compactness)
    distances_to_centroid = np.linalg.norm(pop - centroid, axis=1)
    mean_dist = np.mean(distances_to_centroid)
    max_possible_dist = np.sqrt(self.dim) * (self.upper - self.lower) / 2
    compactness = mean_dist / (max_possible_dist + 1e-10)
    
    # 4. Pairwise distance statistics (clustering detection)
    centroid_dist_matrix = distances_to_centroid[:, np.newaxis] + distances_to_centroid[np.newaxis, :]
    pairwise_dist = np.linalg.norm(pop[:, np.newaxis, :] - pop[np.newaxis, :, :], axis=2)
    np.fill_diagonal(pairwise_dist, np.inf)
    mean_pairwise = np.mean(np.min(pairwise_dist, axis=1))
    max_pairwise = np.max(np.min(pairwise_dist, axis=1))
    clustering_ratio = mean_pairwise / (max_pairwise + 1e-10)
    
    # --- Geometric Regime Classification ---
    # Classify population state purely from spatial geometry
    is_tight = compactness < 0.15
    is_dispersed = compactness > 0.4
    is_corner_trapped = corner_proximity < 0.1 or edge_bias > 0.4
    is_clustered = clustering_ratio < 0.3
    
    # --- Parameter Adjustment via Geometry ---
    if is_tight and is_corner_trapped:
        # Population is compact AND near boundary corner → trapped, need aggressive exploration
        F_adjustment = 1.30
        Cr_adjustment = 0.80
    elif is_tight and not is_corner_trapped:
        # Compact but well-centered → exploit local region
        F_adjustment = 0.85
        Cr_adjustment = 1.10
    elif is_dispersed:
        # Wide spread → reduce F, allow convergence
        F_adjustment = 0.80
        Cr_adjustment = 1.05
    elif is_clustered and not is_tight:
        # Multiple subclusters → moderate F, higher Cr to recombine
        F_adjustment = 1.10
        Cr_adjustment = 0.90
    else:
        # Neutral geometric state → proportional adjustment based on spread
        F_adjustment = 1.0 + 0.20 * (range_normalized_spread - 0.3)
        Cr_adjustment = 1.0 - 0.15 * (range_normalized_spread - 0.3)
    
    # Apply geometrically-motivated adjustments
    self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
    self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
```