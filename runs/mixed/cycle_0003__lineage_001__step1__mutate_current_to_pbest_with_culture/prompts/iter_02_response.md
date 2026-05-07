**Idea: Rank-Weighted Opposition Learning with Dimension Perturbation**

The worst tasks (12, 8, 16, 15) have errors of ~1e+2 to 1e+3 — the algorithm is stuck in local optima and cannot escape. The current operator is **exploitation-heavy** (p_best direction dominates) and provides insufficient **exploration diversity**. This variant replaces the p_best-centric strategy with: (1) rank-based selection using the ENTIRE population, (2) opposition-based learning (reflecting around weighted centroid), and (3) per-dimension Gaussian perturbations for escaping narrow basins.

```python
def _mutate_current_to_pbest_with_culture(self, population, fitness):
    # Use ALL population ranks for selection — fundamentally different from p_best-only
    finite_mask = np.isfinite(fitness)
    valid_fitness = np.where(finite_mask, fitness, np.nanmax(fitness[finite_mask]) * 2)
    ranks = np.argsort(np.argsort(valid_fitness))
    probs = 1.0 / (ranks + 1)
    probs = probs / np.sum(probs)
    
    # Sample weighted centroid from entire population (not just top performers)
    weighted_centroid = np.dot(probs, population)
    
    # Opposition-based learning: reflect current around weighted centroid
    opposite = 2 * weighted_centroid - population
    
    # Add dimension-wise Gaussian perturbations (key for escaping local optima)
    dim_perturb = np.random.randn(self.NP, self.dim) * (self.upper - self.lower) * 0.1
    
    # Blend with cultural memory (when available)
    if self.generation > 5 and np.any(self.cultural_weights > 0):
        weights_norm = self.cultural_weights / (np.sum(self.cultural_weights) + 1e-10)
        cultural_dir = np.dot(weights_norm, self.cultural_memory)
        cultural_boost = 0.1 * cultural_dir
    else:
        cultural_boost = 0.0
    
    # Scale factor: larger for exploration on hard tasks
    F = np.clip(self.F * 1.5, 0.4, 2.0)
    
    # Generate donors from opposition + perturbation + cultural guidance
    donors = opposite + dim_perturb + cultural_boost
    donors = self._clip_to_bounds_batch(donors)
    
    return donors
```