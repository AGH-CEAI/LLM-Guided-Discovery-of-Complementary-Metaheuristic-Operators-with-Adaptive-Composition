Looking at the priority targets, the worst unsolved tasks (Tasks 17, 16, 23, 20, 18, 19) all have errors in the 1e+1 to 5e+02 range, indicating the algorithm is getting trapped in local optima. The `_compute_diversity` function controls restart behavior, which is critical for escaping these traps.

**Analysis of the failure mode:**
- Current ESS-based metric uses covariance matrix eigenvalues - an **indirect** measure
- For ill-conditioned or deceptive problems, the covariance can appear "healthy" (good ESS) even while the population has collapsed to a small region of space
- The restart trigger is miscalibrated: it either fires too late (population already converged to local optimum) or misses the collapse entirely

**Fundamentally different strategy:**
Replace the eigenvalue-based metric with a **direct pairwise distance metric** in decision space. This directly measures whether the population has actually spread through the search space, regardless of covariance matrix structure.

**Idea: Pairwise Distance Diversity**

```python
def _compute_diversity(self):
    """Compute diversity using average pairwise Euclidean distance in decision space.

    Directly measures spatial spread of population rather than relying on
    covariance matrix statistics. This catches population collapse that
    eigenvalue-based metrics miss on ill-conditioned problems.

    Diversity = log(1 + mean_pairwise_dist / expected_dist)
      where expected_dist is the mean distance in a uniformly distributed population

    Log scaling provides sensitivity across orders of magnitude while avoiding
    numerical instability when distances are very small.
    """
    if len(self.population) < 2:
        return 1e-15

    # Compute all pairwise Euclidean distances
    pop = self.population.reshape(len(self.population), -1)
    diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
    pairwise_dists = np.sqrt(np.sum(diffs ** 2, axis=2))

    # Exclude diagonal (self-distances)
    n = len(self.population)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_pairwise_dist = np.mean(pairwise_dists[mask])

    # Normalize by search space diameter
    search_space_diameter = np.sqrt(self.dim) * (self.ub[0] - self.lb[0])
    expected_dist = search_space_diameter / np.sqrt(2.0)

    # Ratio with protection against division by zero
    dist_ratio = mean_pairwise_dist / max(expected_dist, 1e-10)

    # Log-scaled diversity for numerical stability across scales
    diversity = np.log1p(dist_ratio)

    return float(np.clip(diversity, 1e-15, None))
```

**Why this is fundamentally different:**
- **Direct vs indirect**: Measures actual population spread in decision space, not inferred spread from covariance eigenvalues
- **Invariant to rotation/conditioning**: Doesn't depend on eigenvalue distribution, so it won't miss collapse on highly elongated or rotated landscapes
- **Restart timing**: Will correctly trigger restart when population clusters (even if covariance looks fine), helping escape the local optima trapping the worst tasks
- **Numerical robustness**: Log scaling handles the wide range of distances encountered during optimization