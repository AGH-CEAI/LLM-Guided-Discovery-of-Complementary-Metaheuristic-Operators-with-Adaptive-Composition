**Idea: Pairwise Distance Diversity**
Compute diversity as mean pairwise Euclidean distance between all individuals, which is fundamentally more sensitive to clustering and captures multi-dimensional spread holistically rather than averaging per-dimension variances.

```python
def _compute_diversity(self):
    """Compute population diversity as mean pairwise Euclidean distance."""
    n = self.population.shape[0]
    if n < 2:
        return 0.0
    # Compute pairwise squared distances efficiently using broadcasting
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs * diffs, axis=2)
    # Get upper triangle (avoid double counting and diagonal)
    upper_tri_indices = np.triu_indices(n, k=1)
    pairwise_sq_dists = sq_dists[upper_tri_indices]
    # Mean pairwise distance with numerical safety
    mean_sq_dist = np.mean(pairwise_sq_dists)
    # Return Euclidean distance (sqrt of mean squared distance)
    if mean_sq_dist <= 0:
        return 0.0
    return np.sqrt(mean_sq_dist)
```