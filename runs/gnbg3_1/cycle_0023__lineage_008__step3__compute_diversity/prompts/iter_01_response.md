**Idea: Mean Pairwise Distance**
Compute diversity as mean Euclidean distance between all population pairs, capturing global spread rather than per-dimension variance.

```python
def _compute_diversity(self):
    """Compute population diversity as mean pairwise Euclidean distance."""
    pop = self.population
    n = pop.shape[0]
    
    if n < 2:
        return 0.0
    
    # Compute pairwise Euclidean distances using broadcasting
    diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    distances = np.sqrt(sq_dists)
    
    # Extract upper triangle (k=1 excludes diagonal)
    upper_tri_indices = np.triu_indices(n, k=1)
    pairwise_distances = distances[upper_tri_indices]
    
    # Return mean pairwise distance, handle edge cases
    if len(pairwise_distances) == 0:
        return 0.0
    
    return float(np.mean(pairwise_distances))
```