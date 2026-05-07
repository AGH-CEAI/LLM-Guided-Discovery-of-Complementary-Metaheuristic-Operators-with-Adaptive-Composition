Looking at the problem, the `_compute_diversity` function is used to detect stagnation and trigger restarts. The current simple mean-per-dimension-std is too coarse and doesn't capture the actual spatial distribution of individuals.

The worst unsolved tasks (16, 17, 23, etc.) with errors ~10^1-10^2 suggest the algorithm is getting trapped in local optima without proper restart detection.

**Idea: Pairwise Distance + Condition Number Fusion**
Replace the naive per-dimension std with a combination of: (1) mean pairwise Euclidean distance normalized by search space diameter (captures true population spread), and (2) covariance condition number inverse (detects eigenvalue collapse). This captures both global spread and internal ill-conditioning that the current metric completely misses.

```python
def _compute_diversity(self):
    """Compute diversity as pairwise distance + condition number fusion."""
    pop = self.population
    n = pop.shape[0]
    dim = self.dim
    
    if n < 2:
        return 0.0
    
    # Global metric: mean pairwise distance normalized by search space diameter
    diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    triu_idx = np.triu_indices(n, k=1)
    pairwise_dists = dists[triu_idx]
    
    diameter = np.linalg.norm(self.ub - self.lb)
    if diameter < 1e-10:
        diameter = 1.0
    
    global_div = np.mean(pairwise_dists) / diameter
    global_div = np.clip(global_div, 0.0, 1.0)
    
    # Structural metric: inverse condition number of covariance (0=ill-conditioned, 1=well-conditioned)
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.min(eigvals)
    eig_max = np.max(eigvals)
    
    if eig_max < 1e-10:
        eig_max = 1.0
    if eig_min < 1e-10:
        eig_min = 1e-10
    
    cond = eig_max / eig_min
    cond_inv = 1.0 / (1.0 + cond)
    cond_inv = np.clip(cond_inv, 0.0, 1.0)
    
    # Fusion: geometric mean of both metrics for sensitivity to either failure mode
    diversity = np.sqrt(global_div * cond_inv)
    
    return float(diversity)
```