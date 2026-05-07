**Idea: Spectral Condition-Number Restart**
Use eigenvalue-based condition number of the population covariance matrix to detect when the population has collapsed into a low-dimensional subspace, then restart by injecting solutions along the minor eigenvectors (direction of maximum uncertainty).

```python
def _restart_if_needed(self, population, fitness):
    """Restart based on spectral analysis of population covariance matrix.
    
    Uses eigenvalue decomposition to detect subspace collapse (high condition number
    or low effective rank) and injects diversity along minor eigenvectors.
    """
    # Compute population covariance and extract spectral information
    centered = population - population.mean(axis=0)
    cov = np.cov(centered.T)
    
    # Get eigenvalues (sorted ascending)
    try:
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]  # Descending
    except np.linalg.LinAlgError:
        eigvals = None
    
    # Condition number: ratio of max to min eigenvalue
    # High condition number = population elongated in few directions
    if eigvals is not None and len(eigvals) > 0 and eigvals[-1] > 1e-10:
        condition_number = eigvals[0] / max(eigvals[-1], 1e-10)
    else:
        condition_number = 1e10
    
    # Effective rank via normalized Shannon entropy of eigenvalues
    # Low effective rank = population in low-dimensional subspace
    if eigvals is not None and eigvals.sum() > 1e-10:
        probs = eigvals / eigvals.sum()
        probs = probs[probs > 1e-10]
        effective_rank = np.exp(-np.sum(probs * np.log(probs + 1e-10)))
    else:
        effective_rank = 1.0
    
    # Restart triggers: severe anisotropy or low effective dimensionality
    should_restart = (
        condition_number > 500 or
        effective_rank < self.dim * 0.4
    )
    
    if not should_restart:
        return (None, None, None)
    
    # Spectral restart: keep best, inject along minor eigenvectors
    best_idx = np.nanargmin(fitness)
    best_solution = population[best_idx].copy()
    f_opt = fitness[best_idx]
    
    # Re-center and compute eigenvectors
    centered = population - population.mean(axis=0)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
    except np.linalg.LinAlgError:
        # Fallback: random restart
        population = np.random.uniform(-100.0, 100.0, (self.NP, self.dim))
        population[0] = best_solution
        return population, f_opt, best_solution
    
    # Replace worst individuals with perturbations along minor eigenvectors
    n_replace = max(2, self.NP // 3)
    worst_indices = np.argsort(fitness)[-n_replace:]
    
    for i, idx in enumerate(worst_indices):
        # Use eigenvectors with smallest eigenvalues (directions of collapse)
        if i < 3 and eigvals[i] > 1e-10:
            v = eigvecs[:, i]
            # Scale inversely to eigenvalue - smaller eigenvalue = more exploration needed
            scale = 40.0 / max(np.sqrt(eigvals[i]), 1e-6)
        else:
            v = np.random.randn(self.dim)
            v /= (np.linalg.norm(v) + 1e-10)
            scale = 40.0
        
        perturbation = v * scale * np.random.randn()
        population[idx] = np.clip(best_solution + perturbation, -100.0, 100.0)
    
    return population, f_opt, best_solution
```