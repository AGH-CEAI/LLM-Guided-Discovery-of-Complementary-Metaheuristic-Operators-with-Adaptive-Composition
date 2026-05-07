Looking at the context, I need to implement `_crossover_batch` using spectral/linear-algebraic reasoning. The previous Category B variant achieved poor results (1.1746e+02 on task 5), so I need a different spectral approach.

My approach: Use eigendecomposition of the population covariance matrix to compute eigenvalue-weighted crossover probabilities. This aligns crossover with the principal axes of population variance, exploring more aggressively along high-variance directions.

**Idea: Eigenvalue-Weighted Spectral Crossover**
Uses PCA eigendecomposition of population covariance to weight crossover probabilities per dimension, adapting to population anisotropy.
```python
def _crossover_batch(self, population, mutants):
    NP, dim = population
    if dim < 2 or NP < dim + 1:
        return np.where(np.random.rand(NP, dim) < self.CR, mutants, population)
    
    # Compute population covariance and its eigendecomposition
    centered = population - population.mean(axis=0)
    cov = (centered.T @ centered) / max(NP - 1, 1)
    
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
    except Exception:
        return np.where(np.random.rand(NP, dim) < self.CR, mutants, population)
    
    # Condition number for diversity detection
    cond = eigenvalues.max() / eigenvalues.min()
    
    # Spectral adaptive CR: higher CR when population is clustered (low condition number)
    # This encourages exploration of collapsed subspaces
    spectral_cr = self.CR
    if cond < 10.0:
        spectral_cr = min(0.95, self.CR * 1.4)
    elif cond > 100.0:
        spectral_cr = max(0.3, self.CR * 0.8)
    
    # Eigenvalue-weighted probability: dimensions with high variance get higher crossover probability
    ev_sum = eigenvalues.sum()
    if ev_sum > 1e-12:
        ev_prob = eigenvalues / ev_sum
    else:
        ev_prob = np.full(dim, 1.0 / dim)
    
    # Combine spectral weight with base CR, ensuring minimum exploration in all dims
    cr_per_dim = np.clip(0.3 * ev_prob / (ev_prob.max() + 1e-12) + 0.7 * spectral_cr, 0.1, 0.99)
    
    # Generate trial vectors via binomial crossover
    mask = np.random.rand(NP, dim) < cr_per_dim
    mask |= np.arange(dim) == np.random.randint(0, dim, NP)[:, np.newaxis]
    
    trials = np.where(mask, mutants, population)
    return trials
```