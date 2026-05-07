Looking at the task requirements, I need to create a spectral/linear-algebraic variant that uses eigendecomposition, eigenvalues, or singular values to guide mutation. The key is to leverage the population's covariance structure to determine anisotropic mutation scales.

**Idea: PCA-Eigenvalue Scaled Mutation**
Use eigendecomposition of population covariance to compute principal axes and scale mutation strength inversely proportional to eigenvalue magnitude (small eigenvalue = high variance direction = more aggressive exploration).

```python
def _mutate_compass_batch(self, population, fitness):
    """
    Category B: Spectral/linear-algebraic compass mutation.
    Uses eigendecomposition of population covariance matrix to identify
    principal axes and scale mutation inversely to eigenvalue magnitude.
    Directions with low variance (small eigenvalues) are explored more aggressively.
    """
    NP, dim = population.shape
    
    # Compute fitness ranks for selection pressure
    ranks = self._compute_fitness_ranking(fitness)
    best_idx = np.argmin(fitness)
    i_best = np.argmin(ranks)
    
    # Compute population covariance matrix (centered)
    centroid = population.mean(axis=0)
    centered = population - centroid
    cov = (centered.T @ centered) / max(NP - 1, 1)
    
    # Eigendecomposition: V @ diag(λ) @ V.T
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        # Fallback: use standard deviation scaling
        std_pop = np.std(population, axis=0) + 1e-10
        return population + self.F * np.random.randn(NP, dim) * std_pop
    
    # Sort by descending eigenvalue (descending importance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Clamp eigenvalues to avoid numerical issues
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    
    # Compute total variance for normalization
    total_var = np.sum(eigenvalues)
    if total_var > 0:
        explained_ratio = eigenvalues / total_var
    else:
        explained_ratio = np.ones(dim) / dim
    
    # Effective dimensionality: number of PCs needed to explain 95% variance
    cumsum = np.cumsum(explained_ratio)
    n_effective = int(np.searchsorted(cumsum, 0.95)) + 1
    n_effective = min(n_effective, dim)
    
    # Compute mutation scaling per principal axis
    # Low eigenvalue (low variance) -> high scaling (more exploration)
    # High eigenvalue (high variance) -> low scaling (conservative)
    base_scale = 1.0 / np.sqrt(eigenvalues + 1e-10)
    axis_scaling = base_scale / (np.max(base_scale) + 1e-10)
    
    # Generate mutants per individual
    mutants = np.empty_like(population)
    
    for i in range(NP):
        # Direction toward p-best (top 10% by fitness)
        p = 0.1
        top_k = max(1, int(np.ceil(p * NP)))
        pbest_indices = np.argsort(ranks)[:top_k]
        pbest = population[np.random.choice(pbest_indices)]
        
        # Base direction: current to pbest
        direction = pbest - population[i]
        
        # Transform direction to eigenspace: z = V.T @ direction
        z = eigenvectors.T @ direction
        
        # Scale each component inversely to eigenvalue
        z_scaled = z * axis_scaling
        
        # Transform back to original space
        direction_scaled = eigenvectors @ z_scaled
        
        # Normalize scale based on fitness rank (lower rank = more exploration)
        rank_factor = 0.5 + 0.5 * (1.0 - ranks[i])
        
        # Add random perturbation scaled by inverse eigenvalue
        random_perturb = np.zeros(dim)
        for j in range(n_effective):
            # More perturbation along low-variance axes
            random_perturb += eigenvectors[:, j] * np.random.randn() * axis_scaling[j]
        
        # Compute mutant
        mutants[i] = population[i] + self.F * rank_factor * direction_scaled + \
                     0.3 * self.F * random_perturb
    
    return np.clip(mutants, -100.0, 100.0)
```