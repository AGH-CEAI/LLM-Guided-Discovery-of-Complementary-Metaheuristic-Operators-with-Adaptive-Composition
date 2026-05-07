**Idea: Covariance Condition Number Restart**

This restart mechanism analyzes the population's covariance matrix spectral properties (condition number and effective rank) to detect when the population has collapsed into a low-dimensional subspace. When anisotropy is extreme or effective dimensionality drops, a full restart reinitializes the population using eigenvectors to maintain diversity along previously unexplored directions.

```python
def _restart_if_needed(self, population, fitness):
    """Restart based on spectral analysis of population covariance."""
    # Compute population covariance matrix
    centroid = population.mean(axis=0)
    centered = population - centroid
    cov = np.cov(centered.T)
    
    # Get spectral information
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    # Condition number: high value indicates elongated/collapsed distribution
    cond_number = eigenvalues[0] / eigenvalues[-1]
    
    # Effective rank: count dimensions contributing meaningful variance
    total_var = eigenvalues.sum()
    entropy = -np.sum((eigenvalues / total_var) * np.log(eigenvalues / total_var + 1e-12))
    eff_rank = np.exp(entropy)
    
    # Restart if population is too anisotropic (cond > 1000) or collapsed (eff_rank < dim/3)
    dim = self.dim
    should_restart = (cond_number > 1000.0) or (eff_rank < dim / 3.0)
    
    if should_restart:
        # Preserve top 20% individuals
        n_preserve = max(2, self.NP // 5)
        top_indices = np.argsort(fitness)[:n_preserve]
        new_population = population[top_indices].copy()
        
        # Compute eigendecomposition for orthogonal perturbation directions
        eigenvalues_full, eigenvectors_full = np.linalg.eigh(cov)
        eigenvalues_full = np.maximum(eigenvalues_full, 1e-12)
        order = np.argsort(eigenvalues_full)[::-1]
        V = eigenvectors_full[:, order]
        D = np.sqrt(eigenvalues_full[order])
        
        # Generate new individuals using eigenvector perturbations
        while len(new_population) < self.NP:
            # Pick a random preserved individual
            base = new_population[np.random.randint(len(top_indices))]
            
            # Project onto orthogonal complement of current population span
            if len(new_population) > 1:
                current_pop_centered = new_population - new_population.mean(axis=0)
                pop_cov = np.cov(current_pop_centered.T)
                pop_eigvals, pop_eigvecs = np.linalg.eigh(pop_cov)
                pop_eigvals = np.maximum(pop_eigvals, 1e-12)
                pop_order = np.argsort(pop_eigvals)[::-1]
                existing_basis = pop_eigvecs[:, pop_order[:min(len(top_indices), self.dim)]]
                # Project base onto orthogonal complement
                for vec in existing_basis.T:
                    base = base - np.dot(vec, base) * vec
            
            # Add perturbation along eigenvectors scaled by inverse singular values
            noise = np.zeros(dim)
            for j in range(min(dim, len(D))):
                scale = 20.0 / (D[j] + 1.0)  # Larger steps in low-variance directions
                noise += scale * np.random.randn() * V[:, j]
            
            new_individual = np.clip(base + noise, -100.0, 100.0)
            new_population = np.vstack([new_population, new_individual])
        
        return new_population, None, None
    
    return (None, None, None)
```