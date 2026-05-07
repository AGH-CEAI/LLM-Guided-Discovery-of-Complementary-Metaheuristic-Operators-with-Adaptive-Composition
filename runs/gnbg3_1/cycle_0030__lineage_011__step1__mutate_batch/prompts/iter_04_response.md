**Idea: Covariance-Adaptive Mutation with Archive Exploitation**

This approach uses the covariance structure of the population to guide mutations along principal axes (like CMA-ES), combined with an archive of elite solutions for targeted exploitation. This is fundamentally different from standard DE which treats dimensions independently and ignores correlation structure—critical for the rotated/ill-conditioned landscapes causing the worst errors (Tasks 17, 16, 11).

```python
def _mutate_batch(self, population, fitness, selected_operator):
    """Generate mutant vectors using covariance-adaptive mutation with archive."""
    np_pop = len(population)
    dim = self.dim
    
    # Initialize mutant population
    mutants = np.empty_like(population)
    
    # Maintain archive of best solutions
    if not hasattr(self, 'mutation_archive'):
        self.mutation_archive = population[fitness.argsort()[:min(10, np_pop)]].copy()
        self._archive_fitness = np.sort(fitness)[:min(10, np_pop)]
    
    # Update archive with current best solutions
    n_archive = len(self.mutation_archive)
    current_best_idx = np.argmin(fitness)
    if fitness[current_best_idx] < self._archive_fitness[0]:
        if n_archive < 50:
            self.mutation_archive = np.vstack([self.mutation_archive, population[current_best_idx]])
            self._archive_fitness = np.append(self._archive_fitness, fitness[current_best_idx])
        else:
            worst_arch_idx = np.argmax(self._archive_fitness)
            self.mutation_archive[worst_arch_idx] = population[current_best_idx]
            self._archive_fitness[worst_arch_idx] = fitness[current_best_idx]
    
    # Compute covariance matrix of population for adaptive direction
    mean_pop = np.mean(population, axis=0)
    centered = population - mean_pop
    cov_matrix = np.cov(centered.T)
    
    # Regularize for numerical stability
    cov_matrix += np.eye(dim) * 1e-8
    
    # Eigendecomposition with fallback
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
    except np.linalg.LinAlgError:
        eigenvalues = np.ones(dim)
        eigenvectors = np.eye(dim)
    
    # Normalize eigenvalues for scaling
    eigenvalues_norm = eigenvalues / (np.sum(eigenvalues) + 1e-12)
    sqrt_eigen = np.sqrt(eigenvalues_norm)
    
    # Sample random indices for base vectors
    r1 = np.random.randint(0, np_pop, size=np_pop)
    r2 = np.random.randint(0, n_archive, size=np_pop)
    
    # Adaptive F parameter
    f_values = self.f_base + self.f_adapt * np.random.randn(np_pop)
    f_values = np.clip(f_values, 0.1, 2.0)
    
    # Strategy 0: Covariance-adaptive with archive guidance
    if selected_operator == 0:
        base_pop = population[r1]
        target_archive = self.mutation_archive[r2]
        
        # Compute scaled direction using covariance eigendirections
        diff = target_archive - base_pop
        scaled_diff = diff @ eigenvectors * sqrt_eigen[np.newaxis, :] @ eigenvectors.T
        
        # Add correlated noise along principal axes
        noise = np.random.randn(np_pop, dim)
        noise = noise @ eigenvectors * sqrt_eigen[np.newaxis, :] @ eigenvectors.T
        noise *= 0.1 * f_values[:, np.newaxis]
        
        mutants = base_pop + f_values[:, np.newaxis] * scaled_diff + noise
    
    # Strategy 1: Archive-driven exploration with population diversity
    elif selected_operator == 1:
        r3 = np.random.randint(0, np_pop, size=np_pop)
        base_pop = population[r1]
        archive_pop = self.mutation_archive[r2]
        third_pop = population[r3]
        
        # Weighted combination emphasizing archive
        archive_weight = 0.7
        mutants = (archive_pop * archive_weight + 
                  base_pop * (1 - archive_weight) +
                  f_values[:, np.newaxis] * (third_pop - base_pop))
    
    # Strategy 2: Dimension-wise adaptive mutation
    elif selected_operator == 2:
        r3 = np.random.randint(0, np_pop, size=np_pop)
        base_pop = population[r1]
        archive_pop = self.mutation_archive[r2]
        third_pop = population[r3]
        
        # Per-dimension eigenvalue-weighted mutation
        dim_weights = sqrt_eigen / (np.sum(sqrt_eigen) + 1e-12)
        dim_weights = dim_weights / (np.max(dim_weights) + 1e-12)
        
        mutants = (base_pop + 
                  f_values[:, np.newaxis] * dim_weights[np.newaxis, :] * (archive_pop - third_pop))
    
    # Strategy 3: Covariance-based jumping with archive exploitation
    else:
        r3 = np.random.randint(0, np_pop, size=np_pop)
        base_pop = population[r1]
        archive_pop = self.mutation_archive[r2]
        third_pop = population[r3]
        
        # Large covariance jump toward archive
        diff = archive_pop - base_pop
        scaled_diff = diff @ eigenvectors * sqrt_eigen[np.newaxis, :] @ eigenvectors.T
        
        # Secondary perturbation
        perp_diff = third_pop - base_pop
        
        mutants = (base_pop + 
                  f_values[:, np.newaxis] * scaled_diff +
                  0.3 * f_values[:, np.newaxis] * perp_diff)
    
    # Clip to bounds
    mutants = np.clip(mutants, self.lower_bound, self.upper_bound)
    
    return mutants
```