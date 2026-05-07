**Idea: PCA-Axis Compass Mutation**

Category A: Geometry / spatial approach using principal component analysis to determine mutation directions based on the population's actual geometric spread, then guiding mutations along major and minor axes toward/away from extreme points.

```python
def _mutate_compass_batch(self, population, fitness):
    """Vectorized compass mutation using PCA-based geometric analysis."""
    NP, dim = population.shape
    
    # Compute population centroid (geometric center)
    centroid = population.mean(axis=0)
    
    # Compute covariance matrix for PCA
    centered = population - centroid
    cov = np.cov(centered.T)
    
    # Handle edge case: near-degenerate covariance
    if np.linalg.matrix_rank(cov) < dim:
        # Fall back to axis-aligned mutation using coordinate-wise spread
        coordinate_spread = population.std(axis=0)
        coordinate_spread = np.where(coordinate_spread < 1e-10, 1.0, coordinate_spread)
        # Mutate toward random coordinate extremes
        best_idx = np.argmin(fitness)
        mutants = np.empty_like(population)
        for i in range(NP):
            extreme_dim = np.random.randint(dim)
            direction = np.sign(population[best_idx, extreme_dim] - population[i, extreme_dim])
            mutants[i] = population[i] + self.F * direction * coordinate_spread * np.random.rand()
        return mutants
    
    # Eigendecomposition for principal axes
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by descending variance (largest eigenvalue = major axis)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Project population onto principal axes
    projections = centered @ eigenvectors  # Shape: (NP, dim)
    
    # Find extreme individuals along each principal axis
    mutants = np.empty_like(population)
    major_axis = eigenvectors[:, 0]  # Direction of maximum spread
    minor_axis = eigenvectors[:, -1]  # Direction of minimum spread
    
    # Compute axis-aligned spreads for scaling
    major_spread = np.abs(projections[:, 0]).max() + 1e-10
    minor_spread = np.abs(projections[:, -1]).max() + 1e-10
    
    for i in range(NP):
        # Determine mutation strategy based on position relative to centroid
        i_proj_major = projections[i, 0]
        i_proj_minor = projections[i, -1]
        
        # Find individuals far along each axis (geometric extremes)
        # Positive extreme on major axis
        pos_major_idx = np.argmax(projections[:, 0])
        # Negative extreme on major axis
        neg_major_idx = np.argmin(projections[:, 0])
        # Positive extreme on minor axis
        pos_minor_idx = np.argmax(projections[:, -1])
        # Best individual (lowest fitness)
        best_idx = np.argmin(fitness)
        
        # Random selection among strategies
        strategy = np.random.randint(4)
        
        if strategy == 0:
            # Move toward positive major axis extreme
            direction = major_axis
            scale = major_spread
            target = population[pos_major_idx]
        elif strategy == 1:
            # Move toward negative major axis extreme
            direction = -major_axis
            scale = major_spread
            target = population[neg_major_idx]
        elif strategy == 2:
            # Move toward minor axis extreme
            direction = minor_axis if np.random.rand() > 0.5 else -minor_axis
            scale = minor_spread
            target = population[pos_minor_idx] if direction is minor_axis else population[np.argmin(projections[:, -1])]
        else:
            # Move toward geometric centroid
            direction = centroid - population[i]
            scale = np.linalg.norm(direction) + 1e-10
            target = centroid
        
        # Compute mutation vector
        if strategy == 3:
            # Direct centroid pull
            mutants[i] = population[i] + self.F * direction
        else:
            # Directional jump toward extreme point, scaled by geometric distance
            dist_to_target = np.linalg.norm(population[i] - target) + 1e-10
            mutation = self.F * (target - population[i]) * (scale / dist_to_target)
            mutants[i] = population[i] + mutation
    
    return mutants
```