**Idea: Eigendecomposition-Driven Covariance Adaptation**

Use eigendecomposition of the covariance of successful mutations to detect anisotropy and scale differential vectors along principal axes. This gives a fundamentally different angle from the previous Category B variant (which likely used population covariance directly) by tracking *successful mutation direction* covariance specifically, adapting the mutation ellipsoid based on what works rather than just the population shape.

```python
def _mutate_batch(self, population, fitness, ring_prev, ring_next):
    """
    Generate mutation vectors using eigendecomposition-driven covariance adaptation.
    - Track covariance of successful mutations (not just population)
    - Eigendecompose to get principal axes of successful search directions
    - Scale differential vectors inversely proportional to eigenvalue magnitude
    - Blend with centroid for stability
    """
    np_pop, dim = population.shape
    trials = np.zeros((np_pop, dim))
    
    # Compute population centroid for fallback
    centroid = self._compute_population_centroid(population, fitness)
    
    # Select ring-based parents
    r1 = ring_prev
    r2 = ring_next
    
    rand1_vectors = population[r1]
    rand2_vectors = population[r2]
    
    # Adaptive F
    current_F = self.F * (1.0 + 0.1 * np.random.randn())
    current_F = np.clip(current_F, 0.1, 2.0)
    
    # Base mutation: ring-based differential
    base_mutation = rand1_vectors + current_F * (rand2_vectors - population)
    
    # --- NEW: Eigendecomposition-driven scaling ---
    # Build covariance of successful mutations from direction memory
    if len(self.successful_directions) >= dim + 1:
        # Stack successful mutation vectors
        D = np.array(self.successful_directions)
        # Center the data
        D_centered = D - np.mean(D, axis=0)
        # Covariance of successful mutations (not population covariance)
        cov_mutations = np.cov(D_centered, rowvar=False)
        
        # Regularize for numerical stability
        cov_mutations += 0.01 * np.eye(dim)
        
        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_mutations)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            
            # Compute inverse sqrt eigenvalues for scaling
            inv_sqrt_eig = 1.0 / np.sqrt(eigenvalues)
            # Clip to prevent extreme scaling
            inv_sqrt_eig = np.clip(inv_sqrt_eig, 0.1, 10.0)
            # Normalize to prevent overall scale change
            inv_sqrt_eig = inv_sqrt_eig / np.mean(inv_sqrt_eig)
            
            # Build scaling matrix and apply to differential vectors
            scaling_matrix = eigenvectors @ np.diag(inv_sqrt_eig) @ eigenvectors.T
            
            # Transform the differential component
            diff_vectors = rand2_vectors - population
            scaled_diff = diff_vectors @ scaling_matrix
            
            # Eigenscaled mutation component
            eigen_scaled = rand1_vectors + current_F * scaled_diff
            
            # Blend eigenscaled with base mutation
            blend_weight = 0.3 + 0.2 * (1.0 - self.Cr)
            trials = (1 - blend_weight) * base_mutation + blend_weight * eigen_scaled
        except np.linalg.LinAlgError:
            # Fallback on decomposition failure
            trials = base_mutation
    else:
        # Insufficient data for eigendecomposition, use base mutation
        trials = base_mutation
    
    # Clamp F for numerical safety
    current_F = np.clip(current_F, 0.1, 2.0)
    
    return trials, current_F
```