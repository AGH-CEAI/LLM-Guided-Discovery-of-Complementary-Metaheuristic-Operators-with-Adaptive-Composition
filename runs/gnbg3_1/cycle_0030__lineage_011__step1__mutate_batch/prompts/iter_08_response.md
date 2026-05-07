**Idea: Adaptive Orthogonal Multi-Vector Mutation**

A fundamentally different mutation using Gram-Schmidt orthogonalization of top individuals to generate exploration directions that are maximally diverse, combined with adaptive scaling and bounded perturbations. This breaks free from the correlated difference vectors of standard DE that fail on deceptive multi-modal landscapes.

```python
def _mutate_batch(self, population, fitness, selected_operator):
    """Generate mutant vectors using adaptive orthogonal multi-vector mutation."""
    np_pop = len(population)
    dim = population.shape[1]
    
    # Initialize mutant population
    mutants = np.empty_like(population)
    
    # Select top performers for creating orthogonal basis (exploitation)
    n_elite = min(max(3, dim), np_pop // 2)
    sorted_idx = np.argsort(fitness)
    elite_idx = sorted_idx[:n_elite]
    elite_pop = population[elite_idx]
    
    # Build orthogonal directions from elite using Gram-Schmidt
    if n_elite >= 2:
        # Center around best elite
        center = elite_pop[0].copy()
        centered = elite_pop - center
        
        # Orthogonalize using modified Gram-Schmidt
        ortho_vectors = []
        for i in range(min(n_elite - 1, dim)):
            v = centered[i].copy()
            for j in range(len(ortho_vectors)):
                proj = np.dot(v, ortho_vectors[j]) / (np.dot(ortho_vectors[j], ortho_vectors[j]) + 1e-10)
                v = v - proj * ortho_vectors[j]
            norm_v = np.linalg.norm(v)
            if norm_v > 1e-10:
                ortho_vectors.append(v)
        
        # Add random exploration direction
        random_dir = np.random.randn(dim)
        random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-10)
        ortho_vectors.append(random_dir * (self.upper_bound - self.lower_bound) * 0.1)
        
        ortho_matrix = np.array(ortho_vectors) if ortho_vectors else np.zeros((1, dim))
    else:
        ortho_matrix = np.zeros((1, dim))
        center = elite_pop[0] if n_elite > 0 else np.mean(population, axis=0)
    
    # Adaptive F based on diversity
    diversity = np.std(population, axis=0).mean()
    f_base = 0.8 if diversity > 10.0 else 0.5
    f_values = f_base + 0.3 * np.random.randn(np_pop)
    f_values = np.clip(f_values, 0.1, 2.0)
    
    # Generate mutants using orthogonal combination
    for i in range(np_pop):
        # Primary direction from orthogonal basis
        if len(ortho_matrix) > 0:
            coeffs = np.random.randn(len(ortho_matrix)) * f_values[i]
            primary = center + np.dot(coeffs, ortho_matrix)
        else:
            primary = center
        
        # Add secondary direction (random candidate)
        r_idx = np.random.randint(np_pop)
        while r_idx == i:
            r_idx = np.random.randint(np_pop)
        secondary = population[r_idx]
        
        # Combine with bounded perturbation
        alpha = 0.7 + 0.3 * np.random.rand()
        mutants[i] = alpha * primary + (1 - alpha) * secondary
        
        # Add bounded random perturbation for diversity
        perturbation = np.random.randn(dim) * 0.1 * (self.upper_bound - self.lower_bound)
        mutants[i] = mutants[i] + perturbation
    
    # Clip to bounds
    mutants = np.clip(mutants, self.lower_bound, self.upper_bound)
    
    return mutants
```