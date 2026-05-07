**Idea: Convex-Hull Boundary + Centroid-Relative PCA-Directed Mutation**

Uses convex hull vertices to identify extreme individuals, computes centroid-relative directional vectors, scales mutations by axis-aligned population spread, and selects the best candidate by projection onto the first principal component. This is geometrically distinct from prior approaches that used fitness-based direction scoring.

```python
def _mutate_compass_batch(self, population, fitness):
    """Geometry-driven compass mutation using convex hull, centroid, and PCA."""
    NP, dim = population.shape
    fitness_ranks = self._compute_fitness_ranking(fitness)
    
    # Compute convex hull vertices (boundary individuals)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(population)
        hull_vertices = hull.vertices
    except Exception:
        # Fallback: select individuals farthest from centroid
        centroid = population.mean(axis=0)
        dists = np.linalg.norm(population - centroid, axis=1)
        n_boundary = max(3, NP // 4)
        hull_vertices = np.argsort(dists)[-n_boundary:]
    
    # Compute population centroid
    centroid = population.mean(axis=0)
    
    # Compute axis-aligned spread for scaling
    pop_min = population.min(axis=0)
    pop_max = population.max(axis=0)
    spread = np.maximum(pop_max - pop_min, 1e-8)
    
    # PCA: compute principal direction
    centered = population - centroid
    cov = centered.T @ centered / NP
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    except Exception:
        principal_axis = np.ones(dim) / np.sqrt(dim)
    
    # Generate 3 geometric mutation candidates
    candidates = []
    
    # Candidate A: Move toward boundary vertex (exploitation of extremes)
    if len(hull_vertices) > 0:
        boundary_idx = np.random.choice(hull_vertices)
        dir_A = population[boundary_idx] - population  # Shape: (NP, dim)
        # Score by alignment with principal axis (positive = along major variance)
        proj_A = np.dot(dir_A, principal_axis) / (np.linalg.norm(dir_A, axis=1) + 1e-8)
        score_A = proj_A
    else:
        dir_A = np.zeros((NP, dim))
        score_A = np.zeros(NP)
    
    # Candidate B: Move from centroid toward each individual (centroid-relative)
    dir_B = population - centroid  # Each individual relative to centroid
    proj_B = np.dot(dir_B, principal_axis) / (np.linalg.norm(dir_B, axis=1) + 1e-8)
    score_B = proj_B
    
    # Candidate C: Spread-scaled random direction
    # Use random vector scaled by population spread
    random_dir = np.random.randn(NP, dim)
    random_dir = random_dir / (np.linalg.norm(random_dir, axis=1, keepdims=True) + 1e-8)
    dir_C = random_dir * spread  # Scale by axis-aligned spread
    proj_C = np.dot(dir_C, principal_axis) / (np.linalg.norm(dir_C, axis=1) + 1e-8)
    score_C = proj_C
    
    # For each individual, select best candidate direction
    all_dirs = np.stack([dir_A, dir_B, dir_C], axis=1)  # (NP, 3, dim)
    all_scores = np.stack([score_A, score_B, score_C], axis=1)  # (NP, 3)
    
    # Pick best candidate per individual
    best_cand_idx = np.argmax(all_scores, axis=1)  # (NP,)
    
    # Build direction matrix by selecting best candidate for each row
    mutants = np.empty_like(population)
    for i in range(NP):
        mutants[i] = all_dirs[i, best_cand_idx[i]]
    
    # Scale by F and add to current population
    mutants = population + self.F * mutants
    
    return np.clip(mutants, -100.0, 100.0)
```