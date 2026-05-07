**Idea: Convex Hull Boundary Re-seeding**
Use convex hull vertex analysis and k-NN isolation metrics to detect population collapse, then re-seed from the hull boundary (geometric extremes) plus elite perturbation for diversity.

```python
def _restart_if_needed(self, population, fitness):
    """Restart based on convex hull volume and k-NN isolation analysis."""
    NP, dim = population.shape
    
    # Compute axis-aligned bounding box
    mins = population.min(axis=0)
    maxs = population.max(axis=0)
    spans = maxs - mins
    total_span = spans.sum()
    
    # Small span indicates collapse
    if total_span < 1e-6 * dim:
        return (None, None, None)
    
    # Compute k-NN isolation: average distance to k nearest neighbors
    k = max(2, min(5, NP // 5))
    diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    sorted_dists = np.sort(sq_dists, axis=1)
    knn_dist = np.mean(sorted_dists[:, :k], axis=1)
    
    # Points with low k-NN distance are trapped in dense clusters
    isolation_threshold = np.percentile(knn_dist, 25)
    isolated_mask = knn_dist <= isolation_threshold
    
    if not isolated_mask.any():
        return (None, None, None)
    
    # Number of replacements based on isolation extent
    n_replace = max(int(isolated_mask.sum()), min(NP // 4, 10))
    
    # Find indices to replace (lowest isolation = most trapped)
    isolated_indices = np.argsort(knn_dist)[:n_replace]
    
    # Generate replacement candidates via convex hull boundary sampling
    # Sample from convex hull vertices + edge midpoints for geometric diversity
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(population)
        hull_vertices = population[hull.vertices]
        
        # Sample boundary points: convex hull vertices + edge interpolations
        boundary_points = [hull_vertices]
        n_edges = len(hull.vertices)
        for i in range(n_edges):
            v1 = hull_vertices[i]
            v2 = hull_vertices[(i + 1) % n_edges]
            # Sample along edges
            for alpha in [0.25, 0.5, 0.75]:
                boundary_points.append(v1 * alpha + v2 * (1 - alpha))
        boundary_points = np.array(boundary_points)
    except Exception:
        # Fallback: use centroid-perturbed population
        centroid = population.mean(axis=0)
        boundary_points = centroid + (population - centroid) * 1.5
    
    # Also generate elite-perturbed candidates for exploitation
    top_indices = np.argsort(fitness)[:min(3, NP)]
    elite_perturbations = []
    for idx in top_indices:
        for _ in range(3):
            noise_scale = max(5.0, total_span / (dim * 2))
            elite_perturbations.append(
                population[idx] + np.random.randn(dim) * noise_scale
            )
    elite_perturbations = np.array(elite_perturbations)
    
    # Combine boundary and elite candidates
    candidates = np.vstack([boundary_points, elite_perturbations])
    
    # Select diverse replacements from candidates
    n_candidates = len(candidates)
    if n_candidates >= n_replace:
        # Use farthest points from centroid for diversity
        candidate_centroid = candidates.mean(axis=0)
        candidate_dists = np.linalg.norm(candidates - candidate_centroid, axis=1)
        selected_indices = np.argsort(candidate_dists)[-n_replace:]
    else:
        selected_indices = np.arange(n_candidates)
    
    # Create new population
    new_population = population.copy()
    new_fitness = fitness.copy()
    
    for i, cand_idx in enumerate(selected_indices):
        if i < len(isolated_indices):
            replace_idx = isolated_indices[i]
            new_point = np.clip(candidates[cand_idx], -100.0, 100.0)
            new_population[replace_idx] = new_point
            new_fitness[replace_idx] = np.inf
    
    # Update best solution
    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_solution = population[best_idx].copy()
    
    return (new_population, best_fitness, best_solution)
```