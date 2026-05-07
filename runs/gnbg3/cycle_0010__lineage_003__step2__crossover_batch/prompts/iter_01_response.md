**Idea: Centroid-Proximity Weighted Crossover**
Use the geometric distance of each parent from the population centroid to weight crossover contributions per dimension, biasing toward spatial regions where the population is densest.

```python
def _crossover_batch(self, population, mutants):
    """Geometry/spatial crossover: weight by distance to population centroid."""
    NP, dim = population.shape
    
    # Compute population centroid (geometric center of search space)
    centroid = population.mean(axis=0)
    
    # Compute per-parent centroid distances (A: literal geometric layout)
    target_dists = np.linalg.norm(population - centroid, axis=1)
    mutant_dists = np.linalg.norm(mutants - centroid, axis=1)
    
    # Convert distances to proximity weights (closer = higher weight)
    eps = 1e-10
    target_weights = 1.0 / (target_dists + eps)
    mutant_weights = 1.0 / (mutant_dists + eps)
    
    # Normalize per pair
    total_weights = target_weights + mutant_weights
    target_weights /= total_weights
    mutant_weights /= total_weights
    
    # Per-dimension crossover mask (standard binomial)
    CR = getattr(self, 'CR', 0.85)
    j_rand = np.random.randint(dim)
    rand_mask = np.random.rand(NP, dim) < CR
    rand_mask[:, j_rand] = True
    
    # Base trial from binomial crossover
    trials = np.where(rand_mask, mutants, population)
    
    # Apply centroid-proximity weighting: shift toward spatially closer parent
    # For each individual, blend toward the parent nearer the centroid
    for i in range(NP):
        if target_weights[i] > mutant_weights[i]:
            # Target is closer to centroid: bias toward it
            blend = 0.25 * mutant_weights[i]
            trials[i] = trials[i] * (1 - blend) + population[i] * blend
        else:
            # Mutant is closer to centroid: bias toward it
            blend = 0.25 * target_weights[i]
            trials[i] = trials[i] * (1 - blend) + mutants[i] * blend
    
    return trials
```