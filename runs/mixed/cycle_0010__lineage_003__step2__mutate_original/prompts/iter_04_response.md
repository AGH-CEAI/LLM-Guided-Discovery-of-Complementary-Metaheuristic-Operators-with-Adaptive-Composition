**Idea: Distance-Scaled Random Walk with Fitness-Weighted Explorative Mutation**

All existing variants (original, JADE, global-best, multi-donor) use some form of best/centroid-directed mutation, which causes the worst tasks (errors 10^1-10^3) to get trapped in local basins. This variant uses a fundamentally different mechanism: it generates donors by blending random population members with distance-scaled perturbations, completely abandoning the best-directed paradigm that is failing on these stuck tasks.

```python
def _mutate_original(self, population, fitness):
    """Distance-scaled random walk mutation for escaping local basins"""
    # Select two random distinct indices for each individual
    idx_a = np.random.randint(0, self.NP, (self.NP, 2))
    idx_a[:, 1] = np.where(idx_a[:, 1] == np.arange(self.NP),
                           (idx_a[:, 1] + 1) % self.NP, idx_a[:, 1])
    r1, r2 = idx_a[:, 0], idx_a[:, 1]
    
    # Random donor: sample uniformly from entire search space (not from population)
    random_donor = np.random.uniform(self.lower, self.upper, (self.NP, self.dim))
    
    # Distance-based scaling: individuals far from centroid get more exploration
    centroid = np.mean(population, axis=0)
    distances = np.linalg.norm(population - centroid, axis=1) + 1e-10
    max_dist = np.max(distances) + 1e-10
    dist_factors = distances / max_dist  # [0, 1] scale
    
    # Adaptive F: higher for individuals far from centroid (exploration)
    F_explore = np.clip(self.F * (0.5 + 1.5 * dist_factors), 0.2, 2.0)
    
    # Base mutation: blend random donor with population using distance-scaled F
    base = population + F_explore[:, np.newaxis] * (random_donor - population)
    
    # Add random walk perturbation scaled by distance from centroid
    if self.stagnation_count > 3:
        # When stagnant: amplify exploration based on how converged population is
        convergence_factor = 1.0 - (np.mean(distances) / (max_dist + 1e-10))
        walk_scale = 0.5 * convergence_factor * self.stagnation_count
        walk_scale = min(walk_scale, 5.0)
        random_walk = np.random.randn(self.NP, self.dim) * walk_scale
        base = base + random_walk
    
    # Fitness-weighted selection pressure: blend base with random donor
    # Lower fitness (better) individuals get less random influence
    fitness_norm = (fitness - np.min(fitness) + 1e-10) / (np.ptp(fitness) + 1e-10)
    selection_pressure = np.clip(fitness_norm, 0.1, 1.0)[:, np.newaxis]
    
    # Better individuals follow base, worse individuals explore more randomly
    donors = selection_pressure * base + (1 - selection_pressure) * random_donor
    
    return np.clip(donors, self.lower, self.upper)
```