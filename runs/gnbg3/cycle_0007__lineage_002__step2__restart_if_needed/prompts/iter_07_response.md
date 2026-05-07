**Idea: Bootstrap CI Diversity Restart with LHS Reinitialization**

Use bootstrap resampling to estimate confidence intervals on population diversity, triggering restart only when the CI is statistically tight (low variance in diversity estimate). Reinitialize using Latin Hypercube Sampling to maintain space-filling properties while preserving the best solution.

```python
def _restart_if_needed(self, population, fitness):
    """Stochastic restart using bootstrap confidence intervals and LHS reinjection.
    Category G: Monte-Carlo bootstrap + Latin Hypercube Sampling.
    """
    NP, dim = population.shape
    
    # Bootstrap resampling to estimate confidence interval on diversity
    n_bootstrap = 50
    bootstrap_diversities = np.zeros(n_bootstrap)
    
    for b in range(n_bootstrap):
        # Monte Carlo resampling with replacement
        indices = np.random.randint(0, NP, size=NP)
        boot_pop = population[indices]
        boot_centroid = boot_pop.mean(axis=0)
        boot_distances = np.linalg.norm(boot_pop - boot_centroid, axis=1)
        bootstrap_diversities[b] = boot_distances.mean()
    
    # Compute bootstrap confidence interval (90%)
    ci_lower = np.percentile(bootstrap_diversities, 5)
    ci_upper = np.percentile(bootstrap_diversities, 95)
    ci_width = ci_upper - ci_lower
    
    # Diversity threshold
    diversity_threshold = 1.0
    
    # Trigger restart if: low diversity AND tight CI (statistically confident)
    if ci_lower < diversity_threshold and ci_width < 0.5:
        # Latin Hypercube Sampling for space-filling reinitialization
        # Stratified sampling ensures good coverage of the search space
        segments = NP
        lhs_matrix = np.zeros((segments, dim))
        
        for d in range(dim):
            # Generate stratified random samples per dimension
            segment_boundaries = np.linspace(-100.0, 100.0, segments + 1)
            lower = segment_boundaries[:-1]
            upper = segment_boundaries[1:]
            offsets = np.random.rand(segments)
            lhs_matrix[:, d] = lower + offsets * (upper - lower)
        
        # Shuffle columns independently (LHS property)
        for d in range(dim):
            np.random.shuffle(lhs_matrix[:, d])
        
        # Preserve best solution
        best_idx = np.nanargmin(fitness)
        best_fitness = fitness[best_idx]
        best_solution = population[best_idx].copy()
        
        # Replace worst half with LHS samples, keep best
        n_replace = NP // 2
        worst_indices = np.argsort(fitness)[-n_replace:]
        new_pop = population.copy()
        new_pop[worst_indices] = lhs_matrix[worst_indices]
        new_pop[best_idx] = best_solution
        
        # Reset stagnation counter
        self.stagnation_counter = 0
        
        return new_pop, best_fitness, best_solution
    
    return (None, None, None)
```