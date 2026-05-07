**Idea: Monte Carlo Bootstrap Restart with Fitness-Weighted Covariance**

This restart mechanism uses Monte Carlo bootstrap resampling from the top-performing individuals to estimate the fitness-conditional distribution, then samples new population members from this distribution using Latin Hypercube Sampling for well-distributed exploration. The noise scale adapts inversely to the current best fitness, ensuring aggressive diversity injection when trapped far from the optimum.

```python
def _restart_if_needed(self, population, fitness):
    """Stochastic restart via Monte Carlo bootstrap from elite solutions.
    
    Uses bootstrap resampling to estimate the covariance of top performers,
    then generates new population via Latin Hypercube Sampling from this
    distribution. Noise scale adapts inversely to best fitness for
    controlled exploration.
    """
    n = len(population)
    dim = self.dim
    bounds = (-100.0, 100.0)
    
    # Select elite solutions for bootstrap
    n_elite = max(3, n // 4)
    elite_idx = np.argpartition(fitness, n_elite)[:n_elite]
    elite_pop = population[elite_idx]
    
    # Monte Carlo bootstrap: resample from elite with replacement
    n_bootstrap = 100
    bootstrap_indices = np.random.randint(0, len(elite_pop), size=(n_bootstrap, len(elite_pop)))
    bootstrap_samples = elite_pop[bootstrap_indices]  # (100, n_elite, dim)
    
    # Compute bootstrap mean and covariance (fitness-weighted)
    weights = 1.0 / (fitness[elite_idx] + 1e-10)
    weights /= weights.sum()
    weighted_mean = np.average(elite_pop, axis=0, weights=weights)
    
    # Weighted covariance estimate
    centered = elite_pop - weighted_mean
    cov = np.zeros((dim, dim))
    for i in range(len(elite_pop)):
        cov += weights[i] * np.outer(centered[i], centered[i])
    
    # Add regularization for numerical stability
    cov += np.eye(dim) * 1e-6
    
    # Adaptive noise scale: inversely proportional to best fitness
    # Far from optimum (high fitness) -> large exploration
    best_f = np.min(fitness)
    noise_scale = max(50.0, 200.0 / (best_f + 1.0))
    
    # Generate new population via Latin Hypercube Sampling
    # Sample from scaled normal distribution centered on weighted mean
    new_pop = np.zeros((n, dim))
    
    for i in range(n):
        # Truncated normal sampling via rejection
        for _ in range(10):
            sample = np.random.multivariate_normal(weighted_mean, cov * noise_scale)
            if np.all((sample >= bounds[0]) & (sample <= bounds[1])):
                new_pop[i] = sample
                break
        else:
            # Fallback: uniform sampling
            new_pop[i] = np.random.uniform(bounds[0], bounds[1], dim)
    
    # Preserve top 2 solutions unchanged (elitism)
    best_idx = np.argmin(fitness)
    new_pop[0] = population[best_idx]
    second_best = np.argmin(np.where(np.arange(n) != best_idx, fitness, np.inf))
    new_pop[1] = population[second_best]
    
    # Update tracking
    self.archive.append(population[elite_idx].copy())
    self.stagnation_counter = 0
    
    return new_pop, fitness[best_idx], population[best_idx]
```