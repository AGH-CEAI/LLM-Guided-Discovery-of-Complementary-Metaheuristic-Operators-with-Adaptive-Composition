**Idea: Eigenvalue-Spread Modulated Selection**

Category B: Spectral / linear-algebraic — Use eigenvalue spread of population covariance to modulate selection pressure. When the covariance matrix has high condition number (population collapsed to subspace), reduce selection pressure to preserve diversity.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Greedy selection with spectral diversity modulation."""
        improved_mask = trial_fitness < fitness
        improvement = fitness - trial_fitness
        
        # Compute population covariance eigenvalue spread
        centered = population - np.mean(population, axis=0)
        cov = np.cov(centered.T)
        
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        
        # Eigenvalue spread: ratio of largest to smallest non-zero eigenvalue
        # High spread → anisotropic population → reduce selection pressure
        if eigvals[0] > 1e-12 and eigvals[-1] > 1e-12:
            condition = eigvals[0] / eigvals[-1]
        else:
            condition = 1.0
        
        # Modulation: in ill-conditioned populations, accept smaller improvements
        # log(condition) maps: 1→0, 10→2.3, 100→4.6, 1000→6.9
        diversity_bonus = np.log1p(condition)
        
        # Scaled improvement relative to population fitness range
        fit_range = np.ptp(fitness)
        fit_range = max(fit_range, 1e-10)
        scaled_improvement = improvement / fit_range
        
        # Accept if improved OR if scaled improvement exceeds diversity threshold
        accept_threshold = 0.01 / (1.0 + 0.5 * diversity_bonus)
        accept_mask = improved_mask | (scaled_improvement > accept_threshold)
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        new_population[accept_mask] = trials[accept_mask]
        new_fitness[accept_mask] = trial_fitness[accept_mask]
        
        return new_population, new_fitness, accept_mask
```