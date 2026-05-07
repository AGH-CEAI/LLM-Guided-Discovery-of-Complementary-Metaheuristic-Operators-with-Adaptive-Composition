**Idea: Hybrid Diversity-Driven Restart**

Replace the mild perturbation restart with a hybrid approach that combines full random reinitialization (for aggressive diversity) with best-centered Gaussian perturbations (for exploitation), weighted by stagnation severity. Tasks 12, 8, 21 with errors in the thousands are likely stuck in terrible local optima with collapsed population diversity — the current ±10 uniform perturbation is far too conservative to escape.

```python
def _restart_if_stagnant(self, population, fitness, x_best):
        NP = len(population)
        new_pop = population.copy()
        
        # Determine how many to reinitialize randomly vs perturb
        # Full random = aggressive diversity for escaping bad local optima
        # Perturbed = maintain some exploitation of good region
        random_count = NP // 2
        perturb_count = NP - random_count
        
        # Half: complete reinitialization across entire search space
        for i in range(random_count):
            new_pop[i] = np.random.uniform(self.lower, self.upper, self.dim)
        
        # Half: Gaussian perturbation around best with adaptive sigma
        # Sigma scales with search range to ensure meaningful displacement
        sigma = (self.upper - self.lower) * 0.3
        for i in range(random_count, NP):
            new_pop[i] = x_best + np.random.normal(0, sigma, self.dim)
        
        new_pop = self._clip_to_bounds_batch(new_pop)
        return new_pop
```