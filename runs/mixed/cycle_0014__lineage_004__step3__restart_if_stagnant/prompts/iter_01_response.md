**Idea: Random Restart with Elite Preservation**
A fundamentally different diversity injection strategy: completely reinitialize the worst half with new random solutions within bounds, preserving only the current best individual. This provides maximum diversity injection and is a well-known robust baseline for handling stagnation.

```python
def _restart_if_stagnant(self, population, fitness, x_best):
    NP = len(population)
    new_pop = population.copy()
    
    # Preserve the best individual
    best_idx = np.argmin(fitness)
    
    # Replace worst half with completely new random solutions
    n_replace = NP // 2
    worst_indices = np.argpartition(fitness, n_replace)[:n_replace]
    
    for idx in worst_indices:
        new_pop[idx] = np.random.uniform(self.lower, self.upper, self.dim)
    
    # Ensure best individual is retained
    new_pop[best_idx] = x_best.copy()
    
    return new_pop
```