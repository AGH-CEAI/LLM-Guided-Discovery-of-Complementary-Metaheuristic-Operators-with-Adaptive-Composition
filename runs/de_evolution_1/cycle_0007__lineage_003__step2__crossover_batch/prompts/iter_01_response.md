**Idea: Exponential Crossover**
Exponential (two-point modular) crossover that copies a contiguous segment from the mutant, known to work well on separable and block-structured problems.
```python
def _crossover_batch(self, population, mutants, cr_values):
    """Exponential crossover applied to whole population at once."""
    n, dim = population.shape
    trials = population.copy()
    
    for i in range(n):
        # Random starting position
        j = np.random.randint(0, dim)
        L = 0  # number of dimensions taken from mutant
        
        # Copy contiguous segment from mutant while random < CR
        while True:
            trials[i, j] = mutants[i, j]
            L += 1
            j = (j + 1) % dim
            if np.random.random() >= cr_values[i] or L >= dim:
                break
    
    return trials
```