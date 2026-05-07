Looking at the unsolved tasks, the worst ones (Tasks 16, 23, 20, 19, 21) have errors in the range 1e+1 to 1e+2, suggesting the algorithm is getting trapped in local optima on highly multimodal, non-separable functions. The existing variants use standard binomial crossover which inherits dimensions independently — this is poor for non-separable problems where variables are correlated. Tasks closer to solving (6, 7, 12) need finer exploitation.

My key insight: use **exponential crossover** with an adaptive segment length that creates contiguous blocks of donated dimensions. This preserves rotational/correlational structure in the mutant vector much better than binomial crossover for non-separable problems. Additionally, I'll add a rotationally-invariant perturbation mechanism where, for a fraction of individuals, the entire mutant vector is accepted (CR=1.0 effectively), which helps on rotated functions where partial crossover destroys good search directions.

This is fundamentally different from prior variants (which were binomial crossover variants) because exponential crossover creates correlated blocks rather than independent per-dimension decisions, and the full-vector acceptance mechanism preserves the geometric structure of the mutation step.

**Idea: Exponential Crossover with Adaptive Block Length and Full-Vector Acceptance**
Exponential (contiguous-block) crossover that preserves correlations in non-separable functions, with probabilistic full-vector acceptance for rotated problems.

```python
def _crossover_batch(self, population, mutants, cr_values):
    """Exponential crossover with adaptive block length and full-vector acceptance."""
    n, dim = population.shape
    trials = population.copy()
    
    for i in range(n):
        cr = cr_values[i]
        
        # With 30% probability, accept the full mutant vector
        # This preserves rotational structure for non-separable functions
        if np.random.random() < 0.3:
            trials[i] = mutants[i]
            continue
        
        # Exponential crossover: contiguous block of dimensions from mutant
        # Starting position is random
        start = np.random.randint(0, dim)
        
        # The block length follows geometric distribution controlled by CR
        # Higher CR -> longer blocks -> more from mutant
        # For non-separable problems, contiguous blocks preserve correlations
        L = 1  # At least one dimension from mutant
        pos = (start + L) % dim
        
        while np.random.random() < cr and L < dim:
            L += 1
            pos = (start + L) % dim
        
        # Apply the contiguous block
        if L >= dim:
            trials[i] = mutants[i]
        else:
            indices = [(start + j) % dim for j in range(L)]
            trials[i, indices] = mutants[i, indices]
    
    return trials
```