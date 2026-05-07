**Idea: Entropy-Weighted Gaussian Crossover**

Use per-dimension entropy of the population distribution (modeled as Gaussian) to probabilistically select which dimensions inherit from the mutant versus the parent. Dimensions with higher variance/entropy receive more mutant influence, enabling exploration of high-uncertainty regions.

```python
def _crossover_batch(self, population, mutants):
    """Information-theoretic crossover using per-dimension entropy from fitted Gaussian.
    
    Treats the population as a probability distribution, computes entropy per dimension,
    and uses entropy-weighted probabilities to decide whether each dimension inherits
    from the mutant or the parent. High-entropy dimensions favor mutant exploration.
    """
    NP, dim = self.dim if hasattr(self, 'dim') else population.shape
    bounds_low, bounds_high = -100.0, 100.0
    
    # Fit Gaussian to population and compute per-dimension entropy
    pop_mean = np.mean(population, axis=0)
    pop_std = np.std(population, axis=0)
    
    # Prevent division by zero: add small epsilon to zero std
    eps = 1e-10
    pop_std = np.maximum(pop_std, eps)
    
    # Entropy of Gaussian: H = 0.5 * ln(2πe * σ²)
    # Higher variance → higher entropy → more uncertainty → favor mutant exploration
    entropies = 0.5 * np.log(2 * np.pi * np.e * pop_std**2)
    
    # Normalize entropy to get per-dimension crossover probability
    # Dimensions with higher entropy (more spread) get higher CR
    total_entropy = np.sum(entropies)
    if total_entropy > eps:
        entropy_weights = entropies / total_entropy
    else:
        entropy_weights = np.ones(dim) / dim
    
    # Entropy-based CR: base CR modified by entropy contribution
    # Use softmax-like scaling to emphasize high-entropy dimensions
    entropy_CR = self.CR * (1.0 + 0.5 * (entropy_weights / entropy_weights.max()))
    entropy_CR = np.clip(entropy_CR, 0.0, 1.0)
    
    # Initialize trial population as copy of parents
    trials = population.copy()
    
    # Per-dimension crossover based on entropy-weighted probability
    for d in range(dim):
        # Bernoulli trial with entropy-influenced probability
        if np.random.random() < entropy_CR[d]:
            trials[:, d] = mutants[:, d]
    
    # Alternative: dimension-wise sampling where each dimension uses
    # entropy as the probability of inheriting from mutant
    # This creates a distribution-aware crossover mask
    crossover_mask = np.random.random((NP, dim)) < entropy_CR[np.newaxis, :]
    trials = np.where(crossover_mask, mutants, population)
    
    # Clip to bounds for numerical stability
    trials = np.clip(trials, bounds_low, bounds_high)
    
    return trials
```