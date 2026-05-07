Looking at the task, I need to design an information-theoretic crossover that treats the population as a probability distribution. The key insight is to use **entropy** of each dimension to weight the crossover probability — dimensions with higher entropy (more uncertainty) should be more explorative, accepting the mutant more often.

**Idea: Entropy-Weighted Distributional Crossover**

This variant fits a Gaussian model to the population and computes per-dimension entropy from variance. Dimensions with higher entropy contain more information/uncertainty, so they receive higher crossover probability to enable more exploration. This is fundamentally different from geometric (distances) or spectral (eigenvalues) approaches.

```python
def _crossover_batch(self, population, mutants):
    """
    Information-theoretic crossover using entropy-weighted probability.
    
    Treats the population as a probability distribution. Computes per-dimension
    entropy from fitted Gaussian model. Dimensions with higher entropy (more
    uncertainty) get higher crossover probability for exploration; dimensions
    with lower entropy favor exploitation by keeping original values.
    
    Category C: Information-theoretic / distributional.
    """
    NP, dim = population.shape
    
    # Estimate per-dimension variance (zeroth-order distribution fitting)
    var = np.var(population, axis=0)
    var = np.maximum(var, 1e-10)
    
    # Compute Gaussian entropy: H = 0.5 * log(2*pi*e*sigma^2)
    # Higher variance -> higher entropy -> more uncertainty -> more exploration
    ent = 0.5 * np.log(2 * np.pi * np.e * var)
    ent = np.maximum(ent, 1e-8)
    
    # Normalize entropy to get dimension importance weights
    total_ent = ent.sum()
    if total_ent > 0:
        ent_weight = ent / total_ent
    else:
        ent_weight = np.ones(dim) / dim
    
    # Entropy-weighted crossover rate per dimension
    # Scale base CR by entropy weight to boost exploration in uncertain dimensions
    # Base CR ~0.85, scaled range roughly [0.01, 0.99] depending on entropy
    ent_scaled = 0.1 + 5.0 * ent_weight
    CR_per_dim = np.clip(self.CR * ent_scaled, 0.01, 0.99)
    
    # Binomial crossover with entropy-modulated probability
    # Each dimension has its own CR based on how much information it carries
    rand_mask = np.random.rand(NP, dim)
    crossover_mask = rand_mask < CR_per_dim[np.newaxis, :]
    
    # Create trial population: crossover between parent and mutant
    trials = np.where(crossover_mask, mutants, population)
    
    # Clip to bounds for numerical robustness
    trials = np.clip(trials, -100.0, 100.0)
    
    return trials
```