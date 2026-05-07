**Idea: Spread-Adaptive Geometric Crossover**

Crossover probability per dimension is determined by the axis-aligned spread of the population in that dimension. Dimensions with larger spread (indicating exploration potential) receive the mutant more often; dimensions with tight clustering preserve the target. This is purely geometric (A), using spatial variance to self-adapt the recombination mask without any fitness-based reasoning.

```python
def _crossover_batch(self, population, mutants):
    """Geometry-driven crossover: adapt CR per dimension by axis-aligned spread."""
    NP, dim = population.shape
    
    # Compute axis-aligned spread (range) per dimension — purely geometric
    dim_min = population.min(axis=0)
    dim_max = population.max(axis=0)
    spread = dim_max - dim_min
    
    # Prevent division by zero: dimensions with zero spread get minimal mutation
    spread = np.where(spread < 1e-10, 1e-10, spread)
    
    # Normalize to get per-dimension crossover probability
    # Higher spread → higher probability of taking from mutant (more exploration)
    cr_per_dim = spread / spread.sum()
    
    # Generate crossover masks: each dimension has independent probability
    rand_matrix = np.random.rand(NP, dim)
    
    # CR_i for dimension j = cr_per_dim[j]
    # If rand < CR_j, take from mutant; otherwise keep target
    mask = rand_matrix < cr_per_dim[np.newaxis, :]
    
    # Ensure at least one dimension comes from mutant (classic DE guarantee)
    no_mutant_dims = ~mask.any(axis=1)
    forced_dims = np.random.randint(0, dim, size=no_mutant_dims.sum())
    mask[no_mutant_dims, forced_dims] = True
    
    # Build trial vectors
    trials = np.where(mask, mutants, population)
    
    return trials
```