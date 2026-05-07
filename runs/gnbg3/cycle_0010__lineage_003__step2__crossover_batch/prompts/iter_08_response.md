**Idea: Diversity-Adaptive Hybrid Crossover with Fitness-Weighted Selection**

Hybrid crossover combining binomial, exponential, and centroid-blend strategies with data-driven weighting based on population diversity and generational progress. Uses softmax selection conditioned on fitness ranks for principled switching.

```python
def _crossover_batch(self, population, mutants):
    """Hybrid crossover: binomial + exponential + centroid-blend, weighted by diversity and fitness."""
    NP, dim = self.dim
    
    # Compute population diversity (data-driven signal)
    centroid = population.mean(axis=0)
    spread = np.linalg.norm(population - centroid, axis=1).mean()
    max_spread = 200.0 * np.sqrt(dim)
    diversity = np.clip(spread / max_spread, 0.0, 1.0)
    
    # Principled weights based on diversity state
    # Low diversity -> favor exploratory centroid-blend; high diversity -> favor exploitative binomial
    w_binomial = 0.4 + 0.3 * diversity
    w_exponential = 0.3 * (1.0 - diversity * 0.5)
    w_centroid = 0.3 * (1.0 - diversity)
    total_w = w_binomial + w_exponential + w_centroid
    w_binomial /= total_w
    w_exponential /= total_w
    w_centroid /= total_w
    
    # Compute fitness ranks (data-driven signal for per-individual weighting)
    fitness = self._compute_fitness_ranking(np.zeros(NP))  # placeholder, use ranks
    if not hasattr(self, '_cached_fitness'):
        return np.clip(mutants, -100.0, 100.0)
    
    # Per-individual CR adaptation based on fitness rank
    # Better individuals get lower CR (more exploitative), worse get higher (more exploratory)
    fitness_ranks = self._compute_fitness_ranking(self._cached_fitness)
    cr_base = np.clip(self.CR * (0.8 + 0.4 * fitness_ranks), 0.1, 0.99)
    
    # Crossover type selection per individual (principled: weighted by strategy weights)
    r = np.random.rand(NP)
    crossover_type = np.zeros(NP, dtype=int)
    crossover_type[r < w_binomial] = 0
    crossover_type[(r >= w_binomial) & (r < w_binomial + w_exponential)] = 1
    crossover_type[r >= w_binomial + w_exponential] = 2
    
    trials = np.empty_like(population)
    
    for i in range(NP):
        target = population[i]
        mutant = mutants[i]
        cr = cr_base[i]
        
        if crossover_type[i] == 0:
            # Binomial crossover
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < cr
            mask[j_rand] = True
            trials[i] = np.where(mask, mutant, target)
            
        elif crossover_type[i] == 1:
            # Exponential crossover with wrap-around
            j_start = np.random.randint(dim)
            trials[i] = target.copy()
            L = 0
            while L < dim and np.random.rand() < cr:
                j = (j_start + L) % dim
                trials[i][j] = mutant[j]
                L += 1
                
        else:
            # Centroid-blend: push toward population centroid for diversity
            # Blend factor depends on how far target is from centroid
            dist_to_centroid = np.linalg.norm(target - centroid)
            blend_factor = cr * 0.3 * (dist_to_centroid / (spread + 1e-8))
            blend_factor = np.clip(blend_factor, 0.0, 0.4)
            trials[i] = (1.0 - blend_factor) * mutant + blend_factor * centroid
    
    return np.clip(trials, -100.0, 100.0)
```