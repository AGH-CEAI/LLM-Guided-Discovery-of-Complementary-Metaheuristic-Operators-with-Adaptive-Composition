Looking at the problem, I need to design a mutation strategy that specifically targets the worst unsolved tasks (16, 17, 19, 23, 6, 11, 20, 18) which have errors in the 10s-500s range. These massive errors suggest the algorithm is getting stuck in local optima and failing to explore effectively.

**Analysis of failure modes:**
- The worst tasks (error ~10²-10³) are likely highly multi-modal or deceptive
- The current operators use only the current population for mutation guidance
- No archive mechanism is leveraged for escaping local optima
- All existing variants use similar "perturb around best/pbest" approaches

**New approach: Archive-Guided Composite Mutation with Adaptive Scaling**

The key insight is that an archive of historically best solutions can guide mutation toward unexplored promising regions, breaking out of local optima traps. I'll also use multiple scaling factors to balance exploration vs exploitation.

**Idea: Archive-Guided Multi-Scale DE**

```python
def _mutate_batch(self, population, fitness, selected_operator):
    """Generate mutant vectors using archive-guided multi-scale mutation."""
    np_pop = len(population)
    dim = population.shape[1]
    
    # Build ring topology indices
    indices = np.arange(np_pop)
    ring_prev = np.roll(indices, 1)
    ring_next = np.roll(indices, -1)
    
    # Get sorted indices by fitness for pbest selection
    sorted_idx = np.argsort(fitness)
    pbest_idx = sorted_idx[:max(1, int(np_pop * 0.1))]  # Top 10%
    
    # Initialize mutant population
    mutants = np.empty_like(population)
    
    # Maintain archive of best solutions encountered
    if not hasattr(self, 'mutation_archive') or self.mutation_archive is None:
        self.mutation_archive = population[sorted_idx[:min(10, np_pop)]].copy()
    else:
        # Merge current best solutions into archive
        combined = np.vstack([self.mutation_archive, population[sorted_idx[:5]]])
        combined_fitness = np.array(
            list(np.zeros(len(self.mutation_archive)) * 0.1) + 
            list(fitness[sorted_idx[:5]])
        )
        top_indices = np.argsort(combined_fitness)[:min(15, len(combined))]
        self.mutation_archive = combined[top_indices]
    
    # Sample indices for mutation
    r1 = np.array([np.delete(np.arange(np_pop), i) for i in range(np_pop)])
    r1 = np.array([np.random.choice(r1[i]) for i in range(np_pop)])
    r2 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i]])) 
                   for i in range(np_pop)])
    
    # Archive indices
    archive_size = len(self.mutation_archive)
    arch_idx = np.random.randint(0, archive_size, size=np_pop)
    
    # Multi-scale F: small, medium, large for different exploration phases
    # Use different scales based on a rotating pattern and adaptation
    if not hasattr(self, 'mutation_scale_mode'):
        self.mutation_scale_mode = 0
    self.mutation_scale_mode = (self.mutation_scale_mode + 1) % 3
    
    if self.mutation_scale_mode == 0:
        f_values = self.f_base * np.ones(np_pop) * 0.4  # Small scale
    elif self.mutation_scale_mode == 1:
        f_values = self.f_base * np.ones(np_pop) * 0.8  # Medium scale
    else:
        f_values = self.f_base * np.ones(np_pop) * 1.5  # Large scale
    
    # Add individual variation
    f_values = f_values + 0.1 * np.random.randn(np_pop)
    f_values = np.clip(f_values, 0.1, 2.0)
    
    # Apply archive-guided mutation strategy
    if selected_operator == 0:  # Archive-rand/1
        mutants = (self.mutation_archive[arch_idx] + f_values[:, np.newaxis] * 
                  (population[r1] - population[r2]))
    elif selected_operator == 1:  # Archive-to-best/1
        best_idx = np.argmin(fitness)
        mutants = (self.mutation_archive[arch_idx] + f_values[:, np.newaxis] * 
                  (population[best_idx] - self.mutation_archive[arch_idx] + 
                   population[r1] - population[r2]))
    elif selected_operator == 2:  # Current-to-archive-pbest/1
        # Use archive member as pbest
        arch_pbest = self.mutation_archive[arch_idx]
        mutants = (population + f_values[:, np.newaxis] * 
                  (arch_pbest - population[r1] + population[r1] - population[r2]))
    else:  # Archive-rand/2
        arch_idx2 = np.random.randint(0, archive_size, size=np_pop)
        mutants = (self.mutation_archive[arch_idx] + f_values[:, np.newaxis] * 
                  (population[r1] - population[r2] + 
                   population[r2] - self.mutation_archive[arch_idx2]))
    
    # Clip to bounds
    mutants = np.clip(mutants, self.lower_bound, self.upper_bound)
    
    return mutants
```