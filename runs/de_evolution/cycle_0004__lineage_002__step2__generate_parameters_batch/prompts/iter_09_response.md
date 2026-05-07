Looking at the unsolved tasks, I notice a clear pattern:

1. **Tasks 16, 20, 19, 23, 21, 17, 18, 14** have errors stuck at ~1-60, suggesting the algorithm is trapped in local optima on highly multimodal/non-separable functions. These need much more aggressive exploration with large F values and high CR to escape basins.

2. **Tasks 6, 7, 12, 3, 9** have errors at ~1e-8 to 1e-4, suggesting the algorithm gets close but lacks fine-tuning precision. These need very small F values for local refinement.

The key insight: existing variants use moderate F (~0.5 center) with small Cauchy perturbation (scale 0.1). For the worst tasks, we need a **bimodal strategy** — either very aggressive exploration (F near 0.8-1.0, CR near 1.0) to escape deep local optima, OR very fine local search (F near 0.01-0.1) for refinement. I'll use a generation-adaptive approach where early generations use extreme exploration parameters and later generations switch to fine-tuning, with heavy-tailed distributions for both modes.

This is fundamentally different because it ignores the memory entirely for a large fraction of individuals, using fixed extreme parameter regimes instead of adaptive learned values. The memory-based approach converges to moderate parameters that get stuck.

**Idea: Bimodal Extreme Parameters**
Split population into aggressive explorers (F~0.9, CR~1.0) and fine refiners (F~0.05, CR~0.1), ignoring memory for most individuals to escape local optima.

```python
def _generate_parameters_batch(self, memory_f, memory_cr):
    """Generate bimodal F and CR: aggressive exploration + fine local search."""
    n = self.np_size
    f_values = np.zeros(n)
    cr_values = np.zeros(n)
    
    # Determine generation phase from stagnation counter
    stag = getattr(self, 'stagnation_counter', 0)
    gen = getattr(self, 'generation', 0)
    
    # Three subpopulations with different roles
    # 60% aggressive explorers, 20% memory-guided, 20% fine refiners
    explore_frac = 0.6
    memory_frac = 0.2
    # rest is refiners
    
    n_explore = int(n * explore_frac)
    n_memory = int(n * memory_frac)
    n_refine = n - n_explore - n_memory
    
    # === Aggressive explorers: large F, high CR ===
    # Use log-normal distribution centered at high F for heavy tails
    f_explore = np.clip(0.7 + 0.3 * np.abs(np.random.standard_cauchy(n_explore)), 0.4, 1.5)
    f_explore = np.minimum(f_explore, 1.0)
    cr_explore = np.clip(np.random.normal(0.95, 0.05, n_explore), 0.7, 1.0)
    
    # === Memory-guided (standard SHADE-like) ===
    indices = np.random.randint(0, self.memory_size, size=n_memory)
    f_memory = np.zeros(n_memory)
    for i in range(n_memory):
        while True:
            f_val = memory_f[indices[i]] + 0.1 * np.random.standard_cauchy()
            if f_val > 0:
                break
        f_memory[i] = min(f_val, 1.0)
    cr_memory = np.clip(np.random.normal(memory_cr[indices], 0.1), 0.0, 1.0)
    
    # === Fine refiners: very small F, low CR for coordinate-wise refinement ===
    f_refine = np.clip(np.abs(np.random.normal(0.05, 0.03, n_refine)), 0.001, 0.2)
    cr_refine = np.clip(np.random.normal(0.1, 0.1, n_refine), 0.0, 0.3)
    
    # If stagnating hard, shift more toward exploration with even larger F
    if stag > 20:
        f_explore = np.clip(0.9 + 0.2 * np.abs(np.random.standard_cauchy(n_explore)), 0.7, 1.0)
        cr_explore = np.ones(n_explore) * 1.0  # full crossover
    
    # Shuffle assignments randomly
    perm = np.random.permutation(n)
    idx_explore = perm[:n_explore]
    idx_memory = perm[n_explore:n_explore + n_memory]
    idx_refine = perm[n_explore + n_memory:]
    
    f_values[idx_explore] = f_explore
    f_values[idx_memory] = f_memory
    f_values[idx_refine] = f_refine
    
    cr_values[idx_explore] = cr_explore
    cr_values[idx_memory] = cr_memory
    cr_values[idx_refine] = cr_refine
    
    return f_values, cr_values
```