**Idea: Sobol-Compass with Bootstrap Thompson Sampling**

Monte Carlo compass mutation using Sobol low-discrepancy sequences for direction sampling, bootstrap resampling to estimate direction quality, and Thompson sampling for stochastic selection—fundamentally stochastic in its core computation.
```python
def _mutate_compass_batch(self, population, fitness):
    """Stochastic compass mutation using Sobol-guided sampling and bootstrap confidence estimation."""
    NP, dim = population.shape
    
    fitness_ranks = self._compute_fitness_ranking(fitness)
    mutants = np.empty_like(population)
    
    n_samples = max(12, min(25, dim // 2))
    n_subspaces = max(3, min(6, dim // 8))
    n_bootstrap = 8
    
    for i in range(NP):
        i_fitness = fitness_ranks[i]
        i_point = population[i]
        
        others = np.concatenate([np.arange(i), np.arange(i + 1, self.NP)])
        
        # Sobol-guided direction sampling (low-discrepancy sequence)
        sobol_dirs = self._sobol_sample_directions(dim, n_samples)
        
        # Random subspace projections with bootstrap scoring
        subspace_scores = []
        for s in range(n_subspaces):
            active_dims = np.sort(np.random.choice(dim, max(2, dim // 3), replace=False))
            base_dir = sobol_dirs[s % n_samples].copy()
            projected_dir = np.zeros(dim)
            projected_dir[active_dims] = base_dir[active_dims]
            
            # Bootstrap resampling for confidence estimation
            bootstrap_improvements = []
            for _ in range(n_bootstrap):
                n_refs = min(4, len(others))
                sampled_refs = np.random.choice(others, n_refs, replace=False)
                delta = np.mean([population[r] - population[i] for r in sampled_refs], axis=0)
                candidate = i_point + self.F * (projected_dir + delta * 0.5)
                
                # Rank-based fitness estimation
                approx_rank = np.mean([fitness_ranks[r] for r in sampled_refs])
                improvement = i_fitness - approx_rank
                bootstrap_improvements.append(max(0, improvement))
            
            # Bootstrap mean and variance for confidence weighting
            subspace_mean = np.mean(bootstrap_improvements)
            subspace_var = np.var(bootstrap_improvements) + 1e-8
            confidence = 1.0 / (subspace_var + 1e-6)
            subspace_scores.append(subspace_mean * confidence)
        
        # Thompson sampling: draw from posterior and pick max
        scores_array = np.array(subspace_scores)
        posterior_samples = np.random.gamma(scores_array + 0.5, scale=1.0)
        selected_subspace = np.argmax(posterior_samples)
        
        # Build final direction from selected subspace
        active_dims = np.sort(np.random.choice(dim, max(2, dim // 3), replace=False))
        final_dir = sobol_dirs[selected_subspace % n_samples].copy()
        final_dir[active_dims] = 0.0
        
        # Combine with population-based perturbation
        sampled_refs = np.random.choice(others, min(3, len(others)), replace=False)
        delta = np.mean([population[r] - population[i] for r in sampled_refs], axis=0)
        
        mutants[i] = np.clip(i_point + self.F * (final_dir + delta * 0.3), -100.0, 100.0)
    
    return mutants

def _sobol_sample_directions(self, dim, n_samples):
    """Generate quasi-random directions using Sobol sequence."""
    directions = np.zeros((n_samples, dim))
    for s in range(n_samples):
        # Sobol sequence generation (simplified)
        v = 1 << (s + 1)
        primitive = [0, 1, 1, 2, 1, 4, 2, 4, 7, 14, 13, 11, 14, 16, 8, 4, 7, 20, 14, 16]
        p = primitive[min(s, len(primitive) - 1)]
        c = 1.0 / (1 << (s + 1))
        direction = np.zeros(dim)
        for d in range(min(dim, 16)):
            v ^= (v >> 1) ^ ((p & 1) << (d + 1))
            direction[d] = ((v & ((1 << (d + 1)) - 1)) * c) - 0.5
        # Extend to full dimension with random fill
        if dim > 16:
            direction[16:] = np.random.randn(dim - 16) * 0.3
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction /= norm
        directions[s] = direction
    return directions
```