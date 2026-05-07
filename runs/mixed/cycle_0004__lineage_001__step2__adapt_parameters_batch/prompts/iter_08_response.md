**Idea: Performance-Weighted Adaptive Parameters**
Adapt F/CR using weighted statistics from elite performers and stagnation-triggered exploration bursts, rather than purely random sampling.

```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
    sorted_idx = np.argsort(fitness)
    top_k = max(1, self.NP // 4)
    top_indices = sorted_idx[:top_k]
    
    # Compute per-individual improvement
    trial_fitness = self._eval_wrapper_batch(trials, self._func_for_adapt)
    improvements = fitness - trial_fitness
    valid = np.isfinite(improvements) & (improvements > 0)
    
    # Weighted F based on improvement magnitude
    if np.any(valid):
        weights = np.abs(improvements[valid])
        weights = weights / (np.sum(weights) + 1e-30)
        valid_indices = np.where(valid)[0]
        
        # Compute F_i for improved individuals
        improved_mutants = mutants[valid_indices]
        improved_pop = population[valid_indices]
        diff_norms = np.linalg.norm(improved_mutants - improved_pop, axis=1) + 1e-10
        pbest_idx = top_indices[0] if len(top_indices) > 0 else np.argmin(fitness)
        pbest = population[pbest_idx]
        direction_norms = np.linalg.norm(pbest - improved_pop, axis=1) + 1e-10
        
        F_i = diff_norms / (direction_norms + 1e-10)
        F_i = np.clip(F_i, 0.1, 2.0)
        
        # Weighted average F
        self.F = float(np.clip(np.sum(F_i * weights), 0.3, 1.5))
    else:
        # Stagnation: boost exploration
        if self.stagnation_count > 5:
            self.F = float(np.clip(self.F * 1.15, 0.4, 2.0))
        else:
            self.F = float(np.clip(self.F * 0.98 + np.random.uniform(0.5, 0.9) * 0.02, 0.3, 1.5))
    
    # Adaptive CR based on fitness variance
    fit_std = np.std(fitness[np.isfinite(fitness)]) + 1e-10
    fit_range = np.max(fitness[np.isfinite(fitness)]) - np.min(fitness[np.isfinite(fitness)]) + 1e-10
    
    if fit_std < fit_range * 0.05:
        # Low variance: increase CR for broad exploration
        self.CR = float(np.clip(np.random.uniform(0.7, 0.95), 0.1, 0.95))
    elif fit_std > fit_range * 0.3:
        # High variance: decrease CR for fine-tuning
        self.CR = float(np.clip(np.random.uniform(0.2, 0.5), 0.1, 0.95))
    else:
        self.CR = float(np.clip(np.random.uniform(0.4, 0.7), 0.1, 0.95))
    
    # Adaptive p_best_rate
    if self.stagnation_count > 15:
        self.p_best_rate = float(np.clip(self.p_best_rate * 0.9, 0.05, 0.3))
    elif self.stagnation_count > 5:
        self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(-0.02, 0.02), 0.05, 0.3))
```