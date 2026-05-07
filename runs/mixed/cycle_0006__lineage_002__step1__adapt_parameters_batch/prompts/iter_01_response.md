**Idea: Success-Based JADE Adaptation**
Uses actual successful mutations to compute F and CR parameters directly, leveraging weighted mean of successful values instead of random sampling.

```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
    # Identify successful mutations (trials that improved fitness)
    success_mask = trials != population
    improvements = fitness - np.where(success_mask, 
                                      self._eval_wrapper_batch(trials, lambda x: x), 
                                      np.full(len(trials), np.inf))
    
    # Compute actual fitness of trials for success detection
    trial_fitness = self._eval_wrapper_batch(trials, lambda x: x)
    successes = trial_fitness < fitness
    
    if np.sum(successes) > 0:
        # Extract successful F values (estimated from mutation vector differences)
        successful_mutants = mutants[successes]
        successful_trials = trials[successes]
        successful_pop = population[successes]
        
        # Estimate F from successful mutations: F ≈ ||delta|| / ||v1 - v2||
        deltas = successful_trials - successful_pop
        diffs = successful_mutants - successful_pop
        
        f_estimates = []
        for d, v in zip(deltas, diffs):
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-10:
                d_norm = np.linalg.norm(d)
                f_est = np.clip(d_norm / v_norm, 0.0, 2.0)
                f_estimates.append(f_est)
        
        # Extract successful CR values (proportion of dimensions changed in successful trials)
        cr_estimates = np.mean(successful_trials != successful_pop, axis=1)
        
        # Weighted combination: weight by improvement magnitude
        improv = fitness[successes] - trial_fitness[successes]
        weights = np.abs(improv) / (np.sum(np.abs(improv)) + 1e-10)
        
        if len(f_estimates) > 0:
            new_F = float(np.clip(np.average(f_estimates, weights=weights[:len(f_estimates)]), 0.1, 1.5))
            # Momentum: blend with previous F for stability
            self.F = float(0.5 * self.F + 0.5 * new_F)
        
        if len(cr_estimates) > 0:
            new_CR = float(np.clip(np.average(cr_estimates, weights=weights[:len(cr_estimates)]), 0.1, 0.95))
            # Momentum: blend with previous CR for stability
            self.CR = float(0.7 * self.CR + 0.3 * new_CR)
    else:
        # No successes: use moderate defaults with small perturbation
        self.F = float(np.clip(self.F + np.random.uniform(-0.1, 0.1), 0.1, 1.5))
        self.CR = float(np.clip(self.CR + np.random.uniform(-0.05, 0.05), 0.1, 0.95))
    
    # Adapt p_best_rate based on convergence pressure
    sorted_idx = np.argsort(fitness)
    best_fitness = fitness[sorted_idx[0]]
    worst_fitness = fitness[sorted_idx[-1]]
    range_fitness = worst_fitness - best_fitness
    
    if range_fitness > 1e-10:
        # If population is converging, increase exploration (lower p_best)
        convergence_ratio = range_fitness / (np.abs(best_fitness) + 1e-10)
        target_pbest = float(np.clip(0.05 + 0.15 * min(1.0, convergence_ratio), 0.05, 0.3))
        self.p_best_rate = float(0.8 * self.p_best_rate + 0.2 * target_pbest)
```