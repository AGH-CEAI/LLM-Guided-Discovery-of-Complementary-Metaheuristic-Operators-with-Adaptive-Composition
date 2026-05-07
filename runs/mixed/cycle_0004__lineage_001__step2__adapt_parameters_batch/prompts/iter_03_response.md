**Idea: Performance-Feedback Parameter Adaptation**

Instead of random parameter sampling, this variant adapts F and CR based on actual trial improvement rates and fitness distributions. When trials fail to improve (stagnation signal), it aggressively increases F for exploration and adjusts CR for diversity. This directly addresses the priority tasks (errors 10^1-10^3) which are severely stuck—current random adaptation ignores population state entirely.

```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
    # Compute trial improvements (positive = trial better)
    improvements = fitness - self._eval_wrapper_batch(trials, self._obj_func_for_adapt)
    valid_improvements = improvements[np.isfinite(improvements)]
    
    n_improved = int(np.sum(valid_improvements > 0))
    improvement_rate = float(n_improved) / max(float(len(valid_improvements)), 1)
    
    # Base adaptation on fitness distribution
    sorted_idx = np.argsort(fitness)
    top_quarter = sorted_idx[:max(1, self.NP // 4)]
    bottom_quarter = sorted_idx[self.NP - max(1, self.NP // 4):]
    
    top_spread = float(np.std(fitness[top_quarter])) + 1e-10
    bottom_spread = float(np.std(fitness[bottom_quarter])) + 1e-10
    
    # Adapt F: higher when stagnation or when bottom performers are clustered
    if self.stagnation_count > 15:
        F_target = np.random.uniform(0.9, 1.4)  # Large steps when stuck
    elif improvement_rate < 0.1:
        F_target = np.random.uniform(0.7, 1.1)  # Increase exploration
    elif bottom_spread < top_spread * 0.5:
        F_target = np.random.uniform(0.6, 1.0)  # Medium-high when clustered
    else:
        F_target = np.random.uniform(0.4, 0.8)  # Normal refinement
    self.F = float(np.clip(F_target, 0.3, 1.5))
    
    # Adapt CR: higher when population diverse, lower when converged
    if self.stagnation_count > 20:
        CR_target = np.random.uniform(0.7, 0.95)  # Force diversity breakout
    elif improvement_rate < 0.15:
        CR_target = np.random.uniform(0.5, 0.85)  # More diversity
    else:
        CR_target = np.random.uniform(0.3, 0.7)  # Normal
    self.CR = float(np.clip(CR_target, 0.1, 0.95))
    
    # Adapt p_best_rate based on whether top solutions are improving
    top_fitness_mean = float(np.mean(fitness[top_quarter]))
    bottom_fitness_mean = float(np.mean(fitness[bottom_quarter]))
    fitness_gap = bottom_fitness_mean - top_fitness_mean
    
    if fitness_gap < 1.0:  # Top and bottom close = stuck
        p_target = float(np.clip(self.p_best_rate * 1.2, 0.05, 0.3))
    elif fitness_gap > 100.0:  # Large gap = good selection pressure
        p_target = float(np.clip(self.p_best_rate * 0.9, 0.05, 0.3))
    else:
        p_target = self.p_best_rate
    
    if np.random.rand() < 0.25:
        self.p_best_rate = float(np.clip(p_target + np.random.uniform(-0.03, 0.03), 0.05, 0.3))
```

*Note: `self._obj_func_for_adapt` should be set to a simple callable wrapper that returns trial fitness values. If this causes issues, replace the improvement calculation with a direct fitness comparison approach.*