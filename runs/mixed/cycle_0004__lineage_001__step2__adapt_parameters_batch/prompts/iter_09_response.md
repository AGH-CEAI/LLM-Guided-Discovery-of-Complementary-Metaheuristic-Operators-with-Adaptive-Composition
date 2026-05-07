**Idea: Success-Based Adaptive Parameters**
Adapt F and CR using actual improvement signals from top performers, with stagnation-triggered exploration bursts. This replaces random sampling with performance-driven feedback.

```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
    improvements = fitness - trials
    top_half = np.argsort(fitness)[:max(1, self.NP // 2)]
    
    # Track improvement signals from top performers
    top_improvements = improvements[top_half]
    top_improved_mask = top_improvements > 0
    n_top_improved = int(np.sum(top_improved_mask))
    improvement_rate = n_top_improved / max(len(top_half), 1)
    
    # Track global progress
    median_fitness = float(np.median(fitness))
    prev_best = getattr(self, '_prev_best_fitness', float(np.min(fitness)))
    is_improving = float(np.min(fitness)) < prev_best - 1e-12
    self._prev_best_fitness = float(np.min(fitness))
    
    # Adaptive F: based on success rate and stagnation
    if improvement_rate > 0.5 and is_improving:
        # Good progress: use smaller F for exploitation
        F_base = 0.5 + np.random.uniform(0, 0.2)
    elif improvement_rate > 0.2:
        # Moderate progress: balanced
        F_base = 0.7 + np.random.uniform(-0.1, 0.2)
    elif self.stagnation_count > 10:
        # Stagnant: use larger F for exploration burst
        F_base = 1.0 + min(0.5 * np.log1p(self.stagnation_count), 0.8)
    else:
        # Low improvement rate but not stagnant: moderate exploration
        F_base = 0.9 + np.random.uniform(-0.1, 0.2)
    
    self.F = float(np.clip(F_base + np.random.randn() * 0.05, 0.3, 2.0))
    
    # Adaptive CR: based on convergence status
    if improvement_rate > 0.4:
        # Many improving: higher CR for recombination
        CR_base = 0.7 + np.random.uniform(0, 0.2)
    elif improvement_rate > 0.15:
        # Moderate: balanced
        CR_base = 0.5 + np.random.uniform(-0.1, 0.2)
    else:
        # Few improving: lower CR for diversity
        CR_base = 0.3 + np.random.uniform(-0.1, 0.1)
    
    self.CR = float(np.clip(CR_base + np.random.randn() * 0.05, 0.1, 0.95))
    
    # Adaptive p_best_rate: more greedy when stagnant, more diverse when improving
    if self.stagnation_count > 15:
        self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(0.01, 0.03), 0.05, 0.3))
    elif is_improving and improvement_rate > 0.3:
        self.p_best_rate = float(np.clip(self.p_best_rate - np.random.uniform(0.005, 0.015), 0.05, 0.3))
```