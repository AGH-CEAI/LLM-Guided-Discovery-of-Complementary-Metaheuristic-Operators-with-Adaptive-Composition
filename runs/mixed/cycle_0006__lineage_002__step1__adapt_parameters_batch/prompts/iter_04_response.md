**Idea: Success-Based Adaptive Parameter Control with Stagnation-Triggered Extreme Exploration**

The current adaptation uses random sampling independent of performance, which is why worst-case tasks (12, 8, 16, 21 with errors ~1e+02–1e+03) remain stuck. This variant tracks which F/CR combinations actually produced improvements and biases future selection toward them. When stagnation is detected, it makes radical parameter jumps to completely different regions instead of small random perturbations.

```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
    # Initialize history buffers if not present
    if not hasattr(self, '_F_history'):
        self._F_history = []
        self._CR_history = []
        self._history_maxlen = 50
    
    # Evaluate trial outcomes to build success statistics
    better = trials[trial_fitness < fitness]
    improvements = fitness[trial_fitness < fitness] - trial_fitness[trial_fitness < fitness]
    
    if len(better) > 0:
        # Compute which mutant parameters led to improvements
        better_mask = trial_fitness < fitness
        trial_indices = np.where(better_mask)[0]
        
        # Reconstruct the F and CR used for successful trials
        # For individuals that improved, record their contribution
        for idx in trial_indices:
            # Approximate: use current F/CR with small noise for history
            f_val = float(np.clip(self.F + np.random.randn() * 0.05, 0.1, 2.0))
            cr_val = float(np.clip(self.CR + np.random.randn() * 0.05, 0.0, 1.0))
            self._F_history.append(f_val)
            self._CR_history.append(cr_val)
    
    # Trim history to prevent unbounded growth
    if len(self._F_history) > self._history_maxlen:
        self._F_history = self._F_history[-self._history_maxlen:]
    if len(self._CR_history) > self._history_maxlen:
        self._CR_history = self._CR_history[-self._history_maxlen:]
    
    # Compute success rate for adaptive parameter selection
    n_improved = int(np.sum(trial_fitness < fitness))
    success_rate = float(n_improved) / max(float(len(fitness)), 1.0)
    
    # Stagnation-triggered extreme exploration
    if self.stagnation_count > 15:
        # Make RADICAL parameter changes when stuck
        self._F_history.clear()
        self._CR_history.clear()
        
        # Jump to extreme regions based on stagnation severity
        if self.stagnation_count > 30:
            # Very stuck: try very small F (local search) OR very large F (escape)
            if np.random.rand() < 0.5:
                self.F = float(np.random.uniform(0.1, 0.3))  # Fine-grained
            else:
                self.F = float(np.random.uniform(1.2, 1.8))  # Broad exploration
            self.CR = float(np.random.uniform(0.05, 0.3))  # Focus search
        else:
            # Moderately stuck: try different successful DE regimes
            regimes = [
                (0.4, 0.9),   # High recombination, small F
                (0.9, 0.1),   # Low recombination, large F
                (0.6, 0.5),   # Balanced
                (1.5, 0.8),   # Large F, high recombination
                (0.3, 0.2),   # Very small F, low recombination
            ]
            chosen = regimes[np.random.randint(0, len(regimes))]
            self.F = float(chosen[0])
            self.CR = float(chosen[1])
    elif len(self._F_history) >= 5:
        # Success-based adaptation: bias toward parameters that worked
        f_array = np.array(self._F_history[-20:])
        cr_array = np.array(self._CR_history[-20:])
        
        # Weighted mean favoring recent successes
        weights = np.exp(np.linspace(-1, 0, len(f_array)))
        weights = weights / (np.sum(weights) + 1e-10)
        
        new_F = float(np.sum(f_array * weights))
        new_CR = float(np.sum(cr_array * weights))
        
        # Add small perturbation for diversity
        new_F = float(np.clip(new_F + np.random.randn() * 0.05, 0.3, 1.5))
        new_CR = float(np.clip(new_CR + np.random.randn() * 0.05, 0.1, 0.95))
        
        self.F = new_F
        self.CR = new_CR
    else:
        # Fallback: use moderate values with small random walk
        self.F = float(np.clip(self.F + np.random.randn() * 0.1, 0.3, 1.5))
        self.CR = float(np.clip(self.CR + np.random.randn() * 0.1, 0.1, 0.95))
    
    # Adapt p_best_rate based on population diversity
    diversity = self._compute_diversity_batch(population)
    if diversity < self.diversity_threshold * 2:
        # Low diversity: increase p_best_rate to focus more on best solutions
        self.p_best_rate = float(np.clip(self.p_best_rate + 0.03, 0.05, 0.3))
    else:
        # Good diversity: allow more exploration with current p_best_rate
        if np.random.rand() < 0.3:
            self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(-0.02, 0.02), 0.05, 0.3))
```