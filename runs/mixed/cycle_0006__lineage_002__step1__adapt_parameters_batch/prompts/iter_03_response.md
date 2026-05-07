**Idea: Success-Feedback Parameter Adaptation with Convergence Detection**

The worst unsolved tasks (errors ~1e+00 to 1e+03) are likely multimodal/deceptive functions where the population gets stuck in local optima. The current adaptation uses random sampling disconnected from search progress. This variant uses **weighted improvement rate** and **fitness spread** as signals to drive F/CR toward EXPLOITATION when making progress (to refine solutions) and toward EXPLORATION when stagnant (to escape basins).

```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
    # Track weighted improvement rate for parameter adaptation
    if not hasattr(self, '_adapt_weighted_improvement'):
        self._adapt_weighted_improvement = 0.0
        self._adapt_fitness_spread = 1.0
        self._adapt_history_F = []
        self._adapt_history_CR = []
    
    # Compute directional improvement: is best fitness improving?
    current_best = float(np.min(fitness))
    if hasattr(self, '_prev_best_fitness') and np.isfinite(self._prev_best_fitness):
        direction = self._prev_best_fitness - current_best
        # Weighted moving average with exponential decay
        decay = 0.7
        self._adapt_weighted_improvement = decay * self._adapt_weighted_improvement + (1 - decay) * direction
    self._prev_best_fitness = current_best
    
    # Track fitness spread (convergence indicator)
    finite_fitness = fitness[np.isfinite(fitness)]
    if len(finite_fitness) > 1:
        spread = float(np.std(finite_fitness)) + 1e-10
        self._adapt_fitness_spread = 0.9 * self._adapt_fitness_spread + 0.1 * spread
    
    # Normalize improvement by spread to get adaptive signal
    norm_signal = self._adapt_weighted_improvement / (self._adapt_fitness_spread + 1e-10)
    
    # Adaptive F: exploit (lower F) when improving, explore (higher F) when stagnant
    if norm_signal > 0.1:
        # Making progress: focus on exploitation with finer steps
        F_base = 0.4 + 0.3 * np.random.rand()
        F_scale = 0.05
    elif norm_signal < -0.1:
        # Stagnant: increase exploration
        F_base = 0.7 + 0.4 * np.random.rand()
        F_scale = 0.15
    else:
        # Neutral: balanced approach
        F_base = 0.5 + 0.3 * np.random.rand()
        F_scale = 0.1
    
    F_candidate = F_base + np.random.randn() * F_scale
    new_F = float(np.clip(F_candidate, 0.1, 2.0))
    
    # Adaptive CR: higher CR when improving (combine more from mutant), lower when stagnant
    if norm_signal > 0.1:
        CR_base = 0.7 + 0.2 * np.random.rand()
    elif norm_signal < -0.1:
        CR_base = 0.3 + 0.2 * np.random.rand()
    else:
        CR_base = 0.5 + 0.2 * np.random.rand()
    
    CR_candidate = CR_base + np.random.randn() * 0.1
    new_CR = float(np.clip(CR_candidate, 0.05, 0.98))
    
    # Momentum: blend with history to reduce oscillation
    if len(self._adapt_history_F) > 0:
        momentum = 0.3
        new_F = momentum * np.mean(self._adapt_history_F[-5:]) + (1 - momentum) * new_F
        new_CR = momentum * np.mean(self._adapt_history_CR[-5:]) + (1 - momentum) * new_CR
    
    self.F = float(np.clip(new_F, 0.1, 2.0))
    self.CR = float(np.clip(new_CR, 0.05, 0.98))
    
    # Track history
    self._adapt_history_F.append(self.F)
    self._adapt_history_CR.append(self.CR)
    if len(self._adapt_history_F) > 20:
        self._adapt_history_F.pop(0)
        self._adapt_history_CR.pop(0)
    
    # Adapt p_best_rate based on stagnation
    if self.stagnation_count > 15:
        delta = np.random.uniform(0.02, 0.08)
        self.p_best_rate = float(np.clip(self.p_best_rate + delta, 0.05, 0.4))
    elif self.stagnation_count > 5:
        if np.random.rand() < 0.3:
            self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(-0.03, 0.03), 0.05, 0.3))
```