**Idea: Success-Guided Parameter Adaptation with Stagnation Memory**

This approach replaces random sampling with a success-history-driven mechanism that tracks which (F, CR) combinations produced improvements, biases parameters toward successful regions when stagnating, and uses exponentially-weighted moving averages of successful parameters.

```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
    # Track parameter history for success-based adaptation
    if not hasattr(self, 'param_history'):
        self.param_history = []
    if not hasattr(self, 'param_history_fitness'):
        self.param_history_fitness = []
    
    # Evaluate trial success
    improvements = fitness - trials
    success_mask = improvements > 0
    success_rate = float(np.mean(success_mask))
    
    # Store recent parameters weighted by success
    for i in range(self.NP):
        if success_mask[i]:
            self.param_history.append((float(self.F), float(self.CR), float(improvements[i])))
    
    # Keep history bounded
    max_history = 50
    if len(self.param_history) > max_history:
        self.param_history = self.param_history[-max_history:]
    
    # Determine stagnation level
    stagnation_level = min(self.stagnation_count / max(1, self.stagnation_limit), 1.0)
    
    # Adapt F: higher when stagnant (more exploration), lower when succeeding (fine-tuning)
    if len(self.param_history) >= 5:
        # Weighted average of successful F values, weighted by improvement magnitude
        f_vals = np.array([h[0] for h in self.param_history])
        weights = np.array([h[2] for h in self.param_history])
        weights = np.clip(weights, 1e-10, None)
        weights = weights / np.sum(weights)
        success_f_mean = float(np.dot(weights, f_vals))
        
        # Bias toward exploration when stagnant
        base_f = success_f_mean if success_rate > 0.1 else 0.7
        target_f = base_f + stagnation_level * 0.4  # Up to +0.4 when fully stagnant
        target_f = float(np.clip(target_f, 0.3, 2.0))
        
        # Gradual move toward target
        self.F = float(np.clip(0.7 * self.F + 0.3 * target_f, 0.3, 2.0))
    else:
        # Not enough history: use stagnation to guide initial F
        if stagnation_level > 0.5:
            self.F = float(np.clip(self.F + 0.1, 0.3, 2.0))
        else:
            self.F = float(np.clip(self.F + np.random.uniform(-0.05, 0.05), 0.3, 2.0))
    
    # Adapt CR: lower when stagnant (wider search), higher when succeeding (intensify)
    if len(self.param_history) >= 5:
        cr_vals = np.array([h[1] for h in self.param_history])
        weights = np.array([h[2] for h in self.param_history])
        weights = np.clip(weights, 1e-10, None)
        weights = weights / np.sum(weights)
        success_cr_mean = float(np.dot(weights, cr_vals))
        
        # Bias toward lower CR (more exploration) when stagnant
        base_cr = success_cr_mean if success_rate > 0.1 else 0.5
        target_cr = base_cr - stagnation_level * 0.3  # Lower when stagnant
        target_cr = float(np.clip(target_cr, 0.1, 0.95))
        
        self.CR = float(np.clip(0.7 * self.CR + 0.3 * target_cr, 0.1, 0.95))
    else:
        if stagnation_level > 0.5:
            self.CR = float(np.clip(self.CR - 0.05, 0.1, 0.95))
        else:
            self.CR = float(np.clip(self.CR + np.random.uniform(-0.05, 0.05), 0.1, 0.95))
    
    # Adapt p_best_rate: smaller when stagnant (wider selection of best)
    if stagnation_level > 0.3:
        self.p_best_rate = float(np.clip(self.p_best_rate * 0.95, 0.05, 0.3))
    elif success_rate > 0.3:
        self.p_best_rate = float(np.clip(self.p_best_rate * 1.02, 0.05, 0.3))
```