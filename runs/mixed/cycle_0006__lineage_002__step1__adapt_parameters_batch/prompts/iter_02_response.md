**Idea: Success-Based Parameter Adaptation with History Tracking**

Replace naive random parameter sampling with a history-driven approach that tracks which F and CR values led to actual fitness improvements. Use weighted exponential moving averages biased toward successful parameter regions, with restart mechanisms when stagnation is detected.

```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
    # Track parameter history and their associated improvements
    if not hasattr(self, 'param_history'):
        self.param_history = {'F': [], 'CR': [], 'improvements': []}
    
    # Compute per-individual improvements
    trial_fitness = self._eval_wrapper_batch(trials, lambda x: x)
    improvements = fitness - trial_fitness
    
    # Record successful parameter-region samples
    successful_mask = improvements > 0
    n_success = int(np.sum(successful_mask))
    
    if n_success > 0:
        # Extract F and CR from successful individuals
        success_F = self.F * np.ones(n_success)  # Current F was used
        success_CR = self.CR * np.ones(n_success)
        success_improvements = improvements[successful_mask]
        
        self.param_history['F'].extend(success_F.tolist())
        self.param_history['CR'].extend(success_CR.tolist())
        self.param_history['improvements'].extend(success_improvements.tolist())
        
        # Keep history bounded
        max_history = 50
        if len(self.param_history['F']) > max_history:
            self.param_history['F'] = self.param_history['F'][-max_history:]
            self.param_history['CR'] = self.param_history['CR'][-max_history:]
            self.param_history['improvements'] = self.param_history['improvements'][-max_history:]
    
    # Adaptive parameter selection based on history
    history_len = len(self.param_history['F'])
    
    if history_len >= 5:
        # Weighted average favoring high-improvement samples
        weights = np.array(self.param_history['improvements'][-history_len:])
        weights = np.clip(weights, 1e-10, None)
        weights = weights / (np.sum(weights) + 1e-10)
        
        new_F = float(np.dot(weights, np.array(self.param_history['F'][-history_len:])))
        new_CR = float(np.dot(weights, np.array(self.param_history['CR'][-history_len:])))
        
        # Add perturbation scaled by uncertainty
        std_F = float(np.std(self.param_history['F'][-min(10, history_len):]))
        std_CR = float(np.std(self.param_history['CR'][-min(10, history_len):]))
        
        new_F = new_F + np.random.randn() * max(std_F, 0.1)
        new_CR = new_CR + np.random.randn() * max(std_CR, 0.1)
    else:
        # Fallback: aggressive exploration for early generations
        new_F = np.random.uniform(0.2, 1.5)
        new_CR = np.random.uniform(0.1, 0.95)
    
    # Clamp to valid ranges
    self.F = float(np.clip(new_F, 0.1, 2.0))
    self.CR = float(np.clip(new_CR, 0.05, 0.98))
    
    # Stagnation-triggered restart: if no improvement for extended period, reset history
    if self.stagnation_count > 30 and history_len >= 5:
        # Jump to radically different parameter region
        self.F = float(np.clip(np.random.uniform(0.1, 2.0), 0.1, 2.0))
        self.CR = float(np.clip(np.random.uniform(0.05, 0.98), 0.05, 0.98))
        self.param_history = {'F': [], 'CR': [], 'improvements': []}
    else:
        # Gradual p_best_rate adaptation based on stagnation
        if np.random.rand() < 0.3:
            if self.stagnation_count > 15:
                # Increase exploration by lowering p_best_rate
                delta = np.random.uniform(-0.08, 0.02)
            else:
                # Normal operation
                delta = np.random.uniform(-0.03, 0.03)
            self.p_best_rate = float(np.clip(self.p_best_rate + delta, 0.05, 0.4))
```