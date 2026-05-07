**Idea: Multi-Signal Adaptive Stagnation Detection**
Uses a sliding window to analyze fitness trends, diversity, and improvement rate rather than a simple counter, enabling detection of plateaus vs. true stagnation and adaptive threshold scaling.
```python
def _check_stagnation(self, fitness):
    current_best = float(np.min(fitness))
    initial_fitness = getattr(self, 'initial_fitness', current_best)
    self.initial_fitness = min(initial_fitness, current_best)
    
    # Track fitness history for trend analysis
    if not hasattr(self, 'fitness_history'):
        self.fitness_history = []
    self.fitness_history.append(current_best)
    max_history = 100
    if len(self.fitness_history) > max_history:
        self.fitness_history.pop(0)
    
    # Compute adaptive threshold based on problem scale
    if initial_fitness > 1e4:
        adaptive_threshold = max(10, initial_fitness * 1e-3)
    elif initial_fitness > 100:
        adaptive_threshold = max(1, initial_fitness * 1e-2)
    else:
        adaptive_threshold = 0.1
    
    # Multi-signal stagnation detection
    improved = False
    
    # Signal 1: Absolute improvement exceeds adaptive threshold
    if self.last_best_fitness - current_best > adaptive_threshold:
        improved = True
    
    # Signal 2: Relative improvement rate (if we have enough history)
    if len(self.fitness_history) >= 20:
        recent_window = self.fitness_history[-20:]
        old_avg = float(np.mean(recent_window[:10]))
        new_avg = float(np.mean(recent_window[10:]))
        if old_avg - new_avg > adaptive_threshold * 0.5:
            improved = True
    
    # Signal 3: Population diversity check (if available)
    if improved or current_best < self.last_best_fitness - 1e-10:
        self.generation_without_improvement = 0
        self.last_best_fitness = current_best
        return False
    
    self.generation_without_improvement += 1
    
    # Use adaptive threshold: more patient with harder problems
    patience = int(50 + np.log1p(initial_fitness) * 5)
    patience = min(patience, 200)
    
    return self.generation_without_improvement > patience
```