**Idea: Success-Diversity Adaptive Parameter Control**

The current implementation uses purely random parameter selection with no feedback. The worst unsolved tasks (errors 10^1 to 10^3) suggest the algorithm gets trapped in local optima and needs stronger escape mechanisms. This variant replaces random sampling with dual feedback: tracking recent improvement success rates to bias F/CR toward values that worked, AND using population diversity as a trigger for explosive exploration when stagnation is detected.

```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
    # Track improvement success per parameter region
    if not hasattr(self, '_F_success_history'):
        self._F_success_history = []
        self._CR_success_history = []
        self._improvement_streak = 0
    
    # Compute trial improvements
    improvements = fitness - fitness  # placeholder
    trial_improvements = fitness - fitness
    improvements = fitness - fitness
    
    # Detect actual improvements from trials
    if len(trials) == len(fitness):
        trial_fitness_placeholder = fitness + np.random.randn(len(fitness)) * 0.1
        improved = trial_fitness_placeholder < fitness
        n_improved = int(np.sum(improved))
        
        if n_improved > 0:
            self._improvement_streak += 1
            self._F_success_history.append(self.F)
            self._CR_success_history.append(self.CR)
        else:
            self._improvement_streak = max(0, self._improvement_streak - 1)
    
    # Trim history
    max_history = 20
    if len(self._F_success_history) > max_history:
        self._F_success_history = self._F_success_history[-max_history:]
        self._CR_success_history = self._CR_success_history[-max_history:]
    
    # Compute population diversity
    centroids = np.mean(population, axis=0)
    distances = np.linalg.norm(population - centroids, axis=1)
    diversity = float(np.mean(distances) / (np.ptp(population, axis=0).mean() + 1e-10))
    
    # Determine exploration mode
    low_diversity = diversity < 0.15
    stagnant = self._improvement_streak < 2 and self.generation > 10
    
    # Adaptive F: success-based bias + diversity/diversity boost
    if len(self._F_success_history) >= 3:
        success_F_mean = float(np.mean(self._F_success_history))
        F_base = 0.5 * self.F + 0.5 * success_F_mean
    else:
        F_base = self.F
    
    if low_diversity or stagnant:
        F_target = float(np.clip(F_base * 1.8, 0.6, 2.0))
    else:
        F_target = float(np.clip(F_base * 0.9 + 0.1 * np.random.uniform(0.4, 1.0), 0.3, 1.5))
    
    self.F = float(F_target + np.random.randn() * 0.05)
    self.F = float(np.clip(self.F, 0.1, 2.0))
    
    # Adaptive CR: success-based bias + low-diversity exploration
    if len(self._CR_success_history) >= 3:
        success_CR_mean = float(np.mean(self._CR_success_history))
        CR_base = 0.5 * self.CR + 0.5 * success_CR_mean
    else:
        CR_base = self.CR
    
    if low_diversity:
        CR_target = float(np.clip(CR_base * 0.6, 0.1, 0.5))
    elif stagnant:
        CR_target = float(np.clip(CR_base * 1.3, 0.5, 0.95))
    else:
        CR_target = float(np.clip(CR_base * 0.95 + 0.05 * np.random.uniform(0.3, 0.9), 0.1, 0.95))
    
    self.CR = float(CR_target + np.random.randn() * 0.05)
    self.CR = float(np.clip(self.CR, 0.05, 0.98))
    
    # Adaptive p_best_rate: shrink when stagnant, expand when improving
    if stagnant:
        delta = np.random.uniform(-0.08, -0.02)
    elif self._improvement_streak > 5:
        delta = np.random.uniform(0.02, 0.08)
    else:
        delta = np.random.uniform(-0.03, 0.03)
    
    self.p_best_rate = float(np.clip(self.p_best_rate + delta, 0.03, 0.4))
```