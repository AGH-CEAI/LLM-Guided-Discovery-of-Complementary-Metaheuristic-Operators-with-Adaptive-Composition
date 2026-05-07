**Idea: Success-Weighted Adaptive Parameters**

Track recent successful (F, CR) pairs that yielded improvements and bias future selections toward them. This is fundamentally different from random sampling because it uses actual performance feedback to guide parameter search, rather than blind exploration.

```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
    # Initialize success memory on first call
    if not hasattr(self, '_F_history'):
        self._F_history = []
        self._CR_history = []
        self._success_memory_size = 20
    
    # Compute which trials improved
    improvements = fitness - trials
    n_improved = np.sum(improvements > 0)
    
    # Record F/CR that led to improvements
    if n_improved > 0:
        improved_mask = improvements > 0
        for i in np.where(improved_mask)[0]:
            self._F_history.append(float(self.F))
            self._CR_history.append(float(self.CR))
    
    # Keep memory bounded
    if len(self._F_history) > self._success_memory_size:
        self._F_history = self._F_history[-self._success_memory_size:]
        self._CR_history = self._CR_history[-self._success_memory_size:]
    
    # If we have enough success history, sample from successful parameters
    if len(self._F_history) >= 5:
        # Bias toward recent successful F values with some mutation
        F_base = float(np.mean(self._F_history[-10:]))
        F_std = float(np.std(self._F_history[-10:])) + 1e-6
        F_new = float(np.clip(F_base + np.random.randn() * F_std, 0.1, 2.0))
        
        CR_base = float(np.mean(self._CR_history[-10:]))
        CR_std = float(np.std(self._CR_history[-10:])) + 1e-6
        CR_new = float(np.clip(CR_base + np.random.randn() * CR_std, 0.0, 1.0))
        
        self.F = F_new
        self.CR = CR_new
    else:
        # Fallback to random sampling with wider range for exploration
        F_candidates = np.random.uniform(0.2, 1.2, size=5)
        self.F = float(np.clip(np.mean(F_candidates), 0.1, 2.0))
        
        CR_candidates = np.random.uniform(0.1, 0.95, size=5)
        self.CR = float(np.clip(np.mean(CR_candidates), 0.0, 1.0))
    
    # Adaptive p_best_rate based on stagnation
    if self.stagnation_count > 15:
        # Increase exploration by targeting more diverse best individuals
        self.p_best_rate = float(np.clip(self.p_best_rate * 1.1, 0.05, 0.4))
    elif self.stagnation_count == 0 and n_improved > self.NP * 0.3:
        # Good progress, can be more greedy
        self.p_best_rate = float(np.clip(self.p_best_rate * 0.95, 0.05, 0.3))
```