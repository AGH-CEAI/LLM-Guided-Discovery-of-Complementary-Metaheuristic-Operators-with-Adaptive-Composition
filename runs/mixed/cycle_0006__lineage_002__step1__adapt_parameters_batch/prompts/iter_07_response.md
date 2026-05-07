**Idea: Success-History Adaptation with Stagnation Detection**
Uses actual performance feedback to adapt F and CR based on recent successful parameter values, with stagnation-triggered parameter resets to escape local optima traps on the worst unsolved tasks.

```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
    # Initialize success history tracking
    if not hasattr(self, 'F_history'):
        self.F_history = []
        self.CR_history = []
        self.history_max_size = 20
    
    # Track successful parameter values from recent trials
    if trials is not None and len(trials) > 0:
        try:
            trial_fitness = self._eval_wrapper_batch(trials, np.zeros((len(trials), self.dim)))
            improvements = fitness - trial_fitness
            success_mask = improvements > 0
            n_success = int(np.sum(success_mask))
            
            if n_success > 0:
                # Record successful F and CR values
                for i in range(n_success):
                    if len(self.F_history) < self.history_max_size:
                        self.F_history.append(self.F)
                        self.CR_history.append(self.CR)
                
                # Adaptation based on success rate
                success_rate = n_success / max(len(fitness), 1)
                
                # If many successes: exploit (use smaller F for fine-tuning)
                if success_rate > 0.3:
                    self.F = float(np.clip(self.F * 0.95, 0.3, 1.5))
                    self.CR = float(np.clip(self.CR * 1.02, 0.1, 0.95))
                # If few successes: explore (use larger F for escape)
                elif success_rate < 0.1:
                    self.F = float(np.clip(self.F * 1.1, 0.3, 1.5))
                    self.CR = float(np.clip(self.CR * 0.98, 0.1, 0.95))
        except:
            pass
    
    # Stagnation detection with parameter reset
    if self.stagnation_count > 15:
        # Force exploration with diverse parameters
        if len(self.F_history) >= 5:
            # Sample from best performing history entries
            recent_f = self.F_history[-5:]
            recent_cr = self.CR_history[-5:]
            self.F = float(np.clip(np.median(recent_f) + np.random.uniform(-0.1, 0.1), 0.3, 1.5))
            self.CR = float(np.clip(np.median(recent_cr) + np.random.uniform(-0.1, 0.1), 0.1, 0.95))
        else:
            # Random reset for diverse exploration
            self.F = float(np.clip(0.7 + np.random.uniform(-0.3, 0.5), 0.3, 1.5))
            self.CR = float(np.clip(0.5 + np.random.uniform(-0.2, 0.3), 0.1, 0.95))
        
        # Adapt p_best_rate based on stagnation severity
        if self.stagnation_count > 30:
            self.p_best_rate = float(np.clip(self.p_best_rate * 1.1, 0.05, 0.3))
        else:
            self.p_best_rate = float(np.clip(self.p_best_rate * 0.95, 0.05, 0.3))
    else:
        # Normal adaptation with bounded random walk
        if np.random.rand() < 0.3:
            self.F = float(np.clip(self.F + np.random.uniform(-0.05, 0.05), 0.3, 1.5))
        if np.random.rand() < 0.3:
            self.CR = float(np.clip(self.CR + np.random.uniform(-0.05, 0.05), 0.1, 0.95))
        if np.random.rand() < 0.15:
            self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(-0.02, 0.02), 0.05, 0.3))
    
    # Clip all parameters to valid ranges
    self.F = float(np.clip(self.F, 0.3, 1.5))
    self.CR = float(np.clip(self.CR, 0.1, 0.95))
    self.p_best_rate = float(np.clip(self.p_best_rate, 0.05, 0.3))
```