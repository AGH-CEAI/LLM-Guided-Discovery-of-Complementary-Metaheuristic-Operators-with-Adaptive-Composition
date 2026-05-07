Looking at the performance data, I see that ALL previous variants (01-05) produced `-inf` errors (crashed), while the original achieves the best results. The current `_adapt_parameters_batch` uses completely random parameter selection with no learning from success/failure history.

**Analysis of worst unsolved tasks:**
- Tasks 12, 8 (errors ~10³): Likely highly multi-modal or deceptive landscapes requiring aggressive exploration
- Tasks 21, 13, 14, 20, 23 (errors ~10⁰-10²): Multi-modal with local optima traps
- Key insight: The random parameter adaptation provides no feedback-driven learning, causing stagnation on difficult landscapes

**My approach:** Success-based parameter adaptation using exponential moving averages of parameters that produced improvements vs. those that didn't, with explicit stagnation-driven exploration mode.

**Idea: Success-Feedback EMA Adaptation with Stagnation Escape**
```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
        improvements = trials[trial_fitness < fitness]
        regressions = trials[trial_fitness >= fitness]
        
        trial_fitness = np.array([fitness[i] if i >= len(trials) else trials[i] 
                                  for i in range(len(population))])
        improvements_mask = trial_fitness < fitness
        n_improved = np.sum(improvements_mask)
        
        # Success-based EMA adaptation
        alpha_success = 0.3
        alpha_fail = 0.1
        beta = 0.7
        
        if n_improved > 0:
            improved_mutants = mutants[improvements_mask]
            improved_trials = trials[improvements_mask]
            improved_fitness = trial_fitness[improvements_mask]
            base_pop = population[improvements_mask]
            
            inferred_F = np.clip(np.linalg.norm(improved_mutants - base_pop, axis=1) / 
                                 (np.linalg.norm(improved_trials - base_pop, axis=1) + 1e-10), 0.1, 2.0)
            if len(inferred_F) > 0:
                self.F = float(np.clip(self.F * beta + np.mean(inferred_F) * (1 - beta), 0.3, 1.5))
            
            if not hasattr(self, 'F_ema'):
                self.F_ema = self.F
            self.F_ema = float(np.clip(self.F_ema * (1 - alpha_success) + self.F * alpha_success, 0.3, 1.5))
        else:
            self.F = float(np.clip(self.F * (1 + np.random.uniform(0.1, 0.3)), 0.3, 2.0))
            if hasattr(self, 'F_ema'):
                self.F_ema = float(np.clip(self.F_ema * (1 - alpha_fail) + self.F * alpha_fail, 0.3, 1.5))
        
        if n_improved > 0:
            improved_pop = population[improvements_mask]
            improved_trials = trials[improvements_mask]
            diff = np.abs(improved_trials - improved_pop)
            inferred_CR = np.mean(diff > 0, axis=1)
            if len(inferred_CR) > 0:
                self.CR = float(np.clip(self.CR * beta + np.mean(inferred_CR) * (1 - beta), 0.1, 0.95))
            
            if not hasattr(self, 'CR_ema'):
                self.CR_ema = self.CR
            self.CR_ema = float(np.clip(self.CR_ema * (1 - alpha_success) + self.CR * alpha_success, 0.1, 0.95))
        else:
            self.CR = float(np.clip(self.CR * np.random.uniform(0.7, 0.95), 0.1, 0.95))
            if hasattr(self, 'CR_ema'):
                self.CR_ema = float(np.clip(self.CR_ema * (1 - alpha_fail) + self.CR * alpha_fail, 0.1, 0.95))
        
        # Stagnation escape: force extreme parameters
        if self.stagnation_count > 15:
            escape_factor = min(1.0 + 0.05 * (self.stagnation_count - 15), 2.5)
            self.F = float(np.clip(self.F * escape_factor, 0.3, 2.0))
            self.CR = float(np.clip(self.CR * np.random.uniform(0.5, 0.9), 0.1, 0.95))
            self.p_best_rate = float(np.clip(self.p_best_rate * 1.2, 0.05, 0.5))
        elif self.stagnation_count > 5:
            if np.random.rand() < 0.3:
                self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(-0.1, 0.1), 0.05, 0.4))
        
        # Use EMA values as primary
        if hasattr(self, 'F_ema'):
            self.F = self.F_ema
        if hasattr(self, 'CR_ema'):
            self.CR = self.CR_ema
        
        self.F = float(np.clip(self.F, 0.3, 1.5))
        self.CR = float(np.clip(self.CR, 0.1, 0.95))
        self.p_best_rate = float(np.clip(self.p_best_rate, 0.05, 0.5))
```