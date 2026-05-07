**Idea: Success-Feedback EMA Adaptation with Stagnation-Driven Exploration**

The current adaptation randomly samples F/CR without any performance feedback, which is why the worst tasks (12, 8, 16, 21) remain stuck at massive errors. This variant replaces random sampling with exponential moving average (EMA) tracking of which parameter ranges historically produced improvements, PLUS a stagnation-triggered exploration mode that dramatically increases F and p_best when the algorithm gets stuck—directly targeting the local optima traps causing the 1e+02 to 1e+03 errors.

```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
        # Compute per-individual improvements
        improvements = fitness - np.where(trial_fitness < fitness, trial_fitness, fitness)
        n_improved = int(np.sum(improvements > 0))
        total_improvement = float(np.sum(improvements[improvements > 0]))
        
        # Track success metrics for EMA adaptation
        if not hasattr(self, '_F_history'):
            self._F_history = []
            self._CR_history = []
            self._imp_history = []
        
        # Record current parameter quality
        self._F_history.append(float(self.F))
        self._CR_history.append(float(self.CR))
        self._imp_history.append(float(total_improvement))
        
        # Keep bounded history
        if len(self._F_history) > 50:
            self._F_history.pop(0)
            self._CR_history.pop(0)
            self._imp_history.pop(0)
        
        # Compute success-weighted EMAs when we have enough data
        if len(self._imp_history) >= 5:
            imp_arr = np.array(self._imp_history)
            imp_pos = np.maximum(imp_arr, 0)
            weight_sum = float(np.sum(imp_pos)) + 1e-10
            
            # EMA with success-weighted averaging
            F_weights = imp_pos / weight_sum
            CR_weights = imp_pos / weight_sum
            
            # Bias toward parameters that produced improvements
            target_F = float(np.sum(np.array(self._F_history) * F_weights))
            target_CR = float(np.sum(np.array(self._CR_history) * CR_weights))
            
            # Blend toward success-based target
            alpha = 0.3
            self.F = float(np.clip(self.F * (1 - alpha) + target_F * alpha, 0.3, 1.5))
            self.CR = float(np.clip(self.CR * (1 - alpha) + target_CR * alpha, 0.1, 0.95))
        else:
            # Fallback for early generations
            F_candidates = np.random.uniform(0.4, 1.0, size=5)
            self.F = float(np.clip(np.mean(F_candidates), 0.3, 1.5))
            CR_candidates = np.random.uniform(0.3, 0.9, size=5)
            self.CR = float(np.clip(np.mean(CR_candidates), 0.1, 0.95))
        
        # Stagnation-driven exploration: when stuck, radically increase F and p_best
        if self.stagnation_count > 15:
            # Aggressive exploration: large jumps to escape local optima
            self.F = float(np.clip(self.F + np.random.uniform(0.1, 0.4), 0.5, 2.0))
            self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(0.02, 0.08), 0.05, 0.4))
        elif self.stagnation_count > 5:
            # Moderate exploration push
            self.F = float(np.clip(self.F + np.random.uniform(0.05, 0.2), 0.4, 1.8))
            self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(0.01, 0.04), 0.05, 0.35))
        else:
            # Normal adaptation: gentle drift
            if np.random.rand() < 0.2:
                self.p_best_rate = float(np.clip(self.p_best_rate + np.random.uniform(-0.03, 0.03), 0.05, 0.3))
        
        # CR adaptation: lower CR when stagnant (maintain diversity), higher when improving
        if self.stagnation_count > 10:
            self.CR = float(np.clip(self.CR * 0.95, 0.1, 0.9))
        elif n_improved > self.NP // 4:
            self.CR = float(np.clip(self.CR * 1.05, 0.1, 0.95))
```