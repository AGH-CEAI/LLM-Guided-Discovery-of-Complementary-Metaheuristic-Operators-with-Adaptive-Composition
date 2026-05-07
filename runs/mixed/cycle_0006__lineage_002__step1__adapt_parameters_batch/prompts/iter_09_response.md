**Idea: Success-Based Parameter Adaptation with Stagnation Detection**
Adapt parameters based on tracking historically successful (F, CR) pairs and detecting stagnation to trigger exploration mode, using rank-weighted sampling from past successes.

```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
    # Maintain parameter history with success tracking
    if not hasattr(self, '_param_history'):
        self._param_history = []
    if not hasattr(self, '_param_history_fitness'):
        self._param_history_fitness = []
    
    # Track current parameter performance
    improvements = fitness - trials
    n_improved = int(np.sum(improvements > 0))
    if n_improved > 0:
        avg_improvement = float(np.mean(improvements[improvements > 0]))
    else:
        avg_improvement = 0.0
    
    # Store (F, CR, improvement) in history (keep last 50)
    self._param_history.append((self.F, self.CR, avg_improvement))
    if len(self._param_history) > 50:
        self._param_history.pop(0)
    
    # Detect stagnation
    current_best = float(np.min(fitness))
    if not hasattr(self, '_last_best'):
        self._last_best = current_best
        self._stagnation_gens = 0
    else:
        if current_best < self._last_best - 1e-12:
            self._stagnation_gens = 0
        else:
            self._stagnation_gens += 1
        self._last_best = current_best
    
    # Stagnation detection: if no improvement for 8+ generations, switch to exploration mode
    is_stagnant = self._stagnation_gens >= 8
    
    # Compute rank-based weights from history
    if len(self._param_history) >= 5:
        hist_f = np.array([h[0] for h in self._param_history])
        hist_cr = np.array([h[1] for h in self._param_history])
        hist_imp = np.array([h[2] for h in self._param_history])
        
        # Rank transform for robustness
        if len(hist_imp) > 1:
            ranks = len(hist_imp) - np.argsort(np.argsort(hist_imp))
            weights = ranks / float(np.sum(ranks) + 1e-10)
        else:
            weights = np.array([1.0])
        
        # Sample new F and CR weighted by historical success
        if is_stagnant:
            # Exploration mode: wider range, bias toward diversity
            f_range = (0.3, 1.8)
            cr_range = (0.05, 0.95)
            f_candidates = np.random.uniform(f_range[0], f_range[1], size=7)
            cr_candidates = np.random.uniform(cr_range[0], cr_range[1], size=7)
        else:
            # Exploitation mode: sample around weighted historical values
            f_mean = float(np.sum(weights * hist_f))
            cr_mean = float(np.sum(weights * hist_cr))
            f_std = float(np.std(hist_f) + 1e-3)
            cr_std = float(np.std(hist_cr) + 1e-3)
            f_candidates = np.clip(np.random.normal(f_mean, f_std * 0.5, size=7), 0.3, 1.8)
            cr_candidates = np.clip(np.random.normal(cr_mean, cr_std * 0.5, size=7), 0.05, 0.95)
        
        # Take best 3 candidates and average
        f_scores = np.array([np.interp(f, hist_f, weights) if len(hist_f) > 1 else weights[0] 
                            for f in f_candidates])
        cr_scores = np.array([np.interp(cr, hist_cr, weights) if len(hist_cr) > 1 else weights[0] 
                             for cr in cr_candidates])
        
        top_f_idx = np.argsort(f_scores)[-3:]
        top_cr_idx = np.argsort(cr_scores)[-3:]
        
        self.F = float(np.clip(np.mean(f_candidates[top_f_idx]), 0.3, 1.8))
        self.CR = float(np.clip(np.mean(cr_candidates[top_cr_idx]), 0.05, 0.95))
    else:
        # Not enough history, use default adaptive strategy
        if is_stagnant:
            self.F = float(np.clip(self.F * np.random.uniform(1.1, 1.5), 0.3, 2.0))
            self.CR = float(np.random.uniform(0.1, 0.9))
        else:
            F_candidates = np.random.uniform(0.4, 1.0, size=5)
            self.F = float(np.clip(np.mean(F_candidates), 0.3, 1.5))
            CR_candidates = np.random.uniform(0.3, 0.9, size=5)
            self.CR = float(np.clip(np.mean(CR_candidates), 0.1, 0.95))
    
    # Adapt p_best_rate based on population diversity
    diversity = self._compute_diversity_batch(population)
    if diversity < self.diversity_threshold:
        self.p_best_rate = float(np.clip(self.p_best_rate * 0.9, 0.05, 0.3))
    else:
        self.p_best_rate = float(np.clip(self.p_best_rate * 1.02, 0.05, 0.3))
```