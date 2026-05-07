Looking at the unsolved tasks, I see a clear pattern:
- **Tasks 12, 8, 16, 21**: massive errors (500-1200+) — population stuck far from optimum, likely in wrong basins
- **Tasks 14, 20, 13, 23**: moderate errors (1-6) — getting closer but lacking fine convergence

The current `_adapt_parameters_batch` uses purely random parameter sampling with no learning from past success/failure. For the worst tasks stuck at huge errors, the algorithm isn't escaping their local optima basins.

**Idea: Stagnation-Triggered Adaptive Exploration-Exploitation Switching**

When stuck at high errors, the population has converged to wrong regions. I need to:
1. Track convergence via fitness variance and stagnation
2. When stagnant: aggressively switch to exploration (small F, high CR) to escape
3. When improving: use moderate parameters for exploitation
4. Learn from successful parameter regions via exponential moving average of good parameters

```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
    # Track fitness variance to detect convergence state
    fit_variance = float(np.var(fitness)) if len(fitness) > 1 else 1e20
    fit_range = float(np.max(fitness) - np.min(fitness)) if len(fitness) > 1 else 1e20
    
    # Detect if stuck at high error (population converged to wrong region)
    is_stuck_high_error = (self.stagnation_count > 15 or fit_variance < 1e-10) and self.best_fitness > 1e-2
    
    # Detect if making good progress (exploitation phase)
    is_converging = fit_variance < 1e-5 and self.stagnation_count < 5 and self.best_fitness < 1e-2
    
    # Initialize EMA tracking for successful parameters
    if not hasattr(self, '_F_ema'):
        self._F_ema = 0.7
    if not hasattr(self, '_CR_ema'):
        self._CR_ema = 0.5
    if not hasattr(self, '_success_F_history'):
        self._success_F_history = []
    if not hasattr(self, '_success_CR_history'):
        self._success_CR_history = []
    
    # Collect successful parameter pairs from this generation
    improved = fitness - trials < 0
    if np.any(improved):
        n_imp = int(np.sum(improved))
        # Estimate effective F from successful mutations (larger diff = larger F needed)
        fit_improved = fitness[improved]
        trial_improved = trials[improved]
        pop_improved = population[improved]
        
        # Compute per-individual improvement magnitude for adaptive F
        imp_magnitudes = fit_improved - trial_improved
        if np.any(imp_magnitudes > 0):
            # Normalize improvement to estimate appropriate F scale
            avg_imp = float(np.mean(imp_magnitudes))
            fit_scale = max(abs(self.best_fitness), 1.0) + 1e-10
            normalized_imp = avg_imp / fit_scale
            
            # F should scale with improvement: larger improvements need larger F
            inferred_F = float(np.clip(0.3 + normalized_imp * 0.5, 0.2, 1.5))
            self._success_F_history.append(inferred_F)
            
            # CR: use diversity of improved solutions as proxy
            if n_imp >= 2:
                cr_proxy = float(np.clip(np.std(trial_improved) / (fit_range + 1e-10), 0.3, 0.95))
            else:
                cr_proxy = self.CR
            self._success_CR_history.append(cr_proxy)
    
    # Keep history bounded
    max_history = 20
    if len(self._success_F_history) > max_history:
        self._success_F_history = self._success_F_history[-max_history:]
    if len(self._success_CR_history) > max_history:
        self._success_CR_history = self._success_CR_history[-max_history:]
    
    # Update EMA of successful parameters
    ema_alpha = 0.3
    if len(self._success_F_history) >= 3:
        self._F_ema = float(ema_alpha * np.mean(self._success_F_history) + (1 - ema_alpha) * self._F_ema)
    if len(self._success_CR_history) >= 3:
        self._CR_ema = float(ema_alpha * np.mean(self._success_CR_history) + (1 - ema_alpha) * self._CR_ema)
    
    # Determine target parameters based on convergence state
    if is_stuck_high_error:
        # Escaping wrong basin: use aggressive exploration
        target_F = float(np.clip(0.2 + np.random.uniform(0, 0.2), 0.1, 0.5))
        target_CR = float(np.clip(0.85 + np.random.uniform(0, 0.1), 0.8, 0.98))
        # Bias toward historical successes but with exploration kick
        self.F = float(0.3 * target_F + 0.7 * self._F_ema) if len(self._success_F_history) >= 3 else target_F
        self.CR = target_CR
    elif is_converging:
        # Fine convergence: use exploitation with smaller F
        target_F = float(np.clip(0.6 + np.random.uniform(0, 0.2), 0.5, 0.9))
        target_CR = float(np.clip(0.5 + np.random.uniform(0, 0.2), 0.4, 0.7))
        self.F = float(0.2 * target_F + 0.8 * self._F_ema) if len(self._success_F_history) >= 3 else target_F
        self.CR = target_CR
    else:
        # Normal operation: blend between exploration and exploitation
        if len(self._success_F_history) >= 3:
            # Blend EMA with random exploration
            self.F = float(np.clip(self._F_ema + np.random.uniform(-0.15, 0.15), 0.3, 1.5))
            self.CR = float(np.clip(self._CR_ema + np.random.uniform(-0.15, 0.15), 0.1, 0.95))
        else:
            # Fallback to moderate values
            F_candidates = np.random.uniform(0.5, 1.0, size=5)
            self.F = float(np.clip(np.mean(F_candidates), 0.3, 1.5))
            CR_candidates = np.random.uniform(0.4, 0.8, size=5)
            self.CR = float(np.clip(np.mean(CR_candidates), 0.1, 0.95))
    
    # Adjust p_best_rate based on stagnation
    if self.stagnation_count > 20:
        # Increase diversity pressure
        self.p_best_rate = float(np.clip(self.p_best_rate - 0.02, 0.05, 0.3))
    elif self.stagnation_count == 0 and fit_variance > 1e-3:
        # Good progress, can focus more on best
        self.p_best_rate = float(np.clip(self.p_best_rate + 0.01, 0.05, 0.3))
```