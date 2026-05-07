**Idea: Success-Feedback Adaptive Parameters with EMA and Diversity-Aware Bounds**

The current adaptation is pure random averaging with no feedback from actual optimization progress. The worst tasks (12, 8, 16, 21, 13) have errors 10+ decades above target, suggesting premature convergence to local optima or divergence. This variant replaces random sampling with success-rate-weighted exponential moving averages (EMA), adaptive bounds that tighten when stagnant, and explicit diversity-driven parameter perturbation.

```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
    # Compute per-individual improvement
    improvements = fitness - trials
    n_improved = int(np.sum(improvements > 0))
    success_rate = n_improved / max(self.NP, 1)
    
    # Track EMA of F and CR weighted by success
    alpha = 0.1  # EMA smoothing
    if not hasattr(self, '_F_ema'):
        self._F_ema = self.F
        self._CR_ema = self.CR
        self._F_success_buffer = []
        self._CR_success_buffer = []
        self._prev_fitness = fitness.copy()
    else:
        # Buffer last F/CR used for improved individuals
        if n_improved > 0:
            improved_mask = improvements > 0
            avg_F_used = float(np.mean(self._F_buffer[improved_mask])) if hasattr(self, '_F_buffer') and len(self._F_buffer) == self.NP else self.F
            avg_CR_used = float(np.mean(self._CR_buffer[improved_mask])) if hasattr(self, '_CR_buffer') and len(self._CR_buffer) == self.NP else self.CR
            self._F_success_buffer.append(avg_F_used)
            self._CR_success_buffer.append(avg_CR_used)
            if len(self._F_success_buffer) > 20:
                self._F_success_buffer.pop(0)
                self._CR_success_buffer.pop(0)
        
        # Update EMA from success buffer
        if len(self._F_success_buffer) >= 3:
            target_F = float(np.clip(np.mean(self._F_success_buffer), 0.3, 1.5))
            target_CR = float(np.clip(np.mean(self._CR_success_buffer), 0.1, 0.95))
        else:
            target_F = self._F_ema
            target_CR = self._CR_ema
        
        # Blend toward target
        self._F_ema = float(np.clip(alpha * target_F + (1 - alpha) * self._F_ema, 0.1, 2.0))
        self._CR_ema = float(np.clip(alpha * target_CR + (1 - alpha) * self._CR_ema, 0.05, 0.99))
    
    # Diversity-aware perturbation: push F higher when diversity is low
    diversity = self._compute_diversity_batch(population)
    diversity_factor = float(np.clip(1.0 + (self.diversity_threshold - diversity) * 0.5, 1.0, 3.0))
    
    # Stagnation-triggered exploration boost
    if self.stagnation_count > 15:
        exploration_mult = float(np.clip(1.0 + 0.3 * np.log1p(self.stagnation_count - 15), 1.0, 2.5))
    else:
        exploration_mult = 1.0
    
    # Final F: EMA base + diversity boost + stagnation boost + small noise
    F_base = self._F_ema * diversity_factor * exploration_mult
    F_noise = np.random.randn() * 0.05
    self.F = float(np.clip(F_base + F_noise, 0.1, 2.0))
    
    # CR: success-based EMA (higher CR when improving, lower when stagnant)
    if success_rate > 0.3:
        CR_adjust = 0.05  # Encourage recombination when working
    elif self.stagnation_count > 10:
        CR_adjust = -0.05  # Reduce recombination when stuck
    else:
        CR_adjust = 0.0
    self.CR = float(np.clip(self._CR_ema + CR_adjust + np.random.randn() * 0.02, 0.05, 0.99))
    
    # p_best_rate: shrink when stagnant (focus on exploitation), grow when improving (explore best)
    if self.stagnation_count > 5:
        p_adjust = -0.02
    elif success_rate > 0.4:
        p_adjust = 0.02
    else:
        p_adjust = 0.0
    if np.random.rand() < 0.3:  # Probabilistic update
        self.p_best_rate = float(np.clip(self.p_best_rate + p_adjust, 0.03, 0.5))
    
    # Generate per-individual F/CR buffers for next iteration tracking
    self._F_buffer = np.clip(self.F + np.random.randn(self.NP) * 0.1, 0.0, 2.0)
    self._CR_buffer = np.clip(self.CR + np.random.randn(self.NP) * 0.05, 0.0, 1.0)
    self._prev_fitness = fitness.copy()
```