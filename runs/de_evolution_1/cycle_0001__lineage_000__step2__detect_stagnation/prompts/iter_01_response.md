**Idea: Multi-Criteria Stagnation with History-Based Fitness Plateau Detection**
A more comprehensive stagnation detector that tracks fitness history to detect plateaus, checks for TolX/TolFun-like conditions from standard CMA-ES, and uses adaptive thresholds based on generation count.

```python
def _detect_stagnation(self):
    """Check if the algorithm has stagnated and needs a restart using multiple criteria."""
    dim = self.dim
    
    # --- Criterion 1: No improvement counter (adaptive threshold) ---
    stag_limit = 10 + int(30 * dim / self.pop_size)
    if self.stagnation_counter > stag_limit:
        return True
    
    # --- Criterion 2: Sigma collapsed ---
    if self.sigma < 1e-20:
        return True
    
    # --- Criterion 3: Sigma exploded (lost control) ---
    if self.sigma > 1e8:
        return True
    
    # --- Criterion 4: Condition number of C is too large ---
    max_D = np.max(self.D)
    min_D = max(np.min(self.D), 1e-30)
    condition_number = (max_D / min_D) ** 2  # condition of C = (max_D/min_D)^2
    if condition_number > 1e14:
        return True
    
    # --- Criterion 5: TolX - all components of sigma * principal axes are too small ---
    # If the effective step in all coordinates is negligible relative to domain
    tolx = 1e-12
    if self.sigma * np.max(self.D) < tolx:
        return True
    
    # Also check if sigma * sqrt(diag(C)) are all tiny
    diag_C = np.diag(self.C) if self.C.ndim == 2 else np.ones(dim)
    diag_C = np.maximum(diag_C, 0.0)
    if np.all(self.sigma * np.sqrt(diag_C) < tolx):
        return True
    
    # --- Criterion 6: Fitness history plateau detection ---
    # Track fitness history
    if not hasattr(self, '_fitness_history'):
        self._fitness_history = []
    self._fitness_history.append(self.best_fitness)
    
    # Keep a rolling window
    history_len = 10 + int(30 * dim / self.pop_size)
    if len(self._fitness_history) > history_len * 2:
        self._fitness_history = self._fitness_history[-history_len * 2:]
    
    if len(self._fitness_history) >= history_len:
        recent = self._fitness_history[-history_len:]
        old = self._fitness_history[-history_len * 2:-history_len] if len(self._fitness_history) >= history_len * 2 else self._fitness_history[:history_len]
        
        if len(old) > 0 and len(recent) > 0:
            best_recent = np.min(recent)
            best_old = np.min(old)
            # If relative improvement is negligible over the window
            denom = max(abs(best_old), abs(best_recent), 1e-30)
            rel_improvement = abs(best_old - best_recent) / denom
            if rel_improvement < 1e-12 and self.generation > 2 * history_len:
                return True
    
    # --- Criterion 7: NoEffectAxis - adding sigma*D along any principal axis has no effect ---
    if self.generation > 0 and self.generation % max(1, dim) == 0:
        for i in range(dim):
            test_vec = self.mean + 0.1 * self.sigma * self.D[min(i, len(self.D)-1)] * self.B[:, min(i, self.B.shape[1]-1)]
            if np.allclose(test_vec, self.mean, rtol=0, atol=1e-16):
                return True
    
    # --- Criterion 8: NoEffectCoord - adding sigma*sqrt(C_ii) in any coord has no effect ---
    for i in range(dim):
        if self.mean[i] == self.mean[i] + 0.2 * self.sigma * np.sqrt(max(diag_C[i], 0)):
            return True
    
    return False
```