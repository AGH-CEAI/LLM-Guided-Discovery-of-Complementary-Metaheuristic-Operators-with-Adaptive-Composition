Looking at the results, all 23 unsolved tasks have errors many orders of magnitude above the target. The worst tasks (16, 19, 23, 18, 10, 20) have errors in the hundreds or thousands, suggesting the algorithm gets trapped in local optima on multimodal/non-separable functions and doesn't restart aggressively enough. Meanwhile, tasks near convergence (0, 1, 6) with errors ~1e-7 need more generations of fine-tuning before being disrupted.

The current stagnation detection is too conservative — it waits `10 + 30*dim/pop_size` generations (~70 for dim=30, pop=14) before restarting. On hard multimodal problems, the algorithm wastes evaluations stuck in poor basins. The original also only checks the stagnation counter and two degeneration conditions.

My approach: **much more aggressive restart triggering** using multiple signals — relative fitness improvement tracking over a sliding window, sigma explosion detection, and flat fitness landscape detection. This will force frequent restarts on hard multimodal problems, giving more chances to find the global basin. For problems that are nearly solved, the condition ratio check is relaxed to let local search continue.

**Idea: Aggressive Multi-Signal Stagnation**
Trigger restarts much earlier using sliding-window relative improvement, flat fitness detection, sigma explosion, and shorter absolute stagnation limits to maximize basin-hopping on hard multimodal tasks.

```python
def _detect_stagnation(self):
    """Aggressively detect stagnation using multiple signals to force frequent restarts."""
    
    # Track fitness history for window-based checks
    if not hasattr(self, '_fitness_window'):
        self._fitness_window = []
    self._fitness_window.append(self.best_fitness)
    
    # Keep window bounded
    max_window = 200
    if len(self._fitness_window) > max_window:
        self._fitness_window = self._fitness_window[-max_window:]
    
    # 1. Very short absolute stagnation limit — restart quickly
    stag_limit = 5 + int(10 * self.dim / self.pop_size)
    if self.stagnation_counter > stag_limit:
        self._fitness_window = []
        return True
    
    # 2. Sigma collapsed — local search exhausted
    if self.sigma < 1e-16:
        self._fitness_window = []
        return True
    
    # 3. Sigma exploded — search is diverging
    if self.sigma > 1e5:
        self._fitness_window = []
        return True
    
    # 4. Condition number of C is too large
    if np.max(self.D) / max(np.min(self.D), 1e-30) > 1e6:
        self._fitness_window = []
        return True
    
    # 5. Sliding window: check if relative improvement over last N generations is negligible
    window_size = min(15 + self.dim, len(self._fitness_window))
    if window_size >= 10:
        old_f = self._fitness_window[-window_size]
        new_f = self._fitness_window[-1]
        # If both are finite, check relative improvement
        if np.isfinite(old_f) and np.isfinite(new_f):
            denom = max(abs(old_f), 1e-30)
            rel_improvement = (old_f - new_f) / denom
            # If less than tiny relative improvement, restart
            if rel_improvement < 1e-12:
                self._fitness_window = []
                return True
    
    # 6. Flat fitness landscape: if best_fitness is still very large after some generations,
    #    restart sooner to try different basin
    if self.generation > 20 and self.best_fitness > 1e3:
        if self.stagnation_counter > 3:
            self._fitness_window = []
            return True
    
    # 7. Medium fitness but stalled — also restart faster
    if self.generation > 30 and self.best_fitness > 1.0:
        if self.stagnation_counter > max(5, stag_limit // 2):
            self._fitness_window = []
            return True
    
    # 8. Check if eigenvalues of C have any NaN/Inf (numerical breakdown)
    if not np.all(np.isfinite(self.D)):
        self._fitness_window = []
        return True
    
    return False
```