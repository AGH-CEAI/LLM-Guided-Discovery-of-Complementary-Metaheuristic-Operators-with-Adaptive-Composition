Looking at the current `_check_stagnation` implementation and the task error patterns, I need to understand why the worst tasks (17, 16, 20, etc. with errors 10-90) are failing to converge.

**Key Analysis:**
- Current threshold: `self.f_opt < self.f_opt_prev - 1e-12` (absolute, very tight)
- On Task 17 (error ~90): An improvement of 1e-12 is negligible relative to the fitness scale. The algorithm perceives stagnation even when making meaningful progress (e.g., going from 90.0 to 89.999), triggering premature restarts.
- On Task 5 (solved, error 1e-08): The tight threshold works because fitness values are small.
- The algorithm is restarting too aggressively on difficult tasks, never allowing sustained convergence.

**Proposed Solution:**
A **relative improvement threshold** that adapts to the current fitness magnitude, combined with a **population diversity signal** to detect true stagnation vs. mere fitness plateaus with ongoing exploration.

**Idea: Adaptive Multi-Signal Stagnation Detection**
Scale the improvement threshold by `max(1.0, |f_opt|)` so that on high-error tasks, meaningful progress isn't dismissed as stagnation. Also check population diversity to avoid restarting when the algorithm is still exploring effectively.
```python
def _check_stagnation(self):
    """Track stagnation using adaptive relative threshold and diversity signal."""
    # Adaptive threshold: scale to problem difficulty
    # On high-error tasks (error ~90), require proportionally larger improvements
    # On low-error tasks (error ~1e-08), use tighter threshold
    scale = max(1.0, abs(self.f_opt))
    threshold = max(1e-12, 1e-6 * scale)
    
    # Primary signal: best fitness improvement
    improved = self.f_opt < self.f_opt_prev - threshold
    
    # Secondary signal: population diversity (avoid restarting when still exploring)
    if hasattr(self, 'population') and len(self.population) > 1:
        pop_diffs = self.population - self.mean
        max_sq_dist = np.max(np.sum(pop_diffs ** 2, axis=1))
        expected_max_sq = self.dim * ((self.ub[0] - self.lb[0]) / 3.0) ** 2
        spread = np.sqrt(max_sq_dist) / np.sqrt(max(expected_max_sq, 1e-10))
        # Still exploring if population has meaningful spread
        still_exploring = spread > 0.05
    else:
        still_exploring = False
    
    # Reset stagnation if improved OR still exploring (don't restart during exploration)
    if improved or still_exploring:
        self.stagnation_counter = 0
    else:
        self.stagnation_counter += 1
    
    self.f_opt_prev = self.f_opt
```