Looking at the results, I notice that:

1. The **worst unsolved tasks** (16, 19, 23, 18, 20, 10, 13, 22) have errors in the range 1e+1 to 1e+3, suggesting the algorithm gets trapped in local optima on multimodal/complex landscapes and doesn't restart aggressively enough or explore broadly enough.

2. The **original** tends to beat the variants on most tasks, suggesting the current stagnation detection is roughly okay but too conservative — it allows too many wasted evaluations before restarting.

3. **variant_02** catastrophically failed on tasks 5, 6, 16, 17 (likely restarted too aggressively or in wrong ways), while **variant_01** helped on some tasks (11, 12, 14, 15, 23).

The key insight: for hard multimodal problems, we need **much more frequent restarts** to explore more basins, combined with **fitness-history-based detection** that catches not just zero improvement but also negligible improvement (plateau walking). The current approach only checks `stagnation_counter` (no improvement at all) and condition number — it misses cases where fitness improves by tiny amounts but is stuck on a plateau far from the optimum.

**Idea: Aggressive Multi-Signal Stagnation with Relative Progress Tracking**
Detect stagnation using relative fitness progress over a sliding window, median fitness flatness, extremely tight restart triggers, and generation-budget-proportional limits to maximize basin coverage.

```python
def _detect_stagnation(self):
    """Aggressive stagnation detection using multiple signals to trigger frequent restarts."""
    # Very short patience for no-improvement stagnation
    stag_limit = max(5, int(10 * self.dim / self.pop_size))
    if self.stagnation_counter > stag_limit:
        return True
    
    # Check if sigma collapsed
    if self.sigma < 1e-14:
        return True
    
    # Check if sigma exploded (lost control)
    if self.sigma > 1e5:
        return True
    
    # Check condition number of C (degenerate)
    cond = np.max(self.D) / max(np.min(self.D), 1e-30)
    if cond > 1e6:
        return True
    
    # Track fitness history for relative progress detection
    if not hasattr(self, '_fitness_window'):
        self._fitness_window = []
    
    self._fitness_window.append(self.best_fitness)
    
    # Keep a rolling window
    window_size = max(20, int(15 * self.dim / self.pop_size))
    if len(self._fitness_window) > window_size * 2:
        self._fitness_window = self._fitness_window[-window_size * 2:]
    
    # Check relative progress over the window
    if len(self._fitness_window) >= window_size:
        old_fitness = self._fitness_window[-window_size]
        new_fitness = self._fitness_window[-1]
        
        # If fitness is still large (>1.0), demand significant relative improvement
        if abs(old_fitness) > 1e-10:
            rel_improvement = (old_fitness - new_fitness) / (abs(old_fitness) + 1e-30)
            # If less than 0.1% relative improvement over the window, restart
            if rel_improvement < 1e-3:
                self._fitness_window = []
                return True
        else:
            # For small fitness values, check absolute improvement
            abs_improvement = old_fitness - new_fitness
            if abs_improvement < 1e-12:
                self._fitness_window = []
                return True
    
    # Generation-based restart: force restart every N generations to avoid
    # spending too long in any single basin on hard multimodal problems
    max_gens_per_run = max(50, int(200 * self.dim / self.pop_size))
    if self.generation > 0 and self.generation % max_gens_per_run == 0:
        # Only force restart if fitness is still large (not converging well)
        if self.best_fitness > 1e-6:
            self._fitness_window = []
            return True
    
    # Check if p_sigma norm is abnormal (too small = no progress, too large = diverging)
    ps_norm = np.linalg.norm(self.p_sigma)
    if ps_norm < 0.1 * self.chi_n and self.generation > 10:
        return True
    if ps_norm > 3.0 * self.chi_n and self.generation > 10:
        return True
    
    return False
```