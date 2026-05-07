Looking at the results, I notice that:

1. The **original.py** performs best on most tasks, especially the easier ones (Tasks 0-9) where errors are small but not yet at target.
2. The worst unsolved tasks (16, 19, 23, 18, 20, 10, 13, 22) have errors in the range 1e+1 to 1e+3 — these are likely multimodal, ill-conditioned, or compositional functions where CMA-ES gets trapped in local optima.
3. The original stagnation detector triggers restarts too late for hard multimodal problems (stag_limit ~70 for dim=30) AND too eagerly for problems that need patient local refinement.

**Key insight**: For hard multimodal tasks, we need **much more aggressive restarts** — detect stagnation earlier, restart more frequently, and also detect when we're stuck in a local basin (fitness not improving even relatively). For easier tasks close to convergence, we need to be patient. The solution: use a **fitness-history-based relative improvement check** with a short window, combined with very aggressive condition number and sigma monitoring. Also track the ratio of current fitness to detect when we're clearly stuck at a suboptimal level.

**Idea: Aggressive Multi-Signal Stagnation with Short History Window**
Trigger restarts very aggressively using short-window relative fitness improvement, median fitness stalling, sigma-to-eigenvalue ratio collapse, and tight condition number limits to escape local optima fast on multimodal problems.

```python
def _detect_stagnation(self):
    """Aggressively detect stagnation using multiple short-window signals."""
    # Track fitness history
    if not hasattr(self, '_fitness_history'):
        self._fitness_history = []
    self._fitness_history.append(self.best_fitness)
    
    # Very short stagnation counter limit — restart early and often
    stag_limit = 5 + int(10 * self.dim / self.pop_size)
    if self.stagnation_counter > stag_limit:
        self._fitness_history = []
        return True
    
    # Check sigma collapse
    if self.sigma < 1e-14:
        self._fitness_history = []
        return True
    
    # Tight condition number check
    cond = np.max(self.D) / max(np.min(self.D), 1e-30)
    if cond > 1e5:
        self._fitness_history = []
        return True
    
    # Check if sigma * max(D) is too small relative to domain (stuck in tiny basin)
    effective_range = self.sigma * np.max(self.D)
    if effective_range < 1e-10 and self.best_fitness > 1e-6:
        self._fitness_history = []
        return True
    
    # Short-window relative improvement check
    window = min(20, max(5, int(self.dim / 2)))
    if len(self._fitness_history) >= window:
        old_fit = self._fitness_history[-window]
        new_fit = self._fitness_history[-1]
        # If both are finite and positive, check relative improvement
        if np.isfinite(old_fit) and np.isfinite(new_fit) and old_fit > 0:
            rel_improvement = (old_fit - new_fit) / (abs(old_fit) + 1e-30)
            # If less than 0.1% improvement over the window, restart
            if rel_improvement < 1e-3 and self.generation > window:
                self._fitness_history = []
                return True
    
    # Medium-window check for being stuck at high error
    med_window = min(50, len(self._fitness_history))
    if med_window >= 10:
        recent = self._fitness_history[-med_window:]
        if np.std(recent) / (abs(np.mean(recent)) + 1e-30) < 1e-4 and self.best_fitness > 1.0:
            self._fitness_history = []
            return True
    
    # Prevent history from growing unboundedly
    if len(self._fitness_history) > 200:
        self._fitness_history = self._fitness_history[-100:]
    
    return False
```