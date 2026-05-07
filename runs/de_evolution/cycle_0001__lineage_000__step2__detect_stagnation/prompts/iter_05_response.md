Looking at the results, I notice that:

1. The **original.py** performs best on most tasks, suggesting its stagnation detection and restart cycle is reasonable but needs tuning.
2. The worst unsolved tasks (16, 19, 23, 18, 20, etc.) have errors stuck at 10^1 to 10^3, suggesting the algorithm gets trapped in local optima on multimodal/compositional functions and doesn't restart aggressively enough or with enough diversity.
3. Tasks close to solved (0, 1, 6) need more patience for local refinement — the algorithm restarts too early, losing precision.

The key insight: we need a **two-regime** stagnation detector. For high-dimensional multimodal functions (where error is large), we should restart **much more frequently** to explore more basins. But when we're already near a good solution (low error), we should be **very patient** and let the CMA-ES refine. Current variants use a fixed stagnation limit that's either too long for exploration or too short for refinement.

Additionally, I'll add detection of **fitness flatness** (many top individuals having nearly identical fitness, indicating a plateau) and **sigma explosion** (sigma growing too large, indicating the algorithm is lost), which are failure modes not currently handled.

**Idea: Adaptive Two-Regime Stagnation with Fitness Plateau and Sigma Explosion Detection**
Restart quickly when stuck at high error (exploration mode) but be patient at low error (refinement mode); also detect fitness plateaus and sigma explosion.

```python
def _detect_stagnation(self):
    """Adaptive stagnation detection: aggressive restarts when error is large, patient when close to optimum."""
    
    # Track fitness history for plateau detection
    if not hasattr(self, '_fitness_history'):
        self._fitness_history = []
    self._fitness_history.append(self.best_fitness)
    
    # Keep history bounded
    max_hist = 200
    if len(self._fitness_history) > max_hist:
        self._fitness_history = self._fitness_history[-max_hist:]
    
    # Sigma collapsed — need restart
    if self.sigma < 1e-18:
        return True
    
    # Sigma exploded — algorithm is lost
    if self.sigma > 1e5:
        return True
    
    # Condition number too large — C is degenerate
    cond = np.max(self.D) / max(np.min(self.D), 1e-30)
    if cond > 1e8:
        return True
    
    # Two-regime stagnation limit based on current best fitness
    if self.best_fitness < 1e-4:
        # Near optimum: be very patient, allow fine-tuning
        stag_limit = 50 + int(80 * self.dim / self.pop_size)
    elif self.best_fitness < 1.0:
        # Moderate: medium patience
        stag_limit = 20 + int(40 * self.dim / self.pop_size)
    else:
        # Far from optimum: restart aggressively to explore new basins
        stag_limit = 5 + int(15 * self.dim / self.pop_size)
    
    if self.stagnation_counter > stag_limit:
        return True
    
    # Fitness plateau detection: if relative improvement over last N generations is tiny
    window = min(30, len(self._fitness_history))
    if window >= 10:
        old_fit = self._fitness_history[-window]
        new_fit = self._fitness_history[-1]
        if old_fit > 0 and np.isfinite(old_fit) and np.isfinite(new_fit):
            rel_improvement = (old_fit - new_fit) / (abs(old_fit) + 1e-30)
            # If essentially no relative improvement over the window
            if rel_improvement < 1e-12 and self.stagnation_counter > stag_limit // 2:
                return True
    
    # Detect if sigma * max(D) is tiny relative to domain — search volume collapsed
    max_step = self.sigma * np.max(self.D)
    if max_step < 1e-15 * (self.ub - self.lb):
        return True
    
    # Detect if sigma * min(D) is tiny — one direction collapsed while others didn't
    min_step = self.sigma * np.min(self.D)
    if min_step < 1e-20 and cond > 1e4:
        return True
    
    return False
```