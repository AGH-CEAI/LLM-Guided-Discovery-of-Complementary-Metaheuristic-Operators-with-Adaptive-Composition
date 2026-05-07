Looking at the results, I need to analyze what's happening:

1. **Tasks 16, 19, 18, 20, 22, 23** (errors ~10-1100): These are high-dimensional, multimodal problems where the algorithm gets trapped. The current best variants either restart too early (wasting budget) or too late (stuck in local optima).

2. **Tasks 0, 1, 6** (errors ~1e-7): These are nearly solved - need more fine-grained local search time without premature restarts.

3. The key insight: variant_05 does best on many tasks, and the original does best on the near-solved ones. The original has a moderate stagnation limit. variant_05 likely has different restart behavior.

**The core problem**: For hard multimodal tasks, we need AGGRESSIVE restarts with MUCH larger populations to escape deep local optima. For nearly-solved tasks, we need to NEVER restart and let CMA-ES converge. The current stagnation detection doesn't adapt based on progress quality.

My strategy: Use an **adaptive stagnation threshold** that triggers very fast restarts when the algorithm is clearly stuck at poor fitness (allowing many diverse restarts for multimodal tasks), but becomes extremely patient when fitness is already very good (allowing convergence for near-solved tasks). Additionally, detect when the population diversity has collapsed relative to the remaining error magnitude.

**Idea: Error-Proportional Adaptive Patience**
Trigger restarts very quickly when stuck at high error (many restarts for multimodal exploration) but be extremely patient at low error (allow fine convergence).

```python
def _detect_stagnation(self):
    """Adaptive stagnation: fast restarts at high error, patient at low error."""
    # Base stagnation limit scales with how good current best is
    # If best_fitness is large, restart quickly to explore more
    # If best_fitness is tiny, be very patient to allow convergence
    
    if self.best_fitness < 1e-6:
        # Very close to solution - be extremely patient
        stag_limit = 200 + int(100 * self.dim / self.pop_size)
    elif self.best_fitness < 1e-2:
        # Good region - moderate patience
        stag_limit = 50 + int(50 * self.dim / self.pop_size)
    elif self.best_fitness < 1.0:
        # Decent but not great - shorter patience
        stag_limit = 15 + int(20 * self.dim / self.pop_size)
    elif self.best_fitness < 100.0:
        # Poor - restart quickly to try new regions
        stag_limit = 8 + int(10 * self.dim / self.pop_size)
    else:
        # Very poor - restart very aggressively
        stag_limit = 5 + int(5 * self.dim / self.pop_size)
    
    if self.stagnation_counter > stag_limit:
        return True
    
    # Sigma collapse - but only restart if error is still large
    # If error is small and sigma collapsed, that might be fine convergence
    if self.sigma < 1e-16:
        if self.best_fitness > 1e-8:
            return True
        else:
            return False  # Let it stay converged
    
    # Sigma explosion - lost control
    if self.sigma > 1e4:
        return True
    
    # Condition number check - more tolerant when error is small
    if np.min(self.D) > 0:
        cond = np.max(self.D) / np.min(self.D)
        cond_limit = 1e7 if self.best_fitness > 1e-4 else 1e10
        if cond > cond_limit:
            return True
    
    # If we've been running many generations with high error, force restart
    # This catches slow drift that doesn't trigger stagnation counter
    if self.generation > 0 and self.generation % (20 + int(30 * self.dim / self.pop_size)) == 0:
        if self.best_fitness > 10.0:
            # Check recent progress via fitness history
            if hasattr(self, 'best_fitness_history') and len(self.best_fitness_history) > 10:
                recent = self.best_fitness_history[-10:]
                if len(recent) >= 2 and (recent[0] - recent[-1]) / max(abs(recent[0]), 1e-30) < 0.01:
                    return True
    
    # Track fitness history for trend analysis
    if hasattr(self, 'best_fitness_history'):
        if len(self.best_fitness_history) == 0 or self.best_fitness_history[-1] != self.best_fitness:
            self.best_fitness_history.append(self.best_fitness)
    
    return False
```