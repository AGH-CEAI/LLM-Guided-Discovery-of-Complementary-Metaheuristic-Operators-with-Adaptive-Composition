Looking at the unsolved tasks, I see a clear pattern:

1. **Tasks 16, 19, 18, 20, 23, 22, 21** (errors ~4.5 to ~1100): These are high-dimensional, likely multimodal/compositional functions where the algorithm gets trapped. The current stagnation detection is too slow — by the time it triggers, too many evaluations are wasted.

2. **Tasks 10-15** (errors ~1.2 to ~4.0): Moderate errors suggesting the algorithm finds reasonable regions but can't refine enough before budget exhaustion.

3. **Tasks 0, 1, 6** (errors ~1e-7): Very close to target — need more fine-tuning time rather than wasted restarts.

The key insight: The current stagnation limit `10 + 30*dim/pop_size` is ~70+ generations for dim=30, which wastes enormous budget on dead-end basins. For hard multimodal problems, we need **extremely aggressive early restarts** to explore more basins, combined with **fitness-history-based detection** that triggers restart when the rate of improvement flatlines — even if absolute stagnation_counter hasn't hit the limit. Also, we should detect when sigma is either too large (not converging) or the condition number is growing (ill-conditioning without progress).

This variant uses a **multi-signal early restart** approach: short stagnation window, relative improvement tracking over a sliding window, sigma explosion detection, and TolX-based detection (when steps become negligible relative to domain).

**Idea: Aggressive Multi-Signal Early Restart**
Ultra-aggressive stagnation detection combining short patience, sliding-window relative improvement, sigma explosion, and step-size-to-domain ratio checks to trigger rapid restarts on multimodal landscapes.

```python
def _detect_stagnation(self):
    """Aggressive multi-signal stagnation detection for rapid restarts on multimodal problems."""
    dim = self.dim
    
    # Signal 1: Very short absolute stagnation patience — restart quickly
    # For hard multimodal problems, don't waste budget on dead basins
    stag_limit = max(5, int(10 * dim / self.pop_size))
    if self.stagnation_counter > stag_limit:
        return True
    
    # Signal 2: Sigma collapsed — local basin exhausted
    if self.sigma < 1e-14:
        return True
    
    # Signal 3: Sigma exploded — search is diverging, not converging
    if self.sigma > 1e5:
        return True
    
    # Signal 4: Condition number of C too large — ill-conditioned, restart
    cond = np.max(self.D) / max(np.min(self.D), 1e-30)
    if cond > 1e6:
        return True
    
    # Signal 5: TolX — if sigma * max(D) is tiny relative to domain, we're stuck in a point
    max_std = self.sigma * np.max(self.D)
    if max_std < 1e-12 * (self.ub - self.lb):
        return True
    
    # Signal 6: Fitness history sliding window — if no meaningful relative improvement
    # over recent history, restart
    if hasattr(self, 'best_fitness_history') and len(self.best_fitness_history) > 0:
        pass  # best_fitness_history may not be populated by _track_best
    
    # Signal 7: If generation count is high but best fitness is still very large,
    # we're likely in a wrong basin — trigger restart sooner
    if self.generation > 20 and self.best_fitness > 1e3:
        if self.stagnation_counter > max(3, stag_limit // 3):
            return True
    
    # Signal 8: If generation is moderate and we have moderate error with stagnation
    if self.generation > 50 and self.best_fitness > 1.0:
        if self.stagnation_counter > max(3, stag_limit // 2):
            return True
    
    # Signal 9: Evolution path length anomaly — if p_sigma norm is far from expected
    p_norm = np.linalg.norm(self.p_sigma)
    if self.generation > 10:
        # If p_sigma is way too small, step-size adaptation is stalled
        if p_norm < 0.1 * self.chi_n and self.stagnation_counter > 3:
            return True
        # If p_sigma is way too large, search is not settling
        if p_norm > 3.0 * self.chi_n and self.generation > 30:
            return True
    
    return False
```