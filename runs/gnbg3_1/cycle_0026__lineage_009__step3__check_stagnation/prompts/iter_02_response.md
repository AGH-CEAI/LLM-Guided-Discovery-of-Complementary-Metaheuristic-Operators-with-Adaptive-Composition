Looking at the problem, I see:
- 23 of 24 tasks are unsolved, with the worst tasks (17, 16, 11, 10, etc.) having errors in the **10-500 range**
- The current `_check_stagnation` uses a **fixed threshold of 1e-12**, which is completely inadequate for high-error tasks
- For Task 17 (error ~538), a 1e-12 improvement is negligible relative to the problem scale, causing **false non-stagnation** signals
- This means the algorithm never recognizes it's stuck on hard tasks and fails to trigger necessary restarts

The fix: use a **problem-scale-adaptive threshold** that scales with `|f_opt|`, plus a larger stagnation allowance for hard tasks.

**Idea: Scale-Adaptive Stagnation Detection**
Uses relative improvement threshold (`1e-6 * |f_opt|`) so hard tasks (error ~100+) properly detect stagnation, combined with problem-difficulty-aware restart thresholds.
```python
def _check_stagnation(self):
    """Track stagnation counter with scale-adaptive threshold.
    
    Key insight: The original fixed threshold (1e-12) is orders of magnitude
    too small for high-error tasks (error ~10-500). For Task 17 (error ~538),
    even a 0.001 improvement is meaningful, but 1e-12 is numerically invisible.
    
    This version uses: threshold = max(1e-6 * |f_opt|, 1e-10)
    - High-error tasks (|f_opt|~100): threshold ~1e-4 → proper stagnation detection
    - Low-error tasks (|f_opt|~1e-8): threshold ~1e-10 → fine-grained detection
    """
    improvement = self.f_opt_prev - self.f_opt
    
    # Adaptive threshold: scale with problem difficulty
    base_scale = max(abs(self.f_opt), 1.0)
    rel_threshold = 1e-6 * base_scale
    threshold = max(rel_threshold, 1e-10)
    
    if improvement > threshold:
        self.stagnation_counter = 0
    else:
        self.stagnation_counter += 1
    
    # Scale max_stagnation with problem difficulty for harder tasks
    if abs(self.f_opt) > 10.0:
        self.max_stagnation = 150 + self.dim * 5
    elif abs(self.f_opt) > 1.0:
        self.max_stagnation = 100 + self.dim * 4
    else:
        self.max_stagnation = 50 + self.dim * 3
    
    self.f_opt_prev = self.f_opt
```