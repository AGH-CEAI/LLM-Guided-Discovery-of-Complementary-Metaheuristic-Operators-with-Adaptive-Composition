Looking at the unsolved tasks, the worst failures (Tasks 12, 8 with errors ~3000 and ~2000) suggest the algorithm gets trapped in severe local optima. The current `_argmin` always returns the deterministic minimum, which can cause premature convergence—once a mediocre solution becomes "best," it dominates all future guidance, creating a feedback loop that prevents escape.

**Idea: Rank-Based Probabilistic Selection with Temperature Control**

Instead of always selecting the absolute best, use softmax-style rank-based probabilistic selection. This introduces controlled randomness that helps escape local optima. The temperature adapts based on population diversity—high diversity allows greedy selection; low diversity triggers exploration by flattening selection probabilities.

```python
def _argmin(self, arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0
    if arr.size == 1:
        return int(np.argmin(arr))
    
    # Temperature controls exploration vs exploitation
    # Low temp = greedy (only top candidates), High temp = exploratory (any candidate possible)
    if not hasattr(self, '_argmin_temp'):
        self._argmin_temp = 0.5
    
    # Adapt temperature based on population diversity
    fitness_std = np.std(arr)
    if fitness_std < 1e-6:
        # Low diversity: increase temperature to encourage exploration
        self._argmin_temp = min(3.0, self._argmin_temp * 1.2)
    else:
        # Good diversity: can afford greedy selection
        self._argmin_temp = max(0.1, self._argmin_temp * 0.95)
    
    # Rank-based selection: ranks 0=best, n-1=worst
    ranks = np.argsort(np.argsort(arr))
    n = len(arr)
    
    # Exponential weights favoring better ranks, modulated by temperature
    weights = np.exp(-ranks / self._argmin_temp)
    probs = weights / np.sum(weights)
    
    # Probabilistic selection instead of deterministic argmin
    selected_idx = np.random.choice(n, p=probs)
    return int(selected_idx)
```

This fundamentally differs from all prior variants by replacing deterministic minimum selection with stochastic rank-based selection. When trapped in local optima (low diversity), the temperature rises, making selection more exploratory and increasing the chance of promoting a different candidate that could lead the population out of the trap.