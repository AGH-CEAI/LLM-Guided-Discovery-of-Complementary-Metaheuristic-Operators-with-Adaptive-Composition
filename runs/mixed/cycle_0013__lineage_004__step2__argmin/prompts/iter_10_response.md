**Idea: Diversity-Balanced Argmin with Confidence-Based Randomization**

This variant addresses the worst unsolved tasks (12, 8, 21, 14, 23, 20) which show errors stuck at massive values (10³-10⁰), indicating the selection mechanism is repeatedly choosing suboptimal solutions and driving the population toward local optima. The core problem: `argmin` always returns the same deterministic index, causing premature convergence on deceptive landscapes. This approach introduces confidence-weighted randomization—when the population is clustered (low diversity), it deliberately selects a random index to escape local traps and explore new regions.

```python
def _argmin(self, arr):
    # Convert to numpy array for vectorized operations
    arr = np.asarray(arr)
    
    # Compute population statistics for adaptive selection
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    arr_mean = np.mean(arr)
    
    # Confidence score: how well-separated is the best from the rest?
    spread = arr_max - arr_min + 1e-15
    relative_gap = (arr_mean - arr_min) / spread
    
    # High relative_gap (>0.5) means best is significantly better than average
    # Low relative_gap (<0.3) means population is clustered near the best
    # Random selection probability inversely proportional to relative_gap
    random_prob = max(0.0, min(0.7, 0.5 - relative_gap))
    
    # Use random selection when population is clustered (escapes local traps)
    if np.random.random() < random_prob:
        return int(np.random.randint(len(arr)))
    
    # Otherwise, return the minimum index (standard exploitation)
    return int(np.argmin(arr))
```