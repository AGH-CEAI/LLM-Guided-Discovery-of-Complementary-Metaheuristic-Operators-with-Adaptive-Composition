**Idea: Softmax Probabilistic Selection with Temperature Decay**

Replaces deterministic `argmin` with a temperature-scaled softmax that probabilistically selects candidates, introducing controlled exploration even at the selection level. This breaks the pure exploitation pattern that causes the algorithm to get trapped in the worst local optima (Tasks 12, 8 with errors >2000).

```python
def _argmin(self, arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return 0
    if arr.size == 1:
        return int(np.argmin(arr))
    
    # Temperature starts high (exploratory) and decays toward greedy
    # Using generation counter stored on self if available, else default
    gen = getattr(self, '_gen', 0)
    temp = max(0.01, 2.0 / (1.0 + gen * 0.05))
    
    arr_min, arr_max = arr.min(), arr.max()
    arr_range = arr_max - arr_min + 1e-15
    
    # Normalize to [0, 1] where lower fitness = better = higher probability
    normalized = 1.0 - (arr - arr_min) / arr_range
    
    # Softmax with numerical stability
    exp_scores = np.exp((normalized - normalized.max()) / temp)
    probs = exp_scores / (exp_scores.sum() + 1e-15)
    
    # Ensure valid probability distribution
    probs = np.clip(probs, 0, 1)
    probs = probs / (probs.sum() + 1e-15)
    
    return int(np.random.choice(len(arr), p=probs))
```