**Idea: Robust Eval with Bounds and NaN/Inf Handling**

A fundamental improvement: clip inputs to valid bounds and handle non-finite fitness values with a penalty, making the wrapper robust across ill-conditioned or noisy benchmark functions.
```python
def _eval_wrapper(self, x):
    """Robust wrapper with bounds enforcement and NaN/Inf penalty."""
    x = np.atleast_2d(x)
    x_clipped = np.clip(x, self.lower, self.upper)
    result = self._user_func(x_clipped)
    # Penalty for non-finite values (NaN/Inf) to avoid breaking optimization
    result = np.where(np.isfinite(result), result, 1e30)
    return result
```