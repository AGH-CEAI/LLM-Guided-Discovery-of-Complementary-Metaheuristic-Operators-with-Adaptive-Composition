**Idea: Gradient-Descent Escape with Random Restart**
When the wrapper receives a poorly-performing solution, it performs gradient approximation via finite differences and takes a step toward improvement. If that fails, it attempts a random restart. This hybrid local-global search helps escape local optima that trap the PSO.
```python
def _eval_wrapper(self, x):
    """Wrapper with gradient approximation and random restart to escape local optima."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    eps = 1e-5
    f0 = self._user_func(x)
    
    if np.isfinite(f0) and f0 < 1e6:
        return f0
    
    grad = np.zeros_like(x)
    for d in range(x.shape[1]):
        x_plus = x.copy()
        x_plus[:, d] += eps
        f_plus = self._user_func(x_plus)
        grad[:, d] = (f_plus - f0) / (eps + 1e-12)
    
    grad_norm = np.linalg.norm(grad) + 1e-12
    if grad_norm > 1e-6:
        step_size = min(1.0, grad_norm * 0.1)
        x_new = x - grad / grad_norm * step_size
        x_new = np.clip(x_new, self.lower, self.upper)
        f_new = self._user_func(x_new)
        if np.isfinite(f_new) and f_new < f0:
            return f_new
    
    if np.random.random() < 0.3:
        x_random = np.random.uniform(self.lower, self.upper, size=x.shape)
        f_random = self._user_func(x_random)
        if np.isfinite(f_random) and f_random < f0:
            return f_random
    
    x_clipped = np.clip(x, self.lower, self.upper)
    return self._user_func(x_clipped)
```