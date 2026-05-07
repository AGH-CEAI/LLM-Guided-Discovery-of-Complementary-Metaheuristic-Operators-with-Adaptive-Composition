**Idea: Random Coordinate Transformations**
Apply random coordinate transformations (swap, sign-flip, mix) with moderate probability to help particles escape local optima and explore different basins, directly targeting the stagnation causing 10^2-10^3 errors on worst tasks.

```python
def _eval_wrapper(self, x):
    """Apply random coordinate transformations to help escape local optima."""
    x_transformed = x.copy()
    if np.random.random() < 0.3:
        if self.dim >= 2 and np.random.random() < 0.5:
            i, j = np.random.choice(self.dim, 2, replace=False)
            x_transformed[i], x_transformed[j] = x_transformed[j], x_transformed[i]
        elif np.random.random() < 0.5:
            mask = np.random.random(self.dim) < 0.3
            x_transformed[mask] *= -1.0
        else:
            i, j = np.random.choice(self.dim, 2, replace=False)
            alpha = np.random.uniform(-0.3, 0.3)
            x_transformed[i] = x_transformed[i] + alpha * (x_transformed[j] - x_transformed[i])
    x_clipped = np.clip(x_transformed, self.lower, self.upper)
    return self._user_func(x_clipped.reshape(1, -1))[0]
```