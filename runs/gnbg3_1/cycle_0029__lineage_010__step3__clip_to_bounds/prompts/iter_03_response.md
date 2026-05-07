**Idea: Reflective Bounds**
Instead of hard clipping (which loses directional information), use reflection at boundaries to preserve search momentum and enable escape from deceptive local optima near bounds.

```python
def _clip_to_bounds(self, x):
    """Reflect solution to bounds (preserves search direction vs hard clipping)."""
    x = np.asarray(x, dtype=np.float64)
    lb, ub = self.lb, self.ub
    x_shape = x.shape
    
    # Handle 1D and 2D cases
    if x.ndim == 1:
        for _ in range(10):
            mask_lo = x < lb
            mask_hi = x > ub
            if not np.any(mask_lo) and not np.any(mask_hi):
                break
            x = np.where(mask_lo, 2.0 * lb - x, np.where(mask_hi, 2.0 * ub - x, x))
        return x
    else:
        # 2D case (NP x dim)
        for _ in range(10):
            mask_lo = x < lb
            mask_hi = x > ub
            if not np.any(mask_lo) and not np.any(mask_hi):
                break
            x = np.where(mask_lo, 2.0 * lb - x, np.where(mask_hi, 2.0 * ub - x, x))
        return x
```