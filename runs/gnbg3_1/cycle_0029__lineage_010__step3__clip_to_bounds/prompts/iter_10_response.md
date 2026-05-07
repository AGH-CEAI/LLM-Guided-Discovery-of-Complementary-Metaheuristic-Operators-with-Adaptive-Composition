**Idea: Reflection-Based Boundary Handling**

Instead of hard clipping which destroys directional information, use reflection at boundaries to preserve search momentum and enable smoother boundary exploration—critical for the high-error multimodal tasks where solutions get stuck far from the global optimum.

```python
def _clip_to_bounds(self, x):
    """Reflect solutions at boundaries to preserve search direction and momentum."""
    x = np.asarray(x, dtype=np.float64)
    original_shape = x.shape
    x_flat = x.flatten()
    
    overshoot_mask = (x_flat > self.ub) | (x_flat < self.lb)
    if not np.any(overshoot_mask):
        return x
    
    overshoot_upper = x_flat > self.ub
    overshoot_lower = x_flat < self.lb
    
    overshoot_upper_dist = np.where(overshoot_upper, x_flat - self.ub, 0.0)
    overshoot_lower_dist = np.where(overshoot_lower, self.lb - x_flat, 0.0)
    max_overshoot = np.maximum(overshoot_upper_dist, overshoot_lower_dist)
    
    scale = 1.0 - max_overshoot / (self.ub - self.lb + 1e-15)
    scale = np.maximum(scale, 1e-6)
    
    x_flat = np.where(overshoot_upper, self.ub - (x_flat - self.ub) * scale, x_flat)
    x_flat = np.where(overshoot_lower, self.lb + (self.lb - x_flat) * scale, x_flat)
    
    x_flat = np.clip(x_flat, self.lb, self.ub)
    
    return x_flat.reshape(original_shape)
```