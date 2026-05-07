**Idea: Reflective Boundary Handling with Stochastic Perturbation**

Instead of hard clipping, reflect out-of-bounds solutions back into the feasible region with momentum-preserving reflection and small stochastic perturbations. This provides smoother gradients near boundaries and helps escape local optima that cluster near bounds on the worst unsolved tasks (errors ~1e+1 to 8e+1).

```python
def _clip_to_bounds(self, x):
    """Reflective boundary handling with stochastic perturbation."""
    x = np.asarray(x, dtype=np.float64)
    result = np.empty_like(x)
    
    for i in range(self.dim):
        val = x[i]
        lb_i = self.lb[i]
        ub_i = self.ub[i]
        
        if val < lb_i:
            excess = lb_i - val
            result[i] = lb_i + excess + np.random.uniform(0, 1e-6) * (ub_i - lb_i)
        elif val > ub_i:
            excess = val - ub_i
            result[i] = ub_i - excess - np.random.uniform(0, 1e-6) * (ub_i - lb_i)
        else:
            result[i] = val
    
    result = np.clip(result, self.lb, self.ub)
    return result
```