**Idea: Bimodal Heavy-Tailed Parameter Generation**

The worst unsolved tasks (errors 1e+00 to 2e+02) indicate a complete convergence failure, not a precision issue. The current approach uses narrow adaptive Cauchy/Normal perturbations around memory values, which becomes too conservative and prevents escaping poor local optima. This variant uses a bimodal strategy: one mode for **exploitation** (success-history guided, like current) and a second **exploration mode** with heavy-tailed distributions (Levy flight for F, uniform for CR) to aggressively escape stagnation.

```python
def _generate_parameters_batch(self, memory_f, memory_cr):
    """Generate F and CR using bimodal heavy-tailed distributions to escape local optima."""
    n = self.np_size
    
    # Bimodal selection: 40% exploitation, 60% exploration for better coverage
    # Higher exploration fraction to target tasks stuck at high errors
    is_exploit = np.random.random(n) < 0.4
    
    f_values = np.zeros(n)
    cr_values = np.zeros(n)
    
    # Exploitation mode: success-history guided (similar to current but improved)
    exploit_mask = is_exploit
    if np.any(exploit_mask):
        n_exploit = np.sum(exploit_mask)
        indices = np.random.randint(0, self.memory_size, size=n_exploit)
        mu_f = memory_f[indices]
        mu_cr = memory_cr[indices]
        
        # Smaller scale for exploitation
        generation = getattr(self, 'generation', 0)
        scale_f = max(0.05, 0.15 / (1.0 + generation / 150.0))
        scale_cr = max(0.05, 0.15 / (1.0 + generation / 150.0))
        
        # Truncated normal for F in exploit mode
        while True:
            samples = np.random.normal(mu_f, scale_f)
            valid = (samples > 0) & (samples <= 1.0)
            if np.all(valid):
                break
        f_values[exploit_mask] = samples
        
        # Normal for CR in exploit mode
        cr_values[exploit_mask] = np.clip(np.random.normal(mu_cr, scale_cr), 0.0, 1.0)
    
    # Exploration mode: heavy-tailed distributions for escaping local optima
    explore_mask = ~is_exploit
    if np.any(explore_mask):
        n_explore = np.sum(explore_mask)
        
        # Use exponential distribution for F - heavier tail than Cauchy
        # Sample from Exp(rate) shifted to valid range [0.1, 1.0]
        # Lower rate = heavier tail = larger F values for long jumps
        rate_f = np.random.uniform(2.0, 5.0, size=n_explore)
        exp_f = np.random.exponential(1.0 / rate_f)
        f_values[explore_mask] = np.clip(0.8 * exp_f + 0.2, 0.1, 1.0)
        
        # Bimodal CR: either very low (focused search) or very high (diverse search)
        cr_mode = np.random.random(n_explore)
        cr_values[explore_mask] = np.where(
            cr_mode < 0.5,
            np.clip(np.random.normal(0.1, 0.05, size=n_explore), 0.0, 0.3),  # Low CR
            np.clip(np.random.normal(0.9, 0.05, size=n_explore), 0.7, 1.0)   # High CR
        )
    
    return f_values, cr_values
```