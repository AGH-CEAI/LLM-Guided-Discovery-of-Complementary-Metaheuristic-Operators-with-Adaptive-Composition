**Idea: Thompson Sampling F/Cr Adaptation**
Use Thompson sampling with Beta posterior distributions over F and Cr success to make stochastic parameter adaptation decisions, providing principled exploration of the control parameter space.
```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using Thompson sampling with Beta posterior estimation."""
    np = len(improved_mask)
    n_success = np.sum(improved_mask)
    success_rate = n_success / np if np > 0 else 0.5
    
    # Track per-parameter effectiveness via weighted sampling
    self.F_history.append((success_rate, F_used, 1.0 / (np.abs(F_used - 0.5) + 0.1)))
    self.Cr_history.append((success_rate, Cr_used, 1.0 / (np.abs(Cr_used - 0.5) + 0.1)))
    
    # Maintain bounded history
    max_history = self.adaptation_window * 3
    if len(self.F_history) > max_history:
        self.F_history = self.F_history[-max_history:]
    if len(self.Cr_history) > max_history:
        self.Cr_history = self.Cr_history[-max_history:]
    
    min_samples = 8
    if len(self.F_history) < min_samples:
        return
    
    # Thompson sampling: model success probability per parameter as Beta posterior
    # Beta(alpha, beta) where alpha = successes + 1, beta = failures + 1
    
    # Sample candidate F values weighted by historical performance
    n_samples = 50
    f_candidates = np.zeros(n_samples)
    
    # Weight recent entries more heavily
    weights_f = np.array([w for _, _, w in self.F_history])
    weights_f = weights_f / (weights_f.sum() + 1e-30)
    
    for i in range(n_samples):
        # Bootstrap sample from history with replacement
        bootstrap_idx = np.random.choice(len(self.F_history), size=min(len(self.F_history), 20), replace=True, p=weights_f)
        bootstrap_rates = np.array([self.F_history[j][0] for j in bootstrap_idx])
        
        # Thompson sample from Beta distribution fitted to bootstrap
        alpha = np.mean(bootstrap_rates) * 10 + 1
        beta = (1 - np.mean(bootstrap_rates)) * 10 + 1
        f_candidates[i] = np.clip(np.random.beta(alpha, beta), 0.1, 1.5)
    
    # Select F using optimistic sampling (take upper quantile for exploration)
    f_quantile = np.percentile(f_candidates, 75 + 10 * np.random.rand())
    self.F = np.clip(f_quantile, 0.1, 1.5)
    
    # Sample candidate Cr values similarly
    cr_candidates = np.zeros(n_samples)
    weights_cr = np.array([w for _, _, w in self.Cr_history])
    weights_cr = weights_cr / (weights_cr.sum() + 1e-30)
    
    for i in range(n_samples):
        bootstrap_idx = np.random.choice(len(self.Cr_history), size=min(len(self.Cr_history), 20), replace=True, p=weights_cr)
        bootstrap_rates = np.array([self.Cr_history[j][0] for j in bootstrap_idx])
        
        alpha = np.mean(bootstrap_rates) * 10 + 1
        beta = (1 - np.mean(bootstrap_rates)) * 10 + 1
        cr_candidates[i] = np.clip(np.random.beta(alpha, beta), 0.1, 0.9)
    
    cr_quantile = np.percentile(cr_candidates, 75 + 10 * np.random.rand())
    self.Cr = np.clip(cr_quantile, 0.1, 0.9)
```