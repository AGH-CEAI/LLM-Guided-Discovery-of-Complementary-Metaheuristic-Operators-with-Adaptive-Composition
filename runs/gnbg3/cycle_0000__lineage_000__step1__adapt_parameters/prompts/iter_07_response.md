**Idea: Bootstrap CI Parameter Sampler**
Use bootstrap resampling of historical improvement rates to construct confidence intervals, then sample F and CR stochastically from within those intervals. This provides statistically grounded, fundamentally stochastic parameter adaptation.

```python
def _adapt_parameters(self, improvement_rate):
    """Adapt F and CR using bootstrap confidence intervals of improvement rates.
    Category G: Stochastic sampling-based approach using bootstrap estimation."""
    
    # Maintain sliding window of improvement rates
    if not hasattr(self, '_improvement_history'):
        self._improvement_history = []
    
    self._improvement_history.append(improvement_rate)
    if len(self._improvement_history) > 50:
        self._improvement_history.pop(0)
    
    n_history = len(self._improvement_history)
    if n_history < 5:
        # Not enough data: use random walk with small variance
        self.F = np.clip(self.F + np.random.randn() * 0.05, 0.3, 1.5)
        self.CR = np.clip(self.CR + np.random.randn() * 0.03, 0.3, 0.99)
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
        return
    
    # Bootstrap resampling: fundamentally stochastic computation
    history_array = np.array(self._improvement_history)
    n_bootstrap = 100  # Monte Carlo samples
    
    # Bootstrap means: sample with replacement, compute mean for each
    bootstrap_indices = np.random.randint(0, n_history, size=(n_bootstrap, n_history))
    bootstrap_means = history_array[bootstrap_indices].mean(axis=1)
    
    # Construct bootstrap confidence interval
    ci_lower = np.percentile(bootstrap_means, 10)
    ci_upper = np.percentile(bootstrap_means, 90)
    ci_width = max(ci_upper - ci_lower, 0.01)
    
    # Sample F and CR stochastically within confidence interval
    # Higher improvement rate -> scale up F (more exploration) and CR (more recombination)
    baseline_F = 0.6
    baseline_CR = 0.85
    
    # Map CI position to parameter adjustment
    norm_rate = np.clip((ci_lower + ci_width * np.random.rand()) / max(ci_upper, 0.1), 0, 1)
    
    # F: lower improvement -> smaller F (exploit), higher improvement -> larger F (explore)
    target_F = baseline_F * (0.5 + 1.5 * norm_rate)
    # CR: inverse relationship with improvement rate
    target_CR = baseline_CR * (1.3 - 0.5 * norm_rate)
    
    # Stochastic smoothing: blend with momentum
    alpha = 0.3
    self.F = np.clip(alpha * target_F + (1 - alpha) * self.F + np.random.randn() * 0.05, 0.3, 1.5)
    self.CR = np.clip(alpha * target_CR + (1 - alpha) * self.CR + np.random.randn() * 0.03, 0.3, 0.99)
    
    # Record history
    self.F_history.append(self.F)
    self.CR_history.append(self.CR)
    
    # Keep history bounded
    if len(self.F_history) > 100:
        self.F_history = self.F_history[-100:]
        self.CR_history = self.CR_history[-100:]
```