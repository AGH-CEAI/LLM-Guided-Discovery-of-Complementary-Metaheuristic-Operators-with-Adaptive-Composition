**Idea: Bootstrap-Confidence Monte Carlo Integration for Strategy Credit**

This approach uses bootstrap resampling to estimate confidence in each strategy's effectiveness, then Monte Carlo integration over random policy weightings to compute robust expected improvement. The update is confidence-weighted to prevent over-updating on noisy estimates.

```python
def _update_strategy_scores(self, strategy_used, improved):
    """
    Bootstrap-based strategy credit assignment with Monte Carlo integration.
    
    Uses bootstrap resampling to estimate confidence in strategy effectiveness,
    then Monte Carlo integration over random policy weightings to compute
    expected improvement under uncertainty.
    """
    n_strategies = len(self.strategy_scores)
    
    # Compute empirical success rate per strategy
    emp_success_rate = np.zeros(n_strategies)
    for s in range(n_strategies):
        mask = strategy_used == s
        if mask.sum() > 0:
            emp_success_rate[s] = improved[mask].mean()
    
    # Bootstrap resampling: estimate variance in success rates via resampling
    n_bootstrap = 50
    bootstrap_success_rates = np.zeros((n_bootstrap, n_strategies))
    
    for b in range(n_bootstrap):
        # Sample indices with replacement (Monte Carlo sampling of population)
        indices = np.random.randint(0, len(improved), len(improved))
        resampled_improved = improved[indices]
        resampled_strategy = strategy_used[indices]
        
        for s in range(n_strategies):
            mask = resampled_strategy == s
            if mask.sum() > 0:
                bootstrap_success_rates[b, s] = resampled_improved[mask].mean()
    
    # Compute bootstrap statistics
    bootstrap_mean = bootstrap_success_rates.mean(axis=0)
    bootstrap_std = bootstrap_success_rates.std(axis=0)
    
    # Confidence weighting: inverse of coefficient of variation
    # Low variance = high confidence = stronger update signal
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = bootstrap_std / (bootstrap_mean + 1e-10)
        confidence = 1.0 / (cv + 1.0)
        confidence = np.where(np.isfinite(confidence), confidence, 0.5)
    
    # Monte Carlo integration over random policy weightings
    # Sample random convex combinations to estimate expected improvement
    n_mc = 20
    mc_expected_improvement = np.zeros(n_strategies)
    
    for mc in range(n_mc):
        # Sample random Dirichlet weights (policy distribution over simplex)
        raw_weights = np.random.rand(n_strategies)
        policy_weights = raw_weights / raw_weights.sum()
        
        # Expected improvement under this random policy
        for s in range(n_strategies):
            mc_expected_improvement[s] += bootstrap_mean[s] * policy_weights[s]
    
    mc_expected_improvement /= n_mc
    
    # Combine estimates: bootstrap mean weighted by confidence + MC integration
    combined_estimate = 0.6 * bootstrap_mean * confidence + 0.4 * mc_expected_improvement
    
    # Normalize to get update direction
    combined_estimate = combined_estimate / (combined_estimate.sum() + 1e-10)
    
    # Compute delta from uniform baseline (each strategy equally likely)
    delta = combined_estimate - 1.0 / n_strategies
    
    # Apply update with momentum
    lr = 0.1
    momentum = 0.3
    
    if not hasattr(self, '_score_momentum'):
        self._score_momentum = np.zeros(n_strategies)
    
    self._score_momentum = momentum * self._score_momentum + (1 - momentum) * delta
    self.strategy_scores += lr * self._score_momentum
    self.strategy_scores = np.clip(self.strategy_scores, 0.1, 10.0)
```