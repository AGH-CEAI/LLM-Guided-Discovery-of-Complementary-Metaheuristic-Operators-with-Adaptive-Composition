**Idea: Entropy-Driven Adaptive Exploration**

A fundamentally different operator selection strategy that uses Shannon entropy of recent success patterns to dynamically balance exploration vs. exploitation. When the population shows signs of stagnation (low entropy = same operators succeeding repeatedly), it forces aggressive diversification toward untested strategies. This is designed to break through the local optima trapping the worst-performing tasks (errors ~1e+1 to 1e+3), where the current UCB approach fails by over-exploiting known-good operators.

```python
def _select_mutation_operator_batch(self):
    """Entropy-driven adaptive operator selection for escaping local optima."""
    n_ops = self.n_strategies
    
    # Initialize tracking if first call
    if not hasattr(self, 'op_success_history') or self.op_success_history is None:
        self.op_success_history = np.zeros((n_ops, 10))
        self.op_history_idx = 0
        self.last_selected = 0
    
    # Record last operator's success for history tracking
    if hasattr(self, 'last_success_rate'):
        self.op_success_history[self.last_selected, self.op_history_idx] = self.last_success_rate
        self.op_history_idx = (self.op_history_idx + 1) % 10
    
    # Compute empirical success rates from history
    success_rates = np.mean(self.op_success_history, axis=1) + 1e-10
    
    # Compute Shannon entropy of success distribution (measures diversity of recent successes)
    p_normalized = success_rates / (np.sum(success_rates) + 1e-10)
    entropy = -np.sum(p_normalized * np.log(p_normalized + 1e-10))
    max_entropy = np.log(n_ops + 1e-10)
    normalized_entropy = entropy / (max_entropy + 1e-10)
    
    # Compute success rate variance (high variance = some ops working, some not)
    success_variance = np.var(success_rates)
    
    # Adaptive exploration factor: low entropy OR high variance → explore more
    # This handles both "stuck in local optima" and "landscape too complex" cases
    explore_factor = max(0.1, 1.0 - normalized_entropy + min(0.4, success_variance * 10))
    exploit_factor = 1.0 - explore_factor
    
    # Base probabilities from success history (exploitation component)
    exploit_probs = success_rates / np.sum(success_rates)
    
    # Uniform exploration probabilities
    explore_probs = np.ones(n_ops) / n_ops
    
    # Mix exploration and exploitation
    mixed_probs = explore_factor * explore_probs + exploit_factor * exploit_probs
    mixed_probs = mixed_probs / np.sum(mixed_probs)
    
    # Ensure numerical robustness
    mixed_probs = np.clip(mixed_probs, 1e-10, 1.0)
    mixed_probs = mixed_probs / np.sum(mixed_probs)
    
    # Select operator using computed probabilities
    selected = np.random.choice(n_ops, p=mixed_probs)
    
    # Force diversity injection every ~20 calls if entropy is low
    if not hasattr(self, 'call_counter'):
        self.call_counter = 0
    self.call_counter += 1
    
    if self.call_counter % 20 == 0 and normalized_entropy < 0.5:
        # Force selection of least-used operator for diversity
        least_used = np.argmin(self.op_success_history.sum(axis=1))
        if np.random.random() < 0.3:  # 30% chance to force
            selected = least_used
    
    self.last_selected = selected
    return selected
```