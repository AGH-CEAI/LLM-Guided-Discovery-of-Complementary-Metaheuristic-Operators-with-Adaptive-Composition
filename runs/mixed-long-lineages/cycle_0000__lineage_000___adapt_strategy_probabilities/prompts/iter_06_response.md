Looking at the coverage summary, Task 12 is the critical UNCOVERED task (original error=7.319e-05, no variant improves on it). Tasks 3, 12, and 21 are all cases where every variant performs *worse* than the original, suggesting these are tasks where the original strategy distribution is already near-optimal and aggressive adaptation hurts.

**Analysis of the problem:**
- Task 12 has extremely low original error (7.3e-05), meaning the original algorithm is already near the optimum
- Current adaptation uses fitness-weighted selection that can drift away from the original uniform distribution
- All existing variants fail on these "already near-optimal" tasks because they all adapt too aggressively

**Fundamental new approach:**
Instead of always adapting based on success counts, use a **threshold-triggered dual-mode approach**: be conservative (keep original uniform distribution) when best fitness is already very good, and only switch to success-based adaptation when we need to escape local optima. This is fundamentally different from all previous variants which always apply the same adaptation logic.

**Idea: Threshold-Guided Conservative Adaptation**

```python
def _adapt_strategy_probabilities(self):
    # Compute success rates per strategy (avoid division by zero)
    success_rates = np.zeros(self.n_strategies)
    valid_strategies = self.strategy_attempt_count > 0
    success_rates[valid_strategies] = (
        self.strategy_success_count[valid_strategies] / 
        self.strategy_attempt_count[valid_strategies]
    )
    
    # Get current best fitness
    best_fitness = np.min(self.population_fitness) if hasattr(self, 'population_fitness') else np.min(self.archive) if self.archive else 0.0
    
    # Threshold for "already near-optimal" - conservative mode
    # Use a relative threshold: 1.1x the best fitness seen so far
    ref_fitness = getattr(self, 'reference_fitness', 1e-3)
    threshold = ref_fitness * 1.5 if ref_fitness > 0 else 1e-4
    
    # Dual-mode: conservative if already good, adaptive otherwise
    if best_fitness < threshold:
        # Conservative mode: bias toward original uniform distribution
        # This prevents harmful drift on already-converged tasks like #12
        orig_prob = 1.0 / self.n_strategies
        adaptive_prob = success_rates / (np.sum(success_rates) + 1e-10)
        alpha = 0.85  # Strong weight on original distribution
        probs = alpha * orig_prob + (1 - alpha) * adaptive_prob
    else:
        # Adaptive mode: use softmax with temperature on success rates
        # Low temperature = more exploitative, high = more exploratory
        temperature = 0.3
        exp_rates = np.exp(success_rates / (temperature + 1e-10))
        probs = exp_rates / (np.sum(exp_rates) + 1e-10)
    
    # Numerical safety
    probs = np.clip(probs, 1e-10, 1.0)
    probs = probs / (np.sum(probs) + 1e-10)
    
    # Assign strategies
    for i in range(self.NP):
        self.strategy_idx[i] = self.rng.choice(self.n_strategies, p=probs)
```