Looking at the unsolved tasks, the worst ones (Tasks 16, 19, 20, 23, 18, 10, 21, 17) are stuck at errors of 3 to 500 — indicating a fundamental exploration failure where the optimizer gets trapped in local optima. The current sampling method relies solely on the covariance matrix, which can collapse in highly multimodal or deceptive landscapes.

**Idea: Restart-Triggered Diversity Sampling with Heavy-Tailed Perturbation**

The key insight: when stagnation is detected, we need to fundamentally break out of the current distribution by sampling from a much wider, heavy-tailed distribution. I'll use a restart-based exploration strategy that injects diverse solutions from the entire search space and applies larger perturbations when the optimizer is stuck, combined with a t-distribution for heavier tails than the standard Gaussian.

```python
def _sample_population_batch(self):
    """
    Sample a batch of candidates from a mixture of local and diverse exploration.
    When stagnation is detected, inject heavy-tailed solutions across the search space.
    """
    z = np.random.randn(self.pop_size, self.dim)
    
    # Compute stagnation ratio for adaptive diversity injection
    stagnation_threshold = 15 + self.dim
    stagnation_ratio = min(self.stagnation_counter / max(1, stagnation_threshold), 1.0)
    
    # Base scaling: local exploitation
    base_scaling = self.D[np.newaxis, :] * self.sigma
    rotated = z * base_scaling
    population = self.mean[np.newaxis, :] + rotated @ self.B.T
    
    # Inject diverse exploration when stagnant
    if self.stagnation_counter > stagnation_threshold // 2:
        # Compute diversity budget: fraction of population for wide search
        diversity_budget = int(self.pop_size * stagnation_ratio * 0.6)
        diversity_budget = max(1, min(diversity_budget, self.pop_size - 1))
        
        # Generate heavy-tailed samples using t-distribution
        df = max(2.0, 3.0 - stagnation_ratio)  # Lower df = heavier tails
        t_samples = np.random.standard_t(df, size=(diversity_budget, self.dim))
        
        # Scale t-samples to match covariance structure but with amplified variance
        amplify = 3.0 + stagnation_ratio * 7.0
        t_scaled = t_samples * self.D[np.newaxis, :] * self.sigma * amplify
        diverse_pop = self.mean[np.newaxis, :] + t_scaled @ self.B.T
        
        # Also inject completely random solutions for maximum diversity
        random_budget = max(1, diversity_budget // 3)
        random_pop = np.random.uniform(self.lb, self.ub, size=(random_budget, self.dim))
        
        # Combine: place diverse solutions at front (higher selection pressure)
        combined_diverse = np.vstack([diverse_pop, random_pop])[:diversity_budget]
        population[:diversity_budget] = combined_diverse
        
        # Expand step size for the entire batch during exploration phase
        expansion_factor = 1.0 + stagnation_ratio * 2.0
        population = self.mean[np.newaxis, :] + expansion_factor * (population - self.mean[np.newaxis, :])
    
    return population, z
```