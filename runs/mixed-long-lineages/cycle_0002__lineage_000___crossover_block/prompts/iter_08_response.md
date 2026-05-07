**Idea: Ultra-Conservative Single-Dimension Crossover**

A minimally-disruptive crossover designed for sensitive tasks where current operators degrade performance. Uses extremely low CR (0.02-0.08) with single-dimension perturbation and parent-preserving fallback.

```python
def _crossover_block(self, population, trial_population):
    """Op 5: variant_11 - Ultra-conservative crossover for sensitive tasks."""
    n_trials = self.NP
    dim = self.dim
    
    # Ultra-low CR to preserve parent information on sensitive landscapes
    cr_batch = np.clip(self.CR.copy() * 0.15, 0.02, 0.08)
    
    # Single-dimension crossover mask
    cross_mask = self.rng.uniform(size=(n_trials, dim)) < cr_batch[:, np.newaxis]
    
    # Ensure at least one dimension from trial (or fall back to parent-preserving)
    j_rand = self.rng.integers(0, dim, size=n_trials)
    for i in range(n_trials):
        if not cross_mask[i, :].any():
            # No crossover: preserve parent entirely
            cross_mask[i, j_rand[i]] = True
    
    # Blend: small weight to trial vector to minimize disruption
    blend_weight = 0.1
    blended = (
        (1.0 - blend_weight) * population[:n_trials] +
        blend_weight * trial_population
    )
    
    # Apply crossover: mostly parent, minimally perturbed trial
    offspring = np.where(cross_mask, blended, population[:n_trials])
    
    # Clip to bounds
    offspring = np.clip(offspring, self.lower, self.upper)
    return offspring
```