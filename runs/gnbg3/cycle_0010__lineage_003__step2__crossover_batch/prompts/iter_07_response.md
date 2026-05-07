**Idea: LHS-Adaptive CR Crossover**
Use Latin Hypercube Sampling (LHS) to select dimensions for mutation injection, combined with Monte Carlo sampling of CR from a Beta distribution that adapts based on historical crossover success. This ensures diverse exploration of crossover patterns while maintaining principled stochasticity.

```python
def _crossover_batch(self, population, mutants):
    """LHS-guided binomial crossover with adaptive Monte Carlo CR sampling."""
    NP, dim = population.shape
    
    # Initialize or update adaptive CR parameters
    if not hasattr(self, '_cr_success_history'):
        self._cr_success_history = []
        self._cr_alpha = 2.0  # Beta distribution shape param
        self._cr_beta = 5.0
    
    # Monte Carlo: sample CR from Beta distribution per individual
    # Adapt Beta parameters based on recent success
    if len(self._cr_success_history) >= 10:
        recent_success = np.mean(self._cr_success_history[-10:])
        # Higher success -> shift distribution toward exploitation (higher mean CR)
        # Lower success -> shift toward exploration (lower mean CR)
        target_mean = 0.5 + 0.3 * (recent_success - 0.1) / 0.9
        target_mean = np.clip(target_mean, 0.1, 0.9)
        # Solve for Beta params with this mean and fixed variance
        self._cr_alpha = target_mean * 10
        self._cr_beta = (1 - target_mean) * 10
    
    # Sample CR per individual from Beta distribution
    CR_samples = np.random.beta(max(self._cr_alpha, 0.1), max(self._cr_beta, 0.1), size=NP)
    CR_samples = np.clip(CR_samples, 0.1, 0.95)
    
    # Generate LHS points for dimension selection (ensures good coverage)
    # LHS: divide [0,1] into NP strata, sample one point per stratum
    lhs_base = (np.arange(NP) + np.random.rand(NP)) / NP
    np.random.shuffle(lhs_base)
    
    trials = np.empty_like(population)
    
    for i in range(NP):
        # LHS-guided dimension selection: use stratified random sampling
        # Each dimension has a "stratum" assignment to ensure coverage
        j_rand = np.random.randint(dim)
        
        # Binomial crossover with LHS-guided random dimension
        cr = CR_samples[i]
        mask = np.random.rand(dim) < cr
        
        # Ensure at least one dimension from mutant (LHS ensures diversity)
        if not mask.any():
            mask[j_rand] = True
        
        trials[i] = np.where(mask, mutants[i], population[i])
    
    # Track CR success for adaptation (use diversity as proxy signal)
    # High diversity in trial vectors suggests good exploration
    trial_diversity = np.std(trials, axis=0).mean()
    pop_diversity = np.std(population, axis=0).mean()
    cr_success = trial_diversity / (pop_diversity + 1e-10) if pop_diversity > 1e-10 else 0.5
    cr_success = np.clip(cr_success, 0.0, 1.0)
    self._cr_success_history.append(cr_success)
    
    # Keep history bounded
    if len(self._cr_success_history) > 50:
        self._cr_success_history = self._cr_success_history[-50:]
    
    return trials
```