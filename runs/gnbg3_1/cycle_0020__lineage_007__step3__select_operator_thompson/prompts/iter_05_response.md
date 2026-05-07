**Idea: Diversity-Guided Operator Selection**

Instead of pure Thompson Sampling on rewards, this variant detects problem difficulty from runtime metrics (population diversity, condition number, stagnation) and deterministically selects operators designed for that regime. Tasks with errors >10 are stuck in local optima or have collapsed covariance—detecting this and switching to archive-based or exploration-heavy operators is critical.

```python
def _select_operator_thompson(self):
    """Select operator using diversity-guided regime detection."""
    # Compute regime indicators
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min, eig_max = np.min(eigvals), np.max(eigvals)
    cond = eig_max / (max(eig_min, 1e-10))
    eig_spread = eig_min / (eig_max + 1e-10)
    
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    # Regime 1: Severely stuck (high error tasks need this)
    is_severe_trap = (diversity < 0.05) or (cond > 1e6) or (eig_spread < 1e-6)
    
    # Regime 2: Moderate convergence pressure
    is_converging = (rel_var < 0.001) or (diversity < 0.1)
    
    # Regime 3: Normal exploration
    is_normal = not (is_severe_trap or is_converging)
    
    # Map regimes to preferred operators (0=original, 1=archive, 2=diversity, 3=active, 4=temperature)
    if is_severe_trap:
        # Archive-guided (variant_01) for escaping severe local optima
        preferred = 1
    elif is_converging:
        # Diversity-sensitive (variant_06) or active CMA (variant_08)
        preferred = np.random.choice([2, 3])
    else:
        # Thompson sampling with bias toward successful operators
        preferred = None
    
    if preferred is not None:
        self.current_operator = int(preferred)
    else:
        # Fallback: Thompson sampling with regime-aware priors
        regime_prior = np.ones(self.num_operators)
        regime_prior[1] = 2.0  # boost archive for next cycle
        combined_alpha = self.alpha * regime_prior
        samples = np.random.beta(combined_alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
    
    self.operator_counts[self.current_operator] += 1
    return self.current_operator
```