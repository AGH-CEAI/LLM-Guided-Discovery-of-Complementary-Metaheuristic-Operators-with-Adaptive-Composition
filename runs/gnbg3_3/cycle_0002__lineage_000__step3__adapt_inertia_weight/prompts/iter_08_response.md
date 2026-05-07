Looking at the priority targets (Tasks 17, 16, 6, 19 with errors 1e+04 to 1e+02), I need a hybrid approach that combines two orthogonal mechanisms to handle the most challenging problems.

**Idea: Stagnation-Guided Diversity Modulation**

Combine **fitness-stagnation tracking** (temporal, Category F) with **spectral condition number** (Category B) for a principled two-signal hybrid. High condition number = anisotropic landscape = need dispersive exploration. High stagnation = trapped in basin = need escape mechanism. The switching logic uses stagnation state to determine which signal dominates.

```python
def _adapt_inertia_weight(self):
    """Hybrid inertia adaptation: combine stagnation signal with spectral condition."""
    diversity = self._compute_diversity()
    
    # Mechanism 1: stagnation-based signal (temporal)
    stagnation_penalty = np.exp(-self.stagnation_counter / 30.0)
    
    # Mechanism 2: spectral condition number (linear-algebraic)
    centered = self.population - np.mean(self.population, axis=0)
    try:
        _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        cond = singular_values[0] / (singular_values[-1] + 1e-10)
        spectral_stress = np.clip(cond / 50.0, 0.0, 1.0)
    except np.linalg.LinAlgError:
        spectral_stress = 0.5
    
    # Principled switching: stagnation dominates when stuck, spectral guides otherwise
    if self.stagnation_counter > 15:
        # Stagnation regime: use stagnation signal as primary driver
        if diversity < self.diversity_threshold_low:
            self.inertia_weight = max(0.3, self.inertia_weight * 0.85)
        else:
            self.inertia_weight = min(0.9, self.inertia_weight * 1.02)
    else:
        # Normal regime: blend stagnation + spectral signals
        combined_exploration_pressure = 0.6 * (1.0 - stagnation_penalty) + 0.4 * spectral_stress
        
        if combined_exploration_pressure > 0.5:
            self.inertia_weight = max(0.35, 0.729 - 0.15 * combined_exploration_pressure)
        else:
            self.inertia_weight = min(0.95, 0.729 + 0.10 * (0.5 - combined_exploration_pressure))
    
    self.inertia_weight = np.clip(self.inertia_weight, 0.3, 0.95)
```