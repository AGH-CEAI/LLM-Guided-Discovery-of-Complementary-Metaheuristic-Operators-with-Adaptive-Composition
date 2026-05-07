Looking at the worst unsolved tasks (errors of 1e+01 to 1e+05), I need a hybrid approach that combines two distinct mechanisms with principled switching.

**Analysis of failure modes:**
- Tasks 17, 16, 6 have errors > 1e+02, suggesting the population gets trapped or fails to explore properly
- The current spectral-only approach modulates velocity along principal axes but doesn't adapt its strategy based on the population's state
- Category H allows me to combine spectral-based modulation with a complementary mechanism, switching based on a data-driven signal

**Hybrid design:**
- **Mechanism 1 (Spectral):** Project velocity onto population's principal components, scale by condition number (handles anisotropic swarms)
- **Mechanism 2 (Fitness-guided):** Compute direction toward personal best improvements weighted by relative fitness improvement (handles exploitation toward good solutions)
- **Switching signal:** Condition number of the population's covariance — high condition → population is elongated → spectral dominates for exploration; low condition → population is isotropic → fitness-guided dominates for exploitation

This is fundamentally different from all prior categories because it combines geometric analysis (spectral) with fitness-based reasoning (fitness-guided), switching based on a principled geometric signal.

**Idea: Spectral-Fitness Hybrid with Condition-Driven Switching**
Hybrid position update combining spectral velocity modulation with fitness-guided exploitation, switching based on population condition number.
```python
def _position_update_batch(self):
    """Update positions with spectral-fitness hybrid and condition-driven switching."""
    centered = self.population - np.mean(self.population, axis=0)
    cov = np.cov(centered.T)
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        
        # Mechanism 1: Spectral velocity modulation
        # Scale velocity along principal axes based on condition number
        vel_proj = self.velocity @ eigenvectors
        spectral_scale = np.clip(cond / 50.0, 0.3, 3.0)
        vel_spectral = vel_proj * spectral_scale
        
        # Mechanism 2: Fitness-guided exploitation
        # Direction weighted by relative fitness improvement
        fitness_range = np.max(self.current_fitness) - np.min(self.current_fitness) + 1e-10
        fitness_normalized = (self.current_fitness - np.min(self.current_fitness)) / fitness_range
        
        # Particles with worse fitness get stronger pull toward their personal best
        improvement_weight = 1.0 - fitness_normalized
        improvement_weight = np.clip(improvement_weight, 0.0, 1.0)
        
        fitness_direction = (self.personal_best - self.population) * improvement_weight[:, np.newaxis]
        
        # Principled switching: condition number drives mechanism weight
        # High condition -> anisotropic population -> rely on spectral (exploration)
        # Low condition -> isotropic population -> rely on fitness-guided (exploitation)
        spectral_weight = np.clip(cond / 20.0, 0.0, 1.0)
        fitness_weight = 1.0 - spectral_weight
        
        combined_velocity = (
            spectral_weight * vel_spectral +
            fitness_weight * fitness_direction
        )
        
        new_population = self.population + combined_velocity
        
    except np.linalg.LinAlgError:
        # Fallback: simple velocity update with fitness guidance
        fitness_range = np.max(self.current_fitness) - np.min(self.current_fitness) + 1e-10
        fitness_normalized = (self.current_fitness - np.min(self.current_fitness)) / fitness_range
        improvement_weight = np.clip(1.0 - fitness_normalized, 0.0, 1.0)
        fitness_direction = (self.personal_best - self.population) * improvement_weight[:, np.newaxis]
        new_population = self.population + 0.5 * self.velocity + 0.3 * fitness_direction
    
    self.population = self._clip_to_bounds(new_population)
```