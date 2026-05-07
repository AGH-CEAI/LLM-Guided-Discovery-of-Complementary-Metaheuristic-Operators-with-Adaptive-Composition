**Idea: Entropy-Weighted Parameter Adaptation**
Adapt F and CR using Shannon entropy of the fitness distribution and population diversity entropy, treating the population as a probability distribution to be characterised (Category C: Information-theoretic / distributional).

```python
def _adapt_parameters(self, improvement_rate):
    """Adapt F and CR using information-theoretic measures on the population distribution."""
    # Compute fitness distribution entropy (uncertainty in fitness values)
    fitness_vals = np.asarray(self.last_fitness) if hasattr(self, 'last_fitness') and self.last_fitness is not None else np.array([0.0])
    
    # Handle edge cases for fitness entropy
    if len(fitness_vals) < 2:
        fitness_entropy = 0.1
    else:
        fit_min, fit_max = fitness_vals.min(), fitness_vals.max()
        if fit_max - fit_min < 1e-12:
            fitness_entropy = 0.1
        else:
            fit_norm = (fitness_vals - fit_min) / (fit_max - fit_min + 1e-12)
            n_bins = max(5, min(20, len(fitness_vals) // 3))
            hist, _ = np.histogram(fit_norm, bins=n_bins, range=(0, 1))
            probs = hist / (hist.sum() + 1e-12)
            probs = np.clip(probs, 1e-12, 1.0)
            fitness_entropy = -np.sum(probs * np.log2(probs + 1e-12))
    
    # Compute population spatial entropy (diversity in decision space)
    pop_flat = self.population.flatten()
    pop_min, pop_max = pop_flat.min(), pop_flat.max()
    pop_range = pop_max - pop_min
    if pop_range < 1e-12:
        spatial_entropy = 0.1
    else:
        n_bins = max(10, min(50, len(pop_flat) // 5))
        hist, _ = np.histogram(pop_flat, bins=n_bins, range=(pop_min, pop_max))
        probs = hist / (hist.sum() + 1e-12)
        probs = np.clip(probs, 1e-12, 1.0)
        spatial_entropy = -np.sum(probs * np.log2(probs + 1e-12))
    
    # Normalize entropy ratios (0 = all mass in one bin, 1 = uniform)
    fitness_entropy_ratio = fitness_entropy / (np.log2(n_bins) + 1e-12)
    spatial_entropy_ratio = spatial_entropy / (np.log2(n_bins) + 1e-12)
    
    # Combined entropy signal (balance fitness landscape and spatial exploration)
    combined_entropy = 0.6 * fitness_entropy_ratio + 0.4 * spatial_entropy_ratio
    combined_entropy = np.clip(combined_entropy, 0.05, 0.95)
    
    # Adapt F: higher entropy → more exploration (higher F)
    # Use sigmoid-like response for smooth transitions
    target_F = 0.3 + 0.7 * combined_entropy
    self.F = 0.7 * self.F + 0.3 * target_F
    self.F = np.clip(self.F, 0.1, 1.5)
    
    # Adapt CR: improvement rate drives exploitation vs exploration
    # High improvement → increase exploitation (higher CR)
    target_CR = 0.5 + 0.4 * improvement_rate
    self.CR = 0.8 * self.CR + 0.2 * target_CR
    self.CR = np.clip(self.CR, 0.3, 0.98)
    
    # Smooth history tracking
    self.F_history.append(self.F)
    self.CR_history.append(self.CR)
    if len(self.F_history) > 50:
        self.F_history.pop(0)
        self.CR_history.pop(0)
```