**Idea: Gaussian Entropy-Directed Parameter Adaptation**
Adapt F and Cr by fitting a Gaussian to the fitness distribution and using its differential entropy as a diversity signal, combined with entropy of the parameter history to detect convergence regimes. (Category C: Information-theoretic / distributional)

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using information-theoretic measures of the population distribution."""
    success_rate = np.mean(improved_mask)
    
    # Initialize entropy tracking state on first call
    if not hasattr(self, 'fitness_entropy_history'):
        self.fitness_entropy_history = []
        self.fitness_initial_entropy = None
        self.F_entropy_history = []
        self.Cr_entropy_history = []
        self.improvement_magnitudes = []
    
    # Access fitness distribution via stored population
    if hasattr(self, 'fitness') and self.fitness is not None:
        fitness_vals = self.fitness[np.isfinite(self.fitness)]
        
        if len(fitness_vals) > 5:
            # Gaussian differential entropy: H = 0.5*log(2*pi*e*sigma^2)
            # Higher entropy = more diverse population = more exploration
            sigma = np.std(fitness_vals) + 1e-10
            current_fitness_entropy = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
            
            if self.fitness_initial_entropy is None:
                self.fitness_initial_entropy = max(current_fitness_entropy, 1e-6)
            
            self.fitness_entropy_history.append(current_fitness_entropy)
            if len(self.fitness_entropy_history) > 20:
                self.fitness_entropy_history.pop(0)
            
            # Entropy ratio as convergence signal: low ratio = population collapsed
            entropy_ratio = current_fitness_entropy / (self.fitness_initial_entropy + 1e-10)
            entropy_ratio = np.clip(entropy_ratio, 0.01, 10.0)
        else:
            entropy_ratio = 1.0
    else:
        entropy_ratio = 1.0
    
    # Track parameter distribution entropy (diversity of recent F/Cr values)
    self.F_entropy_history.append(F_used)
    self.Cr_entropy_history.append(Cr_used)
    if len(self.F_entropy_history) > 20:
        self.F_entropy_history.pop(0)
        self.Cr_entropy_history.pop(0)
    
    # Parameter diversity via entropy of recent parameter samples
    F_std = np.std(self.F_entropy_history) + 1e-10
    Cr_std = np.std(self.Cr_entropy_history) + 1e-10
    param_diversity = (F_std + Cr_std) / 2.0
    
    # Compute improvement entropy from magnitudes
    if hasattr(self, 'fitness') and self.fitness is not None:
        trial_fitness = getattr(self, 'trial_fitness', None)
        if trial_fitness is not None:
            improvements = self.fitness - trial_fitness
            improvements = improvements[improved_mask & np.isfinite(improvements)]
            if len(improvements) > 3:
                # Entropy of improvement distribution: high = diverse outcomes
                imp_sigma = np.std(improvements) + 1e-10
                imp_entropy = 0.5 * np.log(2 * np.pi * np.e * imp_sigma**2)
                improvement_entropy_normalized = np.clip(imp_entropy / (abs(current_fitness_entropy) + 1e-6), 0.1, 5.0)
            else:
                improvement_entropy_normalized = 1.0
        else:
            improvement_entropy_normalized = 1.0
    else:
        improvement_entropy_normalized = 1.0
    
    # Decision logic: entropy-driven regime detection
    if entropy_ratio < 0.25:
        # CONVERGENCE DETECTED: population collapsed, inject maximum diversity
        F_adjustment = 1.25 * (1.0 / (entropy_ratio + 0.1))
        Cr_adjustment = 0.70
        self.stagnation_counter = getattr(self, 'stagnation_counter', 0) + 3
    elif entropy_ratio < 0.5:
        # EARLY CONVERGENCE: reduce pressure, favor exploration
        F_adjustment = 1.15
        Cr_adjustment = 0.85
        self.stagnation_counter = getattr(self, 'stagnation_counter', 0) + 1
    elif entropy_ratio > 2.5:
        # DIVERSE: population very spread, favor exploitation
        F_adjustment = 0.85
        Cr_adjustment = 1.15
        self.stagnation_counter = max(0, getattr(self, 'stagnation_counter', 0) - 1)
    else:
        # MODERATE: use improvement entropy and parameter diversity
        if improvement_entropy_normalized > 1.5:
            # Diverse improvements = good exploration, can increase exploitation
            F_adjustment = 0.92
            Cr_adjustment = 1.08
        elif improvement_entropy_normalized < 0.5:
            # Clustered improvements = premature convergence
            F_adjustment = 1.12
            Cr_adjustment = 0.90
        else:
            # Neutral: use parameter diversity signal
            if param_diversity < 0.05:
                F_adjustment = 1.10
                Cr_adjustment = 0.92
            else:
                delta = entropy_ratio - 1.0
                F_adjustment = 1.0 + 0.10 * delta
                Cr_adjustment = 1.0 - 0.08 * delta
    
    # Apply entropy-regularized momentum from parameter history
    F_momentum = np.clip(F_used / (np.mean(self.F_entropy_history) + 1e-10), 0.7, 1.4)
    Cr_momentum = np.clip(Cr_used / (np.mean(self.Cr_entropy_history) + 1e-10), 0.7, 1.4)
    
    # Apply adjustments with momentum correction
    self.F = np.clip(self.F * F_adjustment * (0.9 + 0.2 * F_momentum), 0.1, 2.0)
    self.Cr = np.clip(self.Cr * Cr_adjustment * (0.9 + 0.2 * Cr_momentum), 0.1, 0.95)
```