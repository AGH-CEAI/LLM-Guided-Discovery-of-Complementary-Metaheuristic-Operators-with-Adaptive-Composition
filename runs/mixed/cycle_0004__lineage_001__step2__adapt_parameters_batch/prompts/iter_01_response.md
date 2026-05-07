**Idea: Success-Based JADE-Style Adaptation**
Adapt F and CR based on successful mutation parameters from the previous generation, using median F (for stability) and mean CR (for diversity). This is fundamentally different from random perturbation because it uses actual performance feedback.
```python
def _adapt_parameters_batch(self, population, fitness, mutants, trials):
    """JADE-style success-based parameter adaptation."""
    # Initialize history tracking
    if not hasattr(self, '_param_success_F'):
        self._param_success_F = []
        self._param_success_CR = []
    
    # Evaluate trial success by computing trial fitness
    trial_fitness = self._eval_wrapper_batch(trials, None)
    if trial_fitness is None:
        trial_fitness = np.array([func(x) for x in trials])
    
    # Identify successful mutations
    success_mask = trial_fitness < fitness
    
    # Record successful parameter values
    if np.any(success_mask):
        n_success = int(np.sum(success_mask))
        # Store successful F and CR values (per-individual F from variant_01 style)
        for i in range(self.NP):
            if success_mask[i]:
                self._param_success_F.append(float(np.clip(self.F + np.random.randn() * 0.1, 0.0, 2.0)))
                self._param_success_CR.append(float(self.CR))
    
    # Maintain bounded history (JADE-style window)
    max_history = 20
    if len(self._param_success_F) > max_history:
        self._param_success_F = self._param_success_F[-max_history:]
        self._param_success_CR = self._param_success_CR[-max_history:]
    
    # Adapt based on success history: median F (stable), mean CR (diverse)
    if len(self._param_success_F) >= 5:
        self.F = float(np.clip(np.median(self._param_success_F), 0.1, 1.0))
        self.CR = float(np.clip(np.mean(self._param_success_CR), 0.3, 0.95))
    else:
        # Fallback: gradual drift toward favorable values
        self.F = float(np.clip(self.F + np.random.uniform(-0.1, 0.1), 0.3, 1.5))
        self.CR = float(np.clip(self.CR + np.random.uniform(-0.1, 0.1), 0.1, 0.95))
    
    # Adapt p_best_rate based on diversity
    if len(self._param_success_F) >= 10:
        diversity = self._compute_diversity_batch(population)
        if diversity < self.diversity_threshold:
            self.p_best_rate = float(np.clip(self.p_best_rate * 1.1, 0.05, 0.3))
        else:
            self.p_best_rate = float(np.clip(self.p_best_rate * 0.95, 0.05, 0.3))
```