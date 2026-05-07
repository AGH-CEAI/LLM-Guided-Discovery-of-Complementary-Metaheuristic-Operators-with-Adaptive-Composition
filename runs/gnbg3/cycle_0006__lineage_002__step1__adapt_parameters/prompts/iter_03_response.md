**Idea: Entropy-Driven Parameter Adaptation**

Use population entropy estimation (Gaussian-fitted differential entropy per dimension) combined with improvement rate to modulate F and CR. High entropy → increase F for exploration; low entropy → decrease F for exploitation. Improvement rate provides complementary signal.

```python
def _adapt_parameters(self, improvement_rate):
    """Adapt F and CR using information-theoretic measures of population distribution."""
    population = self.population if hasattr(self, 'population') else None
    
    if not hasattr(self, '_ema_entropy'):
        self._ema_entropy = 0.5
        self._ema_improvement = 0.1
        self._entropy_history = []
    
    alpha = 0.2
    
    if population is not None:
        NP, dim = population.shape
        entropies = np.zeros(dim)
        for d in range(dim):
            x = population[:, d]
            std_d = np.std(x)
            if std_d > 1e-10:
                entropies[d] = 0.5 * np.log(2 * np.pi * np.e * std_d**2)
            else:
                entropies[d] = -np.inf
        
        current_entropy = np.mean(entropies[entropies > -np.inf]) if np.any(entropies > -np.inf) else 0.0
        self._entropy_history.append(current_entropy)
        if len(self._entropy_history) > 20:
            self._entropy_history.pop(0)
        
        self._ema_entropy = (1 - alpha) * self._ema_entropy + alpha * max(current_entropy, 0.0)
    
    self._ema_improvement = (1 - alpha) * self._ema_improvement + alpha * improvement_rate
    
    search_range = 200.0
    max_entropy = 0.5 * np.log(2 * np.pi * np.e * (search_range / 2)**2)
    norm_entropy = np.clip(self._ema_entropy / max(max_entropy, 1e-10), 0.0, 1.0)
    
    if norm_entropy > 0.6:
        self.F = np.clip(self.F * 1.08, 0.3, 1.0)
    elif norm_entropy < 0.3:
        self.F = np.clip(self.F * 0.94, 0.3, 1.0)
    
    if norm_entropy > 0.6:
        self.CR = np.clip(self.CR * 0.96, 0.5, 0.98)
    elif norm_entropy < 0.3:
        self.CR = np.clip(self.CR * 1.04, 0.5, 0.98)
    
    if self._ema_improvement > 0.15:
        self.F = np.clip(self.F * 0.95, 0.3, 1.0)
    elif self._ema_improvement < 0.05:
        self.F = np.clip(self.F * 1.1, 0.3, 1.0)
    
    if self._ema_improvement > 0.15:
        self.CR = np.clip(self.CR * 1.05, 0.5, 0.98)
    elif self._ema_improvement < 0.05:
        self.CR = np.clip(self.CR * 0.95, 0.5, 0.98)
    
    self.F_history.append(self.F)
    self.CR_history.append(self.CR)
```