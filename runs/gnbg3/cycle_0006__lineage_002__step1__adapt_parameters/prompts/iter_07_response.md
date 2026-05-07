**Idea: LHS-Bootstrap Stochastic Parameter Adaptation**

Use Latin Hypercube Sampling to generate candidate (F, CR) pairs, evaluate them via bootstrap resampling with random projections onto the historical parameter-performance space, and make a stochastic selection weighted by expected improvement. This provides a principled Monte-Carlo approach to parameter adaptation specifically targeting task 5's optimization landscape.

```python
def _adapt_parameters(self, improvement_rate):
    """Adapt F and CR using Latin Hypercube Sampling with bootstrap confidence estimation."""
    # Initialize history
    if not hasattr(self, '_F_samples'):
        self._F_samples = []
        self._CR_samples = []
        self._improvement_samples = []
        self._max_samples = 100
    
    # Store current observation
    self._F_samples.append(self.F)
    self._CR_samples.append(self.CR)
    self._improvement_samples.append(improvement_rate)
    
    # Maintain bounded history
    if len(self._F_samples) > self._max_samples:
        self._F_samples.pop(0)
        self._CR_samples.pop(0)
        self._improvement_samples.pop(0)
    
    n_samples = len(self._F_samples)
    
    if n_samples < 5:
        # Insufficient data, apply random perturbation
        self.F = np.clip(self.F + np.random.randn() * 0.1, 0.1, 1.5)
        self.CR = np.clip(self.CR + np.random.randn() * 0.05, 0.0, 1.0)
        return
    
    # Latin Hypercube Sampling for candidate generation
    n_candidates = 50
    F_candidates = np.linspace(0.1, 1.5, n_candidates)
    CR_candidates = np.linspace(0.0, 1.0, n_candidates)
    
    np.random.shuffle(F_candidates)
    np.random.shuffle(CR_candidates)
    
    # Bootstrap resampling for stochastic candidate evaluation
    n_bootstrap = 100
    candidate_scores = np.zeros(n_candidates)
    
    for c in range(n_candidates):
        F_c, CR_c = F_candidates[c], CR_candidates[c]
        bootstrap_scores = []
        
        for _ in range(n_bootstrap):
            # Bootstrap resample from history
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            F_boot = np.array([self._F_samples[i] for i in indices])
            CR_boot = np.array([self._CR_samples[i] for i in indices])
            imp_boot = np.array([self._improvement_samples[i] for i in indices])
            
            # Random projection for stochastic evaluation
            angle = np.random.rand() * 2 * np.pi
            proj = np.cos(angle) * (F_boot - F_c) + np.sin(angle) * (CR_boot - CR_c)
            
            # Weight by inverse distance (kernel density estimation style)
            weights = np.exp(-np.abs(proj) / 0.3)
            score = np.sum(weights * imp_boot) / (np.sum(weights) + 1e-8)
            bootstrap_scores.append(score)
        
        candidate_scores[c] = np.mean(bootstrap_scores)
    
    # Probabilistic selection proportional to score
    probs = np.exp(candidate_scores - candidate_scores.max())
    probs /= probs.sum()
    best_idx = np.random.choice(n_candidates, p=probs)
    
    # Apply selected parameters with bounded random noise
    self.F = np.clip(F_candidates[best_idx] + np.random.randn() * 0.05, 0.1, 1.5)
    self.CR = np.clip(CR_candidates[best_idx] + np.random.randn() * 0.02, 0.0, 1.0)
```