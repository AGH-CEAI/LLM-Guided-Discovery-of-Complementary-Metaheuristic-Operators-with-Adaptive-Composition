**Idea: Entropy-Driven Parameter Adaptation**
Use Shannon entropy of the fitness distribution to detect exploration vs. exploitation regimes. High entropy (diverse fitness) → increase F for exploration; low entropy (converged fitness) → decrease F for exploitation. KL divergence from a reference distribution quantifies regime shifts.

```python
def _adapt_parameters(self, improved_mask, F_used, Cr_used):
    """Adapt F and Cr using entropy of the fitness distribution.
    
    Category C: Information-theoretic / distributional.
    Treats fitness values as a probability distribution and uses Shannon entropy
    to detect exploration (high entropy, diverse fitness) vs. exploitation
    (low entropy, converged fitness) regimes. KL divergence from a reference
    distribution quantifies how far the current regime has shifted.
    """
    # Compute entropy of fitness distribution via kernel density estimation proxy
    fitness = np.array([f for f, _ in self.F_success_history[-self.adaptation_window:]] +
                       [np.mean([f for f, _ in self.F_success_history[-self.adaptation_window:]])])
    
    if len(fitness) < 2 or np.all(np.isinf(fitness)):
        return
    
    # Normalize to probability-like distribution using softmax (temperature = 1)
    fitness_min = np.nanmin(fitness)
    fitness_max = np.nanmax(fitness)
    if fitness_max - fitness_min < 1e-10:
        return
    
    # Discretize fitness range into bins for entropy estimation
    n_bins = min(10, max(3, len(fitness) // 2))
    bin_edges = np.linspace(fitness_min - 0.5 * (fitness_max - fitness_min),
                            fitness_max + 0.5 * (fitness_max - fitness_min),
                            n_bins + 1)
    bin_counts, _ = np.histogram(fitness, bins=bin_edges)
    bin_probs = bin_counts / (np.sum(bin_counts) + 1e-10)
    
    # Shannon entropy: H = -sum(p * log(p))
    entropy = -np.sum(np.where(bin_probs > 0, bin_probs * np.log(bin_probs + 1e-10), 0))
    max_entropy = np.log(n_bins)
    normalized_entropy = entropy / (max_entropy + 1e-10)
    
    # Initialize entropy history if needed
    if not hasattr(self, 'fitness_entropy_history'):
        self.fitness_entropy_history = []
    if not hasattr(self, 'last_F_for_entropy'):
        self.last_F_for_entropy = self.F
    
    self.fitness_entropy_history.append(normalized_entropy)
    if len(self.fitness_entropy_history) > self.adaptation_window:
        self.fitness_entropy_history.pop(0)
    
    if len(self.fitness_entropy_history) >= 3:
        recent_entropies = self.fitness_entropy_history[-3:]
        entropy_change = recent_entropies[-1] - recent_entropies[0]
        
        # Reference distribution: uniform over bins (maximum entropy)
        uniform_probs = np.ones(n_bins) / n_bins
        
        # KL divergence from uniform reference: D_KL(P || U)
        # High KL means distribution is concentrated (exploitation regime)
        kl_div = np.sum(np.where(uniform_probs > 0,
                                  uniform_probs * np.log(uniform_probs / (bin_probs + 1e-10),
                                                          uniform_probs),
                                  0))
        
        # Adaptive scaling based on entropy regime
        if recent_entropies[-1] > 0.5:
            # High entropy: exploration regime, increase F
            self.F = np.clip(self.F * (1.0 + 0.15 * entropy_change), 0.1, 1.5)
            self.Cr = np.clip(self.Cr * (1.0 - 0.05 * entropy_change), 0.1, 0.9)
        else:
            # Low entropy / low KL: exploitation regime, decrease F
            self.F = np.clip(self.F * (1.0 - 0.10 * abs(entropy_change)), 0.1, 1.5)
            self.Cr = np.clip(self.Cr * (1.0 + 0.08 * abs(entropy_change)), 0.1, 0.9)
        
        # Momentum term: smooth adaptation based on entropy trend
        momentum = 0.7
        target_F = self.F
        self.F = np.clip(momentum * self.last_F_for_entropy + (1 - momentum) * target_F, 0.1, 1.5)
        self.last_F_for_entropy = self.F
```