**Idea: Entropy-KL Neighborhood Adaptation**
Adapt neighborhood size using entropy of fitness distribution and KL divergence from uniform eigenvalue distribution, treating the population as a probability distribution to characterize.
```python
def _adapt_neighborhood_size(self):
    """Adapt neighborhood size using information-theoretic signals from population distribution.
    
    Treats the population as a probability distribution and uses:
    1. Entropy of discretized fitness distribution
    2. KL divergence of covariance eigenvalues from uniform (spectral entropy)
    
    High entropy + low KL = diverse population → reduce neighborhood for exploitation.
    Low entropy + high KL = clustered/trapped → increase neighborhood for exploration.
    """
    # Signal 1: Entropy of fitness distribution (normalized)
    fitness = self.current_fitness
    if len(fitness) > 1:
        n_bins = min(20, max(3, len(fitness) // 2))
        hist, _ = np.histogram(fitness, bins=n_bins)
        hist = hist / (len(fitness) + 1e-10)
        hist = hist[hist > 0]
        fitness_entropy = -np.sum(hist * np.log(hist + 1e-10))
        max_fitness_entropy = np.log(n_bins)
        norm_fitness_ent = fitness_entropy / (max_fitness_entropy + 1e-10)
    else:
        norm_fitness_ent = 0.5
    
    # Signal 2: KL divergence of covariance eigenvalues from uniform
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues_norm = eigenvalues / (np.sum(eigenvalues) + 1e-10)
        
        # KL(P||U) where U is uniform
        uniform = np.ones(len(eigenvalues_norm)) / len(eigenvalues_norm)
        uniform = uniform + 1e-10
        kl_div = np.sum(eigenvalues_norm * np.log((eigenvalues_norm + 1e-10) / uniform))
        
        max_kl = np.log(len(eigenvalues_norm))
        norm_kl = np.clip(kl_div / (max_kl + 1e-10), 0.0, 1.0)
    except:
        norm_kl = 0.5
    
    # Combined exploration signal: KL pushes toward exploration, entropy pulls toward exploitation
    exploration_signal = 0.4 * (1.0 - norm_fitness_ent) + 0.6 * norm_kl
    exploration_signal = np.clip(exploration_signal, 0.0, 1.0)
    
    # Map to neighborhood size
    min_ns = 1
    max_ns = max(3, self.np // 4)
    target_ns = int(np.round(min_ns + exploration_signal * (max_ns - min_ns)))
    target_ns = np.clip(target_ns, min_ns, max_ns)
    
    # Incremental adjustment
    if target_ns > self.neighborhood_size:
        self.neighborhood_size = min(target_ns, self.neighborhood_size + 1)
    elif target_ns < self.neighborhood_size:
        self.neighborhood_size = max(target_ns, self.neighborhood_size - 1)
```