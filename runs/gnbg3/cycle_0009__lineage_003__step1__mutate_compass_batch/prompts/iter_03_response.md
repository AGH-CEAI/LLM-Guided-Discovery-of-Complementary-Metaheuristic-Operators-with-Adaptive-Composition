**Idea: GMM-Entropy Guided Compass**

This variant fits a Gaussian Mixture Model to the population, computes entropy of the fitness distribution, and selects mutations using KL divergence between candidate directions and the population distribution, plus log-likelihood scoring—all firmly within Category C: Information-theoretic / distributional.

```python
def _mutate_compass_batch(self, population, fitness):
    """Information-theoretic compass mutation using GMM fit and entropy-based scoring."""
    NP, dim = population.shape
    
    # Compute fitness entropy (C: distributional)
    fitness_ranks = self._compute_fitness_ranking(fitness)
    eps = 1e-12
    p_rank = fitness_ranks + eps
    p_rank = p_rank / p_rank.sum()
    fitness_entropy = -np.sum(p_rank * np.log(p_rank + eps))
    entropy_weight = 1.0 / (1.0 + fitness_entropy * 5)
    
    # Fit GMM to population (C: distribution fitting)
    n_components = min(3, NP // 3)
    n_components = max(1, n_components)
    try:
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=n_components, covariance_reg=1e-4, random_state=42)
        gmm.fit(population)
        pop_mean = gmm.means_.mean(axis=0)
        pop_std = np.sqrt(gmm.covariances_).mean(axis=0) + eps
        gmm_weights = gmm.weights_
    except Exception:
        pop_mean = population.mean(axis=0)
        pop_std = population.std(axis=0) + eps
        gmm_weights = np.array([1.0])
    
    # Generate candidate directions per individual (C: distributional sampling)
    mutants = np.empty_like(population)
    for i in range(NP):
        others = np.concatenate([np.arange(i), np.arange(i + 1, self.NP)])
        np.random.shuffle(others)
        r1, r2, r3, r4 = others[:4]
        
        # Candidate directions from different strategies
        dA = population[r1] + self.F * (population[r2] - population[r3])
        best_idx = np.argmin(fitness_ranks)
        dB = population[best_idx] + self.F * (population[r1] - population[r2])
        
        p = 0.1
        top_p = max(1, int(np.ceil(p * self.NP)))
        pbest_idx = np.argsort(fitness_ranks)[:top_p][0]
        dC = population[i] + self.F * (population[pbest_idx] - population[r1]) + self.F * (population[r3] - population[r4])
        
        # Direction D: toward GMM center weighted by component strength
        dD = pop_mean + self.F * (population[r1] - population[r2])
        
        candidates = [dA, dB, dC, dD]
        
        # Score candidates using KL divergence and log-likelihood (C: information-theoretic)
        scores = np.zeros(len(candidates))
        for j, cand in enumerate(candidates):
            # KL divergence proxy: distance from GMM distribution
            diff_from_mean = cand - pop_mean
            kl_proxy = np.sum((diff_from_mean ** 2) / (2 * pop_std ** 2 + eps))
            
            # Log-likelihood under fitted distribution
            ll = -0.5 * np.sum(((cand - pop_mean) / (pop_std + eps)) ** 2)
            
            # Rank improvement potential
            rank_imp = fitness_ranks[i] - min(fitness_ranks[[r1, r2, r3]].min(), fitness_ranks[i] + 0.1)
            
            # Combined score (C: distributional reasoning)
            scores[j] = (1 - entropy_weight) * kl_proxy + entropy_weight * ll + 0.3 * rank_imp
        
        # Select best direction
        best_idx_dir = np.argmax(scores)
        mutants[i] = candidates[best_idx_dir]
    
    return mutants
```