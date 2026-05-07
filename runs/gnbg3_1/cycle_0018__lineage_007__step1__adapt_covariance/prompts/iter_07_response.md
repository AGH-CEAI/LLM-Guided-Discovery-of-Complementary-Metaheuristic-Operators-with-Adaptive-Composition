**Idea: Covariance Reconstruction with Population Empirical Estimation**

Fundamentally different approach: Instead of incrementally updating covariance via rank-one/rank-μ, periodically reconstruct it from scratch using the current population's empirical covariance matrix. This breaks free from accumulated estimation errors that cause the worst tasks (errors 10-70) to be trapped. The key insight is that when the covariance matrix becomes ill-conditioned (cond > 1e6) or the population collapses (diversity < 0.2), the accumulated rank updates have become unreliable and must be refreshed from direct population statistics.

```python
def _adapt_covariance(self):
    """Population-aware covariance reconstruction with empirical estimation."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min, eig_max = np.min(eigvals), np.max(eigvals)
    cond = eig_max / max(eig_min, 1e-10)
    eig_spread = eig_min / (eig_max + 1e-10)
    
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    
    if not hasattr(self, 'reconstruction_count'):
        self.reconstruction_count = 0
        self.last_reconstruction_gen = -1
    
    generations_since_recon = self.generation - self.last_reconstruction_gen
    
    needs_reconstruction = (
        cond > 1e6 or 
        eig_spread < 1e-6 or 
        diversity < 0.05 or
        self.stagnation_counter > self.max_stagnation // 2 or
        (diversity < 0.2 and generations_since_recon > max(self.dim * 2, 50))
    )
    
    if needs_reconstruction and generations_since_recon > max(self.dim, 30):
        pop_centered = (self.population - self.mean) / self.sigma
        C_pop = np.cov(pop_centered.T) + 1e-8 * np.eye(self.dim)
        
        blend_weight = np.clip(0.4 + 0.6 * diversity, 0.3, 0.9)
        self.C = (1.0 - blend_weight) * self.C + blend_weight * C_pop
        
        noise_scale = min(0.15 * np.log10(cond + 1), 0.5) * (1.0 - diversity)
        if noise_scale > 0.01:
            noise = np.random.randn(self.dim, self.dim) * noise_scale
            noise = 0.5 * (noise + noise.T)
            np.fill_diagonal(noise, np.abs(np.diag(noise)))
            self.C += noise
        
        self.pc *= 0.3
        self.reconstruction_count += 1
        self.last_reconstruction_gen = self.generation
    else:
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
    
    self.C = self._ensure_positive_definite(self.C)
```