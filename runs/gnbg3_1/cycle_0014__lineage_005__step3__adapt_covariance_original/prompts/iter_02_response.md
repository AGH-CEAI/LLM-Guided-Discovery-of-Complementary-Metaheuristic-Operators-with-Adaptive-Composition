**Idea: Restart-Triggered Eigendecomposition with Elite Preservation**

The worst tasks (errors 10-85) suggest the optimizer gets permanently trapped in wrong regions. This variant detects convergence traps via relative fitness variance and covariance condition number, then performs a full restart with elite preservation while injecting noise along weak eigendirections to escape deceptive basins.
```python
def _adapt_covariance_original(self):
    """Restart-triggered eigendecomposition with elite preservation for escaping traps."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Compute trap detection metrics
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)
    
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.min(eigvals)
    eig_max = np.max(eigvals)
    cond = eig_max / (max(eig_min, 1e-10))
    
    is_trapped = (rel_var < 1e-3) or (cond > 1e5)
    
    # Adaptive learning rates
    ccov_1 = self.ccov * (1.0 + 10.0 * min(rel_var, 0.1)) / (self.dim + 2.0)
    ccov_mu = self.ccov * self.mueff * (1.0 + 10.0 * min(rel_var, 0.1)) / (self.dim + 2.0) / (self.dim + 4.0)
    ccov_1 = np.clip(ccov_1, 1e-10, 0.5)
    ccov_mu = np.clip(ccov_mu, 1e-10, 0.5)
    
    # Evolution path update
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    rank_one = np.outer(self.pc, self.pc)
    
    # Rank-mu update
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Covariance update
    self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
              ccov_1 * rank_one +
              ccov_mu * rank_mu)
    
    # Inject exploration noise along weak eigendirections when trapped
    if is_trapped:
        eigvecs = np.linalg.eigh(self.C)[1]
        eigvals_safe = np.maximum(eigvals, 1e-10)
        target = np.mean(eigvals_safe)
        for j in range(self.dim):
            if eigvals[j] < target * 0.5:
                scale = (target - eigvals[j]) / target * 0.3
                self.C += scale * np.outer(eigvecs[:, j], eigvecs[:, j])
    
    self.C = self._ensure_positive_definite(self.C)
    
    # Full restart with elite preservation on extreme trapping
    if is_trapped and (rel_var < 1e-4 or cond > 1e7):
        top_elites = min(5, self.NP // 10)
        elite_pop = self.population[:top_elites].copy()
        elite_fit = self.fitness[:top_elites].copy()
        
        self._initialize_population()
        
        insert_count = min(top_elites, len(elite_pop))
        self.population[:insert_count] = elite_pop[:insert_count]
        self.fitness[:insert_count] = elite_fit[:insert_count]
        
        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        self.stagnation_counter = 0
```