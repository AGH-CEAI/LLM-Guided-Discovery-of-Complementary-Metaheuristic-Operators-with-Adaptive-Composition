**Idea: Condition-Aware Covariance Repair with Forced Diversity Injection**

The worst tasks (16, 17, 23, 20, 18, 11, 19, 22) have errors in the 1e+1 to 1e+2 range, indicating severe premature convergence — the covariance matrix is collapsing to a narrow ridge before finding the global basin. The current variant_08 only reacts to stagnation, but this is too late; the optimizer is already trapped. This approach **proactively monitors condition number every generation** and injects fresh variance along principal axes before stagnation even triggers, preventing the collapse that causes 8+ decade errors.

```python
def _adapt_covariance_variant_08(self):
    """Condition-aware covariance repair with proactive diversity injection."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Proactive condition monitoring — detect collapse BEFORE stagnation
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.maximum(np.min(eigvals), 1e-15)
    eig_max = np.maximum(np.max(eigvals), 1e-15)
    cond = eig_max / eig_min
    eig_spread = eig_min / (eig_max + 1e-15)
    
    # Also track population spread vs expected spread
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    rel_diversity = pop_variance / (expected_var + 1e-10)
    
    # Force repair when: ill-conditioned OR collapsed spectrum OR low diversity
    needs_repair = (cond > 1e4) or (eig_spread < 1e-6) or (rel_diversity < 1e-4)
    
    if needs_repair:
        # Compute eigenvectors of current covariance
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-15)
        
        # Redistribute: stretch small eigvals, shrink large eigvals toward median
        target_spread = 1e4  # target condition number
        median_eig = np.median(eigvals)
        
        # Blend each eigenvalue toward median, weighted by how extreme it is
        new_eigvals = np.zeros_like(eigvals)
        for j in range(len(eigvals)):
            extreme_ratio = eigvals[j] / median_eig
            if extreme_ratio < 1.0:
                # Small eigenvalue: boost toward median * sqrt(target_spread)
                blend = min(0.9, 0.3 + 0.2 * np.log1p(1.0 / (extreme_ratio + 1e-10)))
                new_eigvals[j] = eigvals[j] * (1.0 - blend) + median_eig * np.sqrt(target_spread) * extreme_ratio * blend
            else:
                # Large eigenvalue: shrink toward median
                blend = min(0.7, 0.2 * np.log1p(extreme_ratio))
                new_eigvals[j] = eigvals[j] * (1.0 - blend) + median_eig * blend
        
        # Reconstruct repaired covariance
        new_eigvals = np.maximum(new_eigvals, 1e-12)
        self.C = eigvecs @ np.diag(new_eigvals) @ eigvecs.T
        self.C = 0.5 * (self.C + self.C.T)
        
        # Inject fresh axis-aligned variance for uniform exploration
        injection_strength = 0.2 * np.mean(new_eigvals)
        self.C += injection_strength * np.eye(self.dim)
        
        self.pc *= 0.0  # reset evolution path to avoid bias
        self.stagnation_counter = 0
    else:
        # Standard CMA-ES update with adaptive learning rates
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        improv_scale = np.clip(1.0 + 5.0 * np.log1p(improvement * 1e6), 0.5, 3.0)
        
        # Adaptive evolution path update
        cc_adapt = np.clip(self.cc * improv_scale, 0.001, 0.5)
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        # Adaptive covariance learning rate
        ccov_adapt = np.clip(self.ccov * improv_scale, 1e-10, 0.5)
        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * rank_one +
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)
    
    self.C = self._ensure_positive_definite(self.C)
```