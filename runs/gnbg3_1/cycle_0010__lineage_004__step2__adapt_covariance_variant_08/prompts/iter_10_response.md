Looking at the benchmark data, all 24 tasks are unsolved with errors ranging from ~1e-5 to ~1e+2. The worst tasks (16, 17, 11, 20, 23) have errors in the 10-100 range, indicating severe multimodality or deceptive landscapes where the algorithm gets trapped in local optima.

Current variant_08 uses multiple evolution paths but still gets stuck. The key issue: path-based covariance adaptation relies on accumulating gradients over time, which fails when trapped in basins - the accumulated direction becomes meaningless.

**Idea: Explosive Eigendirection Perturbation on Stagnation**

When stagnation is detected, perform eigendecomposition and inject massive perturbations along ALL principal eigenvectors (not just a few). This creates a "star burst" exploration pattern that pushes the search in every direction simultaneously. This is fundamentally different from all prior variants that use small perturbations or learning rate scaling - it triggers a fundamentally different search mode.

```python
def _adapt_covariance_variant_08(self):
    """Explosive eigendirection perturbation on stagnation for multimodal escape."""
    stagnation_threshold = max(30, self.dim * 2)
    is_stagnant = self.stagnation_counter > stagnation_threshold
    
    if is_stagnant:
        # Explosive exploration: perturb along all principal eigendirections
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        
        # Perturb along top k eigenvectors with decreasing strength
        k = min(self.dim, max(3, self.dim // 4))
        for i in range(k):
            strength = 2.0 * (k - i) / k  # stronger for top eigenvectors
            self.C += strength * np.outer(eigvecs[:, i], eigvecs[:, i])
        
        # Also add axis-aligned perturbation for uniform coverage
        self.C += 0.5 * np.eye(self.dim)
        
        self.C = self._ensure_positive_definite(self.C)
        self.pc *= 0.1  # reset evolution path
        self.stagnation_counter = 0
    else:
        # Standard covariance update with improvement-weighted scaling
        y_mean = (self.mean - self.old_mean) / self.sigma
        
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        improv_scale = np.clip(1.0 + 10.0 * np.log1p(improvement * 1e6), 0.1, 5.0)
        
        cc_adapt = np.clip(self.cc * improv_scale, 0.001, 0.3)
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        ccov_adapt = np.clip(self.ccov * improv_scale, 1e-10, 0.5)
        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * rank_one +
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)
        
        self.C = self._ensure_positive_definite(self.C)
        
        eigvals_check = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals_check) / np.min(eigvals_check)
        if cond > 1e7:
            self.C *= 0.5
```