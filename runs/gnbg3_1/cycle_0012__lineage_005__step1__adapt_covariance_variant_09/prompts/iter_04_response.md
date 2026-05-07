Looking at the error magnitudes, the priority tasks (16, 17, 23, 11, 20, 18, 19, 22) are all stuck at ~1e+1 to ~1e+1, indicating severe stagnation/local optima trapping. The existing variants either:
1. React too slowly to stagnation (temperature-based)
2. Only trigger on convergence detection (archive-based)
3. Have conservative perturbation strength

**Idea: Immediate Stagnation-Triggered Covariance Reset with Rank-2 Updates**

A fundamentally different approach: directly check stagnation status inside the covariance method and immediately inject aggressive multi-axis perturbation when stuck, combined with rank-2 Bump-style updates for smoother adaptation.

```python
def _adapt_covariance_variant_09(self):
    """Immediate stagnation-triggered covariance reset with rank-2 updates."""
    stagnation_threshold = max(20, self.dim)
    is_stagnant = self.stagnation_counter > stagnation_threshold

    if is_stagnant:
        # Aggressive multi-axis perturbation: inject fresh variance along eigendirections
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        sorted_idx = np.argsort(eigvals)
        
        # Perturb along top 60% of eigenvectors with decreasing strength
        k = max(1, int(0.6 * self.dim))
        for i in range(k):
            idx = sorted_idx[self.dim - 1 - i]
            strength = 3.0 * (k - i) / k
            self.C += strength * np.outer(eigvecs[:, idx], eigvecs[:, idx])
        
        # Add uniform axis-aligned perturbation
        self.C += 1.5 * np.eye(self.dim)
        
        self.C = self._ensure_positive_definite(self.C)
        self.pc *= 0.01
        self.stagnation_counter = 0
    else:
        # Standard rank-1 update with progress-weighted scaling
        y_mean = (self.mean - self.old_mean) / self.sigma
        
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        improv_scale = np.clip(1.0 + 5.0 * np.log1p(improvement * 1e4), 0.5, 3.0)
        
        cc_adapt = np.clip(self.cc * improv_scale, 0.001, 0.3)
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        # Rank-mu update (normalized)
        rank_mu = np.zeros((self.dim, self.dim))
        total_w = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_w += abs(w)
        
        if total_w > 0:
            rank_mu /= total_w
        
        ccov_adapt = np.clip(self.ccov * improv_scale, 1e-10, 0.5)
        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)
        
        # Condition number check to prevent ill-conditioning
        eigvals_check = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals_check) / np.min(eigvals_check)
        if cond > 1e6:
            self.C *= 0.5
        
        self.C = self._ensure_positive_definite(self.C)
```