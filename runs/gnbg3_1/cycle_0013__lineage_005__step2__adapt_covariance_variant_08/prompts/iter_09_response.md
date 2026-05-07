**Idea: Adaptive Diversity Injection with Conditional Eigenbasis Restart**

This variant proactively monitors population convergence using eigenvalue spread and fitness variance, then injects diversity BEFORE stagnation becomes severe. Unlike the current variant that only reacts after stagnation_threshold generations, this approach uses a continuous "convergence pressure" metric to trigger graduated perturbations along the covariance eigendirections, preventing premature collapse on multimodal tasks.

```python
def _adapt_covariance_variant_08(self):
    """Adaptive diversity injection via convergence pressure and eigendirection restart."""
    y_mean = (self.mean - self.old_mean) / self.sigma

    # Compute convergence pressure metrics
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)

    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = max(np.min(eigvals), 1e-15)
    eig_max = np.max(eigvals)
    eig_spread = eig_min / (eig_max + 1e-15)
    cond = eig_max / eig_min

    # Combined convergence pressure (0 = converged, 1 = diverse)
    pressure = min(1.0, rel_var * 10.0 + eig_spread * 5.0)
    pressure = max(0.0, pressure)

    # Adaptive threshold based on dimension and condition number
    adaptive_threshold = max(15, self.dim * 1.5) * (1.0 + 0.1 * np.log1p(cond))
    is_stagnant = self.stagnation_counter > adaptive_threshold
    is_converged = pressure < 0.05 or cond > 1e6

    # Graduated diversity injection
    if is_stagnant or is_converged:
        # Strong perturbation: restart along eigendirections
        eigvecs = np.linalg.eigh(self.C)[1]

        k = min(self.dim, max(2, self.dim // 3))
        for i in range(k):
            strength = 3.0 * (k - i) / k  # stronger for top eigenvectors
            self.C += strength * np.outer(eigvecs[:, i], eigvecs[:, i])

        self.C += 0.8 * np.eye(self.dim)

        self.C = self._ensure_positive_definite(self.C)
        self.pc *= 0.05
        self.stagnation_counter = 0
    else:
        # Dynamic learning rate scaling based on convergence pressure
        improv_scale = 1.0 + 5.0 * (1.0 - pressure)
        improv_scale = np.clip(improv_scale, 0.2, 10.0)

        cc_adapt = np.clip(self.cc * improv_scale, 0.001, 0.5)
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        ccov_adapt = np.clip(self.ccov * improv_scale, 1e-10, 0.6)
        self.C = ((1.0 - ccov_adapt) * self.C +
                  ccov_adapt * rank_one +
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)

        # Prevent extreme conditioning
        eigvals_check = np.linalg.eigvalsh(self.C)
        cond_check = np.max(eigvals_check) / max(np.min(eigvals_check), 1e-15)
        if cond_check > 1e6:
            self.C *= 0.5
            self.C = self._ensure_positive_definite(self.C)
```