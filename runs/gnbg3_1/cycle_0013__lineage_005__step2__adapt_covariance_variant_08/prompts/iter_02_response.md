**Idea: Multi-Trigger Adaptive Eigenspace Rescaling**

A fundamentally different strategy that monitors multiple failure signals (stagnation, ill-conditioning, fitness collapse) and applies adaptive rescaling across ALL eigendirections with inverse-strength weighting toward minor axes, plus a correlation-matrix repair mechanism to break deceptive ridge structures.

```python
def _adapt_covariance_variant_08(self):
    """Multi-trigger adaptive eigenspace rescaling with minor-axis emphasis."""
    # --- Multi-failure detection ---
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)

    eigvals, eigvecs = np.linalg.eigh(self.C)
    eigvals = np.maximum(eigvals, 1e-12)
    cond = np.max(eigvals) / np.min(eigvals)

    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = pop_variance / (expected_var + 1e-10)

    # Trigger conditions: stagnation OR high condition OR fitness collapse OR low diversity
    is_stagnant = self.stagnation_counter > max(20, self.dim)
    is_ill_cond = cond > 1e5
    is_collapsed = rel_var < 1e-3
    is_low_diversity = diversity < 1e-3
    needs_rescue = is_stagnant or is_ill_cond or is_collapsed or is_low_diversity

    if needs_rescue:
        # Compute adaptive rescue strength based on condition number
        log_cond = np.log10(cond + 1.0)
        rescue_strength = np.clip(0.5 + 0.5 * log_cond, 0.5, 5.0)

        # Inverse-strength weighting: amplify minor axes more
        sorted_indices = np.argsort(eigvals)
        for idx, i in enumerate(sorted_indices):
            # Rank-based strength: minor axes get MORE perturbation
            strength = rescue_strength * (idx + 1) / self.dim
            self.C += strength * np.outer(eigvecs[:, i], eigvecs[:, i])

        # Add uniform isotropic perturbation for baseline exploration
        self.C += 0.3 * np.eye(self.dim)

        # Correlation repair: reduce off-diagonal elements that create deceptive ridges
        diag_C = np.diag(self.C)
        diag_sqrt = np.sqrt(np.maximum(diag_C, 1e-10))
        corr = self.C / np.outer(diag_sqrt, diag_sqrt)
        corr = np.clip(corr, -0.95, 0.95)
        self.C = np.diag(diag_C) + 0.2 * (corr - np.diag(np.diag(corr)))

        self.C = self._ensure_positive_definite(self.C)
        self.pc *= 0.2
        self.stagnation_counter = 0
    else:
        # Standard covariance update with condition-aware scaling
        y_mean = (self.mean - self.old_mean) / self.sigma

        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        improv_scale = np.clip(1.0 + 5.0 * np.log1p(improvement * 1e6), 0.2, 3.0)

        # Reduce learning rate when covariance is already well-conditioned
        cond_factor = np.clip(1.0 / np.log10(cond + 10.0), 0.1, 1.0)

        cc_adapt = np.clip(self.cc * improv_scale * cond_factor, 0.001, 0.3)
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        ccov_adapt = np.clip(self.ccov * improv_scale * cond_factor, 1e-10, 0.5)
        self.C = ((1.0 - ccov_adapt) * self.C +
                  ccov_adapt * rank_one +
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)

        # Passive condition control
        eigvals_check = np.linalg.eigvalsh(self.C)
        cond_check = np.max(eigvals_check) / np.min(eigvals_check)
        if cond_check > 1e7:
            self.C *= 0.5
```