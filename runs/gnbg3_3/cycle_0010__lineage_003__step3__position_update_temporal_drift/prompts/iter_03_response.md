**Idea: KL Divergence Distribution Tracking**
Uses KL divergence between Gaussian-fitted population distributions across generations, differential entropy, and per-particle log-likelihood to modulate velocity. Detects population collapse through distributional shift and guides exploration/exploitation via information-theoretic signals.

```python
def _position_update_temporal_drift(self):
    """Information-theoretic velocity modulation (Category C).

    Uses KL divergence between Gaussian-fitted population distributions,
    differential entropy, and per-particle log-likelihood to modulate velocity.
    Detects distributional collapse/shift via information-theoretic signals.
    """
    try:
        # Current population statistics
        mean = np.mean(self.population, axis=0)
        centered = self.population - mean
        
        # Regularized covariance
        cov = np.cov(centered.T)
        cov_reg = cov + 1e-6 * np.eye(self.dim)
        
        # Eigenvalue decomposition for entropy calculation
        eigenvalues = np.linalg.eigvalsh(cov_reg)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Differential entropy of fitted Gaussian: H = 0.5*log((2*pi*e)^d * det(cov))
        det_cov = np.prod(eigenvalues)
        diff_entropy = 0.5 * (self.dim * np.log(2 * np.pi * np.e) + np.log(det_cov + 1e-20))
        
        # Log-likelihood of each particle under fitted Gaussian
        log_likelihoods = np.zeros(self.np)
        for i in range(self.np):
            diff = centered[i]
            mahalanobis = diff @ np.linalg.solve(cov_reg, diff)
            log_likelihoods[i] = -0.5 * mahalanobis - 0.5 * np.log(det_cov + 1e-20) - 0.5 * self.dim * np.log(2 * np.pi)
        
        # KL divergence from isotropic reference distribution
        ref_variance = np.mean(eigenvalues)
        ref_entropy = 0.5 * self.dim * (1.0 + np.log(2 * np.pi * ref_variance + 1e-20))
        kl_from_ref = max(0.0, ref_entropy - diff_entropy)
        
        # Initialize tracking
        if not hasattr(self, '_prev_mean'):
            self._prev_mean = mean.copy()
        if not hasattr(self, '_prev_cov'):
            self._prev_cov = cov_reg.copy()
        if not hasattr(self, '_ema_kl'):
            self._ema_kl = 0.0
        if not hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        
        # KL divergence between current and previous Gaussian distributions
        prev_eigenvalues = np.linalg.eigvalsh(self._prev_cov)
        prev_eigenvalues = np.clip(prev_eigenvalues, 1e-10, None)
        prev_eigenvalues = np.sort(prev_eigenvalues)[::-1]
        prev_det = np.prod(prev_eigenvalues)
        
        # KL(P||Q) = 0.5*(Tr(Sigma_Q^-1 Sigma_P) + (mu_Q-mu_P)^T Sigma_Q^-1 (mu_Q-mu_P) - d + log(det(Sigma_Q)/det(Sigma_P)))
        prev_inv = np.linalg.inv(self._prev_cov)
        tr_term = np.trace(prev_inv @ cov_reg)
        mean_diff = mean - self._prev_mean
        kl_div = 0.5 * (tr_term + mean_diff @ prev_inv @ mean_diff - self.dim + np.log(prev_det / (det_cov + 1e-20)))
        kl_div = max(kl_div, 0.0)
        
        # Update EMAs
        self._ema_kl = 0.3 * kl_div + 0.7 * self._ema_kl
        
        # Improvement tracking
        if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
            improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_best_fitness = self.global_best_fitness
        self._ema_improvement = 0.3 * improvement + 0.7 * self._ema_improvement
        
        # Store current distribution for next iteration
        self._prev_mean = mean.copy()
        self._prev_cov = cov_reg.copy()
        
        # Information-theoretic signals
        entropy_ratio = np.clip(diff_entropy / (ref_entropy + 1e-10), 0.0, 1.0)
        kl_signal = np.clip(self._ema_kl / (kl_from_ref + 0.1), 0.0, 1.0)
        info_signal = entropy_ratio * 0.5 + kl_signal * 0.5
        
        # Exploration factor from distributional shift
        if self._ema_kl > 0.5:
            explore_factor = 1.2
        elif self._ema_kl < 0.1:
            explore_factor = 0.8
        else:
            explore_factor = 1.0
        
        # Base velocity scale from improvement
        if self._ema_improvement > 1e-6:
            base_scale = 1.2
        elif self._ema_improvement > 1e-10:
            base_scale = 1.0
        else:
            base_scale = 0.7
        
        vel_scale = base_scale * explore_factor * (1.0 + 0.3 * info_signal)
        
        # Per-particle scaling: particles in low-density regions get boost
        ll_range = np.max(log_likelihoods) - np.min(log_likelihoods) + 1e-10
        ll_percentile = (log_likelihoods - np.min(log_likelihoods)) / ll_range
        ll_scale = 0.8 + 0.4 * (1.0 - ll_percentile)
        
        new_population = self.population + vel_scale * self.velocity * ll_scale[:, np.newaxis]
        
    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity
    
    self.population = self._clip_to_bounds(new_population)
```