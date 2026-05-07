**Idea: Spectral Entropy Velocity Modulation**

One-line description: Use eigenvalue-based spectral entropy and condition number from population covariance to modulate velocity scaling, replacing temporal autocorrelation with spectral distribution analysis (Category B: Spectral / linear-algebraic).

```python
def _position_update_temporal_drift(self):
    """Spectral entropy-based velocity modulation (Category B).
    
    Replaces temporal autocorrelation with eigenvalue distribution analysis.
    Spectral entropy detects convergence from covariance shape; condition
    number drives anisotropic scaling. Targets worst tasks with large
    condition numbers (17, 16, 6, 11) where population collapses to subspaces.
    """
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        total_var = np.sum(eigenvalues) + 1e-10
        eigenvalues_norm = eigenvalues / total_var
        
        spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
        max_entropy = np.log(self.dim + 1e-10)
        entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)
        
        cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        cond_log = np.log1p(cond)
        
        cumvar = np.cumsum(eigenvalues) / total_var
        eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
        eff_dim_ratio = eff_dim / self.dim
        
        spectral_signal = entropy_ratio * 0.5 + (1.0 - eff_dim_ratio) * 0.5
        
        if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
            improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_best_fitness = self.global_best_fitness
        
        if not hasattr(self, '_spectral_ema_improvement'):
            self._spectral_ema_improvement = 0.0
        self._spectral_ema_improvement = 0.3 * improvement + 0.7 * self._spectral_ema_improvement
        
        if self._spectral_ema_improvement > 1e-6:
            base_scale = 1.2
        elif self._spectral_ema_improvement > 1e-10:
            base_scale = 1.0
        else:
            base_scale = 0.7
        
        temporal_scale = base_scale * (1.0 + 0.4 * spectral_signal)
        
        if cond > 100.0:
            log_damp = np.log1p(cond) / np.log1p(1000.0)
            log_damp = np.clip(log_damp, 0.0, 1.0)
            temporal_scale *= (1.0 + 0.3 * log_damp)
        
        U = np.linalg.eigvalsh(cov)
        idx = np.argsort(U)[::-1]
        U = U[idx]
        V = np.linalg.eigh(cov)[1][:, idx]
        
        vel_proj = self.velocity @ V
        sv_scale = np.sqrt(U + 1e-10)
        sv_norm = sv_scale / (sv_scale[0] + 1e-10)
        per_comp_scale = np.clip(sv_norm, 0.1, 2.0)
        vel_scaled = vel_proj * per_comp_scale
        new_population = self.population + temporal_scale * (vel_scaled @ V.T)
        
    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity
    
    self.population = self._clip_to_bounds(new_population)
```