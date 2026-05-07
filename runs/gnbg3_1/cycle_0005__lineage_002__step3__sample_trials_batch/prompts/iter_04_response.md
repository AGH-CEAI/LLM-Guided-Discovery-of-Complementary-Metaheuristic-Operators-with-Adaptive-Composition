**Idea: Restart-Injection Hybrid Sampling**

A fundamentally different sampling strategy that periodically injects diversity via restart-based exploration and adaptive scaling to escape local optima traps. This targets the worst tasks (16, 17, 18, 12, 20, 23) where errors in the 10-100 range suggest the current covariance-based sampling is failing to explore effectively, likely due to premature covariance matrix collapse or deceptive basins.

```python
def _sample_trials_batch(self):
    """Sample with restart injection and adaptive scaling for diversity."""
    # Compute current population statistics
    pop_std = np.std(self.population, axis=0)
    pop_mean = np.mean(self.population, axis=0)
    
    # Determine diversity state
    expected_std = (self.ub[0] - self.lb[0]) / 6.0
    diversity_ratio = np.mean(pop_std) / (expected_std + 1e-10)
    
    # Decide sampling strategy based on diversity and stagnation
    inject_restart = (diversity_ratio < 0.1) or (self.stagnation_counter > self.max_stagnation // 2)
    
    if inject_restart:
        # Phase 1: Restart-based exploration (30% of trials)
        n_restart = max(1, self.NP // 3)
        
        # Sample uniformly in bounds with saltation
        restart_trials = np.random.uniform(self.lb, self.ub, (n_restart, self.dim))
        
        # Add perturbation around elite if available
        if hasattr(self, 'x_opt') and self.x_opt is not None:
            n_elite_perturb = max(1, self.NP // 4)
            elite_perturb = self.x_opt + self.sigma * 2.0 * np.random.randn(n_elite_perturb, self.dim)
            elite_perturb = self._clip_to_bounds(elite_perturb)
            restart_trials = np.vstack([restart_trials, elite_perturb])[:n_restart]
        
        # Phase 2: Wide isotropic sampling around mean (30% of trials)
        n_wide = max(1, self.NP // 3)
        z_wide = np.random.randn(n_wide, self.dim)
        wide_trials = pop_mean + self.sigma * 5.0 * z_wide
        
        # Phase 3: Covariance-guided sampling (remaining 40%)
        n_cov = self.NP - n_restart - n_wide
        if n_cov > 0:
            min_eig = np.min(np.linalg.eigvalsh(self.C))
            if min_eig < 1e-8:
                self.C += (1e-7 - min_eig) * np.eye(self.dim)
            try:
                L = np.linalg.cholesky(self.C)
            except np.linalg.LinAlgError:
                diag_C = np.diag(self.C)
                diag_C = np.maximum(diag_C, 1e-10)
                L = np.diag(np.sqrt(diag_C))
            z_cov = np.random.randn(n_cov, self.dim)
            cov_trials = self.mean + self.sigma * (z_cov @ L.T)
        else:
            cov_trials = np.empty((0, self.dim))
        
        self.trials = np.vstack([restart_trials, wide_trials, cov_trials])
    else:
        # Standard covariance-guided sampling with adaptive scaling
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)
        try:
            L = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            diag_C = np.diag(self.C)
            diag_C = np.maximum(diag_C, 1e-10)
            L = np.diag(np.sqrt(diag_C))
        
        # Adaptive sigma scaling based on convergence state
        if self.generation > 0:
            improvement_ratio = abs(self.f_opt - self.f_opt_prev) / (abs(self.f_opt_prev) + 1e-10)
            if improvement_ratio < 1e-6:
                # Stagnation detected: increase exploration radius
                effective_sigma = self.sigma * 3.0
            else:
                effective_sigma = self.sigma
        else:
            effective_sigma = self.sigma
        
        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + effective_sigma * (z @ L.T)
    
    # Clip to bounds
    self.trials = self._clip_to_bounds(self.trials)
    
    # Ensure we have exactly NP trials
    if len(self.trials) < self.NP:
        padding = np.random.uniform(self.lb, self.ub, (self.NP - len(self.trials), self.dim))
        self.trials = np.vstack([self.trials, padding])
    elif len(self.trials) > self.NP:
        self.trials = self.trials[:self.NP]
```