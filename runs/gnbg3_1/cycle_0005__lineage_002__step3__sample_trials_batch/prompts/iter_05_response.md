**Idea: Restart-Driven Diversity Sampling with Adaptive Exploration Escalation**

This variant directly targets the stuck-at-~1e+01 errors by combining restart-history diversity injection with adaptive exploration escalation when stagnant. Instead of relying solely on the slowly-adapting covariance matrix, it periodically injects solutions from previous restarts (which often contain basin-of-attraction information) and exponentially scales the sampling radius when the algorithm fails to improve, enabling escape from local optima that Cholesky-based sampling cannot overcome.

```python
def _sample_trials_batch(self):
    """Sample with restart-history diversity and adaptive exploration escalation."""
    # --- Adaptive exploration scaling based on stagnation ---
    stagnation_ratio = min(self.stagnation_counter / max(self.max_stagnation, 1), 1.0)
    exploration_scale = 1.0 + 15.0 * stagnation_ratio**2
    current_sigma = self.sigma * exploration_scale
    
    # --- Restart history diversity injection ---
    num_restart_points = 0
    if hasattr(self, 'restart_history') and len(self.restart_history) > 0:
        num_restart_points = min(8, len(self.restart_history))
    
    # --- Eigendecomposition for axis-aligned sampling ---
    eigvals, eigvecs = np.linalg.eigh(self.C)
    eigvals = np.maximum(eigvals, 1e-10)
    eigvals_sorted = np.sort(eigvals)[::-1]
    
    # Adaptive conditioning: shrink large eigenvalues, boost small ones
    max_cond = 1e6
    cond = eigvals_sorted[0] / eigvals_sorted[-1]
    if cond > max_cond:
        shrink_factor = (cond / max_cond) ** (1.0 / (self.dim - 1))
        eigvals_sorted[1:] = eigvals_sorted[1:] / shrink_factor
        eigvals_sorted[0] = eigvals_sorted[1] * max_cond
    
    # Reconstruct conditioned covariance
    C_conditioned = eigvecs @ np.diag(eigvals_sorted) @ eigvecs.T
    
    # --- Sample base population from conditioned distribution ---
    n_base = self.NP - num_restart_points
    if n_base > 0:
        L_cond = np.linalg.cholesky(C_conditioned + 1e-10 * np.eye(self.dim))
        z_base = np.random.randn(n_base, self.dim)
        trials_base = self.mean + current_sigma * (z_base @ L_cond.T)
    else:
        trials_base = np.empty((0, self.dim))
    
    # --- Inject restart history diversity ---
    restart_points = np.empty((0, self.dim))
    if num_restart_points > 0:
        indices = np.random.choice(len(self.restart_history), num_restart_points, replace=False)
        for idx in indices:
            history_point = self.restart_history[idx]
            # Perturb around restart point with decreasing scale
            pert_scale = current_sigma * np.random.uniform(0.5, 2.0)
            z_pert = np.random.randn(self.dim)
            pert_point = history_point + pert_scale * z_pert
            restart_points = np.vstack([restart_points, pert_point]) if restart_points.size else pert_point
    
    # --- Combine and clip ---
    self.trials = np.vstack([trials_base, restart_points])
    self.trials = self._clip_to_bounds(self.trials)
```