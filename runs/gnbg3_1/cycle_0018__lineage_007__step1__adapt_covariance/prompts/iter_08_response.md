Looking at the error patterns, I see two distinct failure modes:
1. **Tasks 16-23 (errors 1e+01)**: Getting trapped in local optima with poor exploration
2. **Tasks 0-9 (errors 1e-05 to 1e-01)**: Converging to wrong regions, unable to refine further

All existing variants modify learning rates or add perturbations but never **directly control the exploration geometry**. The worst tasks need a fundamentally different approach: directly manipulating eigenvalues to escape local optima and refine toward global solutions.

**Idea: Eigendirection Expansion with Local Optima Detection**

Instead of tweaking learning rates, this strategy performs explicit eigendecomposition to detect when the covariance matrix has collapsed (small eigenvalues = no exploration in that direction). It then **directly expands along the smallest eigendirections** proportional to the inverse condition number, forcing exploration in directions that would otherwise be ignored. This is fundamentally different because it directly controls the shape of the search distribution rather than using scalar learning rate adjustments.

```python
def _adapt_covariance(self):
    """Eigendirection expansion with local optima escape and fine-grained refinement."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Evolution path update
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    # Rank-one and rank-mu updates
    rank_one = np.outer(self.pc, self.pc)
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Eigendecomposition for direct geometry control
    try:
        eigvals, eigvecs = np.linalg.eigh(self.C)
    except np.linalg.LinAlgError:
        eigvals = np.diag(self.C)
        eigvecs = np.eye(self.dim)
    
    eigvals = np.maximum(eigvals, 1e-10)
    
    # Detect convergence state
    cond = np.max(eigvals) / (np.min(eigvals) + 1e-10)
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
    
    # Detect stagnation
    improvement = max(0.0, self.f_opt_prev - self.f_opt) if hasattr(self, 'f_opt_prev') else 0.0
    is_stagnant = improvement < 1e-10
    
    # Determine exploration mode
    is_collapsed = (cond > 1e4) or (diversity < 0.01) or is_stagnant
    
    if is_collapsed:
        # MODE 1: Escape local optima by expanding smallest eigendirections
        # Scale factor inversely proportional to condition number
        escape_strength = min(0.5, 0.1 / np.log1p(cond))
        
        # Expand along smallest 30% of eigendirections
        n_expand = max(1, self.dim // 3)
        expand_indices = np.argsort(eigvals)[:n_expand]
        
        # Compute perturbation proportional to inverse eigenvalue
        for idx in expand_indices:
            v = eigvecs[:, idx]
            # Larger expansion for smaller eigenvalues
            expansion = escape_strength * np.sqrt(eigvals[idx]) * 10.0
            self.C += expansion * np.outer(v, v)
        
        # Also add isotropic exploration
        self.C += 0.1 * np.eye(self.dim)
        
    else:
        # MODE 2: Normal CMA-ES update with enhanced refinement
        # Use slightly higher learning rate for faster convergence on easy tasks
        ccov_scale = min(1.5, 1.0 + 2.0 * diversity)
        ccov_adaptive = min(self.ccov * ccov_scale, 0.4)
        
        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one +
                  ccov_adaptive * (1.0 - 1.0 / self.mueff) * 2.0 * rank_mu)
        
        # MODE 3: Fine-grained refinement when close to optimum
        if self.f_opt < 1.0:
            # Increase exploration along all eigendirections uniformly
            refinement_strength = 0.02 * (1.0 - np.clip(self.f_opt, 0, 1))
            self.C += refinement_strength * np.eye(self.dim)
    
    # Ensure positive definiteness
    self.C = self._ensure_positive_definite(self.C)
    
    # Emergency reset for extreme ill-conditioning
    eigvals_check = np.linalg.eigvalsh(self.C)
    cond_check = np.max(eigvals_check) / (np.min(eigvals_check) + 1e-10)
    if cond_check > 1e8:
        self.C = np.eye(self.dim) * np.mean(eigvals_check)
        self.pc = np.zeros(self.dim)
```