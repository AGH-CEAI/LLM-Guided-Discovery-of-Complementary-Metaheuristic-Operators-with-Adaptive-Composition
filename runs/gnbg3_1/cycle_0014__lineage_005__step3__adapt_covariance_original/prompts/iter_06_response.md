Looking at the problem:
- **Worst tasks (16-23)**: errors stuck at 10-80, meaning they're ~9 decades from target
- **Original is actually best** on several of these worst tasks, suggesting the current variants are making things worse
- All existing variants use similar matrix-based covariance updates

**Key insight**: The worst tasks likely require **exploration along specific eigendirections** with **aggressive learning** and **periodic restart kicks** to escape deceptive basins. The current approaches all use the same `C = (1-α)C + α*(...)` pattern which is too conservative for these extreme cases.

**Strategy**: Use **eigendecomposition-based covariance shaping** that directly modifies eigenvalues, with adaptive learning rates and periodic "kicks" along the worst-conditioned eigenvectors.

**Idea: Eigen-Decomposition with Directional Kicks**
Uses eigendecomposition to shape covariance along principal axes with adaptive learning rates and periodic exploration kicks.
```python
def _adapt_covariance_original(self):
    """Eigen-decomposition based covariance adaptation with directional kicks."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Compute eigendecomposition of current covariance
    try:
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
    except np.linalg.LinAlgError:
        eigvals = np.ones(self.dim)
        eigvecs = np.eye(self.dim)
    
    # Compute condition number for adaptive learning
    cond = np.max(eigvals) / np.min(eigvals)
    cond = np.clip(cond, 1.0, 1e8)
    
    # Adaptive ccov based on condition
    base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2)
    ccov_adapt = base_ccov * np.sqrt(np.log(cond + 1.0))
    ccov_adapt = np.clip(ccov_adapt, 1e-6, 0.4)
    
    # Adaptive cc based on condition
    cc_adapt = self.cc * np.clip(2.0 / np.log(cond + 2.0), 0.1, 2.0)
    cc_adapt = np.clip(cc_adapt, 0.001, 0.5)
    
    # Evolution path update
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
    
    # Project evolution path to eigenvector basis
    pc_eig = eigvecs.T @ self.pc
    
    # Rank-one update in eigenvector basis (directly modifies eigenvalues)
    rank_one_eig = pc_eig ** 2
    
    # Rank-mu update
    rank_mu_eig = np.zeros(self.dim)
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        diff_eig = eigvecs.T @ diff
        rank_mu_eig += self.weights[i] * (diff_eig ** 2)
    
    # Normalize rank-mu
    rank_mu_sum = np.sum(np.abs(self.weights[:self.mu]))
    if rank_mu_sum > 0:
        rank_mu_eig /= rank_mu_sum
    
    # Adaptive mixing based on condition
    alpha_rank_one = 0.5 + 0.5 / np.log(cond + 2.0)
    alpha_rank_mu = 1.0 - alpha_rank_one
    
    # Update eigenvalues directly
    target_eig = alpha_rank_one * rank_one_eig + alpha_rank_mu * rank_mu_eig
    new_eigvals = (1.0 - ccov_adapt) * eigvals + ccov_adapt * target_eig
    
    # Apply minimum eigenvalue threshold to prevent collapse
    min_eig = np.min(eigvals) * 0.1
    new_eigvals = np.maximum(new_eigvals, min_eig)
    
    # Periodic kick: every 50 generations, boost smallest eigenvalues
    if self.generation > 0 and self.generation % 50 == 0:
        kick_strength = 0.2 * np.mean(eigvals)
        # Kick more along poorly-conditioned directions
        eig_min = np.min(eigvals)
        kick = kick_strength * (1.0 - eigvals / (eig_min + 1e-10))
        kick = np.maximum(kick, 0.0)
        new_eigvals += kick
    
    # Reconstruct covariance matrix
    self.C = eigvecs @ np.diag(new_eigvals) @ eigvecs.T
    self.C = 0.5 * (self.C + self.C.T)
    
    # Full reset on extreme ill-conditioning
    if cond > 1e6:
        self.C = np.eye(self.dim) * np.mean(eigvals)
        self.pc = np.zeros(self.dim)
        self.sigma = min(self.sigma * 2.0, 10.0)
    
    self.C = self._ensure_positive_definite(self.C)
```