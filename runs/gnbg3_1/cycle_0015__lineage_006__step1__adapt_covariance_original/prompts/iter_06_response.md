Looking at the error patterns, tasks 16, 17, 23, 20, 11, 19, 18, 12 are all stuck at errors 10-70, suggesting they're trapped in deceptive local optima or have highly ill-conditioned landscapes. All existing variants use the evolution path `pc` for rank-one updates, which can accumulate momentum in misleading directions when stuck.

**Idea: Natural Evolution Strategy (NES) Gradient Adaptation**

Instead of using the evolution path `pc`, use the **natural gradient** of expected fitness under the sampling distribution. This directly computes which direction maximizes fitness improvement, bypassing the accumulated momentum that traps CMA-ES variants in local optima.

```python
def _adapt_covariance_original(self):
    """Natural Evolution Strategy (NES) gradient-based covariance adaptation."""
    # Compute normalized fitness weights (fitness-shaping) for gradient estimation
    fitness_array = np.array(self.fitness, dtype=np.float64)
    f_min = np.min(fitness_array)
    f_max = np.max(fitness_array)
    fit_range = max(f_max - f_min, 1e-10)
    
    # Rank-based fitness shaping: u_i = max(0, log(mu/2+1) - log(rank_i))
    sorted_indices = np.argsort(fitness_array)
    u_weights = np.zeros(self.NP)
    for rank, idx in enumerate(sorted_indices[:self.mu]):
        u_weights[idx] = max(0.0, np.log(self.mu / 2.0 + 1.0) - np.log(rank + 1.0))
    
    u_sum = np.sum(u_weights)
    if u_sum > 1e-10:
        u_weights /= u_sum
    else:
        u_weights = self.weights[:self.NP]
        u_weights /= np.sum(u_weights)
    
    # Compute NES gradient: sum_i u_i * grad_log_p(x_i) where p is the sampling distribution
    # For Gaussian distribution N(mean, sigma^2 * C):
    # grad_log_p(x) = -0.5 * inv(C) @ (x - mean) @ (x - mean).T + 0.5 * inv(C)
    # Weighted sum gives the natural gradient direction for covariance
    
    # Ensure C is invertible
    min_eig = np.min(np.linalg.eigvalsh(self.C))
    C_reg = self.C.copy()
    if min_eig < 1e-8:
        C_reg += (1e-7 - min_eig) * np.eye(self.dim)
    
    try:
        C_inv = np.linalg.inv(C_reg)
    except np.linalg.LinAlgError:
        diag_C = np.diag(C_reg)
        diag_C = np.maximum(diag_C, 1e-10)
        C_inv = np.diag(1.0 / diag_C)
    
    # Compute mean-centered deviations
    centered = self.population[:self.NP] - self.mean  # (NP, dim)
    
    # Compute covariance gradient: G = sum_i u_i * (y_i @ y_i^T - C) where y_i = C^{-1/2} @ (x_i - mean)
    # Use eigen decomposition for stable C^{-1/2}
    eigvals, eigvecs = np.linalg.eigh(C_reg)
    eigvals = np.maximum(eigvals, 1e-10)
    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    
    # Compute whitened coordinates
    y_coords = centered @ C_inv_sqrt.T  # (NP, dim)
    
    # Natural gradient for mean
    grad_mean = np.zeros(self.dim)
    for i in range(self.NP):
        grad_mean += u_weights[i] * y_coords[i]
    grad_mean = C_inv_sqrt @ grad_mean
    
    # Natural gradient for covariance (Friston's precision update)
    grad_cov = np.zeros((self.dim, self.dim))
    for i in range(self.NP):
        outer_y = np.outer(y_coords[i], y_coords[i])
        grad_cov += u_weights[i] * (outer_y - np.eye(self.dim))
    
    # Learning rates for NES
    eta_mean = 1.0
    eta_cov = 0.5 / (self.dim + 2.0)
    
    # Update mean using natural gradient
    self.mean = self.mean + eta_mean * self.sigma * (C_inv_sqrt @ grad_mean)
    
    # Update covariance using natural gradient with momentum
    delta_cov = eta_cov * grad_cov
    
    # Add momentum to prevent erratic updates
    if not hasattr(self, 'cov_momentum'):
        self.cov_momentum = np.zeros((self.dim, self.dim))
    self.cov_momentum = 0.7 * self.cov_momentum + 0.3 * delta_cov
    
    # Adaptive learning rate based on fitness gradient magnitude
    fitness_improvement = abs(f_min - f_max) / max(abs(f_min), 1.0)
    lr_scale = np.clip(1.0 + np.log1p(fitness_improvement * 10.0), 0.1, 5.0)
    
    # Apply update with adaptive scaling
    self.C = self.C + lr_scale * self.cov_momentum
    
    # Ensure positive definiteness
    self.C = self._ensure_positive_definite(self.C)
```