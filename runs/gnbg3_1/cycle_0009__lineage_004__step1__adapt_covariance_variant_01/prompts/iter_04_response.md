Looking at the error patterns, the worst tasks (16-23 with errors 10-80) are likely suffering from:
1. **Premature convergence** - covariance matrix becoming stale/misleading
2. **Eigenvalue collapse** - eigenvalues drifting to extreme ratios, making the distribution ellipsoid unrepresentative of the fitness landscape
3. **Ill-conditioned covariance** - condition numbers reaching levels where sampling becomes degenerate

The current variant_01 monitors condition number but doesn't actively **smooth eigenvalue drift** over time. The key insight is that the worst tasks need **explicit eigenvalue regularization** that forces the covariance toward a healthier spectrum, preventing the algorithm from getting stuck in degenerate geometries.

**Idea: Eigenvalue History Smoothing with Adaptive Regularization**
Actively track eigenvalue history, smooth the eigenvalue spectrum using exponential moving average, and apply progressive regularization when eigenvalues drift toward pathological distributions. This fundamentally differs from condition-number triggering by implementing proactive spectrum management rather than reactive restarts.

```python
def _adapt_covariance_variant_01(self):
    """Eigenvalue history smoothing with adaptive regularization for ill-conditioned landscapes."""
    y_mean = (self.mean - self.old_mean) / self.sigma

    # Compute current eigenvalues and condition number
    eigvals = np.linalg.eigvalsh(self.C)
    cond = np.max(eigvals) / (np.maximum(np.min(eigvals), 1e-10))
    eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)

    # Track eigenvalue history for drift detection
    if not hasattr(self, 'eigval_history'):
        self.eigval_history = []
        self.eigval_ema = eigvals.copy()
    
    self.eigval_history.append(eigvals.copy())
    if len(self.eigval_history) > 10:
        self.eigval_history.pop(0)

    # Exponential smoothing of eigenvalues (spectrum adaptation)
    alpha_smooth = 0.1
    self.eigval_ema = (1.0 - alpha_smooth) * self.eigval_ema + alpha_smooth * eigvals

    # Detect eigenvalue drift (ratio of current to EMA)
    if len(self.eigval_history) >= 3:
        drift_ratio = np.median(eigvals) / (np.median(self.eigval_ema) + 1e-10)
        drift_ratio = np.clip(drift_ratio, 0.1, 10.0)
    else:
        drift_ratio = 1.0

    # Adaptive regularization based on eigenvalue health
    reg_strength = 0.0
    if cond > 1e6 or eig_spread < 1e-6:
        reg_strength = 0.1
    elif cond > 1e4 or eig_spread < 1e-4:
        reg_strength = 0.05
    elif drift_ratio < 0.5 or drift_ratio > 2.0:
        reg_strength = 0.02

    # Force eigenvalues toward geometric mean for badly conditioned cases
    if cond > 1e7 or eig_spread < 1e-8:
        target_eigvals = np.full(self.dim, np.exp(np.mean(np.log(eigvals + 1e-10))))
        self.eigval_ema = (1.0 - alpha_smooth) * self.eigval_ema + alpha_smooth * target_eigvals

    # Adaptive learning rates with eigenvalue-health scaling
    if cond > 1e6:
        ccov_scale = 0.1
        cc_scale = 0.1
    elif cond > 1e4:
        ccov_scale = 0.3
        cc_scale = 0.3
    elif cond > 1e2:
        ccov_scale = 0.6
        cc_scale = 0.6
    else:
        ccov_scale = 1.0
        cc_scale = 1.0

    # Scale down when eigenvalues are drifting
    drift_penalty = 0.5 if drift_ratio < 0.5 or drift_ratio > 2.0 else 1.0
    ccov_scale *= drift_penalty
    cc_scale *= drift_penalty

    # Fitness-based adaptation signal
    fit_var = np.var(self.fitness)
    fit_scale = max(abs(self.f_opt), 1.0)
    rel_var = fit_var / (fit_scale ** 2 + 1e-10)

    if rel_var < 1e-4:
        ccov_scale *= 0.5
        cc_scale *= 0.5

    # Trigger covariance restart on extreme condition number
    if cond > 1e8 or eig_spread < 1e-8:
        self.C = np.eye(self.dim) * np.mean(eigvals)
        self.pc = np.zeros(self.dim)
        self.L = None
        self.eigval_ema = np.full(self.dim, np.mean(eigvals))
        return

    # Adaptive learning rates
    ccov_1 = ccov_scale * 1.0 / (self.dim + 2.0)
    ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
    cc_adapt = cc_scale * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)

    # Evolution path with adaptive rate
    self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

    # Rank-one update
    rank_one = np.outer(self.pc, self.pc)

    # Rank-mu update
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)

    # Combine updates
    self.C = ((1.0 - ccov_1 - ccov_mu) * self.C + 
              ccov_1 * rank_one + 
              ccov_mu * rank_mu)

    # Apply eigenvalue regularization
    if reg_strength > 0:
        current_trace = np.trace(self.C)
        target_trace = np.sum(self.eigval_ema)
        self.C = (1.0 - reg_strength) * self.C + reg_strength * (target_trace / self.dim) * np.eye(self.dim)

    self.C = self._ensure_positive_definite(self.C)
```