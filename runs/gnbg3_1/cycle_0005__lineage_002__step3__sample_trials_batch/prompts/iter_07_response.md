**Idea: Orthogonal Direction Injection**
Samples trials by combining covariance-guided exploration with fresh orthogonal direction sampling, periodically injecting novel search directions to escape local optima that pure Cholesky sampling cannot break out of.

```python
def _sample_trials_batch(self):
    """Sample trials by injecting fresh orthogonal directions into the covariance-guided search."""
    # Determine if we should inject fresh directions based on stagnation
    inject = (self.stagnation_counter > 10) or (self.generation % 5 == 0)
    
    # Ensure positive definiteness
    min_eig = np.min(np.linalg.eigvalsh(self.C))
    if min_eig < 1e-8:
        self.C += (1e-7 - min_eig) * np.eye(self.dim)
    
    # Compute Cholesky factor
    try:
        L = np.linalg.cholesky(self.C)
    except np.linalg.LinAlgError:
        diag_C = np.diag(self.C)
        diag_C = np.maximum(diag_C, 1e-10)
        L = np.diag(np.sqrt(diag_C))
    
    # Number of fresh orthogonal directions to inject
    n_inject = self.NP // 3 if inject else self.NP // 6
    n_inject = max(1, min(n_inject, self.NP - 1))
    
    # Generate fresh orthogonal directions via QR on random matrix
    Z_fresh = np.random.randn(self.dim, self.dim)
    Q, _ = np.linalg.qr(Z_fresh)
    
    # Sample from standard normal for Cholesky portion
    z_chol = np.random.randn(n_inject, self.dim)
    trials_chol = self.mean + self.sigma * (z_chol @ L.T)
    
    # Sample using fresh orthogonal directions (scaled exploration)
    z_ortho = np.random.randn(n_inject, self.dim)
    trials_ortho = self.mean + self.sigma * 0.5 * (z_ortho @ Q.T)
    
    # Sample remaining from standard normal
    n_remaining = self.NP - 2 * n_inject
    if n_remaining > 0:
        z_rand = np.random.randn(n_remaining, self.dim)
        trials_rand = self.mean + self.sigma * (z_rand @ L.T)
    else:
        trials_rand = np.empty((0, self.dim))
    
    # Combine all trials
    self.trials = np.vstack([trials_chol, trials_ortho, trials_rand])
    
    # Clip to bounds
    self.trials = self._clip_to_bounds(self.trials)
```