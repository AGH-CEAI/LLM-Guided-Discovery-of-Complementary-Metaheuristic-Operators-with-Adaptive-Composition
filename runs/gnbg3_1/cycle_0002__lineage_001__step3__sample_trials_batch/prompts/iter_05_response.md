Looking at the error magnitudes, the worst unsolved tasks (errors 10-100+) likely involve deceptive global structure, narrow basins, or require explosive exploration phases to escape local optima. The current sampling is standard CMA-ES-style multivariate normal around the mean, which can get trapped.

**Idea: Pulsed Explosive Exploration with Stagnation-Triggered Restart Sampling**

This approach fundamentally changes sampling behavior by detecting stagnation and triggering explosive exploration phases where the search radius expands dramatically and momentum is reset. This directly targets the failure mode where CMA-ES converges prematurely on multimodal deceptive landscapes.

```python
def _sample_trials_batch(self):
    """Sample with pulsed explosive exploration on stagnation."""
    # Check for stagnation to trigger exploration burst
    is_stagnant = (self.stagnation_counter > self.max_stagnation // 2)
    
    # Track exploration burst state
    if not hasattr(self, 'in_explosion'):
        self.in_explosion = False
        self.explosion_gen = 0
        self.sigma_backup = None
    
    # Trigger explosive exploration phase
    if is_stagnant and not self.in_explosion:
        self.in_explosion = True
        self.explosion_gen = 0
        # Backup current sigma for later restoration
        self.sigma_backup = self.sigma
        # Explosive expansion: increase sigma by factor of 10-20
        self.sigma = min(self.sigma * 15.0, (self.ub[0] - self.lb[0]) * 0.4)
        # Reset evolution path to remove misleading momentum
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        # Reset covariance to near-identity (isotropic exploration)
        self.C = 0.5 * np.eye(self.dim) + 0.5 * np.diag(np.diag(self.C))
    
    if self.in_explosion:
        self.explosion_gen += 1
        # Decay explosion over ~20 generations
        explosion_decay = max(0.05, 1.0 - self.explosion_gen / 20.0)
        
        # Sample from current covariance (reset to near-isotropic)
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        # Wide isotropic sampling with decaying radius
        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * explosion_decay * (z @ L.T)
        self.trials = self._clip_to_bounds(self.trials)
        
        # Exit explosion phase after decay
        if self.explosion_gen >= 20 and self.sigma_backup is not None:
            self.in_explosion = False
            self.sigma = self.sigma_backup
            self.explosion_gen = 0
            self.sigma_backup = None
    else:
        # Standard CMA-ES sampling
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)
        self.trials = self._clip_to_bounds(self.trials)
```