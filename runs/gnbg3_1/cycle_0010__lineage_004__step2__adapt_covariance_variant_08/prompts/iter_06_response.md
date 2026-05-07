**Idea: Coordinate System Rotation with Stagnation Detection**
A fundamentally different approach that explicitly rotates the covariance matrix's coordinate system when stagnation is detected, targeting non-separable and multi-modal landscapes that trap the current variants.

```python
def _adapt_covariance_variant_08(self):
    """Coordinate system rotation for non-separable and multi-modal landscapes."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    # Stagnation detection for rotation trigger
    if not hasattr(self, 'rotation_stagnation'):
        self.rotation_stagnation = 0
    
    is_improving = self.f_opt < getattr(self, 'prev_f_opt_rot', self.f_opt) - 1e-10
    self.prev_f_opt_rot = self.f_opt
    
    if is_improving:
        self.rotation_stagnation = 0
    else:
        self.rotation_stagnation += 1
    
    # Rotate covariance matrix when stuck (handles non-separability)
    if self.rotation_stagnation >= 3 and np.linalg.norm(self.pc) > 1e-10:
        # Compute rotation from evolution path direction
        pc_norm = np.linalg.norm(self.pc)
        u = self.pc / pc_norm
        
        # Householder reflection for orthogonal rotation
        v = u / (1.0 + abs(np.dot(u, np.ones(self.dim) / np.sqrt(self.dim)) + 1e-10))
        v = v - (np.dot(v, np.ones(self.dim)) / self.dim) * np.ones(self.dim)
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-10:
            v = v / v_norm
            # Apply rotation: C <- Q @ C @ Q.T where Q = I - 2vv^T
            self.C = self.C - 2.0 * np.outer(v, np.dot(v, self.C))
            self.C = self.C - 2.0 * np.outer(np.dot(self.C, v), v) + 4.0 * np.outer(np.dot(v, self.C), v)
        
        # Boost exploration along rotation axes
        self.C += 0.1 * np.eye(self.dim)
        self.rotation_stagnation = 0
    
    # Standard rank-one and rank-mu updates
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    self.C = ((1.0 - self.ccov) * self.C + 
              self.ccov * rank_one +
              (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)
    
    # Inject diagonal exploration on stagnation to escape local optima
    if self.rotation_stagnation >= 2:
        self.C += 0.05 * np.eye(self.dim)
    
    # Ensure positive definiteness and condition number bound
    eigvals = np.linalg.eigvalsh(self.C)
    eig_min = np.min(eigvals)
    eig_max = np.max(eigvals)
    
    if eig_min < 1e-10:
        self.C += (1e-9 - eig_min) * np.eye(self.dim)
    
    cond = eig_max / max(eig_min, 1e-10)
    if cond > 1e7:
        self.C *= 0.5
```