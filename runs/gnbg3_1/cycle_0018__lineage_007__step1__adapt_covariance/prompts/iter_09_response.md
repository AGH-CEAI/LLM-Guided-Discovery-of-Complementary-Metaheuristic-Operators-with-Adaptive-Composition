Looking at the unsolved tasks and the failure modes, I need to understand what's happening:

**Analysis of the Problem:**
- Tasks 16-23 (the worst) have errors of 10-70, meaning the algorithm gets stuck ~9+ decades away from the target
- All existing variants (original, archive-guided, diversity-sensitive, active CMA-ES, exploration temperature) are **CMA-ES-centric** - they adjust learning rates, add perturbations, or use archives but stay within the standard framework
- The consistent failure across ALL variants suggests the population is collapsing into local optima basins and cannot escape

**Why current approaches fail:**
The worst tasks likely have **deceptive local optima** - funnels that lead to local minima far from the global optimum. Archive-based and learning-rate approaches help but don't provide the **directional momentum** needed to break out of these deep basins.

**Proposed Strategy:**
Use **momentum-directed covariance injection** - track the direction of improvement in best-so-far positions, compute a momentum vector, and when stuck, inject a **strong anisotropic perturbation** to the covariance matrix along this momentum direction. This is fundamentally different from archives (which store positions) - it computes and uses the **direction vector** itself to bias exploration.

**Idea: Momentum-Directed Covariance Injection**

```python
def _adapt_covariance(self):
    """Momentum-directed covariance adaptation for escaping persistent local optima."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
    
    # Initialize momentum tracking
    if not hasattr(self, 'best_trajectory'):
        self.best_trajectory = []
        self.momentum_dir = np.zeros(self.dim)
        self.stuck_counter = 0
    
    # Track best position history for momentum
    self.best_trajectory.append(self.x_opt.copy())
    if len(self.best_trajectory) > 20:
        self.best_trajectory.pop(0)
    
    # Detect stagnation (half of max stagnation threshold)
    is_stuck = self.stagnation_counter > self.max_stagnation // 2
    
    # Compute momentum from best trajectory (direction of improvement)
    if len(self.best_trajectory) >= 5:
        recent_best = self.best_trajectory[-1]
        older_best = self.best_trajectory[0]
        raw_momentum = recent_best - older_best
        
        if np.linalg.norm(raw_momentum) > 1e-10:
            self.momentum_dir = 0.9 * self.momentum_dir + 0.1 * raw_momentum
            self.momentum_dir /= np.linalg.norm(self.momentum_dir)
    
    if is_stuck:
        self.stuck_counter += 1
    else:
        self.stuck_counter = max(0, self.stuck_counter - 1)
    
    # Rank-one and rank-mu updates
    rank_one = np.outer(self.pc, self.pc)
    
    rank_mu = np.zeros((self.dim, self.dim))
    for i in range(self.mu):
        diff = (self.population[i] - self.old_mean) / self.sigma
        rank_mu += self.weights[i] * np.outer(diff, diff)
    
    # Adaptive learning rates
    if is_stuck:
        ccov_1 = 0.8 / (self.dim + 2.0)
        ccov_mu = 0.8 * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
    else:
        ccov_1 = self.ccov * 0.5
        ccov_mu = self.ccov * 0.5
    
    # Base covariance update
    self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
              ccov_1 * rank_one +
              ccov_mu * rank_mu)
    
    # Inject momentum-directed perturbation when stuck
    if is_stuck and self.stuck_counter >= 3 and np.linalg.norm(self.momentum_dir) > 1e-10:
        # Strong anisotropic perturbation along momentum direction
        stuck_severity = min(self.stuck_counter / 20.0, 1.0)
        momentum_perturb = stuck_severity * np.outer(self.momentum_dir, self.momentum_dir)
        self.C = self.C + 0.5 * momentum_perturb
    
    self.C = self._ensure_positive_definite(self.C)
```