Looking at the data, all 24 tasks remain unsolved with the worst having errors ~250 (Task 17) and the best unsolved still at ~3.9e-04 (Task 5). The current CSA (Cumulative Step-size Adaptation) relies only on the evolution path `ps` and uses a fixed exponential damping, which fails on multi-modal and ill-conditioned landscapes where the path gets corrupted or misaligned.

**Idea: Multi-Feedback Adaptive Step-Size with Diversity-Sensitive Damping**

Instead of relying solely on the evolution path, this approach combines three orthogonal signals: (1) fitness gradient direction for immediate improvement feedback, (2) population diversity ratio to detect premature convergence and adjust damping, and (3) stagnation-aware step-size recovery that exponentially expands sigma when progress stalls. This breaks the dependency on a single potentially-corrupted signal and provides adaptive damping that responds to the actual search landscape geometry.

```python
def _adapt_step_size(self):
    """Adapt step-size using multi-feedback signals with diversity-sensitive damping."""
    y_mean = (self.mean - self.old_mean) / self.sigma
    
    # Signal 1: Evolution path (CSA baseline)
    self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean
    path_norm = np.linalg.norm(self.ps) / np.sqrt(self.dim)
    
    # Signal 2: Fitness gradient direction
    if hasattr(self, 'prev_f_opt'):
        f_improved = self.f_opt < self.f_opt_prev - 1e-12
        fit_grad = np.sign(self.f_opt_prev - self.f_opt) if hasattr(self, 'prev_f_opt') else 0.0
    else:
        f_improved = False
        fit_grad = 0.0
    
    # Signal 3: Diversity ratio for adaptive damping
    pop_variance = np.mean(np.var(self.population, axis=0))
    expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
    diversity_ratio = pop_variance / (expected_var + 1e-10)
    diversity_ratio = np.clip(diversity_ratio, 1e-4, 10.0)
    
    # Stagnation detection
    stagnation = self.stagnation_counter > self.max_stagnation // 2
    
    # Adaptive damping: increase when diverse (explore more), decrease when converged (exploit)
    base_damping = self.damping
    if stagnation:
        adaptive_damping = base_damping * 0.3  # Lower damping = faster sigma growth
    elif diversity_ratio < 0.1:
        adaptive_damping = base_damping * 0.5  # Low diversity: encourage exploration
    elif diversity_ratio > 0.5:
        adaptive_damping = base_damping * 1.5  # High diversity: be more conservative
    else:
        adaptive_damping = base_damping
    
    # Combined step-size update with multiple feedback signals
    # Primary: evolution path feedback
    csa_term = (path_norm - 1.0) * self.cs / adaptive_damping
    
    # Secondary: fitness-based adjustment
    if f_improved:
        fitness_adjustment = -0.1 / adaptive_damping  # Small reward for improvement
    else:
        fitness_adjustment = 0.05 / adaptive_damping  # Small penalty for no improvement
    
    # Tertiary: diversity-based adjustment (prevent collapse or explosion)
    if diversity_ratio < 0.05:
        diversity_adjustment = 0.2 / adaptive_damping  # Prevent collapse
    elif diversity_ratio > 2.0:
        diversity_adjustment = -0.1 / adaptive_damping  # Slow down if too spread
    else:
        diversity_adjustment = 0.0
    
    # Combine all signals
    total_adjustment = csa_term + fitness_adjustment + diversity_adjustment
    
    # Apply with safeguards
    self.sigma *= np.exp(np.clip(total_adjustment, -2.0, 2.0))
    self.sigma = np.clip(self.sigma, 1e-10, 10.0)
```