Looking at the task coverage, all 24 tasks remain unsolved with errors ranging from ~1e-2 to ~1e+2. The worst tasks (16-23) have errors 10-100x larger than others, suggesting they may be highly multimodal, deceptive, or ill-conditioned problems where the current sampling strategy (multivariate normal with adapted covariance) is failing to explore effectively.

The key insight: All previous `_sample_trials_batch` variants use Gaussian-like sampling from the covariance matrix, which creates a narrow exploration distribution that gets trapped. For the worst tasks, we need fundamentally different sampling that:
1. Uses heavy-tailed distributions to escape local optima
2. Injects orthogonal directions to explore new subspaces
3. Maintains exploration even when covariance suggests convergence

**Idea: Heavy-tailed Power-law Sampling with Orthogonal Injection**

A non-Gaussian sampler using power-law scaling and orthogonal diversity injection to break through stuck states on the hardest multimodal/deceptive tasks.

```python
def _sample_trials_batch(self):
    """Sample using heavy-tailed power-law distribution with orthogonal diversity injection."""
    eigvals, eigvecs = np.linalg.eigh(self.C)
    eigvals = np.maximum(eigvals, 1e-10)
    L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    
    # Base Gaussian sample
    z_gauss = np.random.randn(self.NP, self.dim)
    
    # Heavy-tailed perturbation: power-law scaling breaks narrow exploration
    alpha_tail = 1.5
    u = np.random.uniform(0, 1, (self.NP, self.dim))
    z_power = np.sign(z_gauss) * np.power(np.maximum(np.abs(z_gauss), 1e-10), alpha_tail)
    z_power = z_power / (np.max(np.abs(z_power), axis=1, keepdims=True) + 1e-10)
    
    # Blend: exploration vs exploitation
    mix = np.random.beta(0.3, 1.0)
    z = mix * z_power + (1.0 - mix) * z_gauss
    
    # Orthogonal diversity injection (every 3rd generation)
    if self.generation % 3 == 0 and self.generation > 0:
        n_inject = max(1, self.NP // 4)
        for k in range(n_inject):
            idx = np.random.randint(0, self.NP)
            # Generate vectors orthogonal to current mean movement
            grad_approx = self.mean - self.old_mean if self.generation > 1 else np.zeros(self.dim)
            grad_norm = np.linalg.norm(grad_approx)
            if grad_norm > 1e-10:
                orth_dir = np.random.randn(self.dim)
                orth_dir -= np.dot(orth_dir, grad_approx) / (grad_norm ** 2 + 1e-10) * grad_approx
                orth_dir /= (np.linalg.norm(orth_dir) + 1e-10)
                z[idx] += 0.5 * orth_dir
    
    self.trials = self.mean + self.sigma * (z @ L.T)
    self.trials = self._clip_to_bounds(self.trials)
```