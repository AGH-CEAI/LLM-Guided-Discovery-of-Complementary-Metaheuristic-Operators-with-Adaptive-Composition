Looking at the current `_position_update_svd_whitening` implementation and the worst unsolved tasks, I need a different angle within Category B.

**Analysis of failure modes:**
- Worst tasks (17: 4.6e+04, 16: 2.5e+03, 6: 7.3e+02) likely have extreme anisotropy or ill-conditioning
- Current SVD approach uses inverse-square-law modulation but lacks momentum and proper coordinate system transformation
- The condition number is only used for blending weights, not for direct velocity modulation

**New approach: Eigendecomposition-based Whitened Coordinate Momentum**
- Instead of SVD, use eigendecomposition of the covariance matrix directly
- Transform velocity into the whitened coordinate system (proper decorrelation)
- Add momentum/velocity blending based on eigenvalue conditioning
- Use the condition number to modulate the momentum coefficient more aggressively for high-anisotropy tasks

This differs from the current SVD variant by: (1) proper covariance-based whitening with momentum, (2) more aggressive condition-number-driven exploration, (3) velocity blending in transformed space.

**Idea: Eigencovariance Momentum Whitening**

```python
def _position_update_svd_whitening(self):
    """Eigendecomposition-based whitened coordinate momentum (Category B).
    
    Key insight: transform velocity to eigenspace, apply momentum-based
    updates, and use condition number to aggressively modulate exploration.
    Targets extreme anisotropy in worst unsolved tasks (17, 16, 6).
    """
    centered = self.population - np.mean(self.population, axis=0)

    try:
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        eigenvalues = np.clip(eigenvalues, 1e-12, None)
        
        cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        cond = np.clip(cond, 1.0, 1e6)
        
        total_var = np.sum(eigenvalues) + 1e-10
        
        if not hasattr(self, '_prev_whitened_vel'):
            self._prev_whitened_vel = np.zeros((self.np, self.dim))
        
        vel_proj = self.velocity @ eigenvectors
        
        whiten_factors = np.sqrt(1.0 / eigenvalues + 1e-10)
        whiten_factors = whiten_factors / (np.max(whiten_factors) + 1e-10)
        
        vel_whitened = vel_proj * whiten_factors
        
        momentum_coeff = np.clip(0.3 + 0.5 * np.log1p(cond) / np.log1p(1e6), 0.3, 0.8)
        
        vel_with_momentum = (1.0 - momentum_coeff) * vel_whitened + momentum_coeff * self._prev_whitened_vel
        
        self._prev_whitened_vel = vel_with_momentum.copy()
        
        vel_unwhitened = vel_with_momentum / (whiten_factors + 1e-10)
        
        vel_scaled = vel_unwhitened * np.clip(cond / 100.0, 0.5, 2.5)
        
        new_population = self.population + vel_scaled

    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity

    self.population = self._clip_to_bounds(new_population)
```