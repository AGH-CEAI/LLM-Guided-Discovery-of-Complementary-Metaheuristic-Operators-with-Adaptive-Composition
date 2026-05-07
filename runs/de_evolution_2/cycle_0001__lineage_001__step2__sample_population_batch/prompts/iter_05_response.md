**Idea: Elite-Focused Bimodal Sampling**
Split sampling between tight exploration around the best-known region and diverse exploration around the global mean, with adaptive variance allocation based on search progress.

```python
def _sample_population_batch(self):
        """
        Bimodal sampling: half from elite-biased tight distribution,
        half from global diverse distribution. Designed to escape local
        optima on ill-conditioned/deceptive tasks (16,19,20,23,18,10,14).
        """
        # Compute elite center from current covariance structure
        # Use eigendecomposition to get principal axes and scales
        z_all = np.random.randn(self.pop_size, self.dim)
        
        # Determine exploitation ratio: shrink variance when eigenvalues are extreme
        cond_num = np.max(self.D) / (np.min(self.D) + 1e-20)
        exploit_ratio = np.clip(1.0 / (1.0 + np.log1p(cond_num) / 5.0), 0.1, 0.7)
        
        # Elite center: mean shifted toward best eigenvector direction
        principal_dir = self.B[:, -1]  # direction of largest variance
        elite_center = self.mean + 0.3 * self.sigma * principal_dir * self.D[-1] * 0.1
        
        # Mode 1 (exploit): tight sampling around elite center
        n_exploit = self.pop_size // 2
        exploit_z = z_all[:n_exploit]
        exploit_scaled = exploit_z * (self.D * exploit_ratio)[np.newaxis, :]
        exploit_rotated = exploit_scaled @ self.B.T
        pop_exploit = elite_center[np.newaxis, :] + self.sigma * exploit_rotated
        
        # Mode 2 (explore): standard sampling around global mean
        n_explore = self.pop_size - n_exploit
        explore_z = z_all[n_explore:]
        explore_scaled = explore_z * self.D[np.newaxis, :]
        explore_rotated = explore_scaled @ self.B.T
        pop_explore = self.mean[np.newaxis, :] + self.sigma * explore_rotated
        
        # Combine both modes
        population = np.vstack([pop_exploit, pop_explore])
        z_vectors = z_all.copy()
        
        return population, z_vectors
```