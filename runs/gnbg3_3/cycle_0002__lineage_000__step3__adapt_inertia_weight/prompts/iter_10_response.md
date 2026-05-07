**Idea: Spectral Condition-Number-Driven Inertia Adaptation**

Using the condition number of the population covariance matrix (eigenvalue ratio) and effective dimensionality from SVD to directly modulate inertia weight. High condition number indicates the swarm is trapped in a narrow elongated subspace (needs higher inertia to escape), while well-conditioned populations allow lower inertia for fine-grained exploitation. This is fundamentally different from diversity-threshold approaches.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using spectral condition number and effective dimensionality."""
    centered = self.population - np.mean(self.population, axis=0)
    
    # Compute covariance with numerical safety
    cov = np.cov(centered.T)
    cov += np.eye(self.dim) * 1e-10
    
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-15, None)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # Primary metric: condition number (ratio of largest to smallest eigenvalue)
        cond = eigenvalues[0] / eigenvalues[-1]
        
        # Secondary metric: effective dimensionality from eigenvalue variance
        total_var = np.sum(eigenvalues)
        if total_var > 1e-15:
            var_ratios = eigenvalues / total_var
            entropy = -np.sum(var_ratios * np.log(var_ratios + 1e-15))
            max_entropy = np.log(self.dim)
            eff_dim = np.exp(entropy) / self.dim  # 0 = collapsed, 1 = full-dimensional
        else:
            eff_dim = 0.5
        
        # Track condition number history for stagnation detection
        if not hasattr(self, '_cond_history'):
            self._cond_history = []
        self._cond_history.append(cond)
        if len(self._cond_history) > 20:
            self._cond_history.pop(0)
        
        # Compute condition number trend (positive = increasingly anisotropic)
        if len(self._cond_history) >= 5:
            recent = np.mean(self._cond_history[-3:])
            older = np.mean(self._cond_history[:3])
            cond_trend = recent - older
        else:
            cond_trend = 0.0
        
        # Compute inertia weight based on spectral properties
        if cond > 100:
            # Severely anisotropic: trapped in narrow subspace, need more exploration
            self.inertia_weight = 0.85 + 0.08 * np.tanh(np.log1p(cond) / 10.0 - 1.5)
        elif cond < 10:
            # Well-conditioned: exploit with lower inertia
            self.inertia_weight = 0.6 - 0.15 * (1.0 - eff_dim)
        else:
            # Moderate conditioning: use effective dimensionality
            if eff_dim < 0.5:
                # Low-dimensional: trapped, increase exploration
                self.inertia_weight = 0.729 + 0.15 * (0.5 - eff_dim)
            else:
                # High-dimensional: fine-grained search
                self.inertia_weight = 0.729 - 0.1 * (self.generation / 1000)
        
        # Stagnation detection: if condition number keeps increasing but no improvement
        if cond_trend > 50 and self.stagnation_counter > 10:
            self.inertia_weight = min(0.95, self.inertia_weight + 0.1)
        
        # Clip to valid range
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
        
    except np.linalg.LinAlgError:
        # Fallback to generation-based schedule
        self.inertia_weight = 0.729 - 0.1 * (self.generation / 1000)
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```