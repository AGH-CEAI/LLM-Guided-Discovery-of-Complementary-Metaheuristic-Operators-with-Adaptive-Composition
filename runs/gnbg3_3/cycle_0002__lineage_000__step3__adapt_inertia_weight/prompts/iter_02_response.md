**Idea: Spectral Condition-Number Inertia Modulation**

Adapt inertia weight using the condition number and effective dimensionality of the population covariance matrix. High condition number indicates elongated/anisotropic distribution requiring more exploration; low effective dimensionality indicates population collapse into a subspace requiring stronger perturbation.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using spectral condition number of population covariance."""
    centered = self.population - np.mean(self.population, axis=0)
    
    try:
        eigenvalues, _ = np.linalg.eigh(np.cov(centered.T))
        eigenvalues = np.clip(np.sort(eigenvalues)[::-1], 1e-10, None)
        
        cond = eigenvalues[0] / eigenvalues[-1]
        
        total_var = np.sum(eigenvalues) + 1e-10
        cumvar = np.cumsum(eigenvalues) / total_var
        effective_dim = np.searchsorted(cumvar, 0.95) + 1
        dim_ratio = effective_dim / self.dim
        
        if cond > 100:
            self.inertia_weight = min(0.95, self.inertia_weight * 1.15)
        elif cond < 5:
            self.inertia_weight = max(0.4, self.inertia_weight * 0.85)
        elif dim_ratio < 0.2:
            self.inertia_weight = max(0.4, self.inertia_weight * 0.9)
        elif dim_ratio > 0.8:
            self.inertia_weight = min(0.95, self.inertia_weight * 1.05)
        else:
            self.inertia_weight = 0.729 - 0.1 * (self.generation / 1000)
        
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
        
    except np.linalg.LinAlgError:
        self.inertia_weight = 0.729 - 0.1 * (self.generation / 1000)
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```