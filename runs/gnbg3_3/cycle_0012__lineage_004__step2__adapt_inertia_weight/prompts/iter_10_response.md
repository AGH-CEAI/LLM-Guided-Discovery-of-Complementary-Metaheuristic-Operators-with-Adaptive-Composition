Looking at the worst unsolved tasks (17, 16, 6, 11 with errors 1e+04 to 1e+02), the current condition-number approach isn't effectively handling population collapse. I need a **different spectral mechanism** within Category B.

**Idea: SVD-based Spectral Entropy + Effective Dimensionality Tracking**

Instead of just condition number, I'll use:
1. **Spectral entropy** of normalized singular values - measures how "spread out" the population is across eigen-directions (high entropy = well-distributed = exploit, low entropy = concentrated = explore)
2. **Effective dimensionality ratio** (components to reach 95% variance) - directly detects when population collapses to low-rank subspace
3. **Singular value decay rate** - geometric mean ratio to detect sudden concentration drops

This is fundamentally different from condition number: it captures the *distribution shape* of variance across all directions, not just the extreme ratio.

```python
def _adapt_inertia_weight(self):
    """Adapt inertia weight using SVD-based spectral entropy and effective dimensionality."""
    try:
        centered = self.population - np.mean(self.population, axis=0)
        
        # SVD of centered population (different from covariance eigendecomposition)
        U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
        singular_values = np.clip(singular_values, 1e-10, None)
        
        # Total variance proxy
        total_var = np.sum(singular_values ** 2) + 1e-10
        
        # Normalized singular values (probability distribution over directions)
        sv_norm = singular_values ** 2 / total_var
        sv_norm = np.clip(sv_norm, 1e-10, None)
        
        # Spectral entropy: -sum(p * log(p))
        # High entropy = population spread across many directions = exploit (low inertia)
        # Low entropy = population concentrated in few directions = explore (high inertia)
        spectral_entropy = -np.sum(sv_norm * np.log(sv_norm))
        max_entropy = np.log(min(self.np, self.dim) + 1e-10)
        entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)
        
        # Effective dimensionality: components needed for 95% variance
        cumvar = np.cumsum(sv_norm)
        eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
        eff_dim_ratio = eff_dim / min(self.np, self.dim)
        
        # Singular value decay rate: geometric mean of consecutive ratios
        # High decay = rapid concentration = need more exploration
        if len(singular_values) >= 2:
            sv_ratios = singular_values[1:] / (singular_values[:-1] + 1e-10)
            sv_ratios = np.clip(sv_ratios, 0.0, 1.0)
            decay_rate = np.exp(np.mean(np.log(sv_ratios + 1e-10)))
        else:
            decay_rate = 0.5
        
        # Combined spectral signal: weighted combination of entropy and effective dim
        # Low entropy + low eff_dim_ratio + high decay_rate = collapse → high inertia
        collapse_signal = (1.0 - entropy_ratio) * 0.4 + (1.0 - eff_dim_ratio) * 0.4 + (1.0 - decay_rate) * 0.2
        collapse_signal = np.clip(collapse_signal, 0.0, 1.0)
        
        # Map collapse signal to inertia range [0.4, 0.95]
        # High collapse = more exploration = higher inertia
        target_inertia = 0.4 + 0.55 * collapse_signal
        
        # Blend with generation-based baseline
        decay = 0.729 - 0.15 * (self.generation / 1000)
        target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)
        
        # EMA smoothing for stability
        self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
        
    except np.linalg.LinAlgError:
        decay = 0.729 - 0.15 * (self.generation / 1000)
        self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
        self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
```