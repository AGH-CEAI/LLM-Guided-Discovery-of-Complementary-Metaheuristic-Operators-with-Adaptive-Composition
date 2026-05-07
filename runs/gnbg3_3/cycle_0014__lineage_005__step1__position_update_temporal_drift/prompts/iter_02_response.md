**Idea: SVD Subspace Collapse Recovery**

Category B spectral approach using SVD-based effective rank and singular value entropy to detect population collapse into low-rank subspaces, then applying inverse-collapsed-direction velocity boosting and condition-number-based exploration scaling. Fundamentally different from the original's eigenvalue damping — this focuses on the *effective dimensionality* of the population distribution rather than condition-number ratios.

```python
def _position_update_temporal_drift(self):
    """SVD Subspace Collapse Recovery (Category B: Spectral/Linear-Algebraic).

    Uses singular value entropy and effective rank of the population matrix to
    detect when the population has collapsed into a low-dimensional subspace.
    Applies inverse-collapsed-direction velocity boosting and condition-number-
    based exploration scaling to recover lost degrees of freedom.

    Targets worst tasks (17, 5, 16, 6) where population collapse into disconnected
    local optima leaves exploration directions completely abandoned.
    """
    try:
        # === SPECTRAL ANALYSIS VIA SVD ===
        centered = self.population - np.mean(self.population, axis=0)

        U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
        singular_values = np.clip(singular_values, 1e-10, None)

        # Effective rank from singular value entropy (not eigenvalue entropy)
        sv_norm = singular_values / (singular_values[0] + 1e-10)
        entropy_sv = -np.sum(sv_norm * np.log(sv_norm + 1e-10))
        max_entropy = np.log(len(singular_values))
        effective_rank = np.exp(entropy_sv)
        eff_rank_ratio = effective_rank / len(singular_values)

        # Condition number for anisotropy
        cond = singular_values[0] / (singular_values[-1] + 1e-10)

        # === COLLAPSED DIRECTION DETECTION ===
        # Directions where sv_norm < 0.05 are considered "collapsed"
        collapse_mask = sv_norm < 0.05

        # Boost velocity in collapsed directions (inverse of normalized SV)
        # Low singular value = collapsed direction = needs strong velocity boost
        inverse_boost = np.zeros_like(sv_norm)
        inverse_boost[~collapse_mask] = 1.0 / (sv_norm[~collapse_mask] + 0.01)
        inverse_boost[collapse_mask] = 5.0 / (sv_norm[collapse_mask] + 0.01)
        inverse_boost = np.clip(inverse_boost, 0.5, 10.0)
        inverse_boost = inverse_boost / (np.max(inverse_boost) + 1e-10)

        # === EXPLORATION SCALING FROM CONDITION NUMBER ===
        # Logarithmic mapping: cond varies over many orders of magnitude
        # High cond = ill-conditioned = more exploration needed
        cond_log = np.log1p(cond) / np.log1p(1e6)
        cond_log = np.clip(cond_log, 0.0, 1.0)
        exploration_scale = 0.8 + 0.7 * cond_log

        # === COMBINED EXPLORATION SIGNAL ===
        # Low effective rank ratio = population in low-dim subspace = boost exploration
        if eff_rank_ratio < 0.3:
            exploration_scale *= 1.5
        elif eff_rank_ratio < 0.5:
            exploration_scale *= 1.2

        exploration_scale = np.clip(exploration_scale, 0.3, 2.5)

        # === VELOCITY UPDATE IN SVD BASIS ===
        vel_proj = self.velocity @ Vt.T
        vel_boosted = vel_proj * inverse_boost * exploration_scale
        new_population = self.population + (vel_boosted @ Vt)

    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity

    self.population = self._clip_to_bounds(new_population)
```