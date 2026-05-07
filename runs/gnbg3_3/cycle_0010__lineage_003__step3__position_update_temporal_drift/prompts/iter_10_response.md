Looking at the problem: the current Category B variant uses spectral entropy and eigenvalue-magnitude scaling. I need a DIFFERENT spectral mechanism. The worst tasks (17, 16, 6, 11) have errors ~1e+3–1e+4, suggesting severe population collapse or misalignment. The key insight: **the current approach modulates velocity magnitude but doesn't explicitly correct for misalignment between the velocity direction and the global best direction, nor does it aggressively expand collapsed subspaces**.

My new mechanism will:
1. Compute alignment between each velocity eigenvector direction and the global best vector
2. Use this alignment to drive per-component scaling (reward velocity pointing toward global best, penalize velocity pointing away)
3. When condition number is high (collapsed population), amplify minor eigenvector directions to expand the subspace

This is fundamentally different from the previous Category B approach (SVD whitening) because it uses **directional alignment** rather than just magnitude scaling.

**Idea: Directional Alignment + Subspace Expansion**
Spectral alignment of velocity to global-best direction + condition-number-triggered subspace expansion in minor components.
```python
def _position_update_temporal_drift(self):
    """Directional alignment + subspace expansion (Category B).

    Computes alignment between each eigenvector direction and the global-best
    vector. Velocity components aligned with global-best get boosted; misaligned
    components get dampened. When condition number is high (population collapsed
    to subspace), minor eigenvector directions are amplified to re-expand the
    population. This is orthogonal to SVD whitening (which uses magnitude ratios)
    because it uses DIRECTIONAL alignment with the objective target.
    """
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = np.clip(eigenvalues, 1e-10, None)

        cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
        cond_log = np.log1p(cond)

        sv_scale = np.sqrt(eigenvalues + 1e-10)
        sv_norm = sv_scale / (sv_scale[0] + 1e-10)

        if self.global_best is not None:
            global_dir = self.global_best - np.mean(self.population, axis=0)
            global_dir_norm = np.linalg.norm(global_dir) + 1e-10
            global_dir = global_dir / global_dir_norm

            alignment = np.abs(eigenvectors.T @ global_dir)

            alignment_boost = 0.5 + 0.5 * alignment
            alignment_boost = np.clip(alignment_boost, 0.3, 1.5)
        else:
            alignment_boost = np.ones(self.dim)

        spectral_signal = np.clip(cond_log / 5.0, 0.0, 1.0)

        if spectral_signal > 0.3:
            minor_boost = 1.0 + 1.5 * spectral_signal * (1.0 - sv_norm)
            minor_boost = np.clip(minor_boost, 1.0, 3.0)
        else:
            minor_boost = np.ones(self.dim)

        per_comp_scale = alignment_boost * minor_boost
        per_comp_scale = np.clip(per_comp_scale, 0.3, 3.0)

        vel_proj = self.velocity @ eigenvectors
        vel_scaled = vel_proj * per_comp_scale
        new_population = self.population + (vel_scaled @ eigenvectors.T)

    except np.linalg.LinAlgError:
        new_population = self.population + self.velocity

    self.population = self._clip_to_bounds(new_population)
```