**Idea: Mahalanobis-Entropy Centroid Modulation**
One-line description: Replaces k-NN density with information-theoretic signals (Mahalanobis distance, entropy-based modulation, KL-like local/global divergence) to drive centroid gravity (Category C: Information-theoretic / distributional).

```python
def _position_update_centroid_knn(self):
    """Mahalanobis-entropy centroid modulation (variant_10, Category C).
    
    Reframes centroid gravity within an information-theoretic framework:
    - Mahalanobis distance from fitted Gaussian captures distributional shape
    - Entropy of eigenvalue spectrum measures population collapse
    - Per-particle entropy modulation drives exploration/exploitation balance
    Targets worst tasks (17, 16, 6, 11) where population collapses to subspaces.
    """
    # 1. Fit Gaussian to population for distributional characterization
    centroid = np.mean(self.population, axis=0)
    centered = self.population - centroid
    cov = np.cov(centered.T)
    
    # Regularize covariance for numerical stability
    cov_reg = cov + 0.1 * np.trace(cov) / self.dim * np.eye(self.dim)
    cov_reg = 0.5 * (cov_reg + cov_reg.T)
    
    # 2. Compute precision matrix (inverse covariance) using SVD-based pseudoinverse
    try:
        U_svd, s_svd, Vt_svd = np.linalg.svd(cov_reg, full_matrices=False)
        s_inv = np.where(s_svd > 1e-10, 1.0 / s_svd, 0.0)
        precision = Vt_svd.T @ np.diag(s_inv) @ U_svd.T
    except np.linalg.LinAlgError:
        precision = np.eye(self.dim)
    
    # 3. Compute per-particle Mahalanobis distance (information content)
    mahal_sq = np.sum(centered @ precision * centered, axis=1)
    mahal_dist = np.sqrt(mahal_sq + 1e-10)[:, np.newaxis]
    
    # 4. Entropy of eigenvalue spectrum (detects population collapse)
    eigenvalues = np.linalg.eigvalsh(cov_reg)
    eigenvalues = np.clip(eigenvalues, 1e-10, None)
    eigenvalues = np.sort(eigenvalues)[::-1]
    total_var = np.sum(eigenvalues) + 1e-10
    eigenvalues_norm = eigenvalues / total_var
    
    spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
    max_entropy = np.log(self.dim + 1e-10)
    entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)
    
    # 5. Entropy-based modulation: low entropy (collapsed) → encourage exploration
    # Use entropy_ratio to scale how much particles respond to centroid pull
    exploration_factor = 1.0 + 0.5 * (1.0 - entropy_ratio)
    
    # 6. Per-particle entropy signal: high Mahalanobis distance = low density region
    # Map to [0.5, 2.0] range for modulation
    global_mahal_mean = np.mean(mahal_dist) + 1e-10
    local_entropy_modulation = np.clip(mahal_dist / global_mahal_mean, 0.5, 2.0)
    
    # 7. KL-divergence-like signal: compare local entropy to global entropy
    # Particles in low-entropy (dense) regions get stronger centroid pull
    # Particles in high-entropy (sparse) regions get weaker pull (more exploration)
    kl_signal = np.clip(entropy_ratio / (local_entropy_modulation + 1e-10), 0.3, 1.5)
    
    # 8. Compute centroid attraction with entropy modulation
    dist_to_centroid = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-10
    global_spread = np.mean(dist_to_centroid) + 1e-10
    
    centroid_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread))
    
    # 9. Direction to centroid
    to_centroid_dir = centroid - self.population
    to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
    to_centroid_dir = to_centroid_dir / to_centroid_dist
    
    # 10. Combine signals: entropy_ratio drives global behavior, kl_signal drives per-particle
    correction = centroid_attraction * to_centroid_dir
    correction *= exploration_factor * kl_signal
    
    # 11. Apply update with inertia
    new_population = self.population + self.inertia_weight * self.velocity + 0.3 * correction
    
    self.population = self._clip_to_bounds(new_population)
```