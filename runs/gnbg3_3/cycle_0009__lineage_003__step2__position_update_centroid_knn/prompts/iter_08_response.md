Looking at the worst unsolved tasks (17, 16, 6, 11) with errors in the 10²–10⁴ range, the current centroid_knn (purely geometric) fails to escape large local optima. The key insight: it's blind to fitness information. I need a hybrid that combines **geometric centroid/k-NN** with **fitness-landscape signals**, with a principled (spectral-entropy-driven) switch.

**Idea: Spectral-entropy-gated hybrid centroid/fitness modulation**

```python
def _position_update_centroid_knn(self):
    """Hybrid: Centroid/k-NN + Fitness-rank, gated by spectral entropy (Category H).
    
    Two distinct mechanisms:
      1. Geometric: centroid gravity + k-NN density (from original)
      2. Fitness: rank-based success + directional pull to global best
    
    Principled switching: spectral entropy of population covariance drives
    the blend. Low entropy (clustered) → favor geometric (explore sparse regions).
    High entropy (dispersed) → favor fitness-rank (exploit good directions).
    """
    # --- Mechanism 1: Geometric (centroid + k-NN) ---
    centroid = np.mean(self.population, axis=0)
    dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)

    k = min(5, self.np - 1)
    diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    knn_dists = np.sort(sq_dists, axis=1)[:, :k]
    knn_avg_dist = np.mean(np.sqrt(knn_dists), axis=1, keepdims=True) + 1e-10

    global_spread = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10

    centroid_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))
    to_centroid_dir = centroid - self.population
    to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
    to_centroid_dir = to_centroid_dir / to_centroid_dist
    density_modulation = np.clip(knn_avg_dist / global_spread, 0.2, 2.0)
    geometric_correction = centroid_attraction * to_centroid_dir * density_modulation

    # --- Principled switch: spectral entropy ---
    try:
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]
        total_var = np.sum(eigenvalues) + 1e-10
        eigenvalues_norm = eigenvalues / total_var
        spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
        max_entropy = np.log(self.dim + 1e-10)
        entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)
    except:
        entropy_ratio = 0.5

    # --- Mechanism 2: Fitness-rank signals ---
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

    if not hasattr(self, '_knn_fitness_success'):
        self._knn_fitness_success = np.zeros(self.np)
    improved = self.personal_best_fitness >= self.current_fitness
    self._knn_fitness_success[improved] += 1
    self._knn_fitness_success[~improved] *= 0.9
    success_norm = self._knn_fitness_success / (np.max(self._knn_fitness_success) + 1.0)

    fitness_correction = np.zeros((self.np, self.dim))
    if self.global_best is not None:
        to_best = self.global_best - self.population
        to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
        to_best_dir = to_best / to_best_norm
        fitness_correction = (1.0 - fitness_ranks[:, np.newaxis]) * to_best_dir

    random_perturb = np.random.uniform(-0.3, 0.3, (self.np, self.dim))

    # --- Data-driven blend: entropy gates geometric vs fitness ---
    # Low entropy → population clustered → geometric (sparse exploration)
    # High entropy → population dispersed → fitness (exploit good regions)
    geo_weight = 1.0 - entropy_ratio
    fitness_weight = entropy_ratio

    combined_correction = (
        geo_weight * geometric_correction +
        fitness_weight * (0.3 * fitness_correction + 0.1 * random_perturb)
    )

    # Scale correction by success rate (data-driven, not magic)
    correction_scale = 1.0 + 0.5 * success_norm
    combined_correction = combined_correction * correction_scale

    new_population = self.population + self.inertia_weight * self.velocity + 0.3 * combined_correction
    self.population = self._clip_to_bounds(new_population)
```