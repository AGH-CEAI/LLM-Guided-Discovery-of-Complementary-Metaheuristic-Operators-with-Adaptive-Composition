**Idea: Monte Carlo Importance-Weighted Centroid with Random Projection Exploration**

Category G: Stochastic / sampling-based. Uses Monte Carlo importance-weighted sampling to estimate the optimal centroid direction, combined with random projection exploration to escape local optima. The key insight: instead of a deterministic centroid, sample multiple candidate directions via random projections, weight them by estimated improvement potential using fitness-based importance sampling, and stochastically select the direction. This variance-injection helps the swarm escape the deceptive local optima trapping tasks 17, 16, 6, 5.

```python
def _position_update_centroid_knn(self):
    """Monte Carlo importance-weighted centroid + random projection exploration (Category G).
    
    Fundamentally stochastic: uses Monte Carlo sampling of candidate directions,
    importance-weighted by estimated fitness improvement, with random projection
    exploration to escape local optima. Different from deterministic centroid_knn.
    """
    n_projections = 20
    n_importance_samples = 15
    
    # === MONTE CARLO: Random projection exploration ===
    # Sample random directions in latent space
    random_dirs = np.random.randn(self.dim, n_projections)
    random_dirs = random_dirs / (np.linalg.norm(random_dirs, axis=0, keepdims=True) + 1e-10)
    
    # Evaluate each projection: correlation between position-projection and fitness
    proj_scores = np.zeros(n_projections)
    for j in range(n_projections):
        proj = self.population @ random_dirs[:, j]
        if np.std(proj) > 1e-10 and np.std(self.current_fitness) > 1e-10:
            proj_scores[j] = abs(np.corrcoef(proj, self.current_fitness)[0, 1])
        else:
            proj_scores[j] = 0.0
    
    # Normalize projection scores to weights
    proj_weights = np.exp(proj_scores * 5.0)
    proj_weights = proj_weights / (np.sum(proj_weights) + 1e-10)
    
    # Sample a random projection direction weighted by its score
    selected_proj_idx = np.random.choice(n_projections, p=proj_weights)
    stochastic_dir = random_dirs[:, selected_proj_idx]
    
    # === IMPORTANCE-WEIGHTED CENTROID ESTIMATION ===
    # Weight particles by fitness: better fitness = higher weight
    min_fit = np.min(self.current_fitness)
    max_fit = np.max(self.current_fitness)
    fit_range = max_fit - min_fit + 1e-10
    
    # Use inverse fitness (minimize) for importance weights
    if fit_range > 1e-10:
        inv_fitness = max_fit - self.current_fitness + 1e-10
        importance_weights = inv_fitness / (np.sum(inv_fitness) + 1e-10)
    else:
        importance_weights = np.ones(self.np) / self.np
    
    # Monte Carlo importance sampling of the centroid
    sampled_centroids = []
    for _ in range(n_importance_samples):
        sampled_idx = np.random.choice(self.np, size=self.np, replace=True, p=importance_weights)
        sampled_centroid = np.mean(self.population[sampled_idx], axis=0)
        sampled_centroids.append(sampled_centroid)
    sampled_centroids = np.array(sampled_centroids)
    
    # Select one centroid stochastically (uniform for now)
    selected_centroid = sampled_centroids[np.random.randint(n_importance_samples)]
    
    # === COMBINE DETERMINISTIC CENTROID ATTRACTION WITH STOCHASTIC EXPLORATION ===
    centroid = np.mean(self.population, axis=0)
    to_centroid_dir = centroid - self.population
    to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
    to_centroid_dir = to_centroid_dir / to_centroid_dist
    
    # Project stochastic direction onto particle-wise directions
    stochastic_dir_expanded = stochastic_dir[np.newaxis, :]
    proj_onto_particle = np.sum(stochastic_dir_expanded * to_centroid_dir, axis=1, keepdims=True)
    stochastic_component = stochastic_dir_expanded * proj_onto_particle
    
    # Combine: 70% importance-weighted centroid + 30% stochastic projection
    alpha = np.random.uniform(0.6, 0.8, (self.np, 1))
    combined_dir = alpha * to_centroid_dir + (1 - alpha) * stochastic_component
    
    # Normalize combined direction
    combined_norm = np.linalg.norm(combined_dir, axis=1, keepdims=True) + 1e-10
    combined_dir = combined_dir / combined_norm
    
    # Apply update with inertia
    new_population = self.population + self.inertia_weight * self.velocity + 0.3 * combined_dir
    
    self.population = self._clip_to_bounds(new_population)
```