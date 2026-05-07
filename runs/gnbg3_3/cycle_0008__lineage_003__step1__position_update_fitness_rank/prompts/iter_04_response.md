Looking at the current implementation, I see a critical problem: the existing `_position_update_fitness_rank` is actually implementing **Category E (Topology/graph-based)** mechanisms — building k-NN graphs, computing Laplacian eigenvalues, and performing connected component analysis. This violates the Category D constraint of using "only fitness signals."

I need to replace it with pure **fitness-landscape / rank-based** reasoning: Spearman rank correlations, fitness percentiles, success-history, and fitness-distance correlation in fitness space — **no distances between positions, no graph structures, no covariance**.

**Idea: Spearman-Rank Stagnation Modulation**
Use Spearman correlation between current fitness rank and personal-best rank to detect particles trapped in local optima. Fitness percentile signals identify population convergence. Success-history tracks per-particle improvement rates. These pure fitness signals drive velocity scaling and directional bias — completely orthogonal to geometry/spectral approaches.

```python
def _position_update_fitness_rank(self):
    """Spearman-rank stagnation modulation (Category D: fitness-landscape only).

    Key insight: Use ONLY fitness signals — no position distances, no graph
    structures, no covariance. Spearman correlation between current fitness
    rank and personal-best rank identifies particles stuck at local optima.
    Fitness percentiles detect convergence. Success-history tracks improvement.
    This is orthogonal to graph-based (E) and spectral (B) approaches.
    """
    # Pure fitness-based rank computation
    fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
    
    # Personal best fitness rank (captures "stagnation trap" detection)
    personal_best_rank = np.argsort(np.argsort(self.personal_best_fitness)) / max(1, self.np - 1)
    
    # Spearman-like correlation: high correlation = particle stuck at local optimum
    # (high rank in both current AND personal best = not improving through personal best)
    if np.std(fitness_ranks) > 1e-10 and np.std(personal_best_rank) > 1e-10:
        spearman_corr = np.corrcoef(fitness_ranks, personal_best_rank)[0, 1]
        spearman_corr = np.clip(spearman_corr, -1.0, 1.0)
    else:
        spearman_corr = 0.0
    
    # Fitness percentile signals (no distances used — purely rank-based)
    bottom_percentile = np.percentile(self.current_fitness, 10)
    top_percentile = np.percentile(self.current_fitness, 90)
    
    # Particles at bottom fitness percentile = likely stagnant, need exploration boost
    stagnant_particles = self.current_fitness <= bottom_percentile
    # Particles at top percentile = doing well, can be more exploitative
    top_particles = self.current_fitness >= top_percentile
    
    # Success history: exponential moving average of per-particle improvement
    if not hasattr(self, '_fitness_success_ema'):
        self._fitness_success_ema = np.zeros(self.np)
    
    improvement = self.personal_best_fitness - self.current_fitness
    improved_mask = improvement > 0
    self._fitness_success_ema[improved_mask] = self._fitness_success_ema[improved_mask] * 0.9 + 0.1
    self._fitness_success_ema[~improved_mask] = self._fitness_success_ema[~improved_mask] * 0.9 - 0.05
    self._fitness_success_ema = np.clip(self._fitness_success_ema, -0.5, 1.0)
    
    # Normalize success history
    success_norm = (self._fitness_success_ema - np.min(self._fitness_success_ema) + 1e-10)
    success_norm = success_norm / (np.max(success_norm) + 1e-10)
    
    # Fitness-distance correlation in FITNESS space (not position space)
    # Correlation between fitness rank and distance to global best in fitness space
    if self.global_best_fitness < np.inf:
        # Fitness-distance in fitness space: how correlated is fitness with proximity to best?
        fitness_distance_to_best = np.abs(self.current_fitness - self.global_best_fitness)
        fitness_dist_rank = np.argsort(np.argsort(fitness_distance_to_best)) / max(1, self.np - 1)
        if np.std(fitness_ranks) > 1e-10 and np.std(fitness_dist_rank) > 1e-10:
            fitness_dist_corr = np.corrcoef(fitness_ranks, fitness_dist_rank)[0, 1]
            fitness_dist_corr = np.clip(fitness_dist_corr, -1.0, 1.0)
        else:
            fitness_dist_corr = 0.0
    else:
        fitness_dist_corr = 0.0
    
    # Stagnation detection: particles with high rank in both current AND personal best
    # These are "trapped" — getting stuck at local optima relative to their personal best
    stagnation_signal = spearman_corr * (1.0 - success_norm)
    
    # Exploration/exploitation based on fitness signals
    # Low success + high spearman = stuck at local optimum → exploration
    exploration_signal = np.clip(stagnation_signal * 0.5, 0.0, 1.0)
    # High success + bottom percentile = good trajectory → exploitation
    exploitation_signal = success_norm * stagnant_particles.astype(float)
    
    # Velocity scaling: pure fitness-based
    # Stagnant particles get exploration boost, top particles get exploitation
    vel_scale = 1.0 + 0.5 * exploration_signal - 0.3 * exploitation_signal
    vel_scale = vel_scale * (1.0 + 0.3 * (1.0 - fitness_ranks))
    vel_scale = np.clip(vel_scale, 0.4, 2.5)
    
    # Directional bias: pure fitness-based — top particles pulled toward global best
    if self.global_best is not None:
        # Direction in position space, but decision of HOW MUCH to use it based on fitness
        to_best = self.global_best - self.population
        to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
        to_best_dir = to_best / to_best_norm
        # Top fitness particles get more directional pull; stagnant particles get less
        directional_strength = (1.0 - fitness_ranks[:, np.newaxis]) * (1.0 + success_norm[:, np.newaxis])
        directional = 0.3 * directional_strength * to_best_dir
    else:
        # No global best yet — random perturbation inversely scaled by fitness rank
        directional = 0.3 * (1.0 - fitness_ranks[:, np.newaxis]) * np.random.uniform(-1, 1, (self.np, self.dim))
    
    # Random perturbation inversely proportional to success (more random for stagnant)
    random_scale = 0.3 + 0.4 * exploration_signal[:, np.newaxis]
    random_perturb = random_scale * np.random.uniform(-1, 1, (self.np, self.dim))
    
    # Apply velocity update
    new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + directional + random_perturb
    self.population = self._clip_to_bounds(new_population)
```