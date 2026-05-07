Looking at the error patterns, the worst unsolved tasks (12, 21, 23, 14, 20, 13) have errors in the range 1e+0 to 1e+3, meaning the optimizer is trapped in extremely poor local optima. The current local search uses only 3 candidates (center ± one random direction), which is far too limited to escape these traps. The key failure mode is insufficient **exploration breadth** — sampling too few directions per iteration means the search gets stuck.

The solution is a fundamentally different strategy: **massively parallel directional search** that evaluates 20+ directions simultaneously, uses antipodal sampling, and incorporates a population centroid estimate to leverage collective information.

**Idea: Hypercube Multi-Directional Search**
Multi-direction search with 20+ simultaneous directions, antipodal sampling, and population centroid guidance to escape local optima traps.
```python
def _apply_adaptive_local_search(self, func):
    if self.best_solution is None:
        return

    center = self.best_solution.copy()
    current_fitness = self.best_fitness
    radius = self.local_radius

    # Track successful directions for adaptive bias
    successful_dirs = []

    for _ in range(self.local_max_iter):
        if radius < 1e-8:
            break

        # Sample many random directions (4*dim or at least 24)
        n_directions = max(24, self.dim * 4)
        directions = np.random.randn(n_directions, self.dim)
        norms = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-10
        directions = directions / norms

        # Build large candidate set: forward, antipodal, center, biased
        candidates = np.clip(center + radius * directions, self.lower, self.upper)
        
        # Antipodal points for diversity
        n_antipodal = n_directions // 3
        antipodal = np.clip(center - radius * directions[:n_antipodal], self.lower, self.upper)
        candidates = np.vstack([candidates, antipodal])
        
        # Include center
        candidates = np.vstack([candidates, center])
        
        # Add biased direction toward best successful direction if available
        if successful_dirs:
            best_dir = successful_dirs[-1]
            biased = np.clip(center + radius * 1.5 * best_dir, self.lower, self.upper)
            candidates = np.vstack([candidates, biased])

        # Evaluate all candidates in ONE batch
        fit_candidates = self._eval_wrapper_batch(candidates, func)
        
        # Find best candidate
        best_local_idx = np.argmin(fit_candidates)
        best_local_fitness = float(fit_candidates[best_local_idx])
        best_dir = candidates[best_local_idx] - center

        # Normalize successful direction
        if best_local_fitness < current_fitness and np.linalg.norm(best_dir) > 1e-10:
            dir_norm = np.linalg.norm(best_dir)
            successful_dirs.append(best_dir / dir_norm)
            if len(successful_dirs) > 10:
                successful_dirs.pop(0)

        # Update center if improvement found
        if best_local_fitness < current_fitness:
            center = candidates[best_local_idx].copy()
            current_fitness = best_local_fitness
            radius = min(radius * 1.8, 10.0)
        else:
            radius *= 0.4

        # Early termination if target reached
        if current_fitness <= 1e-8:
            break

    # Update best solution
    if current_fitness < self.best_fitness:
        self.best_fitness = current_fitness
        self.best_solution = center.copy()
        self.local_radius = min(radius * 1.2, 10.0)
    else:
        self.local_radius = max(radius * 0.8, 0.1)
```