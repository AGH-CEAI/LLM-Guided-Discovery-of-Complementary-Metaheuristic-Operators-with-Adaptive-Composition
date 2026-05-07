**Idea: Coordinate-Wise Multi-Scale Pattern Search**

A fundamentally different local search that systematically explores ALL dimensions with coordinated perturbations at multiple scales. Unlike the random direction approach (which tests only 2 directions per iteration), this tests N directions per iteration and uses a multi-scale restart mechanism with memory of historical best solutions to escape deceptive local optima.

```python
def _apply_adaptive_local_search(self, func):
    if self.best_solution is None:
        return

    center = self.best_solution.copy()
    current_fitness = self.best_fitness

    # Adaptive radius starts from current setting, with bounds
    radius = self.local_radius
    max_radius = max(self.upper - self.lower) * 0.1
    min_radius = (self.upper - self.lower).min() * 1e-5

    # Multi-scale restart with memory
    if not hasattr(self, '_local_search_restarts'):
        self._local_search_restarts = 0
    if not hasattr(self, '_local_best_history'):
        self._local_best_history = []

    max_restarts = 3
    max_iter_per_restart = self.local_max_iter

    for restart in range(max_restarts):
        for _ in range(max_iter_per_restart):
            if radius < min_radius:
                break

            # Generate N+1 candidates: N coordinate perturbations + 1 center
            candidates = np.tile(center, (self.dim + 1, 1))
            for d in range(self.dim):
                candidates[d, d] = np.clip(center[d] + radius, self.lower[d], self.upper[d])

            fit_candidates = self._eval_wrapper_batch(candidates, func)
            best_local_idx = int(np.argmin(fit_candidates))
            best_local_fitness = float(fit_candidates[best_local_idx])

            if best_local_idx < self.dim:
                # Improvement found: update and expand radius
                center[best_local_idx] = candidates[best_local_idx, best_local_idx]
                current_fitness = best_local_fitness
                radius = min(radius * 1.5, max_radius)
            else:
                # No improvement: contract radius
                radius *= 0.5

        # Check if converged
        if current_fitness >= self.best_fitness and radius < min_radius * 2:
            break

        # Restart from historical best if stuck
        if restart < max_restarts - 1:
            if len(self._local_best_history) > 0:
                # Pick random historical best (different from current center)
                hist_best = self._local_best_history[np.random.randint(0, len(self._local_best_history))]
                if np.linalg.norm(hist_best - center) > radius:
                    center = hist_best.copy()
                    self._local_search_restarts += 1
                    radius = max(radius * 2, min_radius * 10)
                    continue
            break

    # Update global best if improved
    if current_fitness < self.best_fitness:
        self.best_fitness = current_fitness
        self.best_solution = center.copy()
        # Record in history for restarts
        self._local_best_history.append(center.copy())
        if len(self._local_best_history) > 10:
            self._local_best_history.pop(0)

    # Adapt local radius for next call
    self.local_radius = float(np.clip(radius, min_radius, max_radius))
```