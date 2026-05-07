**Idea: Coordinate-Descent Pattern Search with Directional Memory**

A fundamentally different approach: replace the random direction search with systematic coordinate-wise descent that learns from successful directions. Unlike the current random exploration (which wastes iterations on unproductive directions), this method cycles through dimensions, adapts step sizes per-dimension based on success/failure, and accumulates a memory of successful direction vectors to exploit proven improvement directions. This is especially effective for the high-error tasks (12, 16, 8, 21) which likely have well-structured optima requiring systematic refinement.

```python
def _apply_adaptive_local_search(self, func):
    if self.best_solution is None:
        return

    center = self.best_solution.copy()
    current_fitness = self.best_fitness

    # Per-dimension adaptive step sizes
    if not hasattr(self, '_ls_dim_scales'):
        self._ls_dim_scales = np.ones(self.dim) * self.local_radius

    # Initialize direction history for learning
    if not hasattr(self, '_ls_direction_history'):
        self._ls_direction_history = []

    max_iter = self.local_max_iter * 3

    for iteration in range(max_iter):
        improved = False

        # Exploit learned successful directions first
        if len(self._ls_direction_history) > 0:
            for direction, step_size in reversed(self._ls_direction_history[-5:]):
                cand = np.clip(center + step_size * direction, self.lower, self.upper)
                fit = float(self._eval_wrapper_batch(cand.reshape(1, -1), func)[0])

                if np.isfinite(fit) and fit < current_fitness:
                    self._ls_direction_history.append((direction.copy(), step_size * 1.3))
                    if len(self._ls_direction_history) > 15:
                        self._ls_direction_history.pop(0)

                    center = cand.copy()
                    current_fitness = fit
                    improved = True
                    break

        # Systematic coordinate descent for unexplored improvements
        if not improved:
            dim = iteration % self.dim
            step = self._ls_dim_scales[dim]

            for sign in [1, -1]:
                cand = center.copy()
                cand[dim] = float(np.clip(cand[dim] + sign * step, self.lower[dim], self.upper[dim]))
                fit = float(self._eval_wrapper_batch(cand.reshape(1, -1), func)[0])

                if np.isfinite(fit) and fit < current_fitness:
                    direction = np.zeros(self.dim)
                    direction[dim] = sign
                    self._ls_direction_history.append((direction.copy(), step))
                    if len(self._ls_direction_history) > 15:
                        self._ls_direction_history.pop(0)

                    self._ls_dim_scales[dim] = min(step * 1.5, 10.0)
                    center = cand.copy()
                    current_fitness = fit
                    improved = True
                    break

            if not improved:
                self._ls_dim_scales[dim] = max(step * 0.5, 1e-8)

        if not improved and iteration > self.dim * 2:
            break

    if current_fitness < self.best_fitness:
        self.best_fitness = current_fitness
        self.best_solution = center.copy()
        self.local_radius = float(np.clip(np.mean(self._ls_dim_scales), 0.1, 10.0))
    else:
        self.local_radius = float(np.clip(np.mean(self._ls_dim_scales) * 0.8, 0.1, 10.0))
```