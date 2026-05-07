**Idea: Multi-Pattern Coordinate Descent with Nelder-Mead Hybrid**

A fundamentally different local search that combines systematic coordinate pattern search with a Nelder-Mead simplex. Instead of random Gaussian directions, it uses an expanding coordinate basis and a small simplex that adapts its shape based on function evaluations. This targets the worst tasks (12, 21, 23) which are stuck at massive errors (~10-700), indicating deep local optima basins or deceptive landscapes where random direction search fails to find any improvement path.

```python
def _apply_adaptive_local_search(self, func):
    if self.best_solution is None:
        return

    center = self.best_solution.copy()
    current_fitness = self.best_fitness
    radius = self.local_radius

    # --- Phase 1: Multi-coordinate pattern search with rotating basis ---
    n_dirs = max(4, min(self.dim, 12))
    for _ in range(min(5, self.local_max_iter)):
        if radius < 1e-6:
            break

        # Build an n_dirs-dimensional search basis (cycling coordinates + random dirs)
        basis = np.zeros((n_dirs, self.dim))
        for i in range(n_dirs):
            vec = np.zeros(self.dim)
            coord = i % self.dim
            vec[coord] = 1.0
            if np.random.rand() < 0.3:
                vec = vec + np.random.randn(self.dim) * 0.3
            norm = np.linalg.norm(vec) + 1e-10
            basis[i] = vec / norm

        # Evaluate pattern points (both + and - directions)
        candidates = [center]
        for d in range(n_dirs):
            candidates.append(np.clip(center + radius * basis[d], self.lower, self.upper))
            candidates.append(np.clip(center - radius * basis[d], self.lower, self.upper))

        candidates = np.array(candidates)
        fit_candidates = self._eval_wrapper_batch(candidates, func)
        best_local_idx = int(np.argmin(fit_candidates))
        best_local_fitness = float(fit_candidates[best_local_idx])

        if best_local_fitness < current_fitness - 1e-12:
            center = candidates[best_local_idx].copy()
            current_fitness = best_local_fitness
            radius = min(radius * 1.6, 10.0)
        else:
            radius *= 0.35

    # --- Phase 2: Nelder-Mead simplex refinement ---
    if radius > 1e-5 and current_fitness < self.best_fitness:
        # Initialize simplex around current center
        simplex = np.zeros((self.dim + 1, self.dim))
        simplex[0] = center.copy()
        for i in range(1, self.dim + 1):
            step = radius * 0.5
            simplex[i] = center.copy()
            simplex[i, (i - 1) % self.dim] += step
            simplex[i] = np.clip(simplex[i], self.lower, self.upper)

        simplex_fits = self._eval_wrapper_batch(simplex, func)
        simplex_fits = np.clip(simplex_fits, -1e50, 1e50)

        for _ in range(max(3, self.local_max_iter // 2)):
            # Sort simplex
            sorted_idx = np.argsort(simplex_fits)
            simplex = simplex[sorted_idx]
            simplex_fits = simplex_fits[sorted_idx]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            worst = simplex[-1]
            alpha = 1.0
            reflected = np.clip(centroid + alpha * (centroid - worst), self.lower, self.upper)
            reflected_fit = float(func(reflected.reshape(1, -1))[0])
            reflected_fit = np.clip(reflected_fit, -1e50, 1e50)

            if reflected_fit < simplex_fits[0]:
                # Expansion
                gamma = 2.0
                expanded = np.clip(centroid + gamma * (reflected - centroid), self.lower, self.upper)
                expanded_fit = float(func(expanded.reshape(1, -1))[0])
                expanded_fit = np.clip(expanded_fit, -1e50, 1e50)
                if expanded_fit < reflected_fit:
                    simplex[-1] = expanded
                    simplex_fits[-1] = expanded_fit
                else:
                    simplex[-1] = reflected
                    simplex_fits[-1] = reflected_fit
            elif reflected_fit < simplex_fits[-2]:
                simplex[-1] = reflected
                simplex_fits[-1] = reflected_fit
            else:
                # Contraction
                sigma = 0.5
                contracted = np.clip(centroid + sigma * (worst - centroid), self.lower, self.upper)
                contracted_fit = float(func(contracted.reshape(1, -1))[0])
                contracted_fit = np.clip(contracted_fit, -1e50, 1e50)
                simplex[-1] = contracted
                simplex_fits[-1] = contracted_fit

            # Shrink
            if simplex_fits[-1] >= simplex_fits[0]:
                for i in range(1, self.dim + 1):
                    simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), self.lower, self.upper)
                    simplex_fits[i] = float(np.clip(func(simplex[i].reshape(1, -1))[0], -1e50, 1e50))

        # Update best if simplex found improvement
        best_simplex_idx = int(np.argmin(simplex_fits))
        if simplex_fits[best_simplex_idx] < current_fitness:
            center = simplex[best_simplex_idx].copy()
            current_fitness = simplex_fits[best_simplex_idx]

    if current_fitness < self.best_fitness:
        self.best_fitness = current_fitness
        self.best_solution = center.copy()
        self.local_radius = min(radius * 1.2, 10.0)
    else:
        self.local_radius = max(radius * 0.8, 0.1)
```