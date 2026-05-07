**Idea: Nelder-Mead Simplex Local Search**
A classic derivative-free optimizer using a self-adapting simplex (dim+1 vertices) that performs reflection, expansion, contraction, and shrinkage operations—fundamentally different from the current random direction hill-climbing.

```python
def _apply_adaptive_local_search(self, func):
    """Nelder-Mead Simplex local search - classic derivative-free optimization."""
    if self.best_solution is None:
        return

    dim = self.dim
    # Initialize simplex: vertex 0 is best, others are perturbed in each dimension
    simplex = np.zeros((dim + 1, dim))
    simplex[0] = self.best_solution.copy()

    for i in range(1, dim + 1):
        point = self.best_solution.copy()
        point[i - 1] += self.local_radius
        simplex[i] = np.clip(point, self.lower, self.upper)

    # Evaluate all simplex vertices
    fitness = np.array([func(p) for p in simplex])
    fitness = np.clip(fitness, -1e50, 1e50)

    # Nelder-Mead parameters
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5    # contraction
    sigma = 0.5  # shrinkage

    current_fitness = float(np.min(fitness))
    improved = False

    for _ in range(self.local_max_iter):
        # Sort: simplex[0] = best, simplex[-1] = worst
        sorted_idx = np.argsort(fitness)
        simplex = simplex[sorted_idx]
        fitness = fitness[sorted_idx]

        # Check convergence: simplex diameter
        diameters = np.linalg.norm(simplex - simplex[0], axis=1)
        if np.max(diameters) < 1e-8:
            break

        # Centroid of all points except worst
        centroid = np.mean(simplex[:-1], axis=0)

        # --- Reflection ---
        worst = simplex[-1]
        reflected = centroid + alpha * (centroid - worst)
        reflected = np.clip(reflected, self.lower, self.upper)
        f_reflected = float(np.clip(func(reflected), -1e50, 1e50))

        if f_reflected < fitness[0]:
            # --- Expansion ---
            expanded = centroid + gamma * (reflected - centroid)
            expanded = np.clip(expanded, self.lower, self.upper)
            f_expanded = float(np.clip(func(expanded), -1e50, 1e50))
            if f_expanded < f_reflected:
                simplex[-1] = expanded
                fitness[-1] = f_expanded
            else:
                simplex[-1] = reflected
                fitness[-1] = f_reflected
        elif f_reflected < fitness[-2]:
            # Accept reflection (better than second worst)
            simplex[-1] = reflected
            fitness[-1] = f_reflected
        else:
            # --- Contraction ---
            contracted = centroid + rho * (worst - centroid)
            contracted = np.clip(contracted, self.lower, self.upper)
            f_contracted = float(np.clip(func(contracted), -1e50, 1e50))
            if f_contracted < fitness[-1]:
                simplex[-1] = contracted
                fitness[-1] = f_contracted
            else:
                # --- Shrinkage ---
                for i in range(1, dim + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], self.lower, self.upper)
                    fitness[i] = float(np.clip(func(simplex[i]), -1e50, 1e50))

        # Track improvement
        new_best = float(np.min(fitness))
        if new_best < current_fitness - 1e-12:
            current_fitness = new_best
            improved = True

    # Update best solution if improved
    if current_fitness < self.best_fitness:
        self.best_fitness = current_fitness
        self.best_solution = simplex[np.argmin(fitness)].copy()
        self.local_radius = min(self.local_radius * 1.2, 10.0)
    else:
        self.local_radius = max(self.local_radius * 0.8, 0.1)
```