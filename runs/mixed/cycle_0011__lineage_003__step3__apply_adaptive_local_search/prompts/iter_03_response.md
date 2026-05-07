Looking at the worst unsolved tasks (errors of 1e+01 to 1e+03), the current local search fails because:
1. **Only 3 candidates per iteration** (1 direction ± plus center) is far too sparse for multi-modal landscapes
2. **Random single directions** are inefficient when stuck far from the global optimum
3. **Small step adaptation** (0.4x/1.5x) doesn't scale well for tasks needing orders-of-magnitude improvement

The key insight: tasks with 1e+03 error need a local search that aggressively samples many directions at multiple scales to escape poor local optima and find the basin containing the global optimum.

**Idea: Multi-Scale Multi-Direction Pattern Search**

```python
def _apply_adaptive_local_search(self, func):
    if self.best_solution is None:
        return

    center = self.best_solution.copy()
    current_fitness = self.best_fitness
    radius = self.local_radius

    # Generate dense set of test directions (more candidates = better coverage)
    num_directions = max(16, 2 * self.dim)
    angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)

    for _ in range(self.local_max_iter):
        if radius < 1e-6:
            break

        # Build candidate set: multi-radius, multi-direction grid
        candidates = [center]
        for r in [radius, radius * 3.0]:  # Test fine AND coarse scale
            for angle in angles:
                # Create direction in a random 2D subspace for this angle
                direction = np.zeros(self.dim)
                d1 = np.random.randint(0, self.dim)
                d2 = np.random.randint(0, self.dim - 1)
                if d2 >= d1:
                    d2 += 1
                direction[d1] = np.cos(angle)
                direction[d2] = np.sin(angle)
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                candidates.append(np.clip(center + r * direction, self.lower, self.upper))

        candidates = np.vstack(candidates)
        fit_candidates = self._eval_wrapper_batch(candidates, func)
        best_local_idx = np.argmin(fit_candidates)
        best_local_fitness = float(fit_candidates[best_local_idx])

        if best_local_fitness < current_fitness:
            new_center = candidates[best_local_idx]
            # Compute vector from old center to new center
            step_vector = new_center - center
            step_norm = np.linalg.norm(step_vector) + 1e-10

            if step_norm > 1e-10:
                # Try extrapolating further along the improvement direction
                extrapolated = np.clip(center + step_vector * 2.0, self.lower, self.upper)
                extra_fitness = float(self._eval_wrapper_batch(extrapolated[np.newaxis], func)[0])

                if extra_fitness < best_local_fitness:
                    center = extrapolated
                    current_fitness = extra_fitness
                else:
                    center = new_center
                    current_fitness = best_local_fitness
            else:
                center = new_center
                current_fitness = best_local_fitness

            # Expand radius on success
            radius = min(radius * 1.8, 10.0)
        else:
            # Contract more aggressively on failure
            radius *= 0.3

    if current_fitness < self.best_fitness:
        self.best_fitness = current_fitness
        self.best_solution = center.copy()
        self.local_radius = min(radius * 1.2, 10.0)
    else:
        self.local_radius = max(radius * 0.8, 0.1)
```