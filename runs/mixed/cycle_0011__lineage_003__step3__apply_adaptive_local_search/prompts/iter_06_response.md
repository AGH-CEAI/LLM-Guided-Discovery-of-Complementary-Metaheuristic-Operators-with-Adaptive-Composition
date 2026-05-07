**Idea: ScipyLBFGSB_Local_Search**
Replace random-direction sampling with scipy L-BFGS-B optimizer using numerical gradients, multi-scale restarts, and progressive refinement — fundamentally different from random perturbation search.

```python
def _apply_adaptive_local_search(self, func):
    if self.best_solution is None:
        return

    try:
        from scipy.optimize import minimize
    except ImportError:
        return

    def objective(x):
        return float(func(x.reshape(1, -1))[0])

    center = self.best_solution.copy()
    current_fitness = self.best_fitness
    bounds = [(self.lower[i], self.upper[i]) for i in range(self.dim)]

    # Multi-scale restart: start large, refine progressively
    scales = [max(self.local_radius, 1.0), 0.5, 0.1]
    max_iter_per_scale = [30, 60, 100]

    best_result = None
    best_result_fitness = current_fitness

    for scale, max_iter in zip(scales, max_iter_per_scale):
        x0 = center.copy()

        for restart in range(2):
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': max_iter,
                        'ftol': 0.0,
                        'gtol': 1e-12,
                        'maxfun': max_iter * self.dim * 2,
                        'maxls': 20,
                        'disp': False
                    }
                )

                if result.fun < best_result_fitness:
                    best_result = result
                    best_result_fitness = float(result.fun)
                    if best_result_fitness < current_fitness:
                        center = result.x.copy()

                # Perturb for next restart if no improvement yet
                if restart == 0 and result.fun >= best_result_fitness - 1e-10:
                    perturbation = np.random.randn(self.dim) * scale * 0.5
                    x0 = np.clip(center + perturbation, self.lower, self.upper)
                else:
                    break

            except Exception:
                break

        # Early exit if good enough
        if best_result_fitness <= 1e-10:
            break

    if best_result_fitness < current_fitness:
        self.best_fitness = best_result_fitness
        self.best_solution = center.copy()
        # Adjust radius based on how much improvement achieved
        improvement_ratio = (current_fitness - best_result_fitness) / (abs(current_fitness) + 1e-10)
        if improvement_ratio > 0.5:
            self.local_radius = min(self.local_radius * 1.5, 10.0)
        elif improvement_ratio > 0.1:
            self.local_radius = max(self.local_radius * 0.9, 0.05)
        else:
            self.local_radius = max(self.local_radius * 0.7, 0.01)
```