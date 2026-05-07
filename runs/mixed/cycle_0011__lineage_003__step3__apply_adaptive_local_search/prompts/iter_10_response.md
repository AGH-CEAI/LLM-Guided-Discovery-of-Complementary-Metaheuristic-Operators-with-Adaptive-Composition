**Idea: PCA-Rotated Coordinate Search with Golden Section Line Search**

Replaces random-direction perturbation with systematic coordinate-wise golden-section search, plus PCA-based rotation to escape local basins. This targets the severely-stuck tasks (12, 21, 23 with errors 10-100x) by escaping degenerate local optima through rotation, while the coordinate search provides finer-grained exploitation than the current 3-candidate random approach.

```python
def _apply_adaptive_local_search(self, func):
    if self.best_solution is None:
        return

    center = self.best_solution.copy()
    current_fitness = self.best_fitness
    radius = self.local_radius

    if not hasattr(self, '_ls_direction_history'):
        self._ls_direction_history = np.zeros((20, self.dim))
        self._ls_history_idx = 0
        self._ls_history_count = 0

    def golden_section_search(func, x0, direction, a, b, tol=1e-8, max_iter=30):
        phi = (1 + np.sqrt(5)) / 2
        res_phi = 2 - phi
        x1 = a + res_phi * (b - a)
        x2 = b - res_phi * (b - a)
        f1 = float('inf')
        f2 = float('inf')
        try:
            f1 = float(func(x0 + x1 * direction))
        except:
            pass
        try:
            f2 = float(func(x0 + x2 * direction))
        except:
            pass
        for _ in range(max_iter):
            if b - a < tol:
                break
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = a + res_phi * (b - a)
                try:
                    f1 = float(func(x0 + x1 * direction))
                except:
                    f1 = float('inf')
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = b - res_phi * (b - a)
                try:
                    f2 = float(func(x0 + x2 * direction))
                except:
                    f2 = float('inf')
        if f1 < f2:
            return x1, f1
        return x2, f2

    improved = False
    for _ in range(self.local_max_iter):
        if radius < 1e-8:
            break
        best_pos = center.copy()
        best_local_fit = current_fitness
        step_reduced = True

        for d in range(self.dim):
            direction = np.zeros(self.dim)
            direction[d] = 1.0
            step_pos = radius
            step_neg = -radius

            try:
                fit_pos = float(func(center + step_pos * direction))
            except:
                fit_pos = float('inf')
            try:
                fit_neg = float(func(center + step_neg * direction))
            except:
                fit_neg = float('inf')

            if fit_pos < best_local_fit or fit_neg < best_local_fit:
                if fit_pos < fit_neg:
                    direction[d] = 1.0
                    step_bound = step_pos
                    fit_bound = fit_pos
                else:
                    direction[d] = -1.0
                    step_bound = -step_neg
                    fit_bound = fit_neg
                step_reduced = False
                opt_step, opt_fit = golden_section_search(func, center, direction, 0.0, step_bound)
                if opt_fit < best_local_fit:
                    best_local_fit = opt_fit
                    best_pos = center + opt_step * direction
                    if abs(opt_step) > 1e-10:
                        dir_vec = opt_step * direction
                        self._ls_direction_history[self._ls_history_idx] = dir_vec
                        self._ls_history_idx = (self._ls_history_idx + 1) % 20
                        self._ls_history_count = min(self._ls_history_count + 1, 20)

        if best_local_fit < current_fitness:
            center = best_pos.copy()
            current_fitness = best_local_fit
            improved = True
            radius = min(radius * 1.5, 10.0)
        else:
            radius *= 0.4

    if improved or current_fitness < self.best_fitness:
        if current_fitness < self.best_fitness:
            self.best_fitness = current_fitness
            self.best_solution = center.copy()
            self.local_radius = min(radius * 1.2, 10.0)
        else:
            self.local_radius = max(radius * 0.8, 0.1)

        if self._ls_history_count >= 5:
            history = self._ls_direction_history[:self._ls_history_count]
            hist_mean = np.mean(history, axis=0)
            centered = history - hist_mean
            cov = np.cov(centered.T)
            cov = (cov + cov.T) / 2.0
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
                idx = np.argsort(eigvals)[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                if eigvals[0] > 1e-6 and eigvals[0] / (np.sum(eigvals) + 1e-10) > 0.3:
                    R = eigvecs[:, :min(3, self.dim)]
                    center_rot = np.dot(R.T, center)
                    improved_rot = False
                    for _ in range(2):
                        for d in range(min(3, self.dim)):
                            direction = np.zeros(min(3, self.dim))
                            direction[d] = 1.0
                            step_bounds = [radius * 0.5, -radius * 0.5]
                            for sb in step_bounds:
                                if abs(sb) < 1e-10:
                                    continue
                                opt_step, opt_fit = golden_section_search(
                                    lambda x: func(center + np.dot(R, x)),
                                    center_rot, direction, 0.0, sb)
                                if opt_fit < current_fitness:
                                    center_rot = center_rot + opt_step * direction
                                    current_fitness = opt_fit
                                    improved_rot = True
                        if not improved_rot:
                            break
                    center = np.dot(R, center_rot)
                    center = np.clip(center, self.lower, self.upper)
                    if current_fitness < self.best_fitness:
                        self.best_fitness = current_fitness
                        self.best_solution = center.copy()
            except:
                pass
```