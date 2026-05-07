**Idea: Multi-Direction Pattern Search with Population-Guided Exploration**

A fundamentally different strategy that evaluates MANY directions simultaneously (vs. current 1-2 random), uses orthogonal pattern directions, and incorporates population knowledge to guide exploration toward discovered good regions. This targets the close-but-not-there tasks (1e-08 to 1e-05 range) that need more aggressive local refinement.

```python
def _apply_adaptive_local_search(self, func):
    if self.best_solution is None:
        return

    center = self.best_solution.copy()
    current_fitness = self.best_fitness
    radius = self.local_radius

    # Generate orthogonal pattern directions for systematic exploration
    def _generate_pattern_directions(n_dims, n_dirs):
        dirs = []
        for _ in range(n_dirs):
            d = np.random.randn(n_dims)
            # Orthogonalize against existing directions via Gram-Schmidt
            for existing in dirs:
                d = d - np.dot(existing, d) * existing / (np.dot(existing, existing) + 1e-10)
            norm = np.linalg.norm(d)
            if norm > 1e-8:
                dirs.append(d / norm)
        return dirs

    # Build candidate set: pattern directions + population-guided directions
    n_pattern_dirs = min(8, self.dim)
    pattern_dirs = _generate_pattern_directions(self.dim, n_pattern_dirs)

    candidates = [center.copy()]

    # Add pattern direction perturbations
    for direction in pattern_dirs:
        candidates.append(np.clip(center + radius * direction, self.lower, self.upper))
        candidates.append(np.clip(center - radius * direction, self.lower, self.upper))

    # Add population-guided directions: toward/from best-known solutions
    if hasattr(self, 'cultural_memory') and len(self.cultural_memory) > 0:
        weights = self.cultural_weights[:len(self.cultural_memory)]
        if np.sum(weights) > 1e-10:
            cultural_dir = self.cultural_memory[:len(self.cultural_memory)] * weights[:, np.newaxis]
            cultural_dir = np.sum(cultural_dir, axis=0)
            norm_cd = np.linalg.norm(cultural_dir)
            if norm_cd > 1e-10:
                cultural_dir = cultural_dir / norm_cd
                candidates.append(np.clip(center + radius * cultural_dir, self.lower, self.upper))
                candidates.append(np.clip(center - radius * cultural_dir, self.lower, self.upper))

    # Add intermediate radius candidates for fine-grained search
    for direction in pattern_dirs[:3]:
        candidates.append(np.clip(center + radius * 0.5 * direction, self.lower, self.upper))
        candidates.append(np.clip(center - radius * 0.5 * direction, self.lower, self.upper))

    candidates = np.vstack(candidates)
    fit_candidates = self._eval_wrapper_batch(candidates, func)
    best_local_idx = np.argmin(fit_candidates)
    best_local_fitness = float(fit_candidates[best_local_idx])

    # Update center and radius based on improvement
    if best_local_fitness < current_fitness:
        improvement = current_fitness - best_local_fitness
        center = candidates[best_local_idx].copy()
        current_fitness = best_local_fitness

        # If improvement is small, reduce radius for finer search
        if improvement < 1e-4 * abs(current_fitness + 1e-10):
            radius = max(radius * 0.5, 1e-6)
        else:
            radius = min(radius * 1.2, 10.0)
    else:
        radius *= 0.5

    # Update best if improved
    if current_fitness < self.best_fitness:
        self.best_fitness = current_fitness
        self.best_solution = center.copy()
        self.local_radius = max(radius, 0.1)
    else:
        self.local_radius = max(radius * 0.9, 0.05)
```