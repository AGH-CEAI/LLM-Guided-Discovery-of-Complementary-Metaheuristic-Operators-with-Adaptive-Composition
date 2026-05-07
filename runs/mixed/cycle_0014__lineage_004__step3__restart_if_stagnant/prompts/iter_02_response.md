**Idea: Archive-Guided Anisotropic Restart with Stagnation-Scaled Perturbation**

The worst unsolved tasks (12: 3198, 8: 1761, 21: 50) are stuck in deep local optima. The current restart perturbs only ±10 around x_best—a tiny range that cannot escape large basins. This variant uses (1) stagnation-scaled perturbation radius that grows exponentially with time stuck, (2) anisotropic direction sampling that biases away from the crowded centroid region, and (3) archive-guided sampling to inject diversity from historically promising regions.

```python
def _restart_if_stagnant(self, population, fitness, x_best):
        NP = len(population)
        new_pop = population.copy()
        
        # Stagnation-scaled perturbation: exponentially increasing radius
        stagnation = getattr(self, 'stagnation', 0)
        base_radius = (self.upper - self.lower) * 0.1
        scale = base_radius * (1.5 ** min(stagnation, 10))
        scale = min(scale, (self.upper - self.lower) * 0.8)
        
        # Compute population centroid for anisotropic direction
        centroid = np.mean(population, axis=0)
        
        # Replace worst 40% for stronger diversity
        worst_count = max(1, int(0.4 * NP))
        worst_indices = np.argpartition(fitness, -worst_count)[-worst_count:]
        
        for i, idx in enumerate(worst_indices):
            # Alternate between three strategies
            strategy = i % 3
            
            if strategy == 0:
                # Anisotropic: bias away from centroid
                direction = x_best - centroid
                direction_norm = np.linalg.norm(direction) + 1e-10
                direction_unit = direction / direction_norm
                perp = np.ones(self.dim) - direction_unit
                perp = perp / (np.linalg.norm(perp) + 1e-10)
                new_pop[idx] = x_best + np.random.uniform(-1, 1) * scale * direction_unit + \
                               np.random.uniform(-1, 1) * scale * 0.5 * perp + \
                               np.random.randn(self.dim) * scale * 0.2
            
            elif strategy == 1:
                # Archive-guided: sample from archive with diversity bias
                if len(self.archive) > 0:
                    arch_centroid = np.mean(self.archive, axis=0)
                    arch_dists = np.linalg.norm(self.archive - arch_centroid, axis=1)
                    diversity_weights = arch_dists / (np.sum(arch_dists) + 1e-10)
                    arch_sample = self.archive[np.random.choice(
                        len(self.archive), p=diversity_weights)]
                    new_pop[idx] = arch_sample + np.random.randn(self.dim) * scale * 0.5
                else:
                    new_pop[idx] = np.random.uniform(self.lower, self.upper, self.dim)
            
            else:
                # Long-jump: completely fresh random region
                new_pop[idx] = np.random.uniform(self.lower, self.upper, self.dim)
        
        new_pop = self._clip_to_bounds_batch(new_pop)
        return new_pop
```