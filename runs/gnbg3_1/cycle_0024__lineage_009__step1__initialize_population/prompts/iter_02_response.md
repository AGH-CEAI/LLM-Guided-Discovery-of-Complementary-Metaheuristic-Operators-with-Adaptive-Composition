**Idea: Gradient-Guided Directional Sampling**

Replace uniform LHS with a hybrid approach that adds directional samples along multiple random vectors from the centroid, scaled exponentially from small to large step sizes. This probes the landscape at multiple scales along diverse directions, increasing probability that at least one sample lands in the basin of attraction for the global optimum—especially critical for the severely failing tasks (17, 11, 16) where the optimizer is stuck ~11 decades above target.

```python
def _initialize_population(self):
        """Initialize population using hybrid LHS + directional sampling."""
        half_lhs = self.NP // 2
        half_dir = self.NP - half_lhs
        
        samples = np.empty((self.NP, self.dim))
        
        # Standard LHS for space-filling coverage (half the population)
        for d in range(self.dim):
            bins = np.linspace(self.lb[d], self.ub[d], half_lhs + 1)
            perm = np.random.permutation(half_lhs)
            samples[:half_lhs, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(half_lhs)
        
        # Directional sampling: probe along random directions at multiple scales
        center = (self.lb + self.ub) / 2.0
        range_vec = (self.ub - self.lb) / 2.0
        
        num_dirs = max(3, half_dir // 2)
        for i in range(num_dirs):
            dir_vec = np.random.randn(self.dim)
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm < 1e-10:
                continue
            dir_vec /= dir_norm
            
            # Exponential scale factors: probe from 0.01% to 50% of search space
            scales = np.logspace(-4, np.log10(0.5), num=2)
            for j, scale in enumerate(scales):
                idx = half_lhs + i * 2 + j
                if idx >= self.NP:
                    break
                offset = dir_vec * scale * range_vec
                offset = np.clip(offset, self.lb - center, self.ub - center)
                samples[idx] = center + offset
        
        # Fill remaining slots with uniform samples
        for idx in range(half_lhs + num_dirs * 2, self.NP):
            samples[idx] = np.random.uniform(self.lb, self.ub)
        
        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        
        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        
        self.fitness = self.func(self.population)
        
        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        
        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0
        
        self._reset_operator_state()
```