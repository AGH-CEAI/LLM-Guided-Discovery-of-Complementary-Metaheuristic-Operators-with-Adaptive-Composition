Looking at the problem:
- **Task 17** has error ~2,000 — this is catastrophically far from target, suggesting a function with isolated global optimum that Latin Hypercube Sampling (LHS) likely misses
- The top 8 unsolved tasks all have errors > 10, indicating **multi-modal or deceptive landscapes** where LHS's uniform space-filling doesn't bias toward promising regions
- All 7 previous variants modified `_adapt_covariance_*` methods, but none addressed the initialization phase

**Idea: Elite-Focused Biased Initialization**
Instead of pure space-filling (LHS), use a two-phase approach: (1) generate LHS candidates, (2) evaluate a small subset, then (3) bias the remaining population toward regions near elite solutions using exponential weighting. This directly targets isolated global optima — if the global optimum is in a narrow basin, LHS might place zero individuals there, but biased sampling around the best LHS-found individuals increases coverage of that basin.

```python
def _initialize_population(self):
        """Initialize population with elite-biased sampling for isolated optima."""
        samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        
        # Phase 1: Latin Hypercube for space-filling
        for d in range(self.dim):
            bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
            perm = np.random.permutation(self.NP)
            samples[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(self.NP)
        
        # Phase 2: Quick evaluation of subset to identify promising regions
        eval_size = min(32, self.NP // 4)
        eval_indices = np.random.choice(self.NP, eval_size, replace=False)
        eval_samples = samples[eval_indices]
        eval_fitness = self.func(eval_samples)
        
        # Phase 3: Bias remaining population toward elite regions
        elite_count = min(8, eval_size)
        elite_indices = eval_indices[np.argsort(eval_fitness)[:elite_count]]
        elite_samples = samples[elite_indices]
        
        # Compute adaptive spread based on elite distribution
        if elite_count >= 2:
            elite_spread = np.std(elite_samples, axis=0)
            elite_spread = np.maximum(elite_spread, (self.ub - self.lb) * 0.01)
        else:
            elite_spread = (self.ub - self.lb) * 0.1
        
        # Generate biased samples around elite solutions using exponential weighting
        remaining = self.NP - eval_size
        if remaining > 0:
            # Weight by inverse fitness (better = higher weight)
            eval_fitness_arr = np.array([float(np.asarray(f).flatten()[0]) for f in eval_fitness])
            weights = np.exp(-eval_fitness_arr / (np.std(eval_fitness_arr) + 1e-10))
            weights /= np.sum(weights)
            
            # Sample around elites with exponential decay
            for i in range(remaining):
                # Select elite weighted by fitness
                elite_idx = np.random.choice(elite_count)
                center = elite_samples[elite_idx]
                # Exponential decay: most samples close, some far for exploration
                t = np.random.exponential(0.5)
                t = min(t, 3.0)
                biased = center + t * elite_spread * np.random.randn(self.dim)
                samples[eval_indices[i]] = np.clip(biased, self.lb, self.ub)
        
        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        
        # Covariance from initial population (exponential weighting)
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