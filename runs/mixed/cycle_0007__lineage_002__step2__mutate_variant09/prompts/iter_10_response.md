**Idea: Opposition-Guided Restart with Niching**
A radically different approach using opposition-based learning to escape severe local optima traps, combined with fitness niche preservation to maintain diversity and periodic restart when stagnant.

```python
def _mutate_variant09(self, population, fitness):
    """Variant 09: Opposition-guided mutation with restart and niching"""
    pop_dim = population.shape
    
    # Check if we need a restart (severe stagnation or trapped)
    needs_restart = (self.stagnation_count > 40 or 
                     np.min(fitness) > 1e2)  # Stuck in bad region
    
    if needs_restart and self.best_solution is not None:
        # Generate opposition population
        opp_pop = self.lower + self.upper - population
        
        # Blend best solution with random direction to escape basin
        escape_dir = np.random.randn(self.NP, self.dim)
        escape_dir = escape_dir / (np.linalg.norm(escape_dir, axis=1, keepdims=True) + 1e-10)
        escape_scale = np.random.uniform(10, 50, size=(self.NP, 1))
        escape_pop = np.clip(self.best_solution + escape_dir * escape_scale, self.lower, self.upper)
        
        # Mix: keep some current, some opposition, some escape
        selector = np.random.randint(0, 3, size=self.NP)
        restart_pop = np.where(selector[:, None] == 0, population,
                              np.where(selector[:, None] == 1, opp_pop, escape_pop))
        
        # Fitness-based niche selection: keep diverse solutions
        sorted_idx = np.argsort(fitness)
        n_keep = max(5, self.NP // 4)
        keep_indices = sorted_idx[:n_keep]
        
        # Fill rest with opposition/escape and random
        fill_pop = np.vstack([
            restart_pop[keep_indices],
            self.lower + np.random.rand(self.NP - n_keep, self.dim) * (self.upper - self.lower)
        ])
        population = np.clip(fill_pop, self.lower, self.upper)
        fitness = np.full(self.NP, np.inf)
        self.stagnation_count = 0
    
    # Main mutation: current-to-niche-best with opposition learning
    sorted_idx = np.argsort(fitness)
    n_best = max(1, int(self.NP * self.p_best_rate))
    best_indices = sorted_idx[:n_best]
    
    # Find fitness niche: select from similar-fitness region
    best_fitness_val = fitness[sorted_idx[0]]
    niche_mask = fitness < (best_fitness_val * 1.5 + 1.0)
    niche_indices = np.where(niche_mask)[0]
    if len(niche_indices) < 2:
        niche_indices = best_indices
    
    # p_best from niche
    p_best_idx = niche_indices[np.random.randint(0, len(niche_indices), size=self.NP)]
    p_best = population[p_best_idx]
    
    # Random indices with opposition bias
    idx_a = np.random.randint(0, self.NP, (self.NP, 3))
    idx_a[:, 1:] = np.where(idx_a[:, 1:] == np.arange(self.NP)[:, None],
                            (idx_a[:, 1:] + 1) % self.NP, idx_a[:, 1:])
    idx_a[:, 2] = np.where(idx_a[:, 2] == idx_a[:, 1],
                           (idx_a[:, 2] + 1) % self.NP, idx_a[:, 2])
    r1, r2 = idx_a[:, 1], idx_a[:, 2]
    
    # Opposition vectors: generate opposite of random individuals
    opp_r1 = self.lower + self.upper - population[r1]
    opp_r2 = self.lower + self.upper - population[r2]
    
    # Adaptive F with higher values for exploration
    if self.stagnation_count > 10:
        F_adapt = np.clip(self.F * (1.0 + 0.3 * np.log1p(self.stagnation_count)), 0.5, 2.5)
    else:
        F_adapt = np.clip(self.F, 0.5, 1.5)
    
    # Mutation with opposition and niche guidance
    base_mutation = p_best - population
    diff_opp = opp_r1 - opp_r2
    diff_current = population[r1] - population[r2]
    
    # Blend: 60% niche-guided, 40% opposition-based difference
    donors = population + F_adapt * base_mutation + 0.5 * F_adapt * (0.6 * diff_current + 0.4 * diff_opp)
    
    # Add small opposition perturbation to some individuals
    opp_perturb = np.random.rand(self.NP, self.dim) < 0.2
    perturb_scale = np.random.uniform(1, 10, size=(self.NP, self.dim))
    opp_offset = (self.lower + self.upper - donors) * opp_perturb * perturb_scale * 0.1
    donors = donors + opp_offset
    
    return np.clip(donors, self.lower, self.upper)
```