Looking at the error patterns, the worst unsolved tasks (17, 11, 16, 12, 10, 20, 23, 18) all have errors in the 10-2000 range, indicating the optimizer is getting trapped in **severe local optima**. The key insight is that current LHS initialization treats all regions equally without any fitness-based guidance, so if the global optimum lies in a "dark corner" of the search space, the population may never discover it.

**Idea: Adaptive Basin-Seeking Initialization**
Evaluate an oversampling phase to discover promising basins, then cluster and reinitialize the population around the best basins weighted by fitness quality. This gives CMA-ES a head-start toward better optima instead of starting blind.

```python
def _initialize_population(self):
    """Initialize using adaptive basin-seeking: oversample, evaluate, cluster, reinitialize."""
    lb, ub, dim, NP = self.lb, self.ub, self.dim, self.NP
    
    # Phase 1: Oversample with LHS for space-filling coverage
    oversample_factor = 3
    n_candidates = NP * oversample_factor
    
    candidates = np.zeros((n_candidates, dim))
    for d in range(dim):
        bins = np.linspace(lb[d], ub[d], n_candidates + 1)
        perm = np.random.permutation(n_candidates)
        candidates[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(n_candidates)
    
    # Phase 2: Evaluate to discover promising basins
    candidate_fitness = self.func(candidates)
    
    # Phase 3: Identify cluster centers from top candidates
    n_top = max(10, NP // 4)
    sorted_idx = np.argsort(candidate_fitness)
    top_idx = sorted_idx[:n_top]
    top_candidates = candidates[top_idx]
    
    # Cluster top candidates to find distinct basins
    n_clusters = min(5, n_top)
    if n_top >= n_clusters * 2:
        from scipy.cluster.hierarchy import fcluster, linkage
        Z = linkage(top_candidates, method='ward')
        labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        cluster_centers = []
        cluster_best_fitness = []
        for c in range(1, n_clusters + 1):
            mask = labels == c
            if np.sum(mask) > 0:
                cluster_centers.append(np.mean(top_candidates[mask], axis=0))
                cluster_best_fitness.append(np.min(candidate_fitness[top_idx[mask]]))
    else:
        cluster_centers = [top_candidates[i] for i in range(min(n_top, 5))]
        cluster_best_fitness = [candidate_fitness[top_idx[i]] for i in range(min(n_top, 5))]
    
    if not cluster_centers:
        cluster_centers = [np.mean(candidates, axis=0)]
        cluster_best_fitness = [np.min(candidate_fitness)]
    
    # Phase 4: Reinitialize population around basins weighted by fitness
    init_pop = np.zeros((NP, dim))
    n_basins = len(cluster_centers)
    
    # Weight basins by inverse fitness (better basins get more samples)
    inv_fitness = np.array([1.0 / (max(f, 1e-10)) for f in cluster_best_fitness])
    weights = inv_fitness / np.sum(inv_fitness)
    
    # Allocate population to basins
    basin_alloc = np.zeros(n_basins, dtype=int)
    remainder = NP
    for b in range(n_basins):
        basin_alloc[b] = int(NP * weights[b])
        remainder -= basin_alloc[b]
    # Distribute remainder to best basins
    for b in range(remainder):
        basin_alloc[b % n_basins] += 1
    
    idx = 0
    spread = 0.15 * (ub[0] - lb[0])
    for b in range(n_basins):
        for _ in range(basin_alloc[b]):
            init_pop[idx] = cluster_centers[b] + np.random.randn(dim) * spread
            idx += 1
    
    # Fill remaining with random LHS
    while idx < NP:
        d = idx % dim
        bins = np.linspace(lb[d], ub[d], NP + 1)
        perm = np.random.permutation(NP)
        init_pop[idx] = lb + np.random.rand(dim) * (ub - lb)
        idx += 1
    
    self.population = np.clip(init_pop, lb, ub)
    self.mean = np.mean(self.population, axis=0)
    self.old_mean = self.mean.copy()
    
    self.C = np.cov(self.population.T) + 1e-8 * np.eye(dim)
    self.sigma = np.mean(np.std(self.population, axis=0))
    self.sigma = np.clip(self.sigma, 1e-6, (ub[0] - lb[0]) / 3.0)
    self.ps = np.zeros(dim)
    self.pc = np.zeros(dim)
    
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