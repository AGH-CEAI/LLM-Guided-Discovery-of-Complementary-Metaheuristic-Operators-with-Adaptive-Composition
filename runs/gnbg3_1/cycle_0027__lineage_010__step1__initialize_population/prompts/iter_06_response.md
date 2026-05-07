**Idea: Fitness-Informed LHS with Adaptive Clustering**
One-line description: Use Latin Hypercube sampling followed by fitness evaluation and clustering to bias initial population toward promising basins, providing better starting conditions for multimodal tasks.
```python
def _initialize_population(self):
    """Initialize population using fitness-informed Latin Hypercube with adaptive clustering."""
    # Phase 1: Generate space-filling LHS samples
    try:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(self.dim, scramble=True)
        samples = sampler.random(self.NP)
        samples = qmc.scale(samples, self.lb, self.ub)
    except Exception:
        samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
    
    # Phase 2: Evaluate initial samples to understand fitness landscape
    init_fitness = self.func(samples)
    
    # Phase 3: Adaptive clustering based on problem difficulty
    # For high-dimensional or potentially multimodal problems, use more clusters
    n_clusters = min(max(3, self.dim // 10), 10, self.NP // 15)
    
    if n_clusters >= 2 and self.NP > n_clusters * 3:
        try:
            from scipy.cluster.vq import kmeans2
            # Cluster positions, then analyze cluster fitness
            centroids, labels = kmeans2(samples, n_clusters, minit='points')
            
            # Compute mean fitness per cluster
            cluster_fitness = np.full(n_clusters, np.inf)
            for c in range(n_clusters):
                mask = labels == c
                if np.sum(mask) > 0:
                    cluster_fitness[c] = np.mean(init_fitness[mask])
            
            # Identify best clusters (potential basins of attraction)
            best_clusters = np.argsort(cluster_fitness)[:max(1, n_clusters // 2)]
            
            # Build new population: 60% from best clusters, 40% uniform LHS
            n_from_best = int(0.6 * self.NP)
            n_uniform = self.NP - n_from_best
            
            # Sample from best clusters weighted by inverse fitness
            inv_fitness = 1.0 / (cluster_fitness[best_clusters] - np.min(cluster_fitness[best_clusters]) + 1e-6)
            cluster_probs = inv_fitness / np.sum(inv_fitness)
            
            new_pop = []
            for _ in range(n_from_best):
                c = np.random.choice(best_clusters, p=cluster_probs)
                mask = labels == c
                idx = np.random.choice(np.where(mask)[0])
                # Add small perturbation around cluster member
                new_pop.append(samples[idx] + np.random.randn(self.dim) * 0.1 * (self.ub - self.lb) / 100)
            
            # Fill remaining with uniform LHS samples
            indices = np.random.choice(self.NP, n_uniform, replace=False)
            for idx in indices:
                new_pop.append(samples[idx])
            
            samples = np.array(new_pop)
        except Exception:
            pass  # Keep original LHS samples
    
    # Clip any out-of-bounds values
    samples = np.clip(samples, self.lb, self.ub)
    
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