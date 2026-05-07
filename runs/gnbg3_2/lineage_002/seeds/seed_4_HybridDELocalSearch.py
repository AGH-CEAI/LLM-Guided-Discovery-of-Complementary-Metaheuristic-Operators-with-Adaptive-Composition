import numpy as np


class HybridDELocalSearch:
    """
    Hybrid Differential Evolution with Ring-Topology Mutation and
    Nelder-Mead Local Search Refinement.
    
    Key design choices (different from DE, CMA-ES, PSO variants):
    - Ring neighborhood topology for mutation sources (cellular DE style)
    - Periodic Nelder-Mead simplex refinement on top performers
    - Success-rate based adaptation for F and Cr separately
    - Diversity-triggered restart mechanism
    - Small simplices (dim+1) built from nearest neighbors in Euclidean space
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(120, max(40, 4 * dim))
        self.f_max = 100.0
        self.f_min = -100.0
        
        # Ring topology: each individual connects to 5 nearest in ring
        self.ring_size = min(5, self.np - 1)
        
        # Adaptive parameters
        self.f = 0.5
        self.cr = 0.5
        self.f_history = [0.5]
        self.cr_history = [0.5]
        
        # Tracking
        self.fitness_evals = 0
        self.gen = 0
        self.stagnation_count = 0
        self.last_best = np.inf
        
        # Local search config
        self.ls_interval = 5
        self.ls_fraction = 0.1
    
    def _initialize_population(self):
        pop = np.random.uniform(
            self.f_min, self.f_max, size=(self.np, self.dim)
        )
        pop = np.clip(pop, self.f_min, self.f_max)
        return pop
    
    def _evaluate_batch(self, pop, func):
        if pop.shape[0] == 0:
            return np.array([])
        fitness = func(pop)
        self.fitness_evals += len(fitness)
        if len(fitness) < len(pop):
            self.fitness_evals = np.inf
        return fitness
    
    def _build_ring_neighbors(self):
        half = self.ring_size // 2
        neighbors = np.zeros((self.np, self.ring_size), dtype=int)
        for i in range(self.np):
            neighbors[i] = np.array([
                (i - half + j) % self.np for j in range(self.ring_size)
            ])
        return neighbors
    
    def _mutate_ring_batch(self, pop, neighbors):
        np_r, dim = pop.shape
        
        # Shuffle neighbors for each target to add randomness
        shuffled = np.array([np.random.permutation(neighbors[i]) for i in range(np_r)])
        
        # Use first 3 distinct neighbors for DE/current-to-pbest/1 style mutation
        r0 = shuffled[:, 0]
        r1 = shuffled[:, 1]
        r2 = shuffled[:, 2]
        
        # Find pbest indices (top 20% in fitness)
        n_pbest = max(1, np_r // 5)
        sorted_idx = np.argsort(self.fitness)
        pbest_mask = np.isin(np.arange(np_r), sorted_idx[:n_pbest])
        pbest_pool = np.where(pbest_mask)[0]
        
        # Sample random pbest for each target
        pbest_idx = pbest_pool[np.random.randint(0, len(pbest_pool), size=np_r)]
        
        # Current-to-pbest/1 mutation: X_i + F*(X_pbest - X_i) + F*(X_r0 - X_r1)
        base = pop[np.arange(np_r)]
        pbest = pop[pbest_idx]
        x_r0 = pop[r0]
        x_r1 = pop[r1]
        x_r2 = pop[r2]
        
        # Adaptive F with slight perturbation
        f_adapted = self.f * (0.9 + 0.2 * np.random.random())
        
        mutant = base + f_adapted.reshape(-1, 1) * (pbest - base) + \
                 f_adapted.reshape(-1, 1) * (x_r0 - x_r1)
        
        return mutant
    
    def _crossover_binomial_batch(self, pop, mutant):
        np_r, dim = self.dim
        
        # Binomial crossover
        cr_adapted = np.clip(self.cr + 0.1 * np.random.randn(), 0.1, 0.9)
        mask = np.random.random((np_r, dim)) < cr_adapted.reshape(-1, 1)
        
        # Ensure at least one dimension from mutant
        enforce_idx = np.random.randint(0, dim, size=np_r)
        enforce_mask = np.zeros_like(mask)
        enforce_mask[np.arange(np_r), enforce_idx] = True
        
        mask = mask | enforce_mask
        
        trial = np.where(mask, mutant, pop)
        
        # Clip to bounds
        trial = np.clip(trial, self.f_min, self.f_max)
        
        return trial, cr_adapted
    
    def _select_survivors_batch(self, pop, fitness, trial, trial_fit, cr_adapted):
        # Greedy selection
        better_mask = trial_fit < fitness
        
        new_pop = np.where(better_mask.reshape(-1, 1), trial, pop)
        new_fitness = np.where(better_mask, trial_fit, fitness)
        
        # Track which trials succeeded for adaptation
        self.last_success_mask = better_mask
        self.last_cr_used = cr_adapted
        
        return new_pop, new_fitness
    
    def _find_k_nearest_batch(self, center_idx, candidates, k):
        np_c, dim = candidates.shape
        distances = np.linalg.norm(candidates - candidates[center_idx], axis=1)
        distances[center_idx] = np.inf
        nearest = np.argpartition(distances, k - 1)[:k]
        nearest = nearest[np.argsort(distances[nearest])]
        return nearest
    
    def _nm_refine_batch(self, pop, top_indices):
        if len(top_indices) == 0:
            return pop
        
        np_t = len(top_indices)
        k = self.dim + 1
        
        refined = pop[top_indices].copy()
        
        for i in range(np_t):
            idx = top_indices[i]
            neighbor_idx = self._find_k_nearest_batch(idx, pop, k)
            
            simplex = pop[neighbor_idx]
            centroid = np.mean(simplex[:-1], axis=0)
            
            worst = simplex[-1]
            reflect = 2 * centroid - worst
            
            f_vals = self._local_fitness_cache[neighbor_idx]
            f_worst = f_vals[-1]
            f_reflect = self._eval_single(reflect)
            
            if f_reflect < np.min(f_vals):
                expand = 3 * centroid - 2 * worst
                f_expand = self._eval_single(expand)
                if f_expand < f_reflect:
                    refined[i] = expand
                else:
                    refined[i] = reflect
            elif f_reflect < f_worst:
                contract = 0.5 * centroid + 0.5 * worst
                f_contract = self._eval_single(contract)
                if f_contract < f_worst:
                    refined[i] = contract
                else:
                    refined[i] = centroid
            else:
                inside = 0.5 * centroid + 0.5 * worst
                refined[i] = inside
        
        return refined
    
    def _eval_single(self, x):
        return self._func_ref(x.reshape(1, -1))[0]
    
    def _adapt_parameters(self):
        success_rate = np.mean(self.last_success_mask)
        
        # Adapt F: increase if too few successes, decrease if many successes
        if success_rate > 0.3:
            self.f = min(1.0, self.f * 1.1)
        elif success_rate < 0.1:
            self.f = max(0.1, self.f * 0.9)
        
        # Adapt Cr: track which Cr values worked well
        successful_cr = self.last_cr_used[self.last_success_mask]
        if len(successful_cr) > 0:
            self.cr = 0.9 * self.cr + 0.1 * np.mean(successful_cr)
        
        self.f_history.append(self.f)
        self.cr_history.append(self.cr)
    
    def _compute_diversity(self, pop):
        centroid = np.mean(pop, axis=0)
        distances = np.linalg.norm(pop - centroid, axis=1)
        return np.mean(distances)
    
    def _restart_if_stagnant(self, pop, fitness):
        current_best = np.min(fitness)
        
        if current_best >= self.last_best - 1e-8:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        
        self.last_best = current_best
        
        diversity = self._compute_diversity(pop)
        
        if self.stagnation_count > 50 or diversity < 0.1:
            self.stagnation_count = 0
            # Reinitialize worst half with biased mutation toward best
            sorted_idx = np.argsort(fitness)
            n_replace = self.np // 2
            replace_idx = sorted_idx[-n_replace:]
            
            best_idx = sorted_idx[0]
            best = pop[best_idx]
            
            for idx in replace_idx:
                pop[idx] = best + np.random.randn(self.dim) * diversity
                pop[idx] = np.clip(pop[idx], self.f_min, self.f_max)
        
        return pop, fitness
    
    def _update_best(self, fitness):
        best_local_idx = np.argmin(fitness)
        best_local_fitness = fitness[best_local_idx]
        
        if best_local_fitness < self.best_fitness:
            self.best_fitness = best_local_fitness
            self.best_x = self.population[best_local_idx].copy()
    
    def __call__(self, func, stopping_condition):
        self._func_ref = func
        self.population = self._initialize_population()
        self.fitness = self._evaluate_batch(self.population, func)
        
        if len(self.fitness) == 0:
            return np.inf, np.zeros(self.dim)
        
        self.best_fitness = np.min(self.fitness)
        self.best_x = self.population[np.argmin(self.fitness)].copy()
        
        self.fitness_evals = len(self.fitness)
        self.gen = 0
        neighbors = self._build_ring_neighbors()
        
        while not stopping_condition():
            self.gen += 1
            
            # Mutation using ring topology
            mutant = self._mutate_ring_batch(self.population, neighbors)
            
            # Crossover
            trial, cr_adapted = self._crossover_binomial_batch(self.population, mutant)
            
            # Evaluate trials
            trial_fit = self._evaluate_batch(trial, func)
            
            if len(trial_fit) < len(trial):
                self.population = trial[:len(trial_fit)]
                self.fitness = trial_fit
                self._update_best(self.fitness)
                break
            
            # Check stopping after trial evaluation
            if stopping_condition():
                break
            
            # Selection
            self.population, self.fitness = self._select_survivors_batch(
                self.population, self.fitness, trial, trial_fit, cr_adapted
            )
            
            # Local search refinement on top performers
            if self.gen % self.ls_interval == 0:
                n_ls = max(1, int(self.np * self.ls_fraction))
                top_idx = np.argsort(self.fitness)[:n_ls]
                self._local_fitness_cache = self.fitness[neighbors[top_idx].flatten()].reshape(n_ls, -1)
                refined = self._nm_refine_batch(self.population, top_idx)
                
                refined_fit = self._evaluate_batch(refined, func)
                if len(refined_fit) < len(refined):
                    self.population = np.vstack([self.population, refined[:len(refined_fit)]])
                    self.fitness = np.concatenate([self.fitness, refined_fit])
                    break
                
                improved = refined_fit < self.fitness[top_idx]
                self.population[top_idx[improved]] = refined[improved]
                self.fitness[top_idx[improved]] = refined_fit[improved]
                
                if stopping_condition():
                    self._update_best(self.fitness)
                    break
            
            # Update best
            self._update_best(self.fitness)
            
            # Adapt parameters
            self._adapt_parameters()
            
            # Restart if stagnant
            if self.gen % 10 == 0:
                self.population, self.fitness = self._restart_if_stagnant(
                    self.population, self.fitness
                )
                neighbors = self._build_ring_neighbors()
        
        return self.best_fitness, self.best_x
