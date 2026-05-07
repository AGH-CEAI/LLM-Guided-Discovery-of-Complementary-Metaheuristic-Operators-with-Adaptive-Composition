import numpy as np


class HybridMemeticDifferentialEvolution:
    """Hybrid DE with Nelder-Mead local search and adaptive step-size control."""
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(6 * dim, 60), 400)
        self.lb = np.full(dim, -100.0)
        self.ub = np.full(dim, 100.0)
        self.range = self.ub - self.lb
        
        self.F = 0.7
        self.CR = 0.5
        self.step_size = 0.15 * self.range
        
        self.f_opt = np.inf
        self.x_opt = None
        self.stagnation_gen = 0
        
    def _initialize_population(self):
        r = np.random.default_rng()
        pop = self.lb + r.uniform(0, 1, (self.np, self.dim)) * self.range
        return pop
    
    def _mutate_batch(self, population, best_vec):
        r = np.random.default_rng()
        n = population.shape[0]
        indices = r.choice(self.np, size=(n, 5), replace=False)
        r1, r2, r3, r4, r5 = [indices[:, i] for i in range(5)]
        
        v1 = population[r1]
        v2 = population[r2]
        v3 = population[r3]
        v4 = population[r4]
        v5 = population[r5]
        
        F1 = self.F * (0.9 + 0.2 * r.uniform(0, 1, (n, 1)))
        F2 = 0.5 * F1
        
        mutant = v1 + F1 * (best_vec - v1) + F2 * (v2 - v3) + 0.3 * F1 * (v4 - v5)
        return np.clip(mutant, self.lb, self.ub)
    
    def _crossover_batch(self, targets, mutants):
        r = np.random.default_rng()
        n, d = targets.shape
        mask = r.uniform(0, 1, (n, d)) < self.CR
        
        j_rand = r.integers(0, d, size=n)
        mask[np.arange(n), j_rand] = True
        
        trials = np.where(mask, mutants, targets)
        return np.clip(trials, self.lb, self.ub)
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        better = trial_fitness < fitness
        population = np.where(better[:, np.newaxis], trials, population)
        fitness = np.where(better, trial_fitness, fitness)
        return population, fitness
    
    def _local_search_batch(self, population, fitness, func):
        r = np.random.default_rng()
        top_k = min(self.np // 4, 20)
        top_idx = np.argpartition(fitness, top_k)[:top_k]
        
        if len(top_idx) == 0:
            return population, fitness
        
        candidates = population[top_idx]
        
        centroid = (population.sum(axis=0) - candidates) / (self.np - len(top_idx))
        centroid = np.broadcast_to(centroid, candidates.shape).copy()
        
        alpha = 1.5
        reflections = centroid + alpha * (centroid - candidates)
        reflections = np.clip(reflections, self.lb, self.ub)
        ref_fitness = func(reflections)
        
        improved = ref_fitness < fitness[top_idx]
        if np.any(improved):
            shrink_mask = ~improved
            shrink_factor = 0.3
            
            candidates[improved] = reflections[improved]
            fitness[top_idx[improved]] = ref_fitness[improved]
            
            contracted = candidates[shrink_mask] + shrink_factor * (centroid[shrink_mask] - candidates[shrink_mask])
            contracted = np.clip(contracted, self.lb, self.ub)
            cont_fitness = func(contracted)
            
            shrink_idx = np.where(shrink_mask)[0]
            better_shrink = cont_fitness < fitness[top_idx[shrink_idx]]
            if np.any(better_shrink):
                candidates[shrink_idx[better_shrink]] = contracted[better_shrink]
                fitness[top_idx[shrink_idx[better_shrink]]] = cont_fitness[better_shrink]
            
            population[top_idx] = candidates
        
        return population, fitness
    
    def _adapt_step_size(self, n_improvements):
        rate = n_improvements / self.np
        if rate > 0.3:
            self.step_size *= 1.15
            self.F = min(1.0, self.F * 1.1)
        elif rate < 0.1:
            self.step_size *= 0.85
            self.F = max(0.3, self.F * 0.9)
            self.CR = min(0.95, self.CR * 1.05)
        else:
            self.CR = max(0.3, self.CR * 0.98)
        
        max_step = 0.5 * self.range
        self.step_size = np.clip(self.step_size, 0.01 * self.range, max_step)
    
    def _compute_diversity(self):
        centroid = self.population.mean(axis=0)
        variance = np.mean((self.population - centroid) ** 2)
        return np.sqrt(variance)
    
    def _restart_batch(self, population, fitness):
        r = np.random.default_rng()
        n_restart = self.np // 2
        
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        
        noise = r.uniform(-0.3, 0.3, (n_restart, self.dim)) * self.range
        new_pop = np.clip(best + noise, self.lb, self.ub)
        
        rand_pop = self.lb + r.uniform(0, 1, (n_restart, self.dim)) * self.range
        
        population[:n_restart] = new_pop
        population[n_restart:2*n_restart] = rand_pop
        
        new_fitness = fitness.copy()
        return population, new_fitness
    
    def __call__(self, func, stopping_condition):
        self.population = self._initialize_population()
        self.fitness = func(self.population)
        
        best_idx = np.argmin(self.fitness)
        self.f_opt = self.fitness[best_idx]
        self.x_opt = self.population[best_idx].copy()
        
        while not stopping_condition():
            best_vec = np.broadcast_to(self.x_opt, self.population.shape).copy()
            
            mutants = self._mutate_batch(self.population, best_vec)
            trials = self._crossover_batch(self.population, mutants)
            
            if stopping_condition():
                break
            
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
            
            if stopping_condition():
                break
            
            self.population, self.fitness = self._select_survivors_batch(
                self.population, self.fitness, trials, trial_fitness
            )
            
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.f_opt:
                self.f_opt = self.fitness[best_idx]
                self.x_opt = self.population[best_idx].copy()
                self.stagnation_gen = 0
            else:
                self.stagnation_gen += 1
            
            if not stopping_condition():
                self.population, self.fitness = self._local_search_batch(
                    self.population, self.fitness, func
                )
                
                best_idx = np.argmin(self.fitness)
                if self.fitness[best_idx] < self.f_opt:
                    self.f_opt = self.fitness[best_idx]
                    self.x_opt = self.population[best_idx].copy()
            
            n_improvements = np.sum(trial_fitness < self.fitness) if len(trial_fitness) == len(self.population) else 0
            self._adapt_step_size(n_improvements)
            
            if self.stagnation_gen > 15 or self._compute_diversity() < 0.5:
                self.population, self.fitness = self._restart_batch(self.population, self.fitness)
                self.fitness = func(self.population)
                self.stagnation_gen = 0
        
        return self.f_opt, self.x_opt
