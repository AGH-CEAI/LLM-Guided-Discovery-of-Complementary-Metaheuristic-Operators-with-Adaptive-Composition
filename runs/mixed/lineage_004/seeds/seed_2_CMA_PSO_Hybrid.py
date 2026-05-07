import numpy as np

class CMA_PSO_Hybrid:
    """Hybrid optimizer combining CMA-ES covariance adaptation with PSO velocity updates."""
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = kwargs.get('NP', min(8 * dim, 400))
        self.F = 0.5
        self.CR = 0.7
        self.bounds = np.array([[-100.0, 100.0]])
        self._init_params()
        self.func = None
        
    def _init_params(self):
        self.mean = np.zeros(self.dim)
        self.cov = np.eye(self.dim) * (self.step_size ** 2) if hasattr(self, 'step_size') else np.eye(self.dim) * 400.0
        self.step_size = 10.0 * np.ones(self.dim).mean() if not hasattr(self, 'step_size') else self.step_size
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.vel = np.zeros((self.NP, self.dim))
        self.personal_best = None
        self.global_best = None
        self.iteration = 0
        self.fitness_best = np.inf
        self.x_best = None
        self.success_history = []
        self.mutation_memory = np.ones(5)
        self.stagnation_count = 0
        
    def __call__(self, func, stopping_condition):
        self.func = func
        self._init_params()
        self.iteration = 0
        self.fitness_best = np.inf
        self.stagnation_count = 0
        
        pop = self._initialize_population()
        pop, fitness = self._eval_wrapper(pop)
        self._update_best(fitness, pop)
        self._init_personal_best(pop, fitness)
        
        while not stopping_condition():
            self.iteration += 1
            
            trials, mut_strategies = self._mutate_batch(pop)
            trials = self._crossover_batch(pop, trials, mut_strategies)
            trials = self._clip_to_bounds_batch(trials)
            
            if stopping_condition():
                break
            
            trial_fitness = func(trials)
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
            
            if np.any(np.isnan(trial_fitness)):
                valid_mask = ~np.isnan(trial_fitness)
                trials = trials[valid_mask]
                trial_fitness = trial_fitness[valid_mask]
                if len(trial_fitness) == 0:
                    break
            
            pop, fitness, success_count = self._select_survivors_batch(pop, fitness, trials)
            
            if stopping_condition():
                break
            
            self._update_best(fitness, pop)
            self._update_personal_best(pop, fitness)
            self._compute_reward(success_count, mut_strategies)
            
            success_rate = success_count / self.NP
            self.success_history.append(success_rate)
            if len(self.success_history) > 50:
                self.success_history.pop(0)
            
            self._adapt_step_size(success_rate)
            self._adapt_parameters_batch(success_rate)
            
            self._update_covariance_batch(pop, trials, success_count)
            self._update_velocity_batch(pop)
            
            pop, fitness = self._apply_adaptive_local_search(pop, fitness)
            
            if self._check_stagnation():
                self._restart_if_stagnant(pop, fitness)
        
        return self.fitness_best, self.x_best.copy()
    
    def _initialize_population(self):
        pop = np.random.uniform(-50, 50, size=(self.NP, self.dim))
        return pop
    
    def _clip_to_bounds_batch(self, batch):
        return np.clip(batch, self.bounds[0, 0], self.bounds[0, 1])
    
    def _eval_wrapper(self, pop):
        pop = self._clip_to_bounds_batch(pop)
        fitness = self.func(pop)
        if len(fitness) < len(pop):
            pop = pop[:len(fitness)]
            fitness = fitness[:len(fitness)]
        if np.any(np.isnan(fitness)):
            valid_mask = ~np.isnan(fitness)
            pop = pop[valid_mask]
            fitness = fitness[valid_mask]
        return pop, fitness
    
    def _init_personal_best(self, pop, fitness):
        self.personal_best = pop.copy()
        self.personal_best_fitness = fitness.copy()
        best_idx = np.argmin(fitness)
        self.global_best = pop[best_idx].copy()
        
    def _update_personal_best(self, pop, fitness):
        improved = fitness < self.personal_best_fitness
        self.personal_best[improved] = pop[improved]
        self.personal_best_fitness[improved] = fitness[improved]
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < self._compute_global_best_fitness():
            self.global_best = pop[best_idx].copy()
    
    def _compute_global_best_fitness(self):
        return self.personal_best_fitness.min() if len(self.personal_best_fitness) > 0 else np.inf
    
    def _mutate_batch(self, pop):
        mut_strategies = self._select_mutation_strategy(len(pop))
        trials = np.empty_like(pop)
        
        for i in range(len(pop)):
            idx = mut_strategies[i]
            if idx == 0:
                trials[i] = self._mutate_current_to_pbest(pop[i])
            elif idx == 1:
                trials[i] = self._mutate_weighted_top(pop[i])
            elif idx == 2:
                trials[i] = self._mutate_current_to_best_rand(pop[i])
            elif idx == 3:
                trials[i] = self._mutate_rand(pop[i])
            else:
                trials[i] = self._mutate_best_with_jitter(pop[i])
        
        return trials, mut_strategies
    
    def _select_mutation_strategy(self, n):
        probs = self.mutation_memory / self.mutation_memory.sum()
        strategies = np.random.choice(len(probs), size=n, p=probs)
        return strategies
    
    def _mutate_current_to_pbest(self, x):
        p_best = self.global_best
        diff = p_best - x
        jitter = 0.1 * np.random.randn(self.dim)
        return x + self.F * (diff + jitter)
    
    def _mutate_weighted_top(self, x):
        top_k = min(5, self.NP)
        top_indices = np.argsort(self.personal_best_fitness)[:top_k]
        weights = 1.0 / (self.personal_best_fitness[top_indices] + 1e-10)
        weights /= weights.sum()
        weighted_sum = np.sum(self.personal_best[top_indices] * weights[:, np.newaxis], axis=0)
        diff = weighted_sum - x
        return x + self.F * diff
    
    def _mutate_current_to_best_rand(self, x):
        indices = np.random.choice(self.NP, 3, replace=False)
        r1, r2, r3 = pop[indices[0]], pop[indices[1]], pop[indices[2]]
        diff = self.global_best - x + r1 - r2
        return x + self.F * diff
    
    def _mutate_rand(self, x):
        indices = np.random.choice(self.NP, 3, replace=False)
        return self.personal_best[indices[0]] + self.F * (self.personal_best[indices[1]] - self.personal_best[indices[2]])
    
    def _mutate_best_with_jitter(self, x):
        diff = self.global_best - x
        jitter = 0.2 * np.random.randn(self.dim)
        return x + self.F * (diff + jitter)
    
    def _crossover_batch(self, pop, trials, mut_strategies):
        crossover_trials = np.empty_like(pop)
        for i in range(len(pop)):
            j_rand = np.random.randint(self.dim)
            mask = np.random.rand(self.dim) < self.CR
            mask[j_rand] = True
            crossover_trials[i] = np.where(mask, trials[i], pop[i])
        return crossover_trials
    
    def _select_survivors_batch(self, pop, fitness, trials):
        trial_fitness = self.func(trials) if len(trials) > 0 else np.array([])
        if len(trial_fitness) < len(trials):
            trials = trials[:len(trial_fitness)]
            trial_fitness = trial_fitness[:len(trial_fitness)]
        
        success_mask = trial_fitness < fitness
        success_count = np.sum(success_mask)
        
        new_pop = pop.copy()
        new_fitness = fitness.copy()
        new_pop[success_mask] = trials[success_mask]
        new_fitness[success_mask] = trial_fitness[success_mask]
        
        return new_pop, new_fitness, success_count
    
    def _update_best(self, fitness, pop):
        if len(fitness) == 0:
            return
        valid_mask = ~np.isnan(fitness)
        if not np.any(valid_mask):
            return
        
        valid_fitness = fitness[valid_mask]
        valid_pop = pop[valid_mask]
        idx = np.argmin(valid_fitness)
        
        if valid_fitness[idx] < self.fitness_best:
            self.fitness_best = valid_fitness[idx]
            self.x_best = valid_pop[idx].copy()
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
    
    def _compute_reward(self, success_count, mut_strategies):
        for s in range(len(self.mutation_memory)):
            count = np.sum(mut_strategies == s)
            if count > 0:
                self.mutation_memory[s] = 0.9 * self.mutation_memory[s] + 0.1 * (success_count / count) * self.NP
    
    def _adapt_step_size(self, success_rate):
        cs = 1.0 / (self.dim + 5.0)
        target_rate = 0.5
        self.step_size *= np.exp(cs * (success_rate - target_rate))
        self.step_size = np.clip(self.step_size, 1e-10, 100.0)
    
    def _adapt_parameters_batch(self, success_rate):
        if success_rate > 0.3:
            self.F = min(1.0, self.F * 1.05)
        else:
            self.F = max(0.1, self.F * 0.95)
        
        if success_rate > 0.4:
            self.CR = min(0.95, self.CR * 1.02)
        else:
            self.CR = max(0.1, self.CR * 0.98)
    
    def _update_covariance_batch(self, pop, trials, success_count):
        if len(pop) < 2:
            return
        
        cc = 4.0 / (self.dim + 4.0)
        ccov = 1.0 / (self.dim ** 2 + 6.0)
        chi_n = self.dim ** 0.5 * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim ** 2))
        
        mean_old = self.mean.copy()
        self.mean = np.mean(pop, axis=0)
        
        y = self.mean - mean_old
        self.ps = (1.0 - cc) * self.ps + np.sqrt(cc * (2.0 - cc)) * y / self.step_size
        
        if np.linalg.norm(self.ps) < chi_n * 1.5:
            self.pc = (1.0 - cc) * self.pc + np.sqrt(cc * (2.0 - cc)) * y / self.step_size
        else:
            self.pc *= 0.0
        
        rank_one = np.outer(self.pc, self.pc)
        
        indices = np.argsort(self.personal_best_fitness)[:min(self.NP, len(pop))]
        weighted_pop = self.personal_best[indices]
        weights = 1.0 / (np.arange(len(indices)) + 1)
        weights /= weights.sum()
        y_w = weighted_pop - mean_old
        rank_mu = np.sum([w * np.outer(y_i, y_i) for w, y_i in zip(weights, y_w)], axis=0)
        
        self.cov = (1.0 - ccov) * self.cov + ccov * (rank_one + (1.0 + (1.0 - cc) ** 2) * cc / ccov * rank_mu)
        
        self.cov = 0.5 * (self.cov + self.cov.T)
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        self.cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    def _update_velocity_batch(self, pop):
        w = 0.7 - 0.3 * min(1.0, self.iteration / 500)
        c1 = 1.0
        c2 = 1.5
        
        r1 = np.random.rand(self.NP, self.dim)
        r2 = np.random.rand(self.NP, self.dim)
        
        cognitive = c1 * r1 * (self.personal_best - pop)
        social = c2 * r2 * (self.global_best - pop)
        
        step_scale = self.step_size / (np.sqrt(np.diag(self.cov)) + 1e-10)
        self.vel = w * self.vel + cognitive + social
        self.vel *= step_scale
    
    def _apply_adaptive_local_search(self, pop, fitness):
        if len(self.success_history) < 10:
            return pop, fitness
        
        recent_success = np.mean(self.success_history[-10:])
        if recent_success < 0.05:
            return pop, fitness
        
        top_count = max(1, self.NP // 10)
        top_indices = np.argsort(fitness)[:top_count]
        
        improved_pop = pop.copy()
        improved_fit = fitness.copy()
        
        for idx in top_indices:
            if np.random.rand() < 0.2:
                x = pop[idx]
                step = 0.05 * self.step_size * np.random.randn(self.dim)
                candidates = x + step * np.random.randn(10, self.dim)
                candidates = self._clip_to_bounds_batch(candidates)
                cand_fit = self.func(candidates)
                
                if len(cand_fit) < 10:
                    candidates = candidates[:len(cand_fit)]
                    cand_fit = cand_fit[:len(cand_fit)]
                
                if len(cand_fit) > 0 and not np.all(np.isnan(cand_fit)):
                    cand_fit = np.where(np.isnan(cand_fit), np.inf, cand_fit)
                    best_local = np.argmin(cand_fit)
                    if cand_fit[best_local] < fitness[idx]:
                        improved_pop[idx] = candidates[best_local]
                        improved_fit[idx] = cand_fit[best_local]
        
        return improved_pop, improved_fit
    
    def _check_stagnation(self):
        if len(self.success_history) < 50:
            return False
        recent = np.mean(self.success_history[-20:])
        return recent < 0.01 and self.stagnation_count > 100
    
    def _restart_if_stagnant(self, pop, fitness):
        self.stagnation_count = 0
        self.success_history = []
        self.mutation_memory = np.ones(5) * 0.5
        
        best_idx = np.argmin(fitness)
        self.mean = pop[best_idx].copy()
        self.cov = np.eye(self.dim) * (self.step_size ** 2)
        self.step_size = min(50.0, self.step_size * 2.0)
        self.vel = np.zeros((self.NP, self.dim))
        
        restart_pop = np.random.randn(self.NP, self.dim) * self.step_size + self.mean
        restart_pop = self._clip_to_bounds_batch(restart_pop)
        restart_fit = self.func(restart_pop)
        
        if len(restart_fit) < self.NP:
            restart_pop = restart_pop[:len(restart_fit)]
            restart_fit = restart_fit[:len(restart_fit)]
        
        for i in range(min(len(restart_pop), self.NP - len(pop))):
            pop = np.vstack([pop, restart_pop[i:i+1]])
            fitness = np.append(fitness, restart_fit[i:i+1])
        
        self._init_personal_best(pop, fitness)
