import numpy as np


class CulturalDifferentialEvolution:
    """
    Differential Evolution with Cultural Memory, Opposition-Based Learning,
    Archive-Driven Diversity, and Adaptive Local Search.
    
    Novel components:
    - Cultural memory guides mutation direction (accumulated successful directions)
    - Opposition-based population initialization for better coverage
    - Dynamic archive of superior solutions used in mutation
    - Adaptive local search on pbest individuals
    - Velocity-like F adaptation based on success history
    - Entropy-based population diversity monitoring
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        self.np = min(max(4 * dim, 60), 10 * dim)
        self.p_best_ratio = 0.15
        self.culture_decay = 0.95
        self.archive_size_factor = 2.0
        self.max_archive_size = int(self.np * self.archive_size_factor)
        self.local_search_trigger_prob = 0.08
        self.local_search_radius_init = 2.0
        self.local_search_radius_decay = 0.95
        self.diversity_threshold = 1e-6
        self.stagnation_counter = 0
        self.max_stagnation = 30
        self.f_base = 0.6
        self.cr_base = 0.85
        
        self.population = None
        self.fitness = None
        self.culture_memory = None
        self.archive = None
        self.archive_fitness = None
        self.f_current = None
        self.cr_current = None
        self.success_f_history = []
        self.success_cr_history = []
        self.best_fitness_history = []
        self.generation = 0
        self.local_search_radius = self.local_search_radius_init
        
    def _initialize_population(self, func):
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.np, self.dim)
        )
        self.population = self._clip_to_bounds_batch(self.population)
        self.fitness = func(self.population)
        if len(self.fitness) < self.np:
            self.fitness = np.full(self.np, np.inf)
        self._initialize_culture_memory()
        self._initialize_archive()
        self.f_current = np.full(self.np, self.f_base)
        self.cr_current = np.full(self.np, self.cr_base)
        
    def _initialize_culture_memory(self):
        self.culture_memory = np.zeros((self.dim, self.dim))
        for j in range(self.dim):
            direction = np.zeros(self.dim)
            direction[j] = 1.0
            self.culture_memory[:, j] = direction
            
    def _initialize_archive(self):
        self.archive = np.zeros((0, self.dim))
        self.archive_fitness = np.array([], dtype=np.float64)
        
    def _opposition_based_initialization(self, func):
        opp_pop = 2.0 * (self.population.mean(axis=0, keepdims=True)) - self.population
        opp_pop = self._clip_to_bounds_batch(opp_pop)
        opp_fitness = func(opp_pop)
        if len(opp_fitness) < self.np:
            opp_fitness = np.full(self.np, np.inf)
        better_mask = opp_fitness < self.fitness
        for i in range(self.np):
            if better_mask[i]:
                self.population[i] = opp_pop[i]
                self.fitness[i] = opp_fitness[i]
                
    def _clip_to_bounds_batch(self, pop):
        return np.clip(pop, self.lower_bound, self.upper_bound)
    
    def _compute_pbest_indices(self):
        n_best = max(1, int(self.np * self.p_best_ratio))
        sorted_indices = np.argsort(self.fitness)
        return sorted_indices[:n_best]
    
    def _update_culture_memory_batch(self, successful_directions, successful_factors):
        if len(successful_directions) == 0:
            return
        avg_direction = successful_directions.mean(axis=0)
        norm = np.linalg.norm(avg_direction)
        if norm > 1e-10:
            normalized = avg_direction / norm
            self.culture_memory = (self.culture_decay * self.culture_memory + 
                                   (1 - self.culture_decay) * normalized[:, np.newaxis] * normalized)
            
    def _mutate_batch(self, population, fitness):
        pbest_indices = self._compute_pbest_indices()
        pbest = population[pbest_indices]
        trials = np.empty((self.np, self.dim), dtype=np.float64)
        
        for i in range(self.np):
            idxs = [k for k in range(self.np) if k != i]
            np.random.shuffle(idxs)
            r1, r2, r3, r4 = idxs[:4]
            
            pbest_idx = np.random.randint(len(pbest_indices))
            
            archive_contrib = np.zeros(self.dim)
            if len(self.archive) >= 3:
                arch_indices = np.random.choice(len(self.archive), 3, replace=False)
                archive_contrib = (self.archive[arch_indices[0]] - 
                                   self.archive[arch_indices[1]])
            
            culture_contrib = np.zeros(self.dim)
            if self.generation > 5:
                culture_directions = self.culture_memory.sum(axis=1)
                culture_norm = np.linalg.norm(culture_directions)
                if culture_norm > 1e-10:
                    culture_contrib = culture_directions / culture_norm
                    culture_contrib *= np.random.uniform(0.1, 0.3)
            
            f_i = self.f_current[i]
            base_mutation = (pbest[pbest_idx] - population[i])
            rand_mutation = f_i * (population[r1] - population[r2])
            archive_mutation = f_i * archive_contrib
            
            trials[i] = (population[i] + base_mutation * 0.7 + 
                        rand_mutation * 0.5 + archive_mutation * 0.3 +
                        culture_contrib * f_i)
            
        return trials
    
    def _crossover_batch(self, population, trials):
        cross_pop = np.empty((self.np, self.dim), dtype=np.float64)
        for i in range(self.np):
            cr_i = self.cr_current[i]
            j_rand = np.random.randint(self.dim)
            mask = np.random.random(self.dim) < cr_i
            mask[j_rand] = True
            cross_pop[i] = np.where(mask, trials[i], population[i])
        return cross_pop
    
    def _clip_to_bounds_batch(self, pop):
        return np.clip(pop, self.lower_bound, self.upper_bound)
    
    def _eval_wrapper(self, candidates, func):
        fitness = func(candidates)
        if not isinstance(fitness, np.ndarray):
            fitness = np.array(fitness)
        if len(fitness) < len(candidates):
            valid_len = len(fitness)
            fitness = np.pad(fitness, (0, len(candidates) - valid_len), 
                           constant_values=np.inf)
        return fitness
    
    def _select_survivors_batch(self, population, fitness, trial_pop, trial_fitness):
        improved_mask = trial_fitness < fitness
        new_population = population.copy()
        new_fitness = fitness.copy()
        successful_directions = []
        successful_factors = []
        
        for i in range(self.np):
            if improved_mask[i]:
                new_population[i] = trial_pop[i]
                new_fitness[i] = trial_fitness[i]
                dir_vec = trial_pop[i] - population[i]
                successful_directions.append(dir_vec)
                if fitness[i] > 1e-10:
                    successful_factors.append(trial_fitness[i] / fitness[i])
                else:
                    successful_factors.append(1.0)
                
        if len(successful_directions) > 0:
            self._update_culture_memory_batch(
                np.array(successful_directions), 
                np.array(successful_factors)
            )
            self._adapt_parameters_batch(successful_directions, successful_factors)
            self._update_archive_batch(new_population, new_fitness)
            
        return new_population, new_fitness, improved_mask.sum()
    
    def _adapt_parameters_batch(self, directions, factors):
        if len(factors) == 0:
            return
        success_rate = len(factors) / self.np
        self.success_f_history.append(np.mean(factors))
        self.success_cr_history.append(success_rate)
        
        if len(self.success_f_history) > 5:
            self.success_f_history.pop(0)
            self.success_cr_history.pop(0)
            
        if len(self.success_f_history) >= 5:
            avg_success_f = np.mean(self.success_f_history)
            if avg_success_f < 1.0:
                for i in range(self.np):
                    self.f_current[i] = np.clip(
                        self.f_current[i] * 0.9, 0.3, 1.5
                    )
            else:
                for i in range(self.np):
                    self.f_current[i] = np.clip(
                        self.f_current[i] * 1.05, 0.3, 1.5
                    )
                    
            avg_success_cr = np.mean(self.success_cr_history)
            if avg_success_cr > 0.3:
                for i in range(self.np):
                    self.cr_current[i] = np.clip(
                        self.cr_current[i] * 1.02, 0.3, 1.0
                    )
            else:
                for i in range(self.np):
                    self.cr_current[i] = np.clip(
                        self.cr_current[i] * 0.95, 0.3, 1.0
                    )
                    
    def _update_archive_batch(self, population, fitness):
        sorted_indices = np.argsort(fitness)
        n_archive = min(len(population), self.max_archive_size)
        
        for idx in sorted_indices[:n_archive]:
            if len(self.archive) < self.max_archive_size:
                self.archive = np.vstack([self.archive, population[idx]])
                self.archive_fitness = np.append(self.archive_fitness, fitness[idx])
            else:
                worst_arch_idx = np.argmax(self.archive_fitness)
                if fitness[idx] < self.archive_fitness[worst_arch_idx]:
                    self.archive[worst_arch_idx] = population[idx]
                    self.archive_fitness[worst_arch_idx] = fitness[idx]
                    
    def _apply_adaptive_local_search(self, population, fitness, func):
        best_idx = np.argmin(fitness)
        best_pos = population[best_idx].copy()
        best_fit = fitness[best_idx]
        
        if np.random.random() > self.local_search_trigger_prob:
            return population, fitness
            
        search_radius = self.local_search_radius
        for _ in range(3):
            perturbation = np.random.normal(0, search_radius, self.dim)
            candidate = self._clip_to_bounds_batch(best_pos + perturbation)
            candidate_fit = func(candidate.reshape(1, -1))
            if len(candidate_fit) > 0 and not np.isnan(candidate_fit[0]):
                if candidate_fit[0] < best_fit:
                    best_pos = candidate
                    best_fit = candidate_fit[0]
                    search_radius *= 1.5
                else:
                    search_radius *= self.local_search_radius_decay
                    
        population[best_idx] = best_pos
        fitness[best_idx] = best_fit
        self.local_search_radius = max(0.01, self.local_search_radius * 0.98)
        
        return population, fitness
    
    def _compute_diversity(self, population):
        centroid = population.mean(axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return distances.mean()
    
    def _restart_if_stagnant(self, population, fitness, func):
        current_best = np.min(fitness)
        self.best_fitness_history.append(current_best)
        
        if len(self.best_fitness_history) > self.max_stagnation:
            self.best_fitness_history.pop(0)
            
        if len(self.best_fitness_history) >= self.max_stagnation:
            improvement = (self.best_fitness_history[0] - 
                          self.best_fitness_history[-1])
            if abs(improvement) < self.diversity_threshold:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                
        if self.stagnation_counter > 0:
            n_replace = min(self.np // 3, 20)
            worst_indices = np.argsort(fitness)[-n_replace:]
            for idx in worst_indices:
                population[idx] = np.random.uniform(
                    self.lower_bound, self.upper_bound, self.dim
                )
            population = self._clip_to_bounds_batch(population)
            fitness = func(population)
            if len(fitness) < self.np:
                fitness = np.full(self.np, np.inf)
            self.stagnation_counter = 0
            self.f_current = np.full(self.np, self.f_base)
            self.cr_current = np.full(self.np, self.cr_base)
            
        return population, fitness
    
    def _compute_reward(self, old_fitness, new_fitness):
        improvements = old_fitness - new_fitness
        positive_improvements = improvements[improvements > 0]
        if len(positive_improvements) > 0:
            return positive_improvements.mean()
        return 0.0
    
    def __call__(self, func, stopping_condition):
        self._initialize_population(func)
        self._opposition_based_initialization(func)
        
        best_idx = np.argmin(self.fitness)
        f_opt = self.fitness[best_idx]
        x_opt = self.population[best_idx].copy()
        
        while not stopping_condition():
            old_fitness = self.fitness.copy()
            
            trials = self._mutate_batch(self.population, self.fitness)
            trials = self._crossover_batch(self.population, trials)
            trials = self._clip_to_bounds_batch(trials)
            
            trial_fitness = self._eval_wrapper(trials, func)
            if len(trial_fitness) < len(trials):
                break
                
            if stopping_condition():
                break
                
            (self.population, self.fitness, n_improved) = (
                self._select_survivors_batch(
                    self.population, self.fitness, trials, trial_fitness
                )
            )
            
            reward = self._compute_reward(old_fitness, self.fitness)
            if reward > 0:
                self._apply_adaptive_local_search(
                    self.population, self.fitness, func
                )
            
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < f_opt:
                f_opt = self.fitness[current_best_idx]
                x_opt = self.population[current_best_idx].copy()
                
            if self.generation % 10 == 0:
                self.population, self.fitness = self._restart_if_stagnant(
                    self.population, self.fitness, func
                )
                
            self.generation += 1
            
        return f_opt, x_opt
