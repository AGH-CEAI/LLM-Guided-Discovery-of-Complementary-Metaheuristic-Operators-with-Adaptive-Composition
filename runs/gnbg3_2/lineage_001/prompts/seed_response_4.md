```python
import numpy as np


class AdaptiveBeeDifferentialOptimizer:
    """
    Hybrid optimizer combining DE-style mutation with ABC-inspired selection
    and opposition-based population refresh. Uses success-history adaptation
    for crossover and scaling parameters.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(150, max(4 * dim, 8 * dim // 10 * 10))
        self.bounds = np.array([-100.0, 100.0])
        self._init_memory()
    
    def _init_memory(self):
        """Initialize success history memory for parameter adaptation."""
        self.memory_size = 6
        self.f_memory = np.ones(self.memory_size) * 0.5
        self.cr_memory = np.ones(self.memory_size) * 0.5
        self.memory_index = 0
        self.success_f = []
        self.success_cr = []
        self.generation_counter = 0
    
    def _initialize_population(self, rng):
        """Initialize population uniformly within bounds."""
        lo, hi = self.bounds
        pop = rng.uniform(lo, hi, size=(self.np, self.dim))
        return pop
    
    def _evaluate_batch(self, population, func):
        """Evaluate fitness for entire population in one batched call."""
        clipped = np.clip(population, self.bounds[0], self.bounds[1])
        fitness = func(clipped)
        if len(fitness) < len(population):
            actual_len = len(fitness)
            fitness = fitness[:actual_len]
            clipped = clipped[:actual_len]
        return clipped, fitness
    
    def _generate_indices(self, n_indices, exclude, rng):
        """Generate unique random indices excluding specific index."""
        indices = rng.choice(self.np, size=n_indices, replace=False)
        indices = indices[indices != exclude]
        while len(indices) < n_indices:
            new_idx = rng.integers(self.np)
            if new_idx != exclude and new_idx not in indices:
                indices = np.append(indices, new_idx)
        return indices[:n_indices]
    
    def _mutate_batch(self, population, fitness, rng):
        """
        Generate mutant vectors using DE/current-to-pbest style mutation
        with adaptive scaling factor per individual.
        """
        np_pop, dim = population.shape
        mutants = np.empty_like(population)
        
        sorted_idx = np.argsort(fitness)
        p_best_count = max(2, np_pop // 5)
        p_best_indices = sorted_idx[:p_best_count]
        
        f_values = rng.choice(self.f_memory, size=np_pop)
        f_values = np.clip(f_values + rng.standard_normal(np_pop) * 0.1, 0.1, 1.5)
        
        for i in range(np_pop):
            p_best = rng.choice(p_best_indices)
            r1, r2 = self._generate_indices(2, i, rng)
            
            diff_scale = f_values[i]
            mutants[i] = population[i] + diff_scale * (population[p_best] - population[i]) \
                         + diff_scale * (population[r1] - population[r2])
        
        mutants = np.clip(mutants, self.bounds[0], self.bounds[1])
        return mutants, f_values
    
    def _crossover_batch(self, population, mutants, cr_values, rng):
        """Binomial crossover with dimension-wise adaptive Cr."""
        np_pop, dim = population.shape
        offspring = np.empty_like(mutants)
        
        crossover_rates = rng.choice(self.cr_memory, size=np_pop)
        crossover_rates = np.clip(crossover_rates + rng.standard_normal(np_pop) * 0.1, 0.0, 1.0)
        
        for i in range(np_pop):
            j_rand = rng.integers(dim)
            mask = rng.random(dim) < crossover_rates[i]
            mask[j_rand] = True
            offspring[i] = np.where(mask, mutants[i], population[i])
        
        offspring = np.clip(offspring, self.bounds[0], self.bounds[1])
        return offspring, crossover_rates
    
    def _select_survivors_batch(self, population, fitness, offspring, offspring_fitness):
        """
        ABC-inspired greedy selection: keep better of target and trial.
        Track improvements for adaptation.
        """
        better_mask = offspring_fitness < fitness
        new_population = np.where(better_mask[:, np.newaxis], offspring, population)
        new_fitness = np.minimum(fitness, offspring_fitness)
        
        improvements = np.abs(fitness - offspring_fitness)
        improvement_mask = better_mask & (improvements > 1e-12)
        
        return new_population, new_fitness, improvement_mask, offspring_fitness
    
    def _adapt_parameters(self, improvement_mask, trial_fitness, f_values, cr_values):
        """
        Update success history memory for F and Cr based on successful improvements.
        """
        if not np.any(improvement_mask):
            return
        
        improved_indices = np.where(improvement_mask)[0]
        successful_f = f_values[improved_indices]
        successful_cr = cr_values[improved_indices]
        
        weights = np.ones(len(successful_f)) / len(successful_f)
        
        new_f = np.sum(weights * successful_f)
        new_cr = np.sum(weights * successful_cr)
        
        self.f_memory[self.memory_index] = np.clip(new_f, 0.1, 1.0)
        self.cr_memory[self.memory_index] = np.clip(new_cr, 0.0, 1.0)
        
        self.memory_index = (self.memory_index + 1) % self.memory_size
    
    def _compute_diversity(self, population):
        """Compute population diversity as average pairwise distance."""
        if len(population) < 2:
            return 0.0
        
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def _local_search_batch(self, population, fitness, func, rng):
        """
        Apply Nelder-Mead-like simplex refinement on top performers.
        Operates in batch on multiple simplexes simultaneously.
        """
        n_top = max(3, self.np // 30)
        top_indices = np.argsort(fitness)[:n_top]
        
        simplex_size = self.dim + 1
        step_size = 0.5
        
        for idx in top_indices:
            base = population[idx].copy()
            reflected = np.empty((simplex_size, self.dim))
            
            reflection_scale = 1.0
            for j in range(simplex_size):
                direction = rng.randn(self.dim)
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                reflected[j] = base + step_size * direction * rng.uniform(0.5, 1.5)
            
            reflected = np.clip(reflected, self.bounds[0], self.bounds[1])
            reflected_fitness = func(reflected)
            
            if len(reflected_fitness) < simplex_size:
                reflected_fitness = np.pad(reflected_fitness, 
                                          (0, simplex_size - len(reflected_fitness)),
                                          constant_values=np.inf)
            
            best_local_idx = np.argmin(reflected_fitness)
            if reflected_fitness[best_local_idx] < fitness[idx]:
                population[idx] = reflected[best_local_idx]
                fitness[idx] = reflected_fitness[best_local_idx]
        
        return population, fitness
    
    def _opposition_based_refresh(self, population, fitness, func, rng):
        """
        Generate opposition-based solutions when diversity is low.
        Keep best individuals, regenerate the rest using opposition learning.
        """
        n_keep = max(5, self.np // 5)
        keep_indices = np.argsort(fitness)[:n_keep]
        
        best_indices = keep_indices[:max(1, n_keep // 2)]
        best_pop = population[best_indices]
        best_fitness = fitness[best_indices]
        
        lo, hi = self.bounds
        center = (lo + hi) / 2
        range_val = (hi - lo) / 2
        
        opposite_pop = center - (population - center)
        opposite_pop = np.clip(opposite_pop, lo, hi)
        
        regenerate_mask = np.ones(len(population), dtype=bool)
        regenerate_mask[keep_indices] = False
        
        n_regenerate = np.sum(regenerate_mask)
        if n_regenerate > 0:
            new_pop = rng.uniform(lo, hi, size=(n_regenerate, self.dim))
            combined = np.vstack([opposite_pop[regenerate_mask], new_pop])
            
            combined_fitness = func(combined)
            if len(combined_fitness) < len(combined):
                combined_fitness = np.pad(combined_fitness,
                                         (0, len(combined) - len(combined_fitness)),
                                         constant_values=np.inf)
            
            split_idx = n_regenerate
            opp_fitness = combined_fitness[:split_idx]
            new_fitness = combined_fitness[split_idx:]
            
            use_opposite = opp_fitness < new_fitness
            final_new = np.where(use_opposite[:, np.newaxis],
                               opposite_pop[regenerate_mask],
                               new_pop)
            
            population[regenerate_mask] = final_new
            fitness[regenerate_mask] = np.where(use_opposite, opp_fitness, new_fitness)
        
        return population, fitness
    
    def _check_stagnation(self, fitness_history, current_fitness, window=10):
        """Check if population has stagnated."""
        if len(fitness_history) < window:
            return False
        
        recent_best = np.min(fitness_history[-window:])
        current_best = np.min(current_fitness)
        
        improvement_ratio = (fitness_history[-window] - current_best) / \
                           (fitness_history[-window] + 1e-10)
        
        return improvement_ratio < 1e-6
    
    def __call__(self, func, stopping_condition):
        """
        Main optimization loop orchestrating batched operations.
        Returns (f_opt, x_opt).
        """
        rng = np.random.default_rng()
        
        population = self._initialize_population(rng)
        population, fitness = self._evaluate_batch(population, func)
        
        if len(fitness) == 0:
            return np.inf, np.zeros(self.dim)
        
        best_idx = np.argmin(fitness)
        f_opt = float(fitness[best_idx])
        x_opt = population[best_idx].copy()
        
        fitness_history = [f_opt]
        stagnation_counter = 0
        local_search_interval = 50
        generation = 0
        
        while not stopping_condition():
            self.generation_counter += 1
            
            mutants, f_values = self._mutate_batch(population, fitness, rng)
            offspring, cr_values = self._crossover_batch(population, mutants, f_values, rng)
            
            if stopping_condition():
                break
            
            offspring_population, offspring_fitness = self._evaluate_batch(offspring, func)
            
            if len(offspring_fitness) < len(offspring):
                actual_len = len(offspring_fitness)
                offspring_population = offspring_population[:actual_len]
                offspring_fitness = offspring_fitness[:actual_len]
            
            if len(offspring_fitness) == 0:
                break
            
            population, fitness, improvement_mask, trial_fit = \
                self._select_survivors_batch(population, fitness, 
                                            offspring_population, offspring_fitness)
            
            self._adapt_parameters(improvement_mask, trial_fit, f_values, cr_values)
            
            current_best_idx = np.argmin(fitness)
            current_best_fitness = float(fitness[current_best_idx])
            
            fitness_history.append(current_best_fitness)
            if len(fitness_history) > 50:
                fitness_history.pop(0)
            
            if current_best_fitness < f_opt:
                f_opt = current_best_fitness
                x_opt = population[current_best_idx].copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            if stagnation_counter > 0 and stagnation_counter % 15 == 0:
                diversity = self._compute_diversity(population)
                if diversity < (self.bounds[1] - self.bounds[0]) * 0.05:
                    population, fitness = self._opposition_based_refresh(
                        population, fitness, func, rng)
                    stagnation_counter = 0
            
            if generation > 0 and generation % local_search_interval == 0:
                population, fitness = self._local_search_batch(
                    population, fitness, func, rng)
                
                if fitness[current_best_idx] < current_best_fitness:
                    if fitness[current_best_idx] < f_opt:
                        f_opt = float(fitness[current_best_idx])
                        x_opt = population[current_best_idx].copy()
            
            generation += 1
            
            if generation > 5000:
                break
        
        return f_opt, x_opt
```