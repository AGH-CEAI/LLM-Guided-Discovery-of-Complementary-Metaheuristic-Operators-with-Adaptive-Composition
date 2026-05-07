```python
import numpy as np


class FitnessRankAdaptiveDEWithTemporalDrift:
    """
    Hybrid optimizer combining:
    - DE/current-to-best/1/bin mutation with adaptive F
    - Fitness-rank-based position drift (temporal memory)
    - Adaptive inertia weight for position updates
    - Opposition-based restart on stagnation
    - Nelder-Mead local search on top performers
    
    Key methods (batch operations on population):
    - _initialize_population: setup with random candidates
    - _mutate_batch: DE mutation with adaptive scaling
    - _crossover_batch: binomial recombination
    - _select_survivors_batch: greedy selection
    - _adapt_parameters_batch: update F, CR, inertia based on improvement history
    - _position_update_temporal_drift: fitness-rank weighted movement
    - _apply_local_search_batch: Nelder-Mead on top-k
    - _restart_opposition_batch: opposition-based refresh
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(8 * dim, 400)
        self.bounds = np.array([[-100.0, 100.0]] * dim)
        self.max_iter = kwargs.get('max_iter', 1000)
        
        self.f = 0.5
        self.cr = 0.7
        self.inertia = 0.4
        self.f_archive = []
        self.cr_archive = []
        
        self.population = None
        self.fitness = None
        self.best_fitness = np.inf
        self.best_position = None
        
        self.prev_positions = None
        self.position_drift = None
        self.improvement_history = []
        self.local_search_freq = 10
        
    def __call__(self, func, stopping_condition):
        self._initialize_population(func)
        
        iteration = 0
        while iteration < self.max_iter and not stopping_condition():
            self._adapt_parameters_batch()
            mutants = self._mutate_batch(self.population, self.fitness)
            trials = self._crossover_batch(self.population, mutants)
            trials = self._clip_to_bounds(trial_array=trials)
            
            trial_fitness = func(trials)
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
                if len(trials) == 0:
                    break
            
            if stopping_condition():
                break
                
            self.population, self.fitness = self._select_survivors_batch(
                self.population, self.fitness, trials, trial_fitness
            )
            
            n_improved = np.sum(trial_fitness < self.fitness)
            self.improvement_history.append(n_improved)
            self.improvement_history = self.improvement_history[-10:]
            
            if np.mean(self.improvement_history[-5:]) < 0.05 * self.np and iteration > 20:
                self._restart_opposition_batch(func)
            
            if iteration % self.local_search_freq == 0:
                self._apply_local_search_batch(func)
            
            self._update_best_batch(self.population, self.fitness)
            
            if iteration % 50 == 0:
                print(f"Iter {iteration}: best_fitness={self.best_fitness:.6e}")
            
            iteration += 1
        
        return self.best_fitness, self.best_position
    
    def _initialize_population(self, func):
        """Initialize population with uniform random within bounds."""
        self.population = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], size=(self.np, self.dim)
        )
        self.fitness = func(self.population)
        self.prev_positions = self.population.copy()
        self.position_drift = np.zeros((self.np, self.dim))
        self._update_best_batch(self.population, self.fitness)
    
    def _mutate_batch(self, population, fitness):
        """DE/current-to-best/1/bin mutation with adaptive F."""
        mutants = np.empty_like(population)
        best_idx = np.argmin(fitness)
        best_vec = population[best_idx]
        
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], size=3, replace=False
            )
            r1, r2, r3 = indices
            
            f_adapted = self.f * (0.9 + 0.2 * np.random.random())
            mutants[i] = population[r1] + f_adapted * (best_vec - population[r2]) + f_adapted * (population[r2] - population[r3])
        
        return mutants
    
    def _crossover_batch(self, population, mutants):
        """Binomial crossover with adaptive CR per individual."""
        trials = np.empty_like(population)
        cr_per_individual = np.random.uniform(0.3, 0.9, size=self.np)
        
        for i in range(self.np):
            j_rand = np.random.randint(self.dim)
            mask = np.random.random(self.dim) < cr_per_individual[i]
            mask[j_rand] = True
            trials[i] = np.where(mask, mutants[i], population[i])
        
        return trials
    
    def _clip_to_bounds(self, trial_array):
        """Clip all candidates to search bounds."""
        return np.clip(trial_array, self.bounds[:, 0], self.bounds[:, 1])
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Greedy selection: keep better of target and trial."""
        better = trial_fitness < fitness
        population[better] = trials[better]
        fitness[better] = trial_fitness[better]
        return population, fitness
    
    def _adapt_parameters_batch(self):
        """Adapt F, CR, and inertia based on improvement rate."""
        if len(self.improvement_history) >= 5:
            improvement_rate = np.mean(self.improvement_history[-5:]) / self.np
            
            self.f = np.clip(self.f + 0.1 * (improvement_rate - 0.2), 0.1, 1.0)
            self.cr = np.clip(self.cr + 0.05 * (improvement_rate - 0.3), 0.1, 0.9)
            self.inertia = np.clip(self.inertia * (1.0 + 0.05 * (improvement_rate - 0.25)), 0.1, 0.9)
            
            self.f_archive.append(self.f)
            self.cr_archive.append(self.cr)
            if len(self.f_archive) > 20:
                self.f_archive = self.f_archive[-20:]
                self.cr_archive = self.cr_archive[-20:]
    
    def _position_update_temporal_drift(self, population, fitness):
        """Fitness-rank weighted drift toward better individuals."""
        ranks = np.argsort(np.argsort(fitness))
        rank_weights = 1.0 / (ranks + 1)
        rank_weights /= rank_weights.sum()
        
        centroid = np.sum(population * rank_weights[:, np.newaxis], axis=0)
        self.position_drift = (self.inertia * self.position_drift + 
                               0.3 * (centroid - population))
        
        new_positions = population + self.position_drift + 0.1 * np.random.randn(self.np, self.dim)
        return self._clip_to_bounds(trial_array=new_positions)
    
    def _apply_local_search_batch(self, func):
        """Apply Nelder-Mead refinement to top performers."""
        k = max(1, self.np // 20)
        top_indices = np.argsort(self.fitness)[:k]
        
        for idx in top_indices:
            if np.random.random() > 0.5:
                continue
            x = self.population[idx].copy()
            try:
                from scipy.optimize import minimize
                result = minimize(
                    lambda x_: func(x_.reshape(1, -1))[0],
                    x, method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-3}
                )
                if result.fun < self.fitness[idx]:
                    self.population[idx] = self._clip_to_bounds(trial_array=result.x.reshape(1, -1))[0]
                    self.fitness[idx] = result.fun
            except Exception:
                pass
    
    def _restart_opposition_batch(self, func):
        """Opposition-based restart to restore diversity."""
        center = 0.5 * (self.bounds[:, 0] + self.bounds[:, 1])
        opposition = 2 * center - self.population
        opposition = self._clip_to_bounds(trial_array=opposition)
        
        opp_fitness = func(opposition)
        if len(opp_fitness) < len(opposition):
            opposition = opposition[:len(opp_fitness)]
            opp_fitness = opp_fitness[:len(opp_fitness)]
        
        combined_pop = np.vstack([self.population, opposition])
        combined_fit = np.concatenate([self.fitness, opp_fitness])
        
        top_indices = np.argsort(combined_fit)[:self.np]
        self.population = combined_pop[top_indices]
        self.fitness = combined_fit[top_indices]
        
        self.position_drift = np.zeros((self.np, self.dim))
        self.inertia = min(0.9, self.inertia * 1.5)
        print(f"  [Restart triggered, diversity restored]")
    
    def _update_best_batch(self, population, fitness):
        """Track global best solution."""
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.best_fitness:
            self.best_fitness = fitness[min_idx]
            self.best_position = population[min_idx].copy()
```