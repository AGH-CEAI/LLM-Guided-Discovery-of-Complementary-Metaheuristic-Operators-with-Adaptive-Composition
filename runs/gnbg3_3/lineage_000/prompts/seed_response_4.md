```python
import numpy as np


class ReflectionGuidedMemeticDE:
    """
    A hybrid metaheuristic combining:
    1. Reflection-guided mutation (candidates reflected about current best)
    2. Success-history adaptive archive (learns from successful mutations)
    3. Nelder-Mead simplex refinement (applied to best individuals)
    4. Opposition-based restart (diversity restoration)
    
    Differs from prior proposals:
    - No spectral/FFT operations (unlike MirrorSpectralDE)
    - No covariance matrix (unlike CovarianceGuidedEvolutionStrategy)
    - No velocity/particle swarm topology (unlike AdaptiveTopologyParticleSwarm)
    - Uses reflection geometry + archive-guided perturbation + direct local search
    """
    
    def __init__(self, dim, NP=None, F_init=0.6, CR_init=0.7, 
                 archive_size_factor=2.0, simplex_refresh_rate=5,
                 diversity_threshold=1e-4, max_stagnation=15):
        self.dim = dim
        self.NP = NP if NP is not None else max(4 * dim, 20)
        self.archive_size = int(self.NP * archive_size_factor)
        self.simplex_refresh_rate = simplex_refresh_rate
        self.diversity_threshold = diversity_threshold
        self.max_stagnation = max_stagnation
        
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        self.F = F_init
        self.CR = CR_init
        self.archive = None
        self.stagnation_count = 0
        self.diversity = 1.0
        self.best_fitness = np.inf
        self.best_position = None
        self.iteration = 0
        
        self._trial_buffer = np.empty((self.NP, dim))
        self._mutant_buffer = np.empty((self.NP, dim))
        self._best_indices = []
        
    def __call__(self, func, stopping_condition):
        population, fitness = self._initialize_population(func)
        self.best_fitness, best_idx = self._update_best(population, fitness)
        self.best_position = population[best_idx].copy()
        
        while not stopping_condition():
            self._mutate_batch(population)
            self._crossover_batch(population)
            self._clip_bounds(self._trial_buffer)
            
            trial_fitness = self._evaluate_batch(func, self._trial_buffer)
            if len(trial_fitness) < len(self._trial_buffer):
                self._trial_buffer = self._trial_buffer[:len(trial_fitness)]
                trial_fitness = func(self._trial_buffer)
                break
            
            improved = self._select_survivors_batch(population, fitness, trial_fitness)
            self._update_best(population, fitness)
            self._update_archive(population, fitness)
            self._adapt_parameters(improved)
            
            self._best_indices.append(np.argmin(fitness))
            if len(self._best_indices) >= self.simplex_refresh_rate:
                self._apply_simplex_refinement(func, population)
                self._best_indices = []
            
            if self._check_diversity_low(population):
                self._opposition_based_restart(func, population, fitness)
            
            if self._check_stagnation():
                self._restart_if_stagnant(func, population, fitness)
            
            self.iteration += 1
        
        return self.best_fitness, self.best_position
    
    def _initialize_population(self, func):
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.NP, self.dim)
        )
        population = np.clip(population, self.lower_bound, self.upper_bound)
        fitness = func(population)
        self.archive = []
        self.stagnation_count = 0
        self.iteration = 0
        return population, fitness
    
    def _update_best(self, population, fitness):
        valid = np.isfinite(fitness)
        if not np.any(valid):
            return self.best_fitness, 0
        min_idx = np.argmin(np.where(valid, fitness, np.inf))
        if fitness[min_idx] < self.best_fitness:
            self.best_fitness = fitness[min_idx]
            self.best_position = population[min_idx].copy()
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
        return self.best_fitness, min_idx
    
    def _mutate_batch(self, population):
        perp_scale = 0.3 * (1.0 + np.random.rand(self.NP, 1))
        reflection_vectors = population - self.best_position
        reflection_vectors = reflection_vectors / (
            np.linalg.norm(reflection_vectors, axis=1, keepdims=True) + 1e-10
        )
        
        perp_directions = np.random.randn(self.NP, self.dim)
        for i in range(self.NP):
            proj = np.dot(perp_directions[i], reflection_vectors[i])
            perp_directions[i] -= proj * reflection_vectors[i]
        perp_directions = perp_directions / (
            np.linalg.norm(perp_directions, axis=1, keepdims=True) + 1e-10
        )
        
        self._mutant_buffer[:] = self.best_position + \
            perp_scale * perp_directions * np.random.rand(self.NP, 1)
        
        archive_influence = np.zeros((self.NP, self.dim))
        if len(self.archive) > 0:
            archive_array = np.array(self.archive)
            n_archive = min(len(self.archive), self.NP)
            archive_sample_idx = np.random.choice(
                len(archive_array), n_archive, replace=False
            )
            archive_samples = archive_array[archive_sample_idx]
            archive_influence = (archive_samples[:self.NP] - self.best_position) * 0.2
        
        self._mutant_buffer += archive_influence * np.random.rand(self.NP, 1)
        
        r1, r2, r3 = self._select_random_indices(self.NP, 3)
        DE_component = self.F * (population[r1] - population[r2])
        self._mutant_buffer += DE_component + self.F * (population[r3] - self._mutant_buffer) * 0.1
    
    def _select_random_indices(self, count, k):
        indices = [np.random.permutation(self.NP) for _ in range(k)]
        return tuple(indices[:k])
    
    def _crossover_batch(self, population):
        j_rand = np.random.randint(0, self.dim, self.NP)
        mask = np.random.rand(self.NP, self.dim) < self.CR
        mask[np.arange(self.NP), j_rand] = True
        self._trial_buffer[:] = np.where(mask, self._mutant_buffer, population)
    
    def _clip_bounds(self, array):
        np.clip(array, self.lower_bound, self.upper_bound, out=array)
    
    def _evaluate_batch(self, func, batch):
        return func(batch)
    
    def _select_survivors_batch(self, population, fitness, trial_fitness):
        improved_mask = trial_fitness < fitness
        improved_count = np.sum(improved_mask)
        
        population[:] = np.where(improved_mask[:, np.newaxis], self._trial_buffer, population)
        fitness[:] = np.where(improved_mask, trial_fitness, fitness)
        
        return improved_count
    
    def _update_archive(self, population, fitness):
        current_best_idx = np.argmin(fitness)
        current_best = population[current_best_idx].copy()
        
        if len(self.archive) == 0 or fitness[current_best_idx] < self.best_fitness:
            self.archive.append(current_best_pos := current_best_pos)
        
        if len(self.archive) > self.archive_size:
            remove_count = len(self.archive) - self.archive_size
            remove_indices = np.random.choice(
                len(self.archive), remove_count, replace=False
            )
            self.archive = [x for i, x in enumerate(self.archive) if i not in remove_indices]
    
    def _adapt_parameters(self, improved):
        improvement_rate = improved / self.NP
        
        if improvement_rate > 0.2:
            self.F = min(1.0, self.F * 1.1)
            self.CR = min(0.95, self.CR * 1.05)
        elif improvement_rate < 0.05:
            self.F = max(0.1, self.F * 0.9)
            self.CR = max(0.3, self.CR * 0.95)
        
        self.F += 0.01 * np.random.randn()
        self.F = np.clip(self.F, 0.1, 1.0)
        
        self.CR = 0.5 * self.CR + 0.5 * np.random.rand()
    
    def _apply_simplex_refinement(self, func, population):
        best_idx = np.argmin(self._best_indices) if self._best_indices else 0
        center = np.mean(population, axis=0)
        
        simplex_size = min(self.dim + 1, 6)
        simplex_indices = np.random.choice(self.NP, simplex_size, replace=False)
        simplex_points = population[simplex_indices]
        
        fitnesses = func(simplex_points)
        sorted_idx = np.argsort(fitnesses)
        best_simplex = simplex_points[sorted_idx[0]]
        
        if fitnesses[sorted_idx[0]] < self.best_fitness:
            self.best_fitness = fitnesses[sorted_idx[0]]
            self.best_position = best_simplex.copy()
        
        reflection_point = 2.0 * best_simplex - center
        reflection_point = np.clip(reflection_point, self.lower_bound, self.upper_bound)
        
        trial = reflection_point.reshape(1, -1)
        trial_fitness = func(trial)
        
        if trial_fitness[0] < fitnesses[sorted_idx[-1]]:
            worst_idx = simplex_indices[sorted_idx[-1]]
            if trial_fitness[0] < fitnesses[sorted_idx[0]]:
                population[worst_idx] = reflection_point
            else:
                contraction = 0.5 * (best_simplex + center)
                contraction = np.clip(contraction, self.lower_bound, self.upper_bound)
                population[worst_idx] = contraction
    
    def _compute_diversity(self, population):
        ranges = np.ptp(population, axis=0)
        mean_range = np.mean(ranges)
        if mean_range < 1e-10:
            return 0.0
        cv = ranges / (mean_range + 1e-10)
        diversity = np.std(cv)
        return diversity
    
    def _check_diversity_low(self, population):
        self.diversity = self._compute_diversity(population)
        return self.diversity < self.diversity_threshold
    
    def _opposition_based_restart(self, func, population, fitness):
        opposition = self.lower_bound + self.upper_bound - population
        opposition = np.clip(opposition, self.lower_bound, self.upper_bound)
        
        combined = np.vstack([population, opposition])
        combined_fitness = func(combined)
        
        sorted_indices = np.argsort(combined_fitness)[:self.NP]
        population[:] = combined[sorted_indices]
        fitness[:] = combined_fitness[sorted_indices]
        
        self._update_best(population, fitness)
    
    def _check_stagnation(self):
        return self.stagnation_count >= self.max_stagnation
    
    def _restart_if_stagnant(self, func, population, fitness):
        self.stagnation_count = 0
        self.archive = []
        
        n_replace = self.NP // 2
        replace_idx = np.random.choice(self.NP, n_replace, replace=False)
        population[replace_idx] = np.random.uniform(
            self.lower_bound, self.upper_bound, (n_replace, self.dim)
        )
        
        elite_count = max(1, self.NP // 5)
        elite_indices = np.argsort(fitness)[:elite_count]
        population[elite_indices] = self.best_position + \
            np.random.randn(elite_count, self.dim) * (self.upper_bound - self.lower_bound) * 0.1
        
        self._clip_bounds(population)
        fitness[:] = func(population)
        self._update_best(population, fitness)
```

```python
import numpy as np


class ReflectionGuidedMemeticDE:
    """
    A hybrid metaheuristic combining:
    1. Reflection-guided mutation (candidates reflected about current best)
    2. Success-history adaptive archive (learns from successful mutations)
    3. Nelder-Mead simplex refinement (applied to best individuals)
    4. Opposition-based restart (diversity restoration)
    
    Differs from prior proposals:
    - No spectral/FFT operations (unlike MirrorSpectralDE)
    - No covariance matrix (unlike CovarianceGuidedEvolutionStrategy)
    - No velocity/particle swarm topology (unlike AdaptiveTopologyParticleSwarm)
    - Uses reflection geometry + archive-guided perturbation + direct local search
    """
    
    def __init__(self, dim, NP=None, F_init=0.6, CR_init=0.7, 
                 archive_size_factor=2.0, simplex_refresh_rate=5,
                 diversity_threshold=1e-4, max_stagnation=15):
        self.dim = dim
        self.NP = NP if NP is not None else max(4 * dim, 20)
        self.archive_size = int(self.NP * archive_size_factor)
        self.simplex_refresh_rate = simplex_refresh_rate
        self.diversity_threshold = diversity_threshold
        self.max_stagnation = max_stagnation
        
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        self.F = F_init
        self.CR = CR_init
        self.archive = None
        self.stagnation_count = 0
        self.diversity = 1.0
        self.best_fitness = np.inf
        self.best_position = None
        self.iteration = 0
        
        self._trial_buffer = np.empty((self.NP, dim))
        self._mutant_buffer = np.empty((self.NP, dim))
        self._best_indices = []
        
    def __call__(self, func, stopping_condition):
        population, fitness = self._initialize_population(func)
        self.best_fitness, best_idx = self._update_best(population, fitness)
        self.best_position = population[best_idx].copy()
        
        while not stopping_condition():
            self._mutate_batch(population)
            self._crossover_batch(population)
            self._clip_bounds(self._trial_buffer)
            
            trial_fitness = self._evaluate_batch(func, self._trial_buffer)
            if len(trial_fitness) < len(self._trial_buffer):
                self._trial_buffer = self._trial_buffer[:len(trial_fitness)]
                trial_fitness = func(self._trial_buffer)
                break
            
            improved = self._select_survivors_batch(population, fitness, trial_fitness)
            self._update_best(population, fitness)
            self._update_archive(population, fitness)
            self._adapt_parameters(improved)
            
            self._best_indices.append(np.argmin(fitness))
            if len(self._best_indices) >= self.simplex_refresh_rate:
                self._apply_simplex_refinement(func, population)
                self._best_indices = []
            
            if self._check_diversity_low(population):
                self._opposition_based_restart(func, population, fitness)
            
            if self._check_stagnation():
                self._restart_if_stagnant(func, population, fitness)
            
            self.iteration += 1
        
        return self.best_fitness, self.best_position
    
    def _initialize_population(self, func):
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.NP, self.dim)
        )
        population = np.clip(population, self.lower_bound, self.upper_bound)
        fitness = func(population)
        self.archive = []
        self.stagnation_count = 0
        self.iteration = 0
        return population, fitness
    
    def _update_best(self, population, fitness):
        valid = np.isfinite(fitness)
        if not np.any(valid):
            return self.best_fitness, 0
        min_idx = np.argmin(np.where(valid, fitness, np.inf))
        if fitness[min_idx] < self.best_fitness:
            self.best_fitness = fitness[min_idx]
            self.best_position = population[min_idx].copy()
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
        return self.best_fitness, min_idx
    
    def _mutate_batch(self, population):
        perp_scale = 0.3 * (1.0 + np.random.rand(self.NP, 1))
        reflection_vectors = population - self.best_position
        reflection_norms = np.linalg.norm(reflection_vectors, axis=1, keepdims=True)
        reflection_vectors = reflection_vectors / (reflection_norms + 1e-10)
        
        perp_directions = np.random.randn(self.NP, self.dim)
        for i in range(self.NP):
            proj = np.dot(perp_directions[i], reflection_vectors[i])
            perp_directions[i] -= proj * reflection_vectors[i]
        perp_norms = np.linalg.norm(perp_directions, axis=1, keepdims=True)
        perp_directions = perp_directions / (perp_norms + 1e-10)
        
        self._mutant_buffer[:] = self.best_position + \
            perp_scale * perp_directions * np.random.rand(self.NP, 1)
        
        archive_influence = np.zeros((self.NP, self.dim))
        if len(self.archive) > 0:
            archive_array = np.array(self.archive)
            n_archive = min(len(self.archive), self.NP)
            archive_sample_idx = np.random.choice(
                len(archive_array), n_archive, replace=False
            )
            archive_samples = archive_array[archive_sample_idx]
            archive_influence = (archive_samples[:self.NP] - self.best_position) * 0.2
        
        self._mutant_buffer += archive_influence * np.random.rand(self.NP, 1)
        
        r1, r2, r3 = self._select_random_indices(self.NP, 3)
        DE_component = self.F * (population[r1] - population[r2])
        self._mutant_buffer += DE_component + self.F * (population[r3] - self._mutant_buffer) * 0.1
    
    def _select_random_indices(self, count, k):
        indices = [np.random.permutation(self.NP) for _ in range(k)]
        return tuple(indices[:k])
    
    def _crossover_batch(self, population):
        j_rand = np.random.randint(0, self.dim, self.NP)
        mask = np.random.rand(self.NP, self.dim) < self.CR
        mask[np.arange(self.NP), j_rand] = True
        self._trial_buffer[:] = np.where(mask, self._mutant_buffer, population)
    
    def _clip_bounds(self, array):
        np.clip(array, self.lower_bound, self.upper_bound, out=array)
    
    def _evaluate_batch(self, func, batch):
        return func(batch)
    
    def _select_survivors_batch(self, population, fitness, trial_fitness):
        improved_mask = trial_fitness < fitness
        improved_count = np.sum(improved_mask)
        
        population[:] = np.where(improved_mask[:, np.newaxis], self._trial_buffer, population)
        fitness[:] = np.where(improved_mask, trial_fitness, fitness)
        
        return improved_count
    
    def _update_archive(self, population, fitness):
        current_best_idx = np.argmin(fitness)
        current_best = population[current_best_idx].copy()
        
        if len(self.archive) == 0:
            self.archive.append(current_best)
        elif fitness[current_best_idx] < self.best_fitness:
            self.archive.append(current_best)
        
        if len(self.archive) > self.archive_size:
            remove_count = len(self.archive) - self.archive_size
            remove_indices = np.random.choice(
                len(self.archive), remove_count, replace=False
            )
            self.archive = [x for i, x in enumerate(self.archive) if i not in remove_indices]
    
    def _adapt_parameters(self, improved):
        improvement_rate = improved / self.NP
        
        if improvement_rate > 0.2:
            self.F = min(1.0, self.F * 1.1)
            self.CR = min(0.95, self.CR * 1.05)
        elif improvement_rate < 0.05:
            self.F = max(0.1, self.F * 0.9)
            self.CR = max(0.3, self.CR * 0.95)
        
        self.F += 0.01 * np.random.randn()
        self.F = np.clip(self.F, 0.1, 1.0)
        
        self.CR = 0.5 * self.CR + 0.5 * np.random.rand()
    
    def _apply_simplex_refinement(self, func, population):
        best_idx = np.argmin(self._best_indices) if self._best_indices else 0
        center = np.mean(population, axis=0)
        
        simplex_size = min(self.dim + 1, 6)
        simplex_indices = np.random.choice(self.NP, simplex_size, replace=False)
        simplex_points = population[simplex_indices]
        
        fitnesses = func(simplex_points)
        sorted_idx = np.argsort(fitnesses)
        best_simplex = simplex_points[sorted_idx[0]]
        
        if fitnesses[sorted_idx[0]] < self.best_fitness:
            self.best_fitness = fitnesses[sorted_idx[0]]
            self.best_position = best_simplex.copy()
        
        reflection_point = 2.0 * best_simplex - center
        reflection_point = np.clip(reflection_point, self.lower_bound, self.upper_bound)
        
        trial = reflection_point.reshape(1, -1)
        trial_fitness = func(trial)
        
        if trial_fitness[0] < fitnesses[sorted_idx[-1]]:
            worst_idx = simplex_indices[sorted_idx[-1]]
            if trial_fitness[0] < fitnesses[sorted_idx[0]]:
                population[worst_idx] = reflection_point
            else:
                contraction = 0.5 * (best_simplex + center)
                contraction = np.clip(contraction, self.lower_bound, self.upper_bound)
                population[worst_idx] = contraction
    
    def _compute_diversity(self, population):
        ranges = np.ptp(population, axis=0)
        mean_range = np.mean(ranges)
        if mean_range < 1e-10:
            return 0.0
        cv = ranges / (mean_range + 1e-10)
        diversity = np.std(cv)
        return diversity
    
    def _check_diversity_low(self, population):
        self.diversity = self._compute_diversity(population)
        return self.diversity < self.diversity_threshold
    
    def _opposition_based_restart(self, func, population, fitness):
        opposition = self.lower_bound + self.upper_bound - population
        opposition = np.clip(opposition, self.lower_bound, self.upper_bound)
        
        combined = np.vstack([population, opposition])
        combined_fitness = func(combined)
        
        sorted_indices = np.argsort(combined_fitness)[:self.NP]
        population[:] = combined[sorted_indices]
        fitness[:] = combined_fitness[sorted_indices]
        
        self._update_best(population, fitness)
    
    def _check_stagnation(self):
        return self.stagnation_count >= self.max_stagnation
    
    def _restart_if_stagnant(self, func, population, fitness):
        self.stagnation_count = 0
        self.archive = []
        
        n_replace = self.NP // 2
        replace_idx = np.random.choice(self.NP, n_replace, replace=False)
        population[replace_idx] = np.random.uniform(
            self.lower_bound, self.upper_bound, (n_replace, self.dim)
        )
        
        elite_count = max(1, self.NP // 5)
        elite_indices = np.argsort(fitness)[:elite_count]
        noise_scale = (self.upper_bound - self.lower_bound) * 0.1
        population[elite_indices] = self.best_position + \
            np.random.randn(elite_count, self.dim) * noise_scale
        
        self._clip_bounds(population)
        fitness[:] = func(population)
        self._update_best(population, fitness)
```