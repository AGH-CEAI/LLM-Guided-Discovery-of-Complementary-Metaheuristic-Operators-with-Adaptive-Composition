```python
import numpy as np


class AdaptiveNeighborhoodPSO:
    """
    Adaptive Neighborhood Particle Swarm Optimizer (ANPSO)
    
    A PSO variant featuring:
    - Dynamic ring topology with adaptive neighborhood size
    - Gradient-guided acceleration terms using fitness differences
    - Coevolutionary archive of diverse solutions
    - Adaptive inertia and learning rate based on swarm diversity
    - Stagnation detection with targeted reinitialization
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower = kwargs.get('lower', -100.0)
        self.upper = kwargs.get('upper', 100.0)
        self.NP = min(kwargs.get('NP', 5 * dim), 400)
        
        self.c1 = kwargs.get('c1', 1.8)
        self.c2 = kwargs.get('c2', 1.8)
        self.c3 = kwargs.get('c3', 0.4)
        
        self.inertia_max = kwargs.get('inertia_max', 0.9)
        self.inertia_min = kwargs.get('inertia_min', 0.3)
        
        self.lr_max = kwargs.get('lr_max', 1.0)
        self.lr_min = kwargs.get('lr_min', 0.1)
        
        self.min_neighbors = kwargs.get('min_neighbors', 3)
        self.max_neighbors = kwargs.get('max_neighbors', dim // 2)
        
        self.archive_size = kwargs.get('archive_size', self.NP // 4)
        self.max_velocity_ratio = kwargs.get('max_vel_ratio', 0.15)
        self.stagnation_limit = kwargs.get('stagnation_limit', 80)
        
        self.population = None
        self.velocities = None
        self.personal_best_pos = None
        self.personal_best_val = None
        self.global_best_pos = None
        self.global_best_val = None
        self.archive = None
        
        self.inertia = self.inertia_max
        self.learning_rate = self.lr_max
        self.neighborhood_size = self.min_neighbors
        self.iteration = 0
        self.gbest_history = []
        
    def _clip_to_bounds(self, pop):
        return np.clip(pop, self.lower, self.upper)
    
    def _initialize_population(self):
        self.population = np.random.uniform(
            self.lower, self.upper, (self.NP, self.dim)
        )
        self.population = self._clip_to_bounds(self.population)
        
        vel_range = (self.upper - self.lower) * 0.1
        self.velocities = np.random.uniform(-vel_range, vel_range, (self.NP, self.dim))
        
        self.personal_best_pos = self.population.copy()
        self.personal_best_val = np.full(self.NP, np.inf)
        
        self.global_best_pos = None
        self.global_best_val = np.inf
        
        self.archive = self.population[:1].copy() if self.NP > 0 else np.empty((0, self.dim))
        
        self.inertia = self.inertia_max
        self.learning_rate = self.lr_max
        self.neighborhood_size = self.min_neighbors
        self.iteration = 0
        self.gbest_history = []
    
    def __call__(self, func, stopping_condition):
        self._initialize_population()
        fitness = func(self.population)
        
        if len(fitness) < self.NP:
            fitness = np.pad(fitness, (0, self.NP - len(fitness)), constant_values=np.nan)
        
        self._update_best_from_fitness(self.population, fitness)
        
        while not stopping_condition():
            self._adapt_parameters()
            
            if stopping_condition():
                break
            
            trials, trial_velocities = self._generate_trial_batch()
            
            if stopping_condition():
                break
            
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                valid = len(trial_fitness)
                trials = trials[:valid]
                trial_velocities = trial_velocities[:valid]
                trial_fitness = np.pad(trial_fitness, (0, len(trials) - valid), constant_values=np.nan)
            
            self._update_best_from_fitness(trials, trial_fitness)
            
            self.population, self.velocities, fitness = self._select_survivors_batch(
                trials, trial_velocities, trial_fitness
            )
            
            self._update_archive_batch()
            self._adapt_neighborhood_size()
            
            self.iteration += 1
            
            if self._detect_stagnation():
                self._reinitialize_stagnant_portion()
        
        return self.global_best_val, self.global_best_pos.copy()
    
    def _generate_trial_batch(self):
        nbr_best = self._compute_neighborhood_best_batch()
        archive_sample = self._sample_archive_batch()
        
        r1 = np.random.uniform(0, 1, (self.NP, self.dim))
        r2 = np.random.uniform(0, 1, (self.NP, self.dim))
        r3 = np.random.uniform(0, 1, (self.NP, self.dim))
        r4 = np.random.uniform(0, 1, (self.NP, self.dim))
        
        cognitive = self.c1 * r1 * (self.personal_best_pos - self.population)
        social = self.c2 * r2 * (nbr_best - self.population)
        archive_force = self.c3 * r3 * (archive_sample - self.population)
        
        gradient = self.learning_rate * r4 * (self.personal_best_pos - self.global_best_pos)
        
        velocity = self.inertia * self.velocities + cognitive + social + archive_force + gradient
        
        max_vel = (self.upper - self.lower) * self.max_velocity_ratio
        velocity = np.clip(velocity, -max_vel, max_vel)
        
        trial = self.population + velocity
        trial = self._clip_to_bounds(trial)
        
        return trial, velocity
    
    def _compute_neighborhood_best_batch(self):
        half = self.neighborhood_size // 2
        indices = np.arange(self.NP)
        
        offsets = np.arange(-half, half + 1)
        nbr_idx = (indices[:, np.newaxis] + offsets[np.newaxis, :]) % self.NP
        
        nbr_pbvals = self.personal_best_val[nbr_idx]
        best_in_nbr = np.argmin(nbr_pbvals, axis=1)
        
        row_indices = np.arange(self.NP)
        nbr_best_positions = self.personal_best_pos[nbr_idx[row_indices, best_in_nbr]]
        
        return nbr_best_positions
    
    def _sample_archive_batch(self):
        if len(self.archive) == 0:
            return self.population
        
        n_samples = min(len(self.archive), self.NP)
        
        if n_samples == self.NP:
            return self.archive
        
        selected = np.random.choice(len(self.archive), n_samples, replace=False)
        
        if n_samples < self.NP:
            samples = np.empty((self.NP, self.dim))
            samples[:n_samples] = self.archive[selected]
            samples[n_samples:] = self.population[n_samples:]
            return samples
        
        return self.archive[selected]
    
    def _update_best_from_fitness(self, positions, fitness):
        improved = ~np.isnan(fitness) & (fitness < self.personal_best_val)
        
        self.personal_best_val[improved] = fitness[improved]
        self.personal_best_pos[improved] = positions[improved]
        
        valid_fitness = fitness[~np.isnan(fitness)]
        if len(valid_fitness) > 0:
            best_idx = np.nanargmin(fitness)
            if fitness[best_idx] < self.global_best_val:
                self.global_best_val = fitness[best_idx]
                self.global_best_pos = positions[best_idx].copy()
                self.gbest_history = [self.global_best_val]
            else:
                self.gbest_history.append(self.global_best_val)
                if len(self.gbest_history) > 1000:
                    self.gbest_history = self.gbest_history[-500:]
    
    def _select_survivors_batch(self, trials, trial_velocities, trial_fitness):
        improved = ~np.isnan(trial_fitness) & (trial_fitness < self.personal_best_val)
        
        new_pop = self.population.copy()
        new_vel = self.velocities.copy()
        new_fit = self.personal_best_val.copy()
        
        new_pop[improved] = trials[improved]
        new_vel[improved] = trial_velocities[improved]
        new_fit[improved] = trial_fitness[improved]
        
        return new_pop, new_vel, new_fit
    
    def _update_archive_batch(self):
        combined = np.vstack([self.archive, self.personal_best_pos])
        
        combined_fit = np.concatenate([
            np.full(len(self.archive), -np.inf),
            self.personal_best_val
        ])
        
        pareto_mask = self._compute_pareto_dominance(combined, combined_fit)
        pareto_solutions = combined[pareto_mask]
        pareto_fitness = combined_fit[pareto_mask]
        
        if len(pareto_solutions) == 0:
            best_idx = np.nanargmin(self.personal_best_val)
            self.archive = self.personal_best_pos[best_idx:best_idx+1].copy()
            return
        
        if len(pareto_solutions) <= self.archive_size:
            self.archive = pareto_solutions.copy()
        else:
            crowding_scores = self._compute_crowding_scores(pareto_solutions, pareto_fitness)
            top_indices = np.argsort(crowding_scores)[-self.archive_size:]
            self.archive = pareto_solutions[top_indices].copy()
    
    def _compute_pareto_dominance(self, pop, fitness):
        n = len(pop)
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i != j and not is_dominated[i]:
                    if fitness[j] <= fitness[i]:
                        is_dominated[i] = True
                        break
        
        return ~is_dominated
    
    def _compute_crowding_scores(self, pop, fitness):
        n = len(pop)
        cd = np.zeros(n)
        
        if n <= 2:
            return cd
        
        cd = np.zeros(n)
        
        for d in range(pop.shape[1]):
            sorted_idx = np.argsort(pop[:, d])
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            
            span = pop[sorted_idx[-1], d] - pop[sorted_idx[0], d]
            if span > 1e-10:
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (pop[sorted_idx[i+1], d] - pop[sorted_idx[i-1], d]) / span
        
        return cd
    
    def _adapt_parameters(self):
        diversity = self._compute_diversity_measure()
        
        self.inertia = self.inertia_max - (self.inertia_max - self.inertia_min) * diversity
        
        sigmoid_input = 10.0 * (diversity - 0.5)
        sigmoid_val = 1.0 / (1.0 + np.exp(-sigmoid_input))
        self.learning_rate = self.lr_max - (self.lr_max - self.lr_min) * sigmoid_val
    
    def _compute_diversity_measure(self):
        if self.NP < 2:
            return 1.0
        
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        
        space_diameter = np.sqrt(self.dim) * (self.upper - self.lower)
        
        return np.mean(distances) / (space_diameter + 1e-10)
    
    def _adapt_neighborhood_size(self):
        diversity = self._compute_diversity_measure()
        
        target = int(
            self.min_neighbors +
            (self.max_neighbors - self.min_neighbors) * diversity
        )
        
        self.neighborhood_size += 0.05 * (target - self.neighborhood_size)
        self.neighborhood_size = int(np.clip(
            self.neighborhood_size,
            self.min_neighbors,
            min(self.max_neighbors, self.NP - 1)
        ))
    
    def _detect_stagnation(self):
        if len(self.gbest_history) < self.stagnation_limit:
            return False
        
        recent = self.gbest_history[-self.stagnation_limit:]
        
        if np.nanmin(recent) >= self.global_best_val - 1e-12:
            return True
        
        variance = np.nanvar(recent)
        if variance < 1e-15 and len(set(np.round(recent, 8))) == 1:
            return True
        
        return False
    
    def _reinitialize_stagnant_portion(self):
        n_replace = max(self.NP // 3, 1)
        
        fitness_copy = self.personal_best_val.copy()
        fitness_copy[np.isnan(fitness_copy)] = -np.inf
        worst_indices = np.argsort(fitness_copy)[-n_replace:]
        
        self.population[worst_indices] = np.random.uniform(
            self.lower, self.upper, (n_replace, self.dim)
        )
        
        vel_range = (self.upper - self.lower) * 0.1
        self.velocities[worst_indices] = np.random.uniform(
            -vel_range, vel_range, (n_replace, self.dim)
        )
        
        self.inertia = self.inertia_max
        self.gbest_history = [self.global_best_val]
```