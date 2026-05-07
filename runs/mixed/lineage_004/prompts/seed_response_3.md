```python
import numpy as np


class AdaptiveTopologyPSO:
    """PSO with adaptive ring topology, cultural memory, and guided perturbation.
    
    Key design features:
    - Adaptive neighborhood radius that expands/contracts based on diversity
    - Cultural memory guiding particles toward promising regions
    - pbest-based movement with adaptive inertia
    - Diversity monitoring with strategic restarts
    - Adaptive local search on stagnation
    - All operations vectorized on batches
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.pop_size = kwargs.get('pop_size', min(200, max(60, 6 * dim)))
        self.bounds = np.array([[-100.0, 100.0]] * dim)
        self.bound_range = 200.0
        
        # Topology parameters
        self.min_topology_radius = 2
        self.max_topology_radius = min(20, self.pop_size // 3)
        self.topology_radius = kwargs.get('init_topology_radius', 3)
        
        # PSO parameters
        self.c1 = kwargs.get('c1', 1.496)
        self.c2 = kwargs.get('c2', 1.496)
        self.c3 = kwargs.get('c3', 0.8)
        self.initial_inertia = 0.9
        self.min_inertia = 0.4
        
        # Adaptive parameters
        self.diversity_low_threshold = 0.1
        self.diversity_high_threshold = 1.0
        self.stagnation_threshold = 15
        self.local_search_interval = 12
        
        # State variables
        self.population = None
        self.velocities = None
        self.fitness = None
        self.pbest = None
        self.pbest_fitness = None
        self.gbest = None
        self.gbest_fitness = np.inf
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        self.max_generations = 5000
        
        # Cultural memory
        self.culture_memory = None
        self.culture_fitness = None
        self.culture_size = max(5, self.pop_size // 8)
        
        # Perturbation history for adaptive control
        self.recent_improvements = []
        
    def _initialize_population(self):
        self.population = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            (self.pop_size, self.dim)
        )
        self.velocities = np.random.uniform(
            -0.1 * self.bound_range, 0.1 * self.bound_range,
            (self.pop_size, self.dim)
        )
        self.pbest = self.population.copy()
        self.pbest_fitness = np.full(self.pop_size, np.inf)
        self.gbest = None
        self.gbest_fitness = np.inf
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        self.recent_improvements = []
        
        mem_idx = np.random.choice(self.pop_size, self.culture_size, replace=False)
        self.culture_memory = self.population[mem_idx].copy()
        self.culture_fitness = np.full(self.culture_size, np.inf)
        
    def _clip_velocities_batch(self, velocities):
        max_vel = 0.2 * self.bound_range
        speed = np.linalg.norm(velocities, axis=1, keepdims=True)
        valid_speed = np.maximum(speed, 1e-10)
        scale = np.minimum(1.0, max_vel / valid_speed)
        velocities *= scale
        return velocities
        
    def _clip_positions_batch(self, positions):
        return np.clip(positions, self.bounds[:, 0], self.bounds[:, 1])
        
    def _eval_wrapper(self, pop, func):
        fitness = func(pop)
        if len(fitness) < len(pop):
            actual = len(fitness)
            return pop[:actual], fitness[:actual]
        return pop, fitness
        
    def _compute_inertia_batch(self):
        progress = self.generation / max(1, self.max_generations)
        inertia = self.initial_inertia - (self.initial_inertia - self.min_inertia) * progress
        return inertia
        
    def _compute_neighborhood_best_batch(self):
        n = self.pop_size
        k = self.topology_radius
        nbest = np.zeros((n, self.dim))
        
        for i in range(n):
            distances = np.sum((self.population - self.population[i]) ** 2, axis=1)
            distances[i] = np.inf
            neighbors = np.argpartition(distances, k)[:k]
            best_neighbor = neighbors[np.argmin(self.pbest_fitness[neighbors])]
            nbest[i] = self.pbest[best_neighbor]
            
        return nbest
        
    def _update_velocity_batch(self, nbest):
        inertia = self._compute_inertia_batch()
        r1 = np.random.random((self.pop_size, self.dim))
        r2 = np.random.random((self.pop_size, self.dim))
        
        cognitive = self.c1 * r1 * (self.pbest - self.population)
        social = self.c2 * r2 * (nbest - self.population)
        
        velocities = inertia * self.velocities + cognitive + social
        self.velocities = self._clip_velocities_batch(velocities)
        
    def _mutate_current_to_pbest_with_culture(self):
        n = self.pop_size
        trials = np.zeros((n, self.dim))
        
        r3 = np.random.random((n, self.dim))
        cultural_weight = 0.3 * (1.0 - self.generation / max(1, self.max_generations))
        culture_idx = np.random.randint(0, self.culture_size, n)
        
        for i in range(n):
            f1 = np.random.uniform(0.5, 1.0)
            f2 = np.random.uniform(0.5, 1.0)
            
            trials[i] = (self.population[i] +
                        f1 * (self.pbest[i] - self.population[i]) +
                        f2 * (self.culture_memory[culture_idx[i]] - self.population[i]) +
                        cultural_weight * r3[i] * (self.gbest - self.population[i]))
                        
        return trials
        
    def _apply_guided_perturbation_batch(self):
        n = self.pop_size
        top_elite = max(3, n // 10)
        elite_mask = np.argsort(self.fitness)[:top_elite]
        elite = self.population[elite_mask]
        elite_fit = self.fitness[elite_mask]
        
        base = elite[np.argmin(elite_fit)]
        base_fitness = np.min(elite_fit)
        
        sigma = max(1e-6, base_fitness * 0.1) if base_fitness > 1e-5 else 1.0
        trials = np.zeros((n, self.dim))
        
        for i in range(n):
            if i in elite_mask:
                trials[i] = self.population[i] + np.random.normal(0, sigma * 0.1, self.dim)
            else:
                trials[i] = (self.population[i] +
                           np.random.normal(0, sigma * 0.5, self.dim) +
                           0.2 * (base - self.population[i]))
                           
        return trials
        
    def _crossover_batch(self, target, mutant):
        n = self.pop_size
        dim = self.dim
        cr = 0.7 + 0.2 * np.random.random()
        
        mask = np.random.random((n, dim)) < cr
        mask[np.arange(n), np.random.randint(0, dim, n)] = True
        
        trials = np.where(mask, mutant, target)
        return trials
        
    def _compute_diversity_batch(self):
        if self.population is None or len(self.population) < 2:
            return 0.0
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances) / (0.5 * self.bound_range)
        
    def _adapt_topology_radius(self, diversity):
        if diversity < self.diversity_low_threshold:
            self.topology_radius = min(self.max_topology_radius, self.topology_radius + 1)
        elif diversity > self.diversity_high_threshold:
            self.topology_radius = max(self.min_topology_radius, self.topology_radius - 1)
            
    def _adapt_parameters_batch(self):
        recent_count = min(50, len(self.recent_improvements))
        if recent_count < 5:
            return
            
        recent = np.array(self.recent_improvements[-recent_count:])
        improvement_rate = np.mean(recent > 0)
        
        if improvement_rate < 0.2:
            self.c1 = min(2.0, self.c1 * 1.05)
            self.c2 = min(2.0, self.c2 * 1.05)
        elif improvement_rate > 0.6:
            self.c1 = max(0.5, self.c1 * 0.95)
            self.c2 = max(0.5, self.c2 * 0.95)
            
    def _compute_reward(self, old_fitness, new_fitness):
        if self.gbest_fitness < 1e-8:
            return 1.0
        improvement = (old_fitness - new_fitness) / max(1e-10, abs(self.gbest_fitness))
        return max(0, improvement)
        
    def _apply_adaptive_local_search(self, func):
        if self.stagnation_counter < self.local_search_interval:
            return False
            
        best_idx = np.argmin(self.fitness)
        center = self.population[best_idx].copy()
        center_fitness = self.fitness[best_idx]
        
        radius = max(5.0, center_fitness * 0.5) if center_fitness > 1e-5 else 10.0
        candidates = center + np.random.normal(0, radius * 0.1, (20, self.dim))
        candidates = self._clip_positions_batch(candidates)
        
        cand_pop, cand_fit = self._eval_wrapper(candidates, func)
        
        if len(cand_fit) > 0:
            best_cand_idx = np.argmin(cand_fit)
            if cand_fit[best_cand_idx] < center_fitness:
                self.population[best_idx] = candidates[best_cand_idx]
                self.fitness[best_idx] = cand_fit[best_cand_idx]
                self.pbest[best_idx] = candidates[best_cand_idx]
                self.pbest_fitness[best_idx] = cand_fit[best_cand_idx]
                return True
                
        return False
        
    def _restart_if_stagnant(self):
        if self.stagnation_counter < self.stagnation_threshold:
            return False
            
        diversity = self._compute_diversity_batch()
        if diversity > self.diversity_low_threshold:
            return False
            
        elite_count = max(2, self.pop_size // 20)
        elite_idx = np.argsort(self.fitness)[:elite_count]
        elites = self.population[elite_idx].copy()
        elite_fit = self.fitness[elite_idx]
        
        self.population = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            (self.pop_size, self.dim)
        )
        self.population[:elite_count] = elites
        
        self.velocities = np.random.uniform(
            -0.1 * self.bound_range, 0.1 * self.bound_range,
            (self.pop_size, self.dim)
        )
        
        self.pbest = self.population.copy()
        self.pbest_fitness = np.full(self.pop_size, np.inf)
        self.pbest_fitness[:elite_count] = elite_fit
        
        self.stagnation_counter = 0
        self.topology_radius = self.min_topology_radius
        
        return True
        
    def _select_survivors_batch(self, trials, trial_fitness):
        n = len(trials)
        if n < self.pop_size:
            return trials, trial_fitness
            
        old_fitness = self.fitness.copy()
        old_gbest = self.gbest_fitness
        
        replace = trial_fitness < self.fitness
        self.population[replace] = trials[replace]
        self.fitness[replace] = trial_fitness[replace]
        
        improved = trial_fitness < old_fitness
        reward = self._compute_reward(old_fitness, trial_fitness)
        self.recent_improvements.append(reward)
        if len(self.recent_improvements) > 100:
            self.recent_improvements.pop(0)
            
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.gbest_fitness:
            self.gbest = self.population[best_idx].copy()
            self.gbest_fitness = self.fitness[best_idx]
            self.stagnation_counter = 0
            self.last_improvement_gen = self.generation
        else:
            self.stagnation_counter += 1
            
        return self.population, self.fitness
        
    def _update_personal_best_batch(self):
        improved = self.fitness < self.pbest_fitness
        self.pbest[improved] = self.population[improved]
        self.pbest_fitness[improved] = self.fitness[improved]
        
    def _update_culture_memory_batch(self):
        if self.gbest_fitness < np.min(self.culture_fitness):
            worst_idx = np.argmax(self.culture_fitness)
            self.culture_memory[worst_idx] = self.gbest.copy()
            self.culture_fitness[worst_idx] = self.gbest_fitness
            
    def __call__(self, func, stopping_condition):
        self._initialize_population()
        
        self.population, self.fitness = self._eval_wrapper(self.population, func)
        if len(self.fitness) == 0:
            return self.gbest_fitness, self.gbest if self.gbest is not None else np.zeros(self.dim)
            
        self._update_personal_best_batch()
        best_idx = np.argmin(self.fitness)
        self.gbest = self.population[best_idx].copy()
        self.gbest_fitness = self.fitness[best_idx]
        self.pbest_fitness = self.fitness.copy()
        
        while not stopping_condition():
            self.generation += 1
            
            nbest = self._compute_neighborhood_best_batch()
            self._update_velocity_batch(nbest)
            self.population += self.velocities
            self.population = self._clip_positions_batch(self.population)
            
            pbest_mutant = self._mutate_current_to_pbest_with_culture()
            pbest_trials = self._crossover_batch(self.population, pbest_mutant)
            
            perturbation_trials = self._apply_guided_perturbation_batch()
            perturbation_trials = self._clip_positions_batch(perturbation_trials)
            
            all_trials = np.vstack([pbest_trials, perturbation_trials])
            all_trials = self._clip_positions_batch(all_trials)
            
            if stopping_condition():
                break
                
            trial_pop, trial_fitness = self._eval_wrapper(all_trials, func)
            if len(trial_fitness) == 0:
                break
                
            mid = len(trial_fitness) // 2
            pbest_trial_fit = trial_fitness[:mid]
            pbest_trial_pop = trial_pop[:mid]
            perturb_trial_fit = trial_fitness[mid:]
            perturb_trial_pop = trial_pop[mid:]
            
            if len(pbest_trial_fit) == mid:
                self.population, self.fitness = self._select_survivors_batch(
                    pbest_trial_pop, pbest_trial_fit)
                    
            if len(perturb_trial_fit) > 0 and len(perturb_trial_pop) == len(perturb_trial_fit):
                self.population, self.fitness = self._select_survivors_batch(
                    perturb_trial_pop, perturb_trial_fit)
                    
            if stopping_condition():
                break
                
            self._update_personal_best_batch()
            self._update_culture_memory_batch()
            
            diversity = self._compute_diversity_batch()
            self._adapt_topology_radius(diversity)
            self._adapt_parameters_batch()
            
            self._apply_adaptive_local_search(func)
            
            if self._restart_if_stagnant():
                self.population, self.fitness = self._eval_wrapper(self.population, func)
                if len(self.fitness) < self.pop_size:
                    break
                self._update_personal_best_batch()
                best_idx = np.argmin(self.fitness)
                self.gbest = self.population[best_idx].copy()
                self.gbest_fitness = self.fitness[best_idx]
                
        return self.gbest_fitness, self.gbest
```