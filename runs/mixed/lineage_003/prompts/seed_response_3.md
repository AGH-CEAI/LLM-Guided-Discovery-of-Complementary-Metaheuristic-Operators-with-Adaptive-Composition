```python
import numpy as np

class HybridParticleSwarmCultural:
    """
    Hybrid Particle Swarm with Adaptive Topology and Cultural Memory.
    Uses ring topology with dynamic neighborhood, current-to-pbest mutation,
    elitist cultural belief space, and diversity-triggered restarts.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        self.NP = min(max(4 * dim, 60), min(10 * dim, 300))
        
        # PSO parameters (adaptive)
        self.w = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.2
        self.c1_max = 2.5
        self.c1_min = 1.0
        self.c2_max = 2.5
        self.c2_min = 1.0
        self.v_max = 20.0
        
        # Topology
        self.neighbor_k = max(3, dim // 10)
        
        # Cultural belief space
        self.elite_size = max(5, self.NP // 10)
        
        # State
        self.population = None
        self.velocities = None
        self.fitness = None
        self.personal_best = None
        self.personal_best_fit = None
        self.neighborhood_best = None
        self.neighborhood_best_fit = None
        self.global_best = None
        self.global_best_fit = np.inf
        self.belief_space = None
        self.belief_fitness = None
        
        # Counters
        self.generation = 0
        self.last_improvement_gen = 0
        self.last_diversity_gen = 0
        
    def _initialize_population(self):
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.NP, self.dim)
        )
        self.velocities = np.random.uniform(
            -self.v_max * 0.5, self.v_max * 0.5, (self.NP, self.dim)
        )
        
    def _eval_wrapper(self, batch):
        result = self.func(batch)
        if result is None or len(result) == 0:
            return None
        valid_mask = ~np.isnan(result)
        if not np.any(valid_mask):
            return None
        return result
    
    def _evaluate_initial_batch(self):
        self.fitness = self._eval_wrapper(self.population)
        
    def _build_ring_topology(self):
        n = self.NP
        k = self.neighbor_k
        neighborhoods = np.zeros((n, 2 * k + 1), dtype=int)
        for i in range(n):
            neighbors = [(i + j) % n for j in range(-k, k + 1)]
            neighborhoods[i] = neighbors
        return neighborhoods
    
    def _update_personal_best_batch(self):
        improved = self.fitness < self.personal_best_fit
        self.personal_best[improved] = self.population[improved]
        self.personal_best_fit[improved] = self.fitness[improved]
        
    def _update_neighborhood_best_batch(self, topology):
        for i in range(self.NP):
            nb_indices = topology[i]
            nb_fitness = self.personal_best_fit[nb_indices]
            best_nb = np.argmin(nb_fitness)
            self.neighborhood_best[i] = self.personal_best[nb_indices[best_nb]]
            self.neighborhood_best_fit[i] = nb_fitness[best_nb]
            
    def _update_global_best_batch(self):
        idx = np.argmin(self.personal_best_fit)
        if self.personal_best_fit[idx] < self.global_best_fit:
            self.global_best = self.personal_best[idx].copy()
            self.global_best_fit = self.personal_best_fit[idx]
            self.last_improvement_gen = self.generation
            
    def _select_survivors_batch(self, trials, trial_fitness):
        better = trial_fitness < self.fitness
        self.population[better] = trials[better]
        self.fitness[better] = trial_fitness[better]
        
    def _update_velocity_batch(self):
        r1 = np.random.uniform(0, 1, (self.NP, self.dim))
        r2 = np.random.uniform(0, 1, (self.NP, self.dim))
        cognitive = self.c1 * r1 * (self.personal_best - self.population)
        social = self.c2 * r2 * (self.neighborhood_best - self.population)
        self.velocities = self.w * self.velocities + cognitive + social
        
    def _clip_velocities_batch(self):
        self.velocities = np.clip(self.velocities, -self.v_max, self.v_max)
        
    def _update_position_batch(self):
        self.population = self.population + self.velocities
        self.population = np.clip(self.population, self.lower_bound, self.upper_bound)
        
    def _clip_to_bounds_batch(self, batch):
        return np.clip(batch, self.lower_bound, self.upper_bound)
        
    def _adapt_parameters_batch(self):
        progress = self.generation / max(1, self.max_generations)
        self.w = self.w_max - progress * (self.w_max - self.w_min)
        self.c1 = self.c1_max - progress * (self.c1_max - self.c1_min)
        self.c2 = self.c2_min + progress * (self.c2_max - self.c2_min)
        
    def _compute_diversity(self):
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
        
    def _mutate_current_to_pbest_with_culture(self, topology):
        trials = np.empty((self.NP, self.dim))
        pbest_indices = np.argmin(
            np.where(self.fitness[:, None] < self.personal_best_fit[None, :],
                     self.fitness[:, None], self.personal_best_fit[None, :]),
            axis=1
        )
        pbest_selected = self.personal_best[pbest_indices]
        
        for i in range(self.NP):
            nb_best = self.neighborhood_best[i]
            diff = nb_best - self.population[i]
            F = np.random.uniform(0.5, 1.5)
            mutation = self.population[i] + F * diff
            
            if self.belief_space is not None and len(self.belief_space) > 0:
                elite_influence = self.belief_space[np.random.randint(len(self.belief_space))]
                alpha = 0.2 * np.random.rand()
                mutation = mutation + alpha * (elite_influence - mutation)
            
            trials[i] = mutation
            
        return trials
        
    def _update_belief_space(self):
        sorted_indices = np.argsort(self.fitness)
        new_belief = self.population[sorted_indices[:self.elite_size]]
        new_belief_fit = self.fitness[sorted_indices[:self.elite_size]]
        
        if self.belief_space is None:
            self.belief_space = new_belief
            self.belief_fitness = new_belief_fit
        else:
            combined = np.vstack([self.belief_space, new_belief])
            combined_fit = np.concatenate([self.belief_fitness, new_belief_fit])
            top_k = np.argsort(combined_fit)[:self.elite_size]
            self.belief_space = combined[top_k]
            self.belief_fitness = combined_fit[top_k]
            
    def _restart_if_stagnant(self, diversity):
        stagnation = self.generation - self.last_improvement_gen
        low_diversity = diversity < 1e-6 * self.dim
        
        if stagnation > max(50, self.NP // 4) or low_diversity:
            n_replace = max(self.NP // 4, 5)
            worst_indices = np.argsort(self.fitness)[-n_replace:]
            
            for idx in worst_indices:
                if self.belief_space is not None and len(self.belief_space) > 0:
                    base = self.belief_space[np.random.randint(len(self.belief_space))]
                    noise_scale = max(10.0, (self.upper_bound - self.lower_bound) * 0.1)
                    self.population[idx] = base + np.random.normal(0, noise_scale, self.dim)
                else:
                    self.population[idx] = np.random.uniform(
                        self.lower_bound, self.upper_bound, self.dim
                    )
                self.velocities[idx] = np.random.uniform(
                    -self.v_max * 0.5, self.v_max * 0.5, self.dim
                )
                self.fitness[idx] = np.inf
                
            self.last_improvement_gen = self.generation
            self.last_diversity_gen = self.generation
            
    def __call__(self, func, stopping_condition):
        self.func = func
        self.max_generations = 2000
        self.generation = 0
        self.last_improvement_gen = 0
        self.last_diversity_gen = 0
        
        self._initialize_population()
        self._evaluate_initial_batch()
        
        self.personal_best = self.population.copy()
        self.personal_best_fit = self.fitness.copy()
        self.neighborhood_best = np.empty((self.NP, self.dim))
        self.neighborhood_best_fit = np.full(self.NP, np.inf)
        
        topology = self._build_ring_topology()
        self._update_neighborhood_best_batch(topology)
        self._update_global_best_batch()
        self._update_belief_space()
        
        while not stopping_condition():
            self._adapt_parameters_batch()
            
            trials = self._mutate_current_to_pbest_with_culture(topology)
            trials = self._clip_to_bounds_batch(trials)
            
            trial_fitness = self._eval_wrapper(trials)
            
            if trial_fitness is None:
                break
            if len(trial_fitness) < len(trials):
                valid_len = len(trial_fitness)
                trials = trials[:valid_len]
                trial_fitness = trial_fitness[:valid_len]
                if valid_len == 0:
                    break
                    
            if stopping_condition():
                break
                
            self._select_survivors_batch(trials, trial_fitness)
            self._update_personal_best_batch()
            self._update_neighborhood_best_batch(topology)
            self._update_global_best_batch()
            self._update_belief_space()
            
            self._update_velocity_batch()
            self._clip_velocities_batch()
            self._update_position_batch()
            
            self.generation += 1
            
            if self.generation % 10 == 0:
                diversity = self._compute_diversity()
                self._restart_if_stagnant(diversity)
                topology = self._build_ring_topology()
                
        return self.global_best_fit, self.global_best
```