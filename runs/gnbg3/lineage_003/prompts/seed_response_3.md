```python
import numpy as np


class AdaptiveTopologyParticleSwarmOptimizer:
    """
    Particle Swarm Optimizer with adaptive neighborhood topology and
    multiple co-evolving subpopulations. Uses ring topology with
    performance-based rewiring and adaptive inertia for balance
    between exploration and exploitation.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(120, max(40, 4 * dim))
        self.bounds = np.array([-100.0, 100.0])
        self.max_velocity = 0.2 * (self.bounds[1] - self.bounds[0])
        
        self.chi = 0.7298
        self.c1_init = 1.4962
        self.c2_init = 1.4962
        
        self.history_size = 10
        self.stagnation_threshold = 30
        self.diversity_low = 0.1
        self.diversity_high = 0.5
        
        self._reset_state()
    
    def _reset_state(self):
        self.population = None
        self.velocity = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.neighborhood_best = None
        self.neighborhood_best_fitness = None
        self.global_best = None
        self.global_best_fitness = np.inf
        
        self.c1 = self.c1_init
        self.c2 = self.c2_init
        self.inertia = 0.9
        
        self.topology = None
        self.stagnation_counter = 0
        self.generation = 0
        self.trial_history = []
        self.success_history = []
        self.adapt_trigger_count = 0
    
    def __call__(self, func, stopping_condition):
        self._reset_state()
        self._initialize_population()
        self._initialize_topology()
        self._evaluate_and_update_best_batch(func)
        
        while not stopping_condition():
            trials, trial_velocity = self._generate_trial_batch()
            
            if len(self.trial_history) >= self.history_size:
                self.trial_history.pop(0)
            self.trial_history.append(trials.copy())
            
            if stopping_condition():
                break
            
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                valid_len = len(trial_fitness)
                trials = trials[:valid_len]
                trial_fitness = trial_fitness[:valid_len]
                trial_velocity = trial_velocity[:valid_len]
            
            if stopping_condition():
                break
            
            self._update_personal_best_batch(trials, trial_fitness)
            self._update_neighborhood_best_batch()
            self._update_global_best_batch()
            
            self._select_survivors_batch(trials, trial_velocity, trial_fitness)
            
            if not np.any(np.isfinite(trial_fitness)):
                self._restart_if_needed()
                continue
            
            self._update_success_metric(trial_fitness)
            self._adapt_inertia_weight()
            self._adapt_cognitive_social_factors()
            
            self._check_stagnation()
            if self.stagnation_counter >= self.stagnation_threshold:
                self._rewire_topology()
                self._adapt_parameters()
            
            if self.stagnation_counter >= 2 * self.stagnation_threshold:
                self._restart_if_needed()
            
            self.generation += 1
        
        return self.global_best_fitness, self.global_best.copy()
    
    def _initialize_population(self):
        self.population = np.random.uniform(
            self.bounds[0], self.bounds[1], size=(self.np, self.dim)
        )
        self.velocity = np.random.uniform(
            -self.max_velocity, self.max_velocity, size=(self.np, self.dim)
        )
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.np, np.inf)
        self.neighborhood_best = self.population.copy()
        self.neighborhood_best_fitness = np.full(self.np, np.inf)
    
    def _initialize_topology(self):
        self.topology = np.zeros((self.np, self.np), dtype=bool)
        for i in range(self.np):
            neighbors = [
                (i - 1) % self.np,
                (i + 1) % self.np,
                (i - 2) % self.np,
                (i + 2) % self.np
            ]
            for j in neighbors:
                self.topology[i, j] = True
        np.fill_diagonal(self.topology, False)
    
    def _evaluate_and_update_best_batch(self, func):
        fitness = func(self.population)
        if len(fitness) < self.np:
            fitness = np.pad(fitness, (0, self.np - len(fitness)), 
                           constant_values=np.nan)
        
        self.personal_best_fitness = fitness.copy()
        self.personal_best = self.population.copy()
        
        self.neighborhood_best_fitness = fitness.copy()
        self.neighborhood_best = self.population.copy()
        
        valid_mask = np.isfinite(fitness)
        if np.any(valid_mask):
            best_idx = np.argmin(fitness[valid_mask])
            global_idx = np.where(valid_mask)[0][best_idx]
            self.global_best_fitness = fitness[global_idx]
            self.global_best = self.population[global_idx].copy()
    
    def _generate_trial_batch(self):
        trials = np.empty((self.np, self.dim))
        trial_velocity = np.empty((self.np, self.dim))
        
        diversity = self._compute_diversity()
        adaptive_inertia = self.inertia * (1.0 + 0.5 * (self.diversity_high - diversity))
        adaptive_inertia = np.clip(adaptive_inertia, 0.1, 1.1)
        
        for i in range(self.np):
            r1, r2 = np.random.rand(2)
            cognitive = self.c1 * r1 * (self.personal_best[i] - self.population[i])
            social = self.c2 * r2 * (self.neighborhood_best[i] - self.population[i])
            
            velocity = adaptive_inertia * self.velocity[i] + cognitive + social
            
            velocity_mag = np.linalg.norm(velocity)
            if velocity_mag > self.max_velocity:
                velocity = velocity * (self.max_velocity / velocity_mag)
            
            trial_velocity[i] = velocity
            trials[i] = self.population[i] + velocity
        
        trials = np.clip(trials, self.bounds[0], self.bounds[1])
        trial_velocity = np.clip(trial_velocity, -self.max_velocity, self.max_velocity)
        
        return trials, trial_velocity
    
    def _compute_diversity(self):
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        max_dist = np.sqrt(self.dim) * (self.bounds[1] - self.bounds[0]) / 2
        return np.mean(distances) / max_dist if max_dist > 0 else 1.0
    
    def _update_personal_best_batch(self, trials, trial_fitness):
        improve_mask = (trial_fitness < self.personal_best_fitness) & np.isfinite(trial_fitness)
        self.personal_best_fitness[improve_mask] = trial_fitness[improve_mask]
        self.personal_best[improve_mask] = trials[improve_mask]
    
    def _update_neighborhood_best_batch(self):
        for i in range(self.np):
            neighbors = np.where(self.topology[i])[0]
            if len(neighbors) > 0:
                neighbor_fitness = self.personal_best_fitness[neighbors]
                best_neighbor_idx = neighbors[np.argmin(neighbor_fitness)]
                if self.personal_best_fitness[best_neighbor_idx] < self.neighborhood_best_fitness[i]:
                    self.neighborhood_best_fitness[i] = self.personal_best_fitness[best_neighbor_idx]
                    self.neighborhood_best[i] = self.personal_best[best_neighbor_idx]
    
    def _update_global_best_batch(self):
        valid_mask = np.isfinite(self.personal_best_fitness)
        if np.any(valid_mask):
            best_idx = np.argmin(self.personal_best_fitness[valid_mask])
            global_idx = np.where(valid_mask)[0][best_idx]
            if self.personal_best_fitness[global_idx] < self.global_best_fitness:
                self.global_best_fitness = self.personal_best_fitness[global_idx]
                self.global_best = self.personal_best[global_idx].copy()
                self.stagnation_counter = 0
    
    def _select_survivors_batch(self, trials, trial_velocity, trial_fitness):
        combined_pop = np.vstack([self.population, trials])
        combined_velocity = np.vstack([self.velocity, trial_velocity])
        
        combined_fitness = np.concatenate([
            self.personal_best_fitness,
            trial_fitness
        ])
        
        valid_mask = np.isfinite(combined_fitness)
        sorted_indices = np.argsort(combined_fitness[valid_mask])
        top_indices = np.where(valid_mask)[0][sorted_indices[:self.np]]
        
        self.population = combined_pop[top_indices]
        self.velocity = combined_velocity[top_indices]
        
        current_fitness = combined_fitness[top_indices]
        self.personal_best_fitness = np.minimum(self.personal_best_fitness[:self.np], current_fitness)
        
        for i in range(self.np):
            if current_fitness[i] < self.personal_best_fitness[i]:
                self.personal_best_fitness[i] = current_fitness[i]
                self.personal_best[i] = self.population[i]
    
    def _update_success_metric(self, trial_fitness):
        current_best = np.min(trial_fitness)
        initial_best = self.global_best_fitness
        
        if len(self.success_history) >= self.history_size:
            self.success_history.pop(0)
        
        if current_best < initial_best:
            self.success_history.append(1.0)
        else:
            decay = 0.9 ** self.stagnation_counter
            self.success_history.append(decay)
    
    def _adapt_inertia_weight(self):
        if len(self.success_history) >= 3:
            avg_success = np.mean(self.success_history[-3:])
            diversity = self._compute_diversity()
            
            if diversity < self.diversity_low:
                self.inertia = min(1.1, self.inertia * 1.05)
            elif diversity > self.diversity_high:
                self.inertia = max(0.3, self.inertia * 0.95)
            else:
                target_success = 0.2
                if avg_success > target_success:
                    self.inertia = max(0.3, self.inertia * 0.98)
                else:
                    self.inertia = min(1.1, self.inertia * 1.02)
    
    def _adapt_cognitive_social_factors(self):
        if len(self.success_history) >= 5:
            recent = np.array(self.success_history[-5:])
            
            cognitive_success = np.mean(recent[:2])
            social_success = np.mean(recent[-2:])
            
            if cognitive_success > 0.3:
                self.c1 = min(2.5, self.c1 * 1.02)
            elif cognitive_success < 0.1:
                self.c1 = max(0.5, self.c1 * 0.98)
            
            if social_success > 0.3:
                self.c2 = min(2.5, self.c2 * 1.02)
            elif social_success < 0.1:
                self.c2 = max(0.5, self.c2 * 0.98)
    
    def _check_stagnation(self):
        current_best = self.global_best_fitness
        if hasattr(self, '_prev_best'):
            if abs(current_best - self._prev_best) < 1e-10 * max(1.0, abs(self._prev_best)):
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        self._prev_best = current_best
    
    def _rewire_topology(self):
        fitness_scores = self.personal_best_fitness.copy()
        sorted_indices = np.argsort(fitness_scores)
        
        num_rewire = max(2, self.np // 10)
        
        for _ in range(num_rewire):
            i = np.random.randint(self.np)
            
            current_neighbors = np.where(self.topology[i])[0]
            if len(current_neighbors) > 0:
                worst_neighbor = current_neighbors[np.argmax(fitness_scores[current_neighbors])]
                self.topology[i, worst_neighbor] = False
                self.topology[worst_neighbor, i] = False
            
            candidates = sorted_indices[fitness_scores[sorted_indices] < fitness_scores[i]]
            if len(candidates) > 0:
                k = min(2, len(candidates))
                new_neighbors = np.random.choice(candidates, size=k, replace=False)
                for j in new_neighbors:
                    if j != i:
                        self.topology[i, j] = True
                        self.topology[j, i] = True
        
        self.stagnation_counter = 0
        self.adapt_trigger_count += 1
    
    def _adapt_parameters(self):
        self.c1 = self.c1_init + 0.2 * np.random.randn()
        self.c2 = self.c2_init + 0.2 * np.random.randn()
        self.c1 = np.clip(self.c1, 0.5, 2.5)
        self.c2 = np.clip(self.c2, 0.5, 2.5)
        
        self.inertia = 0.5 + 0.4 * np.random.rand()
        
        self.stagnation_counter = 0
        self.success_history = []
    
    def _restart_if_needed(self):
        self._initialize_population()
        self._initialize_topology()
        
        if self.global_best is not None:
            replace_idx = np.random.randint(self.np)
            self.population[replace_idx] = self.global_best.copy()
            self.personal_best[replace_idx] = self.global_best.copy()
        
        self.stagnation_counter = 0
        self.success_history = []
        self.inertia = 0.9
        self.c1 = self.c1_init
        self.c2 = self.c2_init
```