```python
import numpy as np


class TopologyAdaptiveParticleSwarmOptimizer:
    """
    Particle Swarm Optimizer with adaptive topology switching and 
    opposition-based restart mechanism.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.bounds = np.array([-100.0, 100.0])
        self.search_range = self.bounds[1] - self.bounds[0]
        
        # Population sizing: 5*dim with reasonable bounds
        self.NP = min(max(5 * dim, 50), min(300, 10 * dim))
        
        # PSO control parameters
        self.max_generations = kwargs.get('max_generations', 1000)
        self.w_max = 0.9
        self.w_min = 0.2
        self.c1_max = 2.0
        self.c1_min = 0.5
        self.c2_max = 2.0
        self.c2_min = 0.5
        self.chi = 0.729
        
        # Current adaptive parameters
        self.w = self.w_max
        self.c1 = self.c1_max
        self.c2 = self.c2_max
        
        # Topology configuration
        self.available_topologies = ['ring', 'star', 'random_sparse', 'von_neumann']
        self.current_topology = None
        self.topology_switch_interval = 25
        
        # Diversity and stagnation thresholds
        self.min_diversity_ratio = 0.001
        self.stagnation_patience = 30
        self.replacement_ratio = 0.4
        
        # Internal state
        self.population = None
        self.velocity = None
        self.fitness = None
        self.pbest = None
        self.pbest_fitness = None
        self.gbest = None
        self.gbest_fitness = np.inf
        self.generation = 0
        self.eval_count = 0
        self.best_history = []
        self.diversity_history = []
        self.stagnation_counter = 0
        
    def __call__(self, func, stopping_condition):
        self._initialize_optimizer(func)
        
        while not stopping_condition():
            self._update_adaptive_parameters()
            
            if self.generation > 0 and self.generation % self.topology_switch_interval == 0:
                self._switch_topology()
            
            trials, trial_velocities = self._generate_trial_vectors_batch()
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                trial_fitness = self._handle_budget_exhaustion(trial_fitness, trials)
                if stopping_condition():
                    break
            
            if stopping_condition():
                break
                
            self.population, self.fitness, self.velocity = self._select_survivors_batch(
                trials, trial_fitness, trial_velocities
            )
            
            self._update_personal_and_global_best()
            
            current_diversity = self._compute_population_diversity()
            self.diversity_history.append(current_diversity)
            self.best_history.append(self.gbest_fitness)
            
            self._handle_stagnation_or_diversity(func)
            self.generation += 1
            
        return self.gbest_fitness, self.gbest.copy()
    
    def _initialize_optimizer(self, func):
        self.population = self._initialize_population()
        self.velocity = self._initialize_velocity()
        self.fitness = func(self.population)
        self.eval_count = len(self.fitness)
        
        valid_mask = np.isfinite(self.fitness)
        self.pbest = self.population[valid_mask].copy()
        self.pbest_fitness = self.fitness[valid_mask].copy()
        
        if len(self.pbest_fitness) > 0:
            best_idx = np.argmin(self.pbest_fitness)
            self.gbest = self.pbest[best_idx].copy()
            self.gbest_fitness = self.pbest_fitness[best_idx]
        
        self._switch_topology()
        self.generation = 0
        self.best_history = [self.gbest_fitness]
        self.diversity_history = []
        self.stagnation_counter = 0
    
    def _initialize_population(self):
        u = np.random.rand(self.NP, self.dim)
        population = self.bounds[0] + u * self.search_range
        return self._clip_to_bounds(population)
    
    def _clip_to_bounds(self, candidates):
        return np.clip(candidates, self.bounds[0], self.bounds[1])
    
    def _initialize_velocity(self):
        v_max = 0.2 * self.search_range
        velocity = np.random.uniform(-v_max, v_max, size=(self.NP, self.dim))
        return velocity
    
    def _create_connectivity_matrix(self, topology_type):
        adj = np.zeros((self.NP, self.NP), dtype=bool)
        
        if topology_type == 'ring':
            adj = self._build_ring_topology(adj)
        elif topology_type == 'star':
            adj = self._build_star_topology(adj)
        elif topology_type == 'random_sparse':
            adj = self._build_random_sparse_topology(adj)
        elif topology_type == 'von_neumann':
            adj = self._build_von_neumann_topology(adj)
        
        np.fill_diagonal(adj, True)
        return adj
    
    def _build_ring_topology(self, adj):
        for i in range(self.NP):
            adj[i, i] = True
            adj[i, (i - 1) % self.NP] = True
            adj[i, (i + 1) % self.NP] = True
        return adj
    
    def _build_star_topology(self, adj):
        hub = self.NP // 2
        adj[hub, :] = True
        adj[:, hub] = True
        return adj
    
    def _build_random_sparse_topology(self, adj):
        np.random.seed(hash(str(self.generation)) % 2**32)
        k = max(2, self.NP // 4)
        for i in range(self.NP):
            neighbors = np.random.choice(
                [j for j in range(self.NP) if j != i], 
                size=min(k, self.NP - 1), 
                replace=False
            )
            adj[i, neighbors] = True
        return adj
    
    def _build_von_neumann_topology(self, adj):
        grid_size = int(np.ceil(np.sqrt(self.NP)))
        for i in range(self.NP):
            row, col = i // grid_size, i % grid_size
            neighbors = [
                (row, (col - 1) % grid_size),
                (row, (col + 1) % grid_size),
                ((row - 1) % grid_size, col),
                ((row + 1) % grid_size, col)
            ]
            for nr, nc in neighbors:
                j = nr * grid_size + nc
                if j < self.NP:
                    adj[i, j] = True
        return adj
    
    def _switch_topology(self):
        new_topology = np.random.choice(self.available_topologies)
        self.current_topology = self._create_connectivity_matrix(new_topology)
    
    def _update_adaptive_parameters(self):
        progress = min(self.generation / max(self.max_generations, 1), 1.0)
        ease = progress ** 0.7
        
        self.w = self.w_max - (self.w_max - self.w_min) * ease
        self.c1 = self.c1_max - (self.c1_max - self.c1_min) * ease
        self.c2 = self.c2_min + (self.c2_max - self.c2_min) * ease
    
    def _generate_trial_vectors_batch(self):
        r1 = np.random.rand(self.NP, self.dim)
        r2 = np.random.rand(self.NP, self.dim)
        r3 = np.random.rand(self.NP, self.dim)
        
        neighborhood_best = self._compute_neighborhood_best_positions()
        
        social_choice = (r3[:, 0] < 0.7).reshape(-1, 1)
        lbest_component = np.where(social_choice, neighborhood_best, self.gbest)
        
        velocity = self.chi * (
            self.w * self.velocity +
            self.c1 * r1 * (self.pbest - self.population) +
            self.c2 * r2 * (lbest_component - self.population)
        )
        
        v_max = 0.25 * self.search_range
        velocity = np.clip(velocity, -v_max, v_max)
        
        trials = self.population + velocity
        trials = self._clip_to_bounds(trials)
        
        return trials, velocity
    
    def _compute_neighborhood_best_positions(self):
        neighborhood_best = np.zeros((self.NP, self.dim))
        
        for i in range(self.NP):
            neighbors = self.current_topology[i]
            neighbor_fitness = np.where(neighbors, self.pbest_fitness, np.inf)
            best_neighbor_idx = np.argmin(neighbor_fitness)
            neighborhood_best[i] = self.pbest[best_neighbor_idx]
        
        return neighborhood_best
    
    def _select_survivors_batch(self, trials, trial_fitness, trial_velocities):
        improvement_mask = trial_fitness < self.fitness
        improvement_mask &= np.isfinite(trial_fitness)
        
        new_population = self.population.copy()
        new_velocity = self.velocity.copy()
        new_fitness = self.fitness.copy()
        
        new_population[improvement_mask] = trials[improvement_mask]
        new_velocity[improvement_mask] = trial_velocities[improvement_mask]
        new_fitness[improvement_mask] = trial_fitness[improvement_mask]
        
        return new_population, new_fitness, new_velocity
    
    def _update_personal_and_global_best(self):
        should_update_pbest = (self.fitness < self.pbest_fitness) & np.isfinite(self.fitness)
        
        self.pbest[should_update_pbest] = self.population[should_update_pbest]
        self.pbest_fitness[should_update_pbest] = self.fitness[should_update_pbest]
        
        if np.any(should_update_pbest):
            current_best_idx = np.argmin(self.pbest_fitness)
            if self.pbest_fitness[current_best_idx] < self.gbest_fitness:
                self.gbest = self.pbest[current_best_idx].copy()
                self.gbest_fitness = self.pbest_fitness[current_best_idx]
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
        else:
            self.stagnation_counter += 1
    
    def _compute_population_diversity(self):
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        mean_distance = np.mean(distances)
        diversity = 2.0 * mean_distance / self.search_range
        return diversity
    
    def _handle_budget_exhaustion(self, partial_fitness, trials):
        n_received = len(partial_fitness)
        if n_received < len(trials):
            valid_fitness = np.full(len(trials), np.inf)
            valid_fitness[:n_received] = partial_fitness
        else:
            valid_fitness = partial_fitness
        return valid_fitness
    
    def _handle_stagnation_or_diversity(self, func):
        current_diversity = self.diversity_history[-1] if self.diversity_history else 1.0
        is_stagnated = self.stagnation_counter >= self.stagnation_patience
        is_low_diversity = current_diversity < self.min_diversity_ratio
        
        if is_stagnated or is_low_diversity:
            self._perform_guided_restart(func)
    
    def _perform_guided_restart(self, func):
        n_replace = max(1, int(self.NP * self.replacement_ratio))
        replace_indices = np.random.choice(self.NP, size=n_replace, replace=False)
        
        best_particle = self.gbest
        range_size = self.search_range
        
        for idx in replace_indices:
            if np.random.rand() < 0.5:
                reflected = 2.0 * best_particle - self.population[idx]
                reflected = self._clip_to_bounds(reflected)
                
                fitness_original = func(self.population[idx:idx+1])[0]
                fitness_reflected = func(reflected.reshape(1, -1))[0]
                
                if np.isfinite(fitness_reflected) and fitness_reflected < fitness_original:
                    self.population[idx] = reflected
                    self.fitness[idx] = fitness_reflected
                else:
                    self.population[idx] = self._initialize_single_particle()
                    self.fitness[idx] = func(self.population[idx:idx+1])[0]
            else:
                self.population[idx] = self._initialize_single_particle()
                self.fitness[idx] = func(self.population[idx:idx+1])[0]
        
        self.pbest[replace_indices] = self.population[replace_indices].copy()
        self.pbest_fitness[replace_indices] = self.fitness[replace_indices].copy()
        
        self.stagnation_counter = 0
        self._switch_topology()
    
    def _initialize_single_particle(self):
        u = np.random.rand(self.dim)
        return self.bounds[0] + u * self.search_range
```