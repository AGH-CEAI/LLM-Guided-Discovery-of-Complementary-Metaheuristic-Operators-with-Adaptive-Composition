```python
import numpy as np

class QuantumTopologyParticleSwarm:
    """
    Quantum-Behaved Particle Swarm Optimization with Adaptive Topology.
    
    Key innovations:
    - Quantum-inspired position sampling replaces classical velocity
    - Adaptive ring topology rewires based on performance and diversity
    - Memory-guided exploration using historical best positions
    - Adaptive sampling radius that evolves with search progress
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(10 * dim, 300)  # Population size
        self.bounds = np.array([-100.0, 100.0])
        self.lb = self.bounds[0]
        self.ub = self.bounds[1]
        
        # Adaptive parameters
        self.inertia = 0.729
        self.cognitive = 1.494
        self.social = 1.494
        self.quantum_radius = 2.0
        
        # Memory for historical best positions
        self.memory_size = 5
        self.position_memory = None
        
        # Topology state
        self.topology = None  # Will be initialized as ring
        
        # Tracking
        self.f_opt = np.inf
        self.x_opt = None
        self.stagnation_count = 0
        self.max_stagnation = 50
        
    def __call__(self, func, stopping_condition):
        self._initialize_optimization(func)
        
        while not stopping_condition():
            self._update_personal_best_batch(func)
            self._update_topology_batch()
            self._update_global_best_batch()
            
            self._adapt_parameters_batch()
            
            trials = self._generate_quantum_trials_batch()
            trials = self._clip_to_bounds_batch(trials)
            
            if stopping_condition():
                break
                
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
            
            if len(trial_fitness) == 0:
                break
            
            self._update_memory_batch()
            
            self.population, self.fitness, self.trials, trial_fitness = \
                self._select_survivors_batch(
                    self.population, self.fitness, 
                    trials, trial_fitness
                )
            
            self._check_stagnation_and_restart()
            
            if stopping_condition():
                break
        
        return self.f_opt, self.x_opt
    
    def _initialize_optimization(self, func):
        """Initialize population, velocity, and tracking structures."""
        self.population = np.random.uniform(
            self.lb, self.ub, size=(self.np, self.dim)
        )
        self.population = np.clip(self.population, self.lb, self.ub)
        
        self.velocity = np.random.uniform(
            -0.1 * (self.ub - self.lb),
            0.1 * (self.ub - self.lb),
            size=(self.np, self.dim)
        )
        
        self.fitness = func(self.population)
        
        if len(self.fitness) < self.np:
            self.population = self.population[:len(self.fitness)]
            self.velocity = self.velocity[:len(self.fitness)]
            self.np = len(self.fitness)
        
        self.personal_best_pos = self.population.copy()
        self.personal_best_fit = self.fitness.copy()
        
        self.f_opt = np.min(self.fitness)
        self.x_opt = self.population[np.argmin(self.fitness)].copy()
        
        self._initialize_ring_topology()
        self._initialize_memory()
        
        self.stagnation_count = 0
    
    def _initialize_ring_topology(self):
        """Create initial ring topology connections."""
        self.topology = np.zeros((self.np, self.np), dtype=bool)
        for i in range(self.np):
            self.topology[i, (i - 1) % self.np] = True
            self.topology[i, i] = True
            self.topology[i, (i + 1) % self.np] = True
    
    def _initialize_memory(self):
        """Initialize memory of good positions."""
        self.position_memory = np.zeros((self.memory_size, self.dim))
        self.memory_fitness = np.full(self.memory_size, np.inf)
        self.memory_counter = 0
    
    def _update_memory_batch(self):
        """Update memory with current best positions."""
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < np.max(self.memory_fitness):
            replace_idx = np.argmax(self.memory_fitness)
            self.position_memory[replace_idx] = self.population[best_idx].copy()
            self.memory_fitness[replace_idx] = self.fitness[best_idx]
    
    def _update_personal_best_batch(self, func):
        """Update personal best positions where improvement occurred."""
        improved = self.fitness < self.personal_best_fit
        self.personal_best_fit[improved] = self.fitness[improved]
        self.personal_best_pos[improved] = self.population[improved].copy()
    
    def _update_topology_batch(self):
        """Adapt topology based on diversity and performance."""
        diversity = self._compute_diversity_batch()
        
        performance = np.zeros(self.np)
        for i in range(self.np):
            neighbors = np.where(self.topology[i])[0]
            if len(neighbors) > 0:
                neighbor_fitness = self.fitness[neighbors]
                performance[i] = np.mean(neighbor_fitness)
            else:
                performance[i] = self.fitness[i]
        
        if diversity < 0.1 * (self.ub - self.lb):
            self._rewire_topology_for_diversity()
        elif np.std(performance) < 0.1 * np.mean(np.abs(performance) + 1):
            self._rewire_topology_for_performance()
    
    def _rewire_topology_for_diversity(self):
        """Add long-range connections to increase diversity."""
        best_idx = np.argmin(self.fitness)
        worst_indices = np.argsort(self.fitness)[-max(3, self.np // 10):]
        
        for w_idx in worst_indices:
            if not self.topology[w_idx, best_idx]:
                self.topology[w_idx, best_idx] = True
                self.topology[best_idx, w_idx] = True
    
    def _rewire_topology_for_performance(self):
        """Add connections to high-performing particles."""
        good_indices = np.argsort(self.fitness)[:max(3, self.np // 5)]
        
        for i in range(self.np):
            if self.fitness[i] > np.median(self.fitness):
                connections_added = 0
                for g_idx in good_indices:
                    if not self.topology[i, g_idx] and connections_added < 2:
                        self.topology[i, g_idx] = True
                        self.topology[g_idx, i] = True
                        connections_added += 1
    
    def _update_global_best_batch(self):
        """Update neighborhood bests based on current topology."""
        self.neighborhood_best_pos = np.zeros((self.np, self.dim))
        self.neighborhood_best_fit = np.full(self.np, np.inf)
        
        for i in range(self.np):
            neighbors = np.where(self.topology[i])[0]
            neighbor_best_idx = neighbors[np.argmin(self.fitness[neighbors])]
            self.neighborhood_best_pos[i] = self.personal_best_pos[neighbor_best_idx]
            self.neighborhood_best_fit[i] = self.personal_best_fit[neighbor_best_idx]
    
    def _generate_quantum_trials_batch(self):
        """Generate quantum-inspired trial positions."""
        trials = np.zeros((self.np, self.dim))
        
        for i in range(self.np):
            center = self.neighborhood_best_pos[i]
            p_best = self.personal_best_pos[i]
            
            alpha = np.random.rand()
            attractor = alpha * p_best + (1 - alpha) * center
            
            quantum_offset = np.random.normal(0, self.quantum_radius, self.dim)
            trials[i] = attractor + quantum_offset
            
            if np.random.rand() < 0.2 and np.any(self.memory_fitness < np.inf):
                memory_idx = np.argmin(self.memory_fitness)
                memory_influence = np.random.rand(self.dim)
                trials[i] = trials[i] * (1 - 0.3 * memory_influence) + \
                           0.3 * memory_influence * self.position_memory[memory_idx]
        
        return trials
    
    def _adapt_parameters_batch(self):
        """Adapt inertia, quantum radius, and cognitive/social coefficients."""
        current_best = np.min(self.fitness)
        current_worst = np.max(self.fitness)
        
        if current_worst > current_best + 1e-10:
            spread_ratio = (current_best - np.mean(self.fitness)) / (current_worst - current_best + 1e-10)
        else:
            spread_ratio = 0.0
        
        self.inertia = 0.729 * (1.0 - 0.5 * spread_ratio)
        self.inertia = np.clip(self.inertia, 0.3, 0.9)
        
        self.cognitive = 1.494 * (0.5 + 0.5 * np.random.rand(self.np)[:, np.newaxis])
        self.social = 1.494 * (0.5 + 0.5 * np.random.rand(self.np)[:, np.newaxis])
        
        self.quantum_radius = 2.0 * self.inertia
        self.quantum_radius = np.clip(self.quantum_radius, 0.5, 5.0)
    
    def _clip_to_bounds_batch(self, positions):
        """Clip all positions to search bounds."""
        return np.clip(positions, self.lb, self.ub)
    
    def _compute_diversity_batch(self):
        """Compute population diversity as average pairwise distance."""
        if self.np < 2:
            return 0.0
        
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Select survivors based on fitness improvement."""
        improved = trial_fitness < fitness
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        new_population[improved] = trials[improved]
        new_fitness[improved] = trial_fitness[improved]
        
        self.velocity[improved] = trials[improved] - population[improved]
        
        return new_population, new_fitness, trials, trial_fitness
    
    def _check_stagnation_and_restart(self):
        """Check for stagnation and trigger restart if needed."""
        current_best = np.min(self.fitness)
        
        if current_best >= self.f_opt - 1e-10:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
            self.f_opt = current_best
            self.x_opt = self.population[np.argmin(self.fitness)].copy()
        
        if self.stagnation_count >= self.max_stagnation:
            self._adaptive_restart()
    
    def _adaptive_restart(self):
        """Perform adaptive restart with diversity injection."""
        self.stagnation_count = 0
        
        current_best_idx = np.argmin(self.fitness)
        current_best_pos = self.population[current_best_idx].copy()
        current_best_fit = self.fitness[current_best_idx]
        
        n_inject = max(5, self.np // 4)
        injection_range = 0.3 * (self.ub - self.lb)
        
        injected = np.random.uniform(
            current_best_pos - injection_range,
            current_best_pos + injection_range,
            size=(n_inject, self.dim)
        )
        injected = self._clip_to_bounds_batch(injected)
        
        self.population[:n_inject] = injected
        
        if n_inject < self.np:
            self.population[n_inject:] = np.random.uniform(
                self.lb, self.ub, size=(self.np - n_inject, self.dim)
            )
            self.population[n_inject:] = self._clip_to_bounds_batch(
                self.population[n_inject:]
            )
        
        self.velocity = np.random.uniform(
            -0.2 * (self.ub - self.lb),
            0.2 * (self.ub - self.lb),
            size=(self.np, self.dim)
        )
        
        self.fitness = np.full(self.np, np.inf)
        self.personal_best_fit = np.full(self.np, np.inf)
        
        self._initialize_ring_topology()
        
        self.f_opt = current_best_fit
        self.x_opt = current_best_pos.copy()
```