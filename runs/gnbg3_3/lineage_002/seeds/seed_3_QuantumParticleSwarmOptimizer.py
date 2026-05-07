import numpy as np


class QuantumParticleSwarmOptimizer:
    """
    Quantum-Inspired Particle Swarm with Adaptive Topology and Multi-Strategy Adaptation.
    
    Key features:
    - Quantum tunneling effect via position update with delta parameter
    - Adaptive ring topology that adjusts connectivity based on swarm diversity
    - Multiple adaptation targets: inertia weight, delta parameter, velocity clamping
    - Stagnation detection with strategic mutation restart
    - Batched operations for maximum efficiency
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 120), 300)  # Population size 120-300 for dim=30
        self.bounds = np.array([[-100.0, 100.0]] * dim)
        self.max_iters = kwargs.get('max_iters', 1000)
        
        # Topology parameters
        self.neighbor_connectivity = 2  # Ring connectivity (2 = immediate neighbors only)
        self.min_connectivity = 1
        self.max_connectivity = min(10, self.np // 10)
        
        # Adaptation state
        self.omega = 0.9
        self.delta = 0.5  # Quantum tunneling parameter
        self.fitness_history = []
        self.stagnation_counter = 0
        self.adaptation_count = 0
        
        # Best solution tracking
        self.g_best_fitness = np.inf
        self.g_best_position = None
        
    def __call__(self, func, stopping_condition):
        """Main optimization loop orchestrating batched operations."""
        self._initialize_population()
        
        iteration = 0
        while not stopping_condition():
            # Evaluate current population
            fitness = func(self.population)
            if len(fitness) < len(self.population):
                break
            
            # Update personal and global bests
            self._update_best_batch(fitness)
            
            # Check stagnation and adapt if needed
            self._adapt_inertia_weight(fitness)
            self._adapt_delta_parameter()
            self.stagnation_counter = self._detect_stagnation()
            
            if self.stagnation_counter >= 15:
                self._strategic_restart(func)
            
            # Build and evaluate trial population
            trials = self._generate_trials_batch()
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                break
            
            if stopping_condition():
                break
            
            # Selection and adaptation
            self._select_survivors_batch(trial_fitness, trials)
            self._adapt_connectivity()
            
            iteration += 1
            
            if iteration >= self.max_iters:
                break
        
        return self.g_best_fitness, self.g_best_position.copy()
    
    def _initialize_population(self):
        """Initialize population with uniform random sampling."""
        self.population = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], 
            size=(self.np, self.dim)
        )
        self.velocity = np.random.uniform(-20, 20, size=(self.np, self.dim))
        
        # Personal best tracking
        self.p_best_fitness = np.full(self.np, np.inf)
        self.p_best_position = self.population.copy()
        
        # Initialize inertia with problem-dependent scaling
        self.omega = 1.0 / (1.0 + 0.1 * self.dim)
        self.delta = 0.5
        
        # Reset tracking state
        self.g_best_fitness = np.inf
        self.g_best_position = None
        self.fitness_history = []
        self.stagnation_counter = 0
    
    def _velocity_update_batch(self):
        """Compute velocity with cognitive, social, and quantum components."""
        r1 = np.random.uniform(0, 1, size=(self.np, self.dim))
        r2 = np.random.uniform(0, 1, size=(self.np, self.dim))
        
        cognitive = self.omega * self.velocity
        social = 1.5 * r1 * (self.p_best_position - self.population)
        
        # Topology-based social influence
        neighbor_best = self._compute_neighborhood_best()
        social_component = 1.5 * r2 * (neighbor_best - self.population)
        
        # Quantum tunneling component
        quantum = self.delta * np.random.choice([-1, 1], size=(self.np, self.dim)) * np.log(
            1.0 / np.random.uniform(0.01, 1.0, size=(self.np, self.dim))
        )
        
        velocity = cognitive + social + social_component + quantum
        return velocity
    
    def _position_update_batch(self, velocity):
        """Update positions with velocity and quantum perturbation."""
        delta_pos = velocity.copy()
        
        # Add quantum position jump
        quantum_jump = self.delta * np.random.randn(self.np, self.dim)
        
        new_position = self.population + delta_pos + quantum_jump
        new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])
        
        return new_position
    
    def _generate_trials_batch(self):
        """Generate trial population using PSO with adaptations."""
        velocity = self._velocity_update_batch()
        trials = self._position_update_batch(velocity)
        return trials
    
    def _compute_neighborhood_best(self):
        """Compute neighborhood best positions using adaptive ring topology."""
        neighbor_best_pos = np.zeros((self.np, self.dim))
        
        for i in range(self.np):
            neighbors = []
            for offset in range(1, self.neighbor_connectivity + 1):
                neighbors.append((i - offset) % self.np)
                neighbors.append((i + offset) % self.np)
            
            neighbor_indices = list(set(neighbors))
            all_indices = neighbor_indices + [i]
            
            neighbor_fitness = self.p_best_fitness[all_indices]
            best_local_idx = all_indices[np.argmin(neighbor_fitness)]
            neighbor_best_pos[i] = self.p_best_position[best_local_idx]
        
        return neighbor_best_pos
    
    def _update_best_batch(self, fitness):
        """Update personal and global bests from fitness values."""
        improved = fitness < self.p_best_fitness
        self.p_best_fitness[improved] = fitness[improved]
        self.p_best_position[improved] = self.population[improved].copy()
        
        current_best_idx = np.argmin(self.p_best_fitness)
        if self.p_best_fitness[current_best_idx] < self.g_best_fitness:
            self.g_best_fitness = self.p_best_fitness[current_best_idx]
            self.g_best_position = self.p_best_position[current_best_idx].copy()
            self.stagnation_counter = 0
            self.adaptation_count += 1
        
        self.fitness_history.append(self.g_best_fitness)
        if len(self.fitness_history) > 50:
            self.fitness_history.pop(0)
    
    def _select_survivors_batch(self, trial_fitness, trials):
        """Select survivors between current population and trials."""
        combined_fitness = np.column_stack([self.p_best_fitness, trial_fitness])
        best_idx = np.argmin(combined_fitness, axis=1)
        
        # Update population with better solutions
        for i in range(self.np):
            if best_idx[i] == 1:  # Trial is better
                self.population[i] = trials[i].copy()
                self.p_best_fitness[i] = trial_fitness[i]
                self.p_best_position[i] = trials[i].copy()
    
    def _adapt_inertia_weight(self, fitness):
        """Adapt inertia weight based on fitness trajectory and diversity."""
        if len(self.fitness_history) < 5:
            return
        
        recent_improvement = self.fitness_history[-5] - self.fitness_history[-1]
        diversity = self._compute_diversity()
        
        # Sigmoid-based adaptation
        target_omega = 0.4 if recent_improvement > 0 else 0.729
        
        # Blend with diversity consideration
        if diversity < 0.1:
            target_omega = min(target_omega + 0.2, 0.95)
        
        self.omega = 0.9 * self.omega + 0.1 * target_omega
        self.omega = np.clip(self.omega, 0.1, 0.95)
    
    def _adapt_delta_parameter(self):
        """Adapt quantum tunneling delta based on stagnation."""
        if self.stagnation_counter > 10:
            self.delta = min(self.delta * 1.1, 2.0)
        else:
            self.delta = max(self.delta * 0.98, 0.1)
    
    def _adapt_connectivity(self):
        """Adapt neighborhood connectivity based on swarm diversity."""
        diversity = self._compute_diversity()
        
        if diversity < 0.05:
            self.neighbor_connectivity = min(
                self.neighbor_connectivity + 1, 
                self.max_connectivity
            )
        elif diversity > 0.3:
            self.neighbor_connectivity = max(
                self.neighbor_connectivity - 1,
                self.min_connectivity
            )
    
    def _compute_diversity(self):
        """Compute population diversity using coefficient of variation."""
        if self.g_best_position is None:
            return 1.0
        
        distances = np.linalg.norm(
            self.population - self.g_best_position, 
            axis=1
        )
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        if mean_dist < 1e-10:
            return 0.0
        
        return std_dist / (mean_dist + 1e-10)
    
    def _detect_stagnation(self):
        """Detect stagnation based on fitness history."""
        if len(self.fitness_history) < 20:
            return 0
        
        recent_window = self.fitness_history[-20:]
        if max(recent_window) - min(recent_window) < 1e-8:
            return 20
        
        return 0
    
    def _strategic_restart(self, func):
        """Perform strategic restart with guided mutation."""
        # Keep best solution
        best_idx = np.argmin(self.p_best_fitness)
        elite_position = self.p_best_position[best_idx].copy()
        elite_fitness = self.p_best_fitness[best_idx]
        
        # Compute perturbation based on fitness
        perturbation_scale = max(elite_fitness * 0.1, 1.0)
        
        # Generate new population around elite
        for i in range(self.np):
            if i != best_idx:
                angle = np.random.uniform(0, 2 * np.pi, self.dim)
                radius = np.random.exponential(perturbation_scale, size=self.dim)
                
                self.population[i] = elite_position + radius * np.cos(angle)
                self.population[i] = np.clip(
                    self.population[i], 
                    self.bounds[:, 0], 
                    self.bounds[:, 1]
                )
        
        self.stagnation_counter = 0
        self.delta = 0.5  # Reset quantum parameter
        self.neighbor_connectivity = 2  # Reset topology
