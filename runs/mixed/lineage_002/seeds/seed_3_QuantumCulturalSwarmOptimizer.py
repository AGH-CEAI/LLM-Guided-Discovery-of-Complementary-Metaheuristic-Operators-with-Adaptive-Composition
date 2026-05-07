import numpy as np


class QuantumCulturalSwarmOptimizer:
    """
    Quantum-Behaved Cultural Swarm Optimizer (QCSO).
    
    Combines:
    - Quantum potential well model for position updates (global attractor + mean best)
    - Small-world neighborhood topology with adaptive rewiring based on diversity
    - Lévy-flight inspired perturbations for exploration
    - Adaptive parameter control (inertia, quantum radius)
    - Cultural memory tracking for historical best positions
    - Stagnation detection with adaptive restarts
    """
    
    def __init__(self, dim, npop=None, w_init=0.9, w_end=0.4, c1=2.0, c2=2.0,
                 quantum_beta=1.0, levy_scale=0.01, bounds=(-100, 100)):
        self.dim = dim
        self.bounds = bounds
        self.npop = npop if npop is not None else max(50, min(5 * dim, 300))
        
        # Inertia weight parameters
        self.w_init = w_init
        self.w_end = w_end
        
        # Acceleration coefficients
        self.c1 = c1
        self.c2 = c2
        
        # Quantum model parameter
        self.quantum_beta = quantum_beta
        
        # Lévy flight scale
        self.levy_scale = levy_scale
        
        # State variables
        self.population = None
        self.velocity = None
        self.fitness = None
        self.personal_best_pos = None
        self.personal_best_fit = None
        self.global_best_pos = None
        self.global_best_fit = np.inf
        
        # Cultural memory (history of best positions)
        self.culture_memory = None
        self.culture_fitness = None
        self.culture_counter = None
        
        # Neighborhood topology
        self.neighbors = None
        self.local_best_pos = None
        self.local_best_fit = None
        
        # Diversity tracking
        self.diversity_history = []
        self.stagnation_count = 0
        
        # Iteration counter
        self.iteration = 0
        
    def _initialize_population(self):
        """Initialize population uniformly in bounds."""
        lo, hi = self.bounds
        self.population = np.random.uniform(lo, hi, (self.npop, self.dim))
        self.velocity = np.random.uniform(-(hi - lo), (hi - lo), (self.npop, self.dim))
        
        self.personal_best_pos = self.population.copy()
        self.personal_best_fit = np.full(self.npop, np.inf)
        
        # Initialize local bests for ring topology
        self.local_best_pos = self.population.copy()
        self.local_best_fit = np.full(self.npop, np.inf)
        
        # Initialize culture memory
        self.culture_memory = np.zeros((10, self.dim))
        self.culture_fitness = np.full(10, np.inf)
        self.culture_counter = np.zeros(10, dtype=int)
        
        # Initialize neighborhood (ring topology)
        self._initialize_ring_topology()
        
        self.iteration = 0
        self.diversity_history = []
        self.stagnation_count = 0
        
    def _initialize_ring_topology(self):
        """Initialize ring neighborhood topology."""
        self.neighbors = np.zeros((self.npop, 3), dtype=int)
        for i in range(self.npop):
            self.neighbors[i] = [(i - 1) % self.npop, i, (i + 1) % self.npop]
            
    def _clip_to_bounds_batch(self, pop):
        """Clip all individuals to search bounds."""
        lo, hi = self.bounds
        return np.clip(pop, lo, hi)
        
    def _eval_wrapper(self, pop):
        """Evaluate population with budget awareness."""
        clipped = self._clip_to_bounds_batch(pop)
        fitness = yield clipped
        return fitness
        
    def _evaluate_batch(self, pop):
        """Evaluate population batch and return fitness."""
        return self._clip_to_bounds_batch(pop)
        
    def _update_personal_best_batch(self):
        """Update personal best positions for improved individuals."""
        improved = self.fitness < self.personal_best_fit
        self.personal_best_pos[improved] = self.population[improved]
        self.personal_best_fit[improved] = self.fitness[improved]
        
    def _update_local_best_batch(self):
        """Update local best for each neighborhood."""
        for i in range(self.npop):
            nb_indices = self.neighbors[i]
            nb_fits = self.personal_best_fit[nb_indices]
            best_local_idx = nb_indices[np.argmin(nb_fits)]
            if self.personal_best_fit[best_local_idx] < self.local_best_fit[i]:
                self.local_best_pos[i] = self.personal_best_pos[best_local_idx]
                self.local_best_fit[i] = self.personal_best_fit[best_local_idx]
                
    def _update_global_best_batch(self):
        """Update global best position."""
        best_idx = np.argmin(self.personal_best_fit)
        if self.personal_best_fit[best_idx] < self.global_best_fit:
            self.global_best_pos = self.personal_best_pos[best_idx].copy()
            self.global_best_fit = self.personal_best_fit[best_idx]
            
    def _update_culture_memory_batch(self):
        """Update cultural memory with new best positions."""
        if self.global_best_fit >= self.culture_fitness[0]:
            return
            
        # Shift memory and insert new best
        if self.global_best_fit < self.culture_fitness[-1]:
            self.culture_memory[:-1] = self.culture_memory[1:]
            self.culture_fitness[:-1] = self.culture_fitness[1:]
            self.culture_counter[:-1] = self.culture_counter[1:]
            
            insert_pos = np.searchsorted(self.culture_fitness[1:], self.global_best_fit)
            self.culture_memory[insert_pos] = self.global_best_pos.copy()
            self.culture_fitness[insert_pos] = self.global_best_fit
            self.culture_counter[insert_pos] = 1
            
    def _compute_adaptive_inertia(self):
        """Compute adaptive inertia weight based on iteration."""
        progress = self.iteration / max(1, 1000)
        w = self.w_init - (self.w_init - self.w_end) * min(progress, 1.0)
        return w
        
    def _compute_quantum_radius(self):
        """Compute quantum potential radius (beta decreases over time)."""
        base_radius = self.quantum_beta * (1.0 - self.iteration / max(1, 2000))
        return max(0.1, base_radius)
        
    def _compute_mean_best_position(self):
        """Compute mean of personal best positions."""
        return np.mean(self.personal_best_pos, axis=0)
        
    def _generate_levy_step(self, size):
        """Generate Lévy flight step using Mantegna algorithm."""
        sigma_u = (0.696 * self.levy_scale) ** (1.0 / 1.5)
        sigma_v = 1.0 ** (1.0 / 1.5)
        
        u = np.random.normal(0, sigma_u, size)
        v = np.random.normal(0, sigma_v, size)
        
        step = u / (np.abs(v) ** (1.0 / 1.5))
        return step
        
    def _mutate_quantum_attractor_batch(self, inertia):
        """Generate mutation vectors using quantum attractor model."""
        mean_best = self._compute_mean_best_position()
        quantum_r = self._compute_quantum_radius()
        
        # Quantum mutation: attract toward mean best with random direction
        mutation = np.zeros_like(self.population)
        
        for i in range(self.npop):
            # Direction from particle to mean best
            to_mean = mean_best - self.population[i]
            dist_mean = np.linalg.norm(to_mean)
            
            if dist_mean > 1e-10:
                direction = to_mean / dist_mean
                # Quantum collapse: particle appears within radius
                quantum_shift = np.random.uniform(-1, 1) * quantum_r * dist_mean
                mutation[i] = direction * quantum_shift
            else:
                mutation[i] = np.zeros(self.dim)
                
        # Add cultural influence from memory
        if self.culture_fitness[0] < np.inf:
            best_culture = self.culture_memory[0]
            culture_influence = 0.1 * (best_culture - self.population)
            mutation += culture_influence
            
        return mutation
        
    def _mutate_local_best_batch(self):
        """Generate mutation toward local best positions."""
        return self.local_best_pos - self.population
        
    def _mutate_global_best_batch(self):
        """Generate mutation toward global best position."""
        return self.global_best_pos - self.population
        
    def _update_velocity_batch(self, inertia):
        """Update velocity using adaptive PSO with quantum influence."""
        mutation_quantum = self._mutate_quantum_attractor_batch(inertia)
        mutation_local = self._mutate_local_best_batch()
        mutation_global = self._mutate_global_best_batch()
        
        # Combine quantum and classical velocity updates
        r1 = np.random.uniform(0, 1, (self.npop, self.dim))
        r2 = np.random.uniform(0, 1, (self.npop, self.dim))
        r3 = np.random.uniform(0, 1, (self.npop, self.dim))
        
        # Inertia term
        velocity_inertia = inertia * self.velocity
        
        # Cognitive term (toward personal best)
        velocity_cognitive = self.c1 * r1 * (self.personal_best_pos - self.population)
        
        # Social term (toward local best)
        velocity_local = 0.5 * r2 * mutation_local
        
        # Global term (toward global best)
        velocity_global = 0.5 * r3 * mutation_global
        
        # Quantum term
        velocity_quantum = 0.3 * mutation_quantum
        
        self.velocity = velocity_inertia + velocity_cognitive + velocity_global + velocity_quantum
        self.velocity = np.clip(self.velocity, -200, 200)
        
    def _apply_levy_perturbation_batch(self, rate=0.2):
        """Apply Lévy-flight perturbations to a subset of population."""
        n_perturbed = int(rate * self.npop)
        if n_perturbed == 0:
            return
            
        indices = np.random.choice(self.npop, n_perturbed, replace=False)
        
        for idx in indices:
            levy_steps = self._generate_levy_step(self.dim)
            self.population[idx] += levy_steps * 10.0
            
    def _crossover_batch(self, other_pop):
        """Perform binomial crossover between current and trial population."""
        cr = np.random.uniform(0.7, 0.95, self.npop)
        cross_mask = np.random.random((self.npop, self.dim)) < cr[:, np.newaxis]
        
        # Ensure at least one dimension is crossed
        cross_mask[np.arange(self.npop), np.random.randint(0, self.dim, self.npop)] = True
        
        trials = np.where(cross_mask, other_pop, self.population)
        return trials
        
    def _select_survivors_batch(self, trials, trial_fitness):
        """Select survivors between current population and trials."""
        better_mask = trial_fitness < self.fitness
        
        new_population = self.population.copy()
        new_population[better_mask] = trials[better_mask]
        
        new_fitness = self.fitness.copy()
        new_fitness[better_mask] = trial_fitness[better_mask]
        
        # Also update velocity for improved individuals
        improved_mask = better_mask
        velocity_change = trials[improved_mask] - self.population[improved_mask]
        self.velocity[improved_mask] = 0.5 * velocity_change + 0.5 * self.velocity[improved_mask]
        
        return new_population, new_fitness
        
    def _compute_diversity(self):
        """Compute population diversity as average pairwise distance."""
        if self.population is None or len(self.population) < 2:
            return 0.0
            
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
        
    def _rewire_topology_adaptive(self):
        """Adaptively rewire neighborhood connections based on diversity."""
        current_diversity = self._compute_diversity()
        self.diversity_history.append(current_diversity)
        
        if len(self.diversity_history) > 20:
            recent_div = np.mean(self.diversity_history[-20:])
            
            # If diversity is low, rewire to increase exploration
            if recent_div < 5.0 and self.stagnation_count > 3:
                # Randomly rewire some connections
                n_rewire = max(1, self.npop // 10)
                rewire_indices = np.random.choice(self.npop, n_rewire, replace=False)
                
                for idx in rewire_indices:
                    # Replace one random neighbor with a random particle
                    neighbor_slot = np.random.randint(3)
                    new_neighbor = np.random.randint(self.npop)
                    self.neighbors[idx, neighbor_slot] = new_neighbor
                    
                self.stagnation_count = 0
                
    def _check_stagnation(self):
        """Check for stagnation and trigger restart if needed."""
        current_diversity = self._compute_diversity()
        
        if len(self.diversity_history) > 50:
            old_best = self.culture_fitness[0] if self.culture_fitness[0] < np.inf else self.global_best_fit
            if abs(old_best - self.global_best_fit) < 1e-8:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
                
            if self.stagnation_count > 10 or current_diversity < 1.0:
                return True
                
        return False
        
    def _restart_if_stagnant(self):
        """Perform adaptive restart to escape local optima."""
        if not self._check_stagnation():
            return False
            
        # Save best solution
        best_saved = self.global_best_pos.copy()
        best_fit_saved = self.global_best_fit
        
        # Reinitialize portion of population
        lo, hi = self.bounds
        n_replace = max(10, self.npop // 3)
        replace_indices = np.random.choice(self.npop, n_replace, replace=False)
        
        self.population[replace_indices] = np.random.uniform(lo, hi, (n_replace, self.dim))
        self.velocity[replace_indices] = np.random.uniform(-(hi - lo), (hi - lo), (n_replace, self.dim))
        
        # Restore best solution to population
        restore_idx = np.random.randint(self.npop)
        self.population[restore_idx] = best_saved
        self.velocity[restore_idx] = np.zeros(self.dim)
        
        self.stagnation_count = 0
        self.diversity_history = []
        
        return True
        
    def __call__(self, func, stopping_condition):
        """
        Run optimization.
        
        Args:
            func: Fitness function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        self._initialize_population()
        
        # Initial evaluation
        self.fitness = func(self.population)
        if len(self.fitness) < self.npop:
            # Budget exhausted
            best_idx = np.argmin(self.fitness[:len(self.fitness)])
            return self.fitness[best_idx], self.population[best_idx]
            
        self._update_personal_best_batch()
        self._update_local_best_batch()
        self._update_global_best_batch()
        self._update_culture_memory_batch()
        
        # Main optimization loop
        while not stopping_condition():
            self.iteration += 1
            
            # Adaptive inertia weight
            inertia = self._compute_adaptive_inertia()
            
            # Update velocities
            self._update_velocity_batch(inertia)
            
            # Update positions
            trial_population = self.population + self.velocity
            
            # Apply Lévy perturbation occasionally
            if self.iteration % 10 == 0:
                self._apply_levy_perturbation_batch(rate=0.15)
                
            # Clip to bounds
            trial_population = self._clip_to_bounds_batch(trial_population)
            
            # Evaluate trials
            trial_fitness = func(trial_population)
            
            # Handle budget exhaustion
            if len(trial_fitness) < len(trial_population):
                if len(trial_fitness) > 0:
                    trial_population = trial_population[:len(trial_fitness)]
                    trial_fitness = trial_fitness[:len(trial_fitness)]
                break
                
            if stopping_condition():
                break
                
            # Selection
            self.population, self.fitness = self._select_survivors_batch(
                trial_population, trial_fitness
            )
            
            # Update bests
            self._update_personal_best_batch()
            self._update_local_best_batch()
            self._update_global_best_batch()
            self._update_culture_memory_batch()
            
            # Adaptive topology rewiring
            if self.iteration % 20 == 0:
                self._rewire_topology_adaptive()
                
            # Check for stagnation and restart
            if self.iteration % 50 == 0:
                self._restart_if_stagnant()
                
        return self.global_best_fit, self.global_best_pos
