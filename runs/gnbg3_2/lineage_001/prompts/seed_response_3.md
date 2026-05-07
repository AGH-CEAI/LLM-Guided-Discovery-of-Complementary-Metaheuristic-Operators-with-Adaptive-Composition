```python
import numpy as np


class TopologyAwareQuantumParticleSwarm:
    """
    Particle Swarm Optimizer with ring topology neighborhoods, quantum-inspired
    perturbations, adaptive parameters, and stagnation-triggered restarts.
    All operations are batched (vectorized) for efficiency.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        self.np = min(max(4 * dim, 60), 300)  # Population size
        
        # PSO parameters with adaptive components
        self.inertia = 0.729
        self.cognitive = 1.49445
        self.social = 1.49445
        self.quantum_strength = 0.5
        
        # Adaptation state
        self._adapt_history = []
        self._successful_trials = 0
        self._total_trials = 0
        self._last_improvement_gen = 0
        
        # Diversity and restart
        self.diversity_threshold = 1e-6
        self.stagnation_limit = 150
        self.min_diversity = 1e-10
        
        # Topology: ring structure
        self._build_topology()
        
    def _build_topology(self):
        """Create ring neighborhood topology: each particle connects to 2 neighbors."""
        self.topology = np.arange(self.np)
        self.neighbor_left = np.roll(self.topology, 1)
        self.neighbor_right = np.roll(self.topology, -1)
        
    def __call__(self, func, stopping_condition):
        """Main optimization loop orchestrating batched operations."""
        population = self._initialize_population()
        fitness = self._evaluate_batch(func, population)
        
        personal_best_pos, personal_best_fit = self._init_personal_best(population, fitness)
        neighborhood_best_pos, neighborhood_best_fit = self._compute_neighborhood_best_batch(
            population, fitness
        )
        
        velocity = self._initialize_velocity(population)
        f_opt, x_opt = self._update_global_best(population, fitness)
        
        generation = 0
        while not stopping_condition():
            generation += 1
            
            # Generate trial positions using PSO + quantum perturbation
            trials = self._generate_trials_batch(population, velocity, 
                                                 personal_best_pos, neighborhood_best_pos)
            trials = self._clip_to_bounds_batch(trials)
            
            if len(trials) == 0:
                break
                
            trial_fitness = self._evaluate_batch(func, trials)
            
            if len(trial_fitness) < len(trials):
                valid_len = len(trial_fitness)
                trials = trials[:valid_len]
                trial_fitness = trial_fitness[:valid_len]
                if valid_len == 0:
                    break
                    
            if stopping_condition():
                break
                
            # Selection and update
            population, fitness, velocity, improved = self._select_and_update_batch(
                population, fitness, velocity, trials, trial_fitness,
                personal_best_pos, personal_best_fit
            )
            
            # Update personal and neighborhood bests
            personal_best_pos, personal_best_fit = self._update_personal_best_batch(
                population, fitness, personal_best_pos, personal_best_fit
            )
            neighborhood_best_pos, neighborhood_best_fit = self._compute_neighborhood_best_batch(
                population, fitness
            )
            
            # Track best solution
            f_opt, x_opt = self._update_global_best(population, fitness)
            
            # Adaptation and restart logic
            if improved:
                self._last_improvement_gen = generation
                
            self._adapt_parameters(improved, generation)
            
            diversity = self._compute_diversity(population)
            if self._should_restart(generation, diversity):
                population, velocity, personal_best_pos, personal_best_fit = \
                    self._restart_population(population, fitness, personal_best_pos)
                neighborhood_best_pos, neighborhood_best_fit = \
                    self._compute_neighborhood_best_batch(population, fitness)
                    
        return f_opt, x_opt
    
    def _initialize_population(self):
        """Sample initial population uniformly in search space."""
        return np.random.uniform(
            self.lower_bound, self.upper_bound, size=(self.np, self.dim)
        )
    
    def _initialize_velocity(self, population):
        """Initialize velocity as fraction of search space range."""
        v_range = (self.upper_bound - self.lower_bound) * 0.1
        return np.random.uniform(-v_range, v_range, size=(self.np, self.dim))
    
    def _evaluate_batch(self, func, population):
        """Evaluate fitness for entire population in single call."""
        return func(np.clip(population, self.lower_bound, self.upper_bound))
    
    def _init_personal_best(self, population, fitness):
        """Initialize personal best positions and fitness values."""
        return population.copy(), fitness.copy()
    
    def _clip_to_bounds_batch(self, population):
        """Ensure all candidates stay within search bounds."""
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def _generate_trials_batch(self, population, velocity, pbest, nbest):
        """Generate trial positions using PSO velocity + quantum perturbation."""
        # Standard PSO velocity update
        r1 = np.random.rand(self.np, self.dim)
        r2 = np.random.rand(self.np, self.dim)
        r3 = np.random.uniform(0, 1, size=(self.np, self.dim))
        
        cognitive_component = self.cognitive * r1 * (pbest - population)
        social_component = self.social * r2 * (nbest - population)
        
        velocity_new = self.inertia * velocity + cognitive_component + social_component
        
        # Quantum-inspired perturbation for exploration
        quantum_perturbation = self._quantum_perturbation_batch(population, r3)
        
        trials = population + velocity_new + quantum_perturbation
        
        # Update velocity reference
        velocity[:] = velocity_new
        
        return trials
    
    def _quantum_perturbation_batch(self, population, rand_vals):
        """Apply quantum-inspired delta based on distance to attractor."""
        # Quantum rotation angle based on random threshold
        quantum_mask = rand_vals < self.quantum_strength
        delta = np.random.randn(self.np, self.dim) * 2.0
        
        perturbation = np.where(quantum_mask, delta, 0.0)
        return perturbation
    
    def _select_and_update_batch(self, population, fitness, velocity, 
                                  trials, trial_fitness, pbest, pbest_fit):
        """Select survivors and update velocity based on improvement."""
        improved_mask = trial_fitness < fitness
        
        # Update population with improved candidates
        new_population = np.where(
            improved_mask[:, np.newaxis], trials, population
        )
        new_fitness = np.where(improved_mask, trial_fitness, fitness)
        
        # Update velocity only for improved particles
        new_velocity = velocity.copy()
        improved_idx = np.where(improved_mask)[0]
        if len(improved_idx) > 0:
            new_velocity[improved_idx] = new_population[improved_idx] - population[improved_idx]
        
        return new_population, new_fitness, new_velocity, np.any(improved_mask)
    
    def _update_personal_best_batch(self, population, fitness, pbest_pos, pbest_fit):
        """Update personal best positions where current fitness is better."""
        improved = fitness < pbest_fit
        new_pbest_pos = pbest_pos.copy()
        new_pbest_pos[improved] = population[improved]
        new_pbest_fit = np.where(improved, fitness, pbest_fit)
        return new_pbest_pos, new_pbest_fit
    
    def _compute_neighborhood_best_batch(self, population, fitness):
        """Compute best position in each particle's neighborhood (ring topology)."""
        nbest_pos = np.empty((self.np, self.dim))
        nbest_fit = np.empty(self.np)
        
        for i in range(self.np):
            neighbors = [self.neighbor_left[i], i, self.neighbor_right[i]]
            neighbor_fits = fitness[neighbors]
            best_neighbor = neighbors[np.argmin(neighbor_fits)]
            nbest_pos[i] = population[best_neighbor]
            nbest_fit[i] = fitness[best_neighbor]
            
        return nbest_pos, nbest_fit
    
    def _update_global_best(self, population, fitness):
        """Track the overall best solution found so far."""
        best_idx = np.argmin(fitness)
        return fitness[best_idx], population[best_idx].copy()
    
    def _adapt_parameters(self, improved, generation):
        """Adapt PSO parameters based on recent performance history."""
        self._total_trials += self.np
        
        if improved:
            self._successful_trials += self.np
            self._adapt_history.append(1)
        else:
            self._adapt_history.append(0)
            
        # Keep history window manageable
        if len(self._adapt_history) > 50:
            self._adapt_history = self._adapt_history[-50:]
            
        # Compute recent success rate
        if len(self._adapt_history) >= 10:
            success_rate = np.mean(self._adapt_history[-10:])
            
            # Adapt inertia: higher when successful, lower when struggling
            if success_rate > 0.5:
                self.inertia = min(0.95, self.inertia * 1.02)
            else:
                self.inertia = max(0.3, self.inertia * 0.98)
                
            # Adapt quantum strength inversely to success rate
            if success_rate < 0.3:
                self.quantum_strength = min(0.9, self.quantum_strength * 1.05)
            else:
                self.quantum_strength = max(0.1, self.quantum_strength * 0.97)
                
            # Adapt social/cognitive balance
            if generation - self._last_improvement_gen > 50:
                self.cognitive *= 1.01
                self.social *= 1.01
            else:
                self.cognitive = min(2.0, self.cognitive * 0.999)
                self.social = min(2.0, self.social * 0.999)
    
    def _compute_diversity(self, population):
        """Compute population diversity as average pairwise distance."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def _should_restart(self, generation, diversity):
        """Determine if population should be restarted due to stagnation."""
        generations_since_improvement = generation - self._last_improvement_gen
        
        if generations_since_improvement > self.stagnation_limit:
            return True
            
        if diversity < self.min_diversity:
            return True
            
        return False
    
    def _restart_population(self, population, fitness, pbest_pos):
        """Reinitialize population while preserving best solutions."""
        # Keep top 20% of particles
        n_keep = max(1, self.np // 5)
        keep_indices = np.argsort(fitness)[:n_keep]
        
        new_population = np.empty((self.np, self.dim))
        new_velocity = np.empty((self.np, self.dim))
        new_pbest_pos = np.empty((self.np, self.dim))
        
        # Preserve best particles
        new_population[:n_keep] = population[keep_indices]
        new_pbest_pos[:n_keep] = pbest_pos[keep_indices]
        
        # Reinitialize remaining particles with diversity
        if n_keep < self.np:
            best_particle = population[np.argmin(fitness)]
            spread = (self.upper_bound - self.lower_bound) * 0.5
            
            new_population[n_keep:] = np.random.uniform(
                best_particle - spread, best_particle + spread, 
                size=(self.np - n_keep, self.dim)
            )
            new_population[n_keep:] = np.clip(
                new_population[n_keep:], self.lower_bound, self.upper_bound
            )
            new_pbest_pos[n_keep:] = new_population[n_keep:]
            
        # Reset velocity
        v_range = (self.upper_bound - self.lower_bound) * 0.1
        new_velocity[:] = np.random.uniform(-v_range, v_range, size=(self.np, self.dim))
        
        # Reset adaptation tracking
        self._last_improvement_gen = 0
        self._adapt_history.clear()
        
        return new_population, new_velocity, new_pbest_pos, fitness[keep_indices]
```