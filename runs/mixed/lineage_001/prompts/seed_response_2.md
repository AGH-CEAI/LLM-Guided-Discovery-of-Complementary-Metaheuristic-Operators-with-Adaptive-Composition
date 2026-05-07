```python
import numpy as np


class ArchipelagoDifferentialParticleSwarm:
    """
    Archipelago Differential Particle Swarm (ADPSO) optimizer.
    
    Key design decisions:
    - Island model (subpopulations) with ring topology migration
    - Velocity-based mutation inspired by PSO combined with DE-style differential terms
    - Adaptive step sizes and crossover rates per island based on success rates
    - Diversity-driven restart mechanism for stagnant islands
    - Batch evaluation for all individuals across all islands per generation
    """
    
    def __init__(self, dim, n_islands=4, np_per_island=None, *args, **kwargs):
        self.dim = dim
        self.n_islands = n_islands
        # Population size per island: 5*dim total, divided among islands
        self.np_per_island = np_per_island if np_per_island else max(20, 5 * dim // n_islands)
        self.total_pop = self.np_per_island * n_islands
        
        # Bounds
        self.lb = -100.0
        self.ub = 100.0
        
        # Initialize adaptive parameters per island
        self.step_sizes = np.full(n_islands, 0.5)
        self.crossover_rates = np.full(n_islands, 0.7)
        
        # Tracking
        self.best_fitness_history = []
        
    def __call__(self, func, stopping_condition):
        """Main entry point: optimize func until stopping_condition is met."""
        # Initialize population across all islands
        population = self._initialize_population()
        fitness = self._eval_wrapper(func, population)
        
        # Track best across all islands
        best_idx = np.argmin(fitness)
        f_opt = fitness[best_idx]
        x_opt = population[best_idx].copy()
        
        # Per-island best tracking
        island_best_fitness = np.full(self.n_islands, np.inf)
        island_best_positions = np.zeros((self.n_islands, self.dim))
        
        # Per-island success tracking for adaptation
        recent_success = np.zeros(self.n_islands)
        recent_trials = np.zeros(self.n_islands)
        
        # Velocity initialization
        velocity = self._initialize_velocity(population)
        
        # Main optimization loop
        while not stopping_condition():
            # Evaluate current population
            fitness = self._eval_wrapper(func, population)
            
            # Handle budget exhaustion
            if len(fitness) < len(population):
                population = population[:len(fitness)]
                velocity = velocity[:len(fitness)]
                fitness = fitness[:len(fitness)]
                if len(fitness) == 0:
                    break
            
            # Update per-island bests
            island_best_fitness, island_best_positions = self._update_island_best(
                population, fitness, island_best_fitness, island_best_positions
            )
            
            # Check global best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < f_opt:
                f_opt = fitness[current_best_idx]
                x_opt = population[current_best_idx].copy()
            
            # Build trial population using velocity-based mutation
            trial_population = self._build_trial_population_batch(
                population, velocity, island_best_fitness, island_best_positions
            )
            
            # Check stopping after trial generation
            if stopping_condition():
                break
            
            # Evaluate trials in batch
            trial_fitness = self._eval_wrapper(func, trial_population)
            
            # Handle budget exhaustion
            if len(trial_fitness) < len(trial_population):
                trial_population = trial_population[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
                if len(trial_fitness) == 0:
                    break
            
            # Track successes for adaptation
            for i in range(self.n_islands):
                start, end = i * self.np_per_island, (i + 1) * self.np_per_island
                island_pop = population[start:end]
                island_trials = trial_population[start:end]
                island_fit = fitness[start:end]
                island_trial_fit = trial_fitness[start:end]
                
                successes = np.sum(island_trial_fit < island_fit)
                recent_success[i] += successes
                recent_trials[i] += len(island_trial_fit)
            
            # Select survivors (DE-style selection)
            population, fitness, velocity = self._select_survivors_batch(
                population, fitness, trial_population, trial_fitness, velocity
            )
            
            # Adapt step size and crossover rate per island
            self._adapt_parameters_batch(recent_success, recent_trials)
            
            # Reset success counters periodically
            if np.sum(recent_trials) > self.total_pop * 5:
                recent_success.fill(0)
                recent_trials.fill(0)
            
            # Periodic migration between islands
            if len(self.best_fitness_history) % 5 == 0:
                population, velocity = self._migrate_between_islands(
                    population, velocity, island_best_positions
                )
            
            # Restart stagnant islands
            population, velocity = self._restart_stagnant_islands(
                population, velocity, func
            )
            
            # Track diversity for monitoring
            diversity = self._compute_diversity(population)
            
            # Update history
            self.best_fitness_history.append(f_opt)
        
        return f_opt, x_opt
    
    def _initialize_population(self):
        """Initialize uniform random population across all islands."""
        pop = np.random.uniform(
            self.lb, self.ub, size=(self.total_pop, self.dim)
        )
        return self._clip_to_bounds_batch(pop)
    
    def _initialize_velocity(self, population):
        """Initialize velocity vectors for all individuals."""
        range_val = self.ub - self.lb
        velocity = np.random.uniform(
            -0.1 * range_val, 0.1 * range_val, size=(self.total_pop, self.dim)
        )
        return velocity
    
    def _eval_wrapper(self, func, population):
        """Evaluate fitness for entire population batch."""
        # Clip to bounds before evaluation
        population = self._clip_to_bounds_batch(population)
        fitness = func(population)
        # Handle NaN values
        nan_mask = np.isnan(fitness)
        if np.any(nan_mask):
            fitness[nan_mask] = np.inf
        return fitness
    
    def _clip_to_bounds_batch(self, population):
        """Clip all individuals to search bounds."""
        return np.clip(population, self.lb, self.ub)
    
    def _update_island_best(self, population, fitness, island_best_fitness, island_best_positions):
        """Update best position found in each island."""
        for i in range(self.n_islands):
            start, end = i * self.np_per_island, (i + 1) * self.np_per_island
            island_fit = fitness[start:end]
            min_idx = np.argmin(island_fit)
            if island_fit[min_idx] < island_best_fitness[i]:
                island_best_fitness[i] = island_fit[min_idx]
                island_best_positions[i] = population[start + min_idx].copy()
        return island_best_fitness, island_best_positions
    
    def _build_trial_population_batch(self, population, velocity, island_best_fitness, island_best_positions):
        """Build trial population using velocity-based mutation across all islands."""
        trials = np.empty_like(population)
        
        for i in range(self.n_islands):
            start, end = i * self.np_per_island, (i + 1) * self.np_per_island
            island_pop = population[start:end]
            island_vel = velocity[start:end]
            
            # Get island-specific parameters
            step_size = self.step_sizes[i]
            cr = self.crossover_rates[i]
            
            # Get reference points for mutation
            global_best_idx = np.argmin(island_best_fitness)
            
            # Velocity-based mutation: combine inertia, cognitive, social terms
            mutated = self._velocity_mutate_batch(
                island_pop, island_vel, island_best_positions, 
                global_best_idx, step_size, i
            )
            
            # Crossover with current population
            trials[start:end] = self._crossover_batch(island_pop, mutated, cr)
        
        return self._clip_to_bounds_batch(trials)
    
    def _velocity_mutate_batch(self, pop, vel, island_best_pos, global_best_idx, step_size, island_id):
        """Velocity-based mutation combining PSO and DE concepts."""
        n = len(pop)
        
        # Inertia weight (decreases over time)
        w = 0.9 - 0.3 * min(1.0, len(self.best_fitness_history) / 100)
        
        # Cognitive component: attraction to personal best
        pbest = island_best_pos[island_id]
        cognitive = np.tile(pbest, (n, 1)) - pop
        
        # Social component: attraction to best island (ring topology)
        best_neighbor_idx = (island_id - 1) % self.n_islands
        social = np.tile(island_best_pos[best_neighbor_idx], (n, 1)) - pop
        
        # DE-style differential term: random difference between two individuals
        idx1, idx2 = np.random.choice(n, 2, replace=False)
        differential = pop[idx1] - pop[idx2]
        
        # Update velocity
        new_vel = w * vel + step_size * cognitive + step_size * 0.5 * social
        
        # Apply mutation: position + differential + velocity influence
        mutated = pop + step_size * differential + new_vel * 0.5
        
        return mutated
    
    def _crossover_batch(self, target, mutant, cr):
        """Binomial crossover between target and mutant."""
        n, dim = target.shape
        trials = np.empty_like(target)
        
        # Random crossover mask
        mask = np.random.random((n, dim)) < cr
        
        # Ensure at least one dimension comes from mutant
        enforce_idx = np.random.randint(dim, size=n)
        mask[np.arange(n), enforce_idx] = True
        
        trials = np.where(mask, mutant, target)
        return trials
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness, velocity):
        """DE-style selection: keep better of target and trial for each individual."""
        # Determine which trials improved
        improved = trial_fitness < fitness
        
        # Update population with successful trials
        new_pop = np.where(improved[:, np.newaxis], trials, population)
        new_fit = np.where(improved, trial_fitness, fitness)
        
        # Update velocity: keep old velocity for non-improvers
        new_vel = np.where(improved[:, np.newaxis], 
                          velocity * 0.5,  # dampen velocity on success
                          velocity * 1.1)  # increase exploration on failure
        
        return new_pop, new_fit, new_vel
    
    def _adapt_parameters_batch(self, recent_success, recent_trials):
        """Adapt step size and crossover rate per island based on success rates."""
        for i in range(self.n_islands):
            if recent_trials[i] > 0:
                success_rate = recent_success[i] / recent_trials[i]
                
                # Adapt step size: increase if success rate is low, decrease if high
                if success_rate > 0.2:
                    self.step_sizes[i] *= 0.9
                elif success_rate < 0.05:
                    self.step_sizes[i] *= 1.1
                
                # Adapt crossover rate: slight random walk
                self.crossover_rates[i] += np.random.uniform(-0.05, 0.05)
                
                # Keep in valid range
                self.step_sizes[i] = np.clip(self.step_sizes[i], 0.01, 2.0)
                self.crossover_rates[i] = np.clip(self.crossover_rates[i], 0.3, 0.95)
    
    def _migrate_between_islands(self, population, velocity, island_best_positions):
        """Ring topology migration: each island receives best from previous island."""
        # Find best individual in each island
        migrants = []
        migrant_velocities = []
        
        for i in range(self.n_islands):
            start, end = i * self.np_per_island, (i + 1) * self.np_per_island
            island_pop = population[start:end]
            island_vel = velocity[start:end]
            best_idx = np.argmin(self._eval_wrapper(lambda x: x[:, 0], island_pop))
            migrants.append(island_pop[best_idx].copy())
            migrant_velocities.append(island_vel[best_idx].copy())
        
        # Migrate: island i receives from island (i-1)
        for i in range(self.n_islands):
            donor_idx = (i - 1) % self.n_islands
            start = i * self.np_per_island
            # Replace worst individual in target island with migrant
            worst_idx = start + np.argmax(
                self._eval_wrapper(lambda x: x[:, 0], population[start:start + self.np_per_island])
            )
            population[worst_idx] = migrants[donor_idx].copy()
            velocity[worst_idx] = migrant_velocities[donor_idx].copy() * 0.5
        
        return population, velocity
    
    def _restart_stagnant_islands(self, population, velocity, func):
        """Restart islands that have stagnated (no improvement over many generations)."""
        if len(self.best_fitness_history) < 20:
            return population, velocity
        
        # Check improvement in recent history
        recent_window = min(20, len(self.best_fitness_history))
        recent_fitness = self.best_fitness_history[-recent_window:]
        
        # Detect stagnation
        improved_recently = any(
            recent_fitness[i] < recent_fitness[i-1] 
            for i in range(1, len(recent_fitness))
        )
        
        if not improved_recently and len(self.best_fitness_history) % 20 == 0:
            # Restart the island with worst current fitness
            island_fitness = []
            for i in range(self.n_islands):
                start, end = i * self.np_per_island, (i + 1) * self.np_per_island
                island_fit = func(population[start:end])
                island_fitness.append(np.min(island_fit))
            
            worst_island = np.argmax(island_fitness)
            start = worst_island * self.np_per_island
            end = (worst_island + 1) * self.np_per_island
            
            # Reinitialize worst island with diversity
            range_val = self.ub - self.lb
            new_pop = np.random.uniform(
                self.lb + 0.2 * range_val,
                self.ub - 0.2 * range_val,
                size=(self.np_per_island, self.dim)
            )
            population[start:end] = self._clip_to_bounds_batch(new_pop)
            
            # Reset velocity for restarted island
            velocity[start:end] = np.random.uniform(
                -0.05 * range_val, 0.05 * range_val,
                size=(self.np_per_island, self.dim)
            )
            
            # Reset adaptive parameters
            self.step_sizes[worst_island] = 0.5
            self.crossover_rates[worst_island] = 0.7
        
        return population, velocity
    
    def _compute_diversity(self, population):
        """Compute population diversity as average Euclidean distance to centroid."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
```