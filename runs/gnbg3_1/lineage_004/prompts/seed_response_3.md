```python
import numpy as np


class TopologyAwareSwarmOptimizer:
    """
    Particle Swarm Optimizer with ring topology neighborhoods and adaptive mechanisms.
    Key features:
    - Ring topology for neighborhood best communication
    - Adaptive inertia weight based on diversity
    - Covariance-based trial sampling for directional exploration
    - Adaptive step size control
    - Automatic restart on stagnation
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.pop_size = min(max(4 * dim, 20), 300)
        self.bounds = np.array([[-100.0, 100.0]] * dim)
        self.range = self.bounds[0, 1] - self.bounds[0, 0]
        
        # Population state
        self.population = None
        self.velocity = None
        self.personal_best_pos = None
        self.personal_best_fit = None
        self.neighborhood_best_pos = None
        self.neighborhood_best_fit = None
        
        # Global best tracking
        self.g_best_pos = None
        self.g_best_fit = np.inf
        
        # Adaptive parameters
        self.inertia = 0.729
        self.step_size = 0.1 * self.range
        self.cov_matrix = np.eye(dim) * (self.step_size ** 2)
        self.cognitive_coeff = 1.494
        self.social_coeff = 1.494
        
        # Diversity and stagnation
        self.diversity = 0.0
        self.stagnation_count = 0
        self.fitness_history = []
        
        # Adaptation counters
        self.trial_count = 0
        self.step_adapt_success = 0
        self.cov_adapt_success = 0
        self.step_adapt_rate = 0.1
        self.cov_adapt_rate = 0.1
        
        # Velocity bounds
        self.max_velocity = 0.2 * self.range
    
    def _initialize_population(self):
        """Initialize swarm with random positions and zero velocity."""
        self.population = np.random.uniform(
            self.bounds[0, 0], self.bounds[0, 1], (self.pop_size, self.dim)
        )
        self.velocity = np.zeros((self.pop_size, self.dim))
        self.personal_best_pos = self.population.copy()
        self.personal_best_fit = np.full(self.pop_size, np.inf)
        self.g_best_fit = np.inf
        self.g_best_pos = None
        self.fitness_history = []
        self.stagnation_count = 0
        
    def _compute_neighborhood_best_batch(self):
        """Compute ring-topology neighborhood bests for all particles."""
        n = self.pop_size
        nb_pos = np.empty((n, self.dim))
        nb_fit = np.empty(n)
        
        for i in range(n):
            left = (i - 1) % n
            right = (i + 1) % n
            neighbors = [left, i, right]
            neighbor_fits = self.personal_best_fit[neighbors]
            best_idx = neighbors[np.argmin(neighbor_fits)]
            nb_pos[i] = self.personal_best_pos[best_idx]
            nb_fit[i] = self.personal_best_fit[best_idx]
        
        self.neighborhood_best_pos = nb_pos
        self.neighborhood_best_fit = nb_fit
    
    def _update_personal_best_batch(self, fitness):
        """Update personal bests where current positions are better."""
        improved = fitness < self.personal_best_fit
        self.personal_best_fit[improved] = fitness[improved]
        self.personal_best_pos[improved] = self.population[improved]
        
        any_improved = np.any(improved)
        if any_improved:
            best_local_idx = np.argmin(self.personal_best_fit)
            if self.personal_best_fit[best_local_idx] < self.g_best_fit:
                self.g_best_fit = self.personal_best_fit[best_local_idx]
                self.g_best_pos = self.personal_best_pos[best_local_idx].copy()
    
    def _update_velocity_batch(self):
        """Update particle velocities with cognitive and social components."""
        r1 = np.random.uniform(0, 1, (self.pop_size, self.dim))
        r2 = np.random.uniform(0, 1, (self.pop_size, self.dim))
        
        cognitive = self.cognitive_coeff * r1 * (self.personal_best_pos - self.population)
        social = self.social_coeff * r2 * (self.neighborhood_best_pos - self.population)
        
        self.velocity = self.inertia * self.velocity + cognitive + social
        self.velocity = np.clip(self.velocity, -self.max_velocity, self.max_velocity)
    
    def _update_position_batch(self):
        """Update positions and clip to bounds."""
        self.population = self.population + self.velocity
        self._clip_to_bounds()
    
    def _clip_to_bounds(self):
        """Clip all candidates to search bounds."""
        np.clip(self.population, self.bounds[0, 0], self.bounds[0, 1], out=self.population)
    
    def _adapt_inertia_weight(self):
        """Adapt inertia weight based on improvement and diversity."""
        if len(self.fitness_history) >= 5:
            recent_improvement = self.fitness_history[-1] < self.fitness_history[-5]
            if recent_improvement and self.diversity > 0.1 * self.range:
                self.inertia = min(self.inertia * 1.1, 0.9)
            elif not recent_improvement:
                self.inertia = max(self.inertia * 0.9, 0.4)
    
    def _sample_trials_batch(self):
        """Generate trial positions using adaptive covariance sampling."""
        self.trial_count += 1
        
        # Sample from multivariate normal with current covariance
        L = np.linalg.cholesky(self.cov_matrix + np.eye(self.dim) * 1e-8)
        noise = np.random.standard_normal((self.pop_size, self.dim))
        perturbations = L @ noise.T.T * self.step_size
        
        # Combine with PSO-informed direction
        r3 = np.random.uniform(0, 1, (self.pop_size, self.dim))
        pso_direction = r3 * (self.neighborhood_best_pos - self.population)
        
        trials = self.population + 0.5 * perturbations + 0.3 * pso_direction
        np.clip(trials, self.bounds[0, 0], self.bounds[0, 1], out=trials)
        
        return trials
    
    def _adapt_covariance(self, trials, trial_fitness):
        """Adapt covariance matrix based on successful perturbations."""
        if len(trial_fitness) < self.pop_size * 0.5:
            return
        
        valid_mask = np.isfinite(trial_fitness)
        if np.sum(valid_mask) < 10:
            return
        
        current_mean_fit = np.mean(self._get_current_fitness())
        trial_mean_fit = np.mean(trial_fitness[valid_mask])
        
        if trial_mean_fit < current_mean_fit:
            valid_trials = trials[valid_mask]
            valid_pop = self.population[valid_mask]
            diff = valid_trials - valid_pop
            cov_update = np.cov(diff.T, bias=True)
            
            self.cov_matrix = (1 - self.cov_adapt_rate) * self.cov_matrix + self.cov_adapt_rate * cov_update
            self.cov_adapt_success += 1
            
            if self.cov_adapt_success % 5 == 0:
                self.cov_adapt_rate = min(self.cov_adapt_rate * 1.2, 0.3)
        else:
            self.cov_adapt_rate = max(self.cov_adapt_rate * 0.9, 0.02)
    
    def _adapt_step_size(self, trial_fitness):
        """Adapt step size based on success ratio."""
        if len(trial_fitness) < self.pop_size:
            return
        
        valid_mask = np.isfinite(trial_fitness)
        success_ratio = np.mean(trial_fitness[valid_mask] < self._get_current_fitness()[valid_mask])
        
        if success_ratio > 0.3:
            self.step_size *= 1.1
            self.step_adapt_success += 1
        elif success_ratio < 0.1:
            self.step_size *= 0.9
        
        self.step_size = np.clip(self.step_size, 0.01 * self.range, 0.5 * self.range)
        self.max_velocity = 0.3 * self.step_size
    
    def _adapt_velocity_bounds(self):
        """Adapt velocity bounds based on step size."""
        self.max_velocity = np.clip(self.step_size * 2, 0.01 * self.range, 0.5 * self.range)
    
    def _compute_diversity(self):
        """Compute swarm diversity as mean distance to centroid."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        self.diversity = np.mean(distances)
    
    def _get_current_fitness(self):
        """Return current personal best fitness values."""
        return self.personal_best_fit.copy()
    
    def _select_survivors_batch(self, trials, trial_fitness):
        """Select better individuals between current and trials."""
        if len(trial_fitness) < self.pop_size:
            padding = np.full(self.pop_size - len(trial_fitness), np.inf)
            trial_fitness = np.concatenate([trial_fitness, padding])
            trials = np.vstack([trials, np.zeros((self.pop_size - len(trials), self.dim))])
        
        current_fit = self._get_current_fitness()
        better_mask = trial_fitness < current_fit
        self.population[better_mask] = trials[better_mask]
        self.velocity[better_mask] = 0.0
    
    def _restart_if_stagnant(self):
        """Reinitialize portion of swarm if stagnant."""
        self.fitness_history.append(self.g_best_fit)
        
        if len(self.fitness_history) >= 2:
            improvement = self.fitness_history[-2] - self.fitness_history[-1]
            if improvement < 1e-6 * abs(self.fitness_history[-2] + 1e-8):
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
        
        if self.stagnation_count >= 3:
            n_replace = self.pop_size // 3
            replace_idx = np.random.choice(self.pop_size, n_replace, replace=False)
            
            self.population[replace_idx] = np.random.uniform(
                self.bounds[0, 0], self.bounds[0, 1], (n_replace, self.dim)
            )
            self.velocity[replace_idx] = np.zeros((n_replace, self.dim))
            self.personal_best_pos[replace_idx] = self.population[replace_idx]
            self.personal_best_fit[replace_idx] = np.inf
            
            self.stagnation_count = 0
            self.step_size = min(self.step_size * 1.5, 0.3 * self.range)
            self.cov_matrix = np.eye(self.dim) * (self.step_size ** 2)
    
    def __call__(self, func, stopping_condition):
        """
        Run optimization loop.
        
        Args:
            func: Fitness function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        self._initialize_population()
        
        # Initial evaluation
        fitness = func(self.population)
        if len(fitness) < self.pop_size:
            fitness = np.pad(fitness, (0, self.pop_size - len(fitness)), constant_values=np.inf)
        
        self._update_personal_best_batch(fitness)
        self._compute_neighborhood_best_batch()
        
        # Main optimization loop
        while not stopping_condition():
            # Generate and evaluate trials
            trials = self._sample_trials_batch()
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
            
            if stopping_condition():
                break
            
            # Update swarm state
            self._select_survivors_batch(trials, trial_fitness)
            self._update_velocity_batch()
            self._update_position_batch()
            
            # Evaluate updated positions
            fitness = func(self.population)
            if len(fitness) < self.pop_size:
                fitness = np.pad(fitness, (0, self.pop_size - len(fitness)), constant_values=np.inf)
            
            self._update_personal_best_batch(fitness)
            self._compute_neighborhood_best_batch()
            
            # Adaptive mechanisms
            self._adapt_inertia_weight()
            self._adapt_velocity_bounds()
            self._adapt_step_size(trial_fitness)
            self._adapt_covariance(trials, trial_fitness)
            self._compute_diversity()
            self._restart_if_stagnant()
            
            if stopping_condition():
                break
        
        return self.g_best_fit, self.g_best_pos
```