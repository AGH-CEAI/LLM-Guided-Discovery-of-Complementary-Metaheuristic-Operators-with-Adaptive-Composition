```python
import numpy as np


class TopologyAdaptiveParticleSwarmOptimizer:
    """
    Particle Swarm Optimizer with topology-aware neighborhoods and adaptive parameters.
    Uses ring topology with adaptive inertia/acceleration and covariance-guided perturbations.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 120), 300)  # Population size 120-300 for dim=30
        
        # Default bounds
        self.lb = -100.0
        self.ub = 100.0
        
        # PSO parameters (adaptive)
        self.chi = 0.729  # Constriction factor
        self.c1_init = 1.496  # Cognitive coefficient
        self.c2_init = 1.496  # Social coefficient
        
        # Adaptive parameters
        self.omega = 0.9  # Inertia weight (decreases over time)
        self.omega_min = 0.4
        self.omega_decay = 0.99
        
        # Topology parameters
        self.topology_type = 'ring'
        self.neighbor_count = 3
        
        # Covariance adaptation parameters
        self.cov_adapt_rate = 0.1
        self.perturbation_strength = 0.1
        
        # Diversity management
        self.min_diversity = 0.01
        self.diversity_penalty = 0.5
        
        # Stagnation detection
        self.stagnation_threshold = 50
        self.improvement_threshold = 1e-8
        
        # Operator pool for trial generation
        self.trial_operators = ['pso', 'cov_perturb', 'best_perturb', 'diversity_boost']
        self.operator_weights = np.ones(len(self.trial_operators))
        self.operator_rewards = np.zeros(len(self.trial_operators))
        self.operator_uses = np.zeros(len(self.trial_operators))
        
        # Internal state
        self.population = None
        self.velocity = None
        self.fitness = None
        self.personal_best = None
        self.personal_best_fit = None
        self.global_best = None
        self.global_best_fit = np.inf
        self.neighborhood_best = None
        self.neighborhood_best_fit = None
        
        # Covariance matrix for adaptation
        self.cov_matrix = np.eye(dim)
        self.cov_learning_rate = 0.05
        
        # History for adaptation
        self.best_history = []
        self.generation = 0
        
    def _initialize_population(self):
        """Initialize population with random positions within bounds."""
        self.population = np.random.uniform(
            self.lb, self.ub, size=(self.np, self.dim)
        )
        self.velocity = np.random.uniform(
            -0.1 * (self.ub - self.lb), 0.1 * (self.ub - self.lb),
            size=(self.np, self.dim)
        )
        self.personal_best = self.population.copy()
        self.personal_best_fit = np.full(self.np, np.inf)
        
    def _setup_ring_topology(self):
        """Set up ring neighborhood topology."""
        n = self.np
        k = self.neighbor_count
        
        # For each particle, find k/2 neighbors on each side in ring
        self.neighbor_indices = np.zeros((n, k), dtype=int)
        for i in range(n):
            left = [(i - j) % n for j in range(1, k // 2 + 1)]
            right = [(i + j) % n for j in range(1, k // 2 + 1)]
            self.neighbor_indices[i] = left + right
            
    def _evaluate_population_batch(self, population):
        """Evaluate fitness for entire population batch."""
        # Clip to bounds
        population = np.clip(population, self.lb, self.ub)
        
        # Call fitness function
        fitness = self.func(population)
        
        # Handle budget exhaustion
        if len(fitness) < len(population):
            actual_len = len(fitness)
            population = population[:actual_len]
            fitness = fitness[:actual_len]
            
        return population, fitness
    
    def _update_personal_best(self):
        """Update personal best positions for each particle."""
        improved = self.fitness < self.personal_best_fit
        self.personal_best[improved] = self.population[improved]
        self.personal_best_fit[improved] = self.fitness[improved]
        
    def _update_neighborhood_best_batch(self):
        """Update neighborhood best for all particles using topology."""
        n = self.np
        self.neighborhood_best = np.zeros((n, self.dim))
        self.neighborhood_best_fit = np.full(n, np.inf)
        
        for i in range(n):
            neighbor_ids = self.neighbor_indices[i]
            all_candidates = np.concatenate([self.personal_best[neighbor_ids], 
                                              self.personal_best[i:i+1]])
            all_fits = np.concatenate([self.personal_best_fit[neighbor_ids],
                                        self.personal_best_fit[i:i+1]])
            best_idx = np.argmin(all_fits)
            self.neighborhood_best[i] = all_candidates[best_idx]
            self.neighborhood_best_fit[i] = all_fits[best_idx]
            
    def _update_global_best(self):
        """Update global best position."""
        best_idx = np.argmin(self.personal_best_fit)
        if self.personal_best_fit[best_idx] < self.global_best_fit:
            self.global_best = self.personal_best[best_idx].copy()
            self.global_best_fit = self.personal_best_fit[best_idx]
            
    def _compute_velocity_pso_batch(self):
        """Compute PSO velocity updates for all particles."""
        r1 = np.random.uniform(0, 1, size=(self.np, self.dim))
        r2 = np.random.uniform(0, 1, size=(self.np, self.dim))
        
        cognitive = self.c1_init * r1 * (self.personal_best - self.population)
        social = self.c2_init * r2 * (self.neighborhood_best - self.population)
        
        self.velocity = self.omega * self.velocity + cognitive + social
        
        # Limit velocity
        v_max = 0.2 * (self.ub - self.lb)
        velocity_norm = np.linalg.norm(self.velocity, axis=1, keepdims=True)
        velocity_norm = np.maximum(velocity_norm, 1e-10)
        scale = np.minimum(velocity_norm, v_max) / velocity_norm
        self.velocity *= scale
        
    def _move_particles_batch(self):
        """Move particles using velocity."""
        self.population = self.population + self.velocity
        self.population = np.clip(self.population, self.lb, self.ub)
        
    def _generate_cov_perturbation_batch(self):
        """Generate covariance-guided perturbations around current positions."""
        # Sample from multivariate normal using adapted covariance
        perturbations = np.random.multivariate_normal(
            mean=np.zeros(self.dim),
            cov=self.cov_matrix,
            size=self.np
        )
        perturbations *= self.perturbation_strength * (self.ub - self.lb)
        trials = self.population + perturbations
        return np.clip(trials, self.lb, self.ub)
    
    def _generate_best_perturbation_batch(self):
        """Generate perturbations directed toward global best."""
        direction = self.global_best - self.population
        direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction_norm = np.maximum(direction_norm, 1e-10)
        direction_unit = direction / direction_norm
        
        # Random step sizes
        steps = np.random.uniform(0.1, 0.5, size=(self.np, 1)) * (self.ub - self.lb)
        
        trials = self.population + direction_unit * steps
        return np.clip(trials, self.lb, self.ub)
    
    def _generate_diversity_boost_batch(self):
        """Generate perturbations that increase diversity."""
        # Move toward random directions away from swarm center
        centroid = np.mean(self.population, axis=0)
        away_vectors = self.population - centroid
        away_norm = np.linalg.norm(away_vectors, axis=1, keepdims=True)
        away_norm = np.maximum(away_norm, 1e-10)
        away_unit = away_vectors / away_norm
        
        # Add random orthogonal component
        random_component = np.random.randn(self.np, self.dim)
        random_component -= away_unit * np.sum(random_component * away_unit, axis=1, keepdims=True)
        random_norm = np.linalg.norm(random_component, axis=1, keepdims=True)
        random_norm = np.maximum(random_norm, 1e-10)
        random_unit = random_component / random_norm
        
        # Combine away-from-center with random
        steps = np.random.uniform(0.2, 0.8, size=(self.np, 1)) * (self.ub - self.lb)
        trials = self.population + away_unit * steps * 0.3 + random_unit * steps * 0.7
        return np.clip(trials, self.lb, self.ub)
    
    def _select_trial_operator(self):
        """Select trial generation operator based on weights."""
        weights = self.operator_weights / np.sum(self.operator_weights)
        op_idx = np.random.choice(len(self.trial_operators), p=weights)
        return op_idx
    
    def _generate_trials_batch(self):
        """Generate trial population using selected operators."""
        op_idx = self._select_trial_operator()
        self.operator_uses[op_idx] += 1
        
        if self.trial_operators[op_idx] == 'pso':
            self._compute_velocity_pso_batch()
            trials = self.population + self.velocity
        elif self.trial_operators[op_idx] == 'cov_perturb':
            trials = self._generate_cov_perturbation_batch()
        elif self.trial_operators[op_idx] == 'best_perturb':
            trials = self._generate_best_perturbation_batch()
        else:  # diversity_boost
            trials = self._generate_diversity_boost_batch()
            
        return np.clip(trials, self.lb, self.ub), op_idx
    
    def _update_covariance_batch(self, accepted_improvements):
        """Update covariance matrix based on successful perturbations."""
        if len(accepted_improvements) < 2:
            return
            
        # Compute covariance from successful improvement directions
        improvements = accepted_improvements - self.population[accepted_improvements]
        
        # Update covariance matrix using rank-1 update
        mean_diff = improvements - np.mean(improvements, axis=0)
        cov_update = np.dot(mean_diff.T, mean_diff) / len(improvements)
        
        # Blend with current covariance
        self.cov_matrix = (1 - self.cov_learning_rate) * self.cov_matrix + \
                          self.cov_learning_rate * cov_update
        
        # Ensure positive definiteness
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        self.cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
    def _compute_diversity(self):
        """Compute population diversity measure."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances) / (self.ub - self.lb)
    
    def _adapt_step_size(self):
        """Adapt inertia weight based on progress."""
        if len(self.best_history) < 5:
            return
            
        recent_improvement = self.best_history[-1] - self.best_history[-5]
        if recent_improvement > -self.improvement_threshold:
            # Stagnation - increase exploration
            self.omega = min(self.omega * 1.1, 0.9)
        else:
            # Good progress - fine-tune exploitation
            self.omega = max(self.omega * 0.99, self.omega_min)
            
    def _adapt_acceleration_coefficients(self):
        """Adapt cognitive and social coefficients."""
        diversity = self._compute_diversity()
        
        if diversity < self.min_diversity:
            # Low diversity - increase cognitive component for exploration
            self.c1_init = min(self.c1_init * 1.05, 2.5)
            self.c2_init = max(self.c2_init * 0.95, 0.5)
        else:
            # Good diversity - balance coefficients
            self.c1_init = max(min(self.c1_init * 0.99, 2.0), 1.0)
            self.c2_init = max(min(self.c2_init * 1.01, 2.0), 1.0)
            
    def _update_operator_rewards(self, op_idx, improvement_ratio):
        """Update operator rewards based on performance."""
        # Reward operator if it produced improvements
        if improvement_ratio > 0.1:
            self.operator_rewards[op_idx] += 1.0
        elif improvement_ratio < -0.1:
            self.operator_rewards[op_idx] -= 0.5
            
        # Decay rewards
        self.operator_rewards *= 0.95
        
        # Update weights using exponential moving average
        for i in range(len(self.trial_operators)):
            if self.operator_uses[i] > 0:
                avg_reward = self.operator_rewards[i] / max(self.operator_uses[i], 1)
                self.operator_weights[i] = 0.9 * self.operator_weights[i] + 0.1 * (avg_reward + 1)
                
        # Ensure positive weights
        self.operator_weights = np.maximum(self.operator_weights, 0.1)
        
    def _select_survivors_batch(self, trials, trial_fitness, op_idx):
        """Select survivors based on fitness comparison."""
        # Track improvements for adaptation
        improvements = self.fitness < trial_fitness
        improvement_count = np.sum(improvements)
        improvement_ratio = improvement_count / self.np
        
        # Accept improvements
        accept_mask = self.fitness > trial_fitness
        self.population[accept_mask] = trials[accept_mask]
        self.fitness[accept_mask] = trial_fitness[accept_mask]
        
        # Update operator rewards
        self._update_operator_rewards(op_idx, improvement_ratio)
        
        return improvement_count
    
    def _restart_if_stagnant(self):
        """Restart population if stuck in local optimum."""
        if len(self.best_history) < self.stagnation_threshold:
            return False
            
        recent_best = self.best_history[-self.stagnation_threshold]
        if self.global_best_fit >= recent_best - self.improvement_threshold:
            # Stagnation detected - reinitialize portion of population
            restart_count = self.np // 4
            restart_indices = np.random.choice(self.np, restart_count, replace=False)
            
            # Reinitialize stagnant particles
            self.population[restart_indices] = np.random.uniform(
                self.lb, self.ub, size=(restart_count, self.dim)
            )
            self.velocity[restart_indices] = np.random.uniform(
                -0.1 * (self.ub - self.lb), 0.1 * (self.ub - self.lb),
                size=(restart_count, self.dim)
            )
            
            # Reset personal best for restarted particles
            self.personal_best_fit[restart_indices] = np.inf
            
            # Add diversity boost to covariance
            self.cov_matrix = 0.5 * self.cov_matrix + 0.5 * np.eye(self.dim)
            
            return True
        return False
    
    def _track_best_history(self):
        """Track global best history for stagnation detection."""
        self.best_history.append(self.global_best_fit)
        if len(self.best_history) > 200:
            self.best_history.pop(0)
            
    def _handle_budget_exhaustion(self, fitness, expected_len):
        """Handle case where function returns fewer values than expected."""
        if len(fitness) < expected_len:
            # Budget exhausted - return what we have
            return True
        return False
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer.
        
        Parameters:
            func: Fitness function accepting (N, dim) array, returning (N,) array
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        self.func = func
        
        # Initialize
        self._initialize_population()
        self._setup_ring_topology()
        
        # Initial evaluation
        self.population, self.fitness = self._evaluate_population_batch(self.population)
        
        if self._handle_budget_exhaustion(self.fitness, self.np):
            return self.global_best_fit, self.global_best
            
        self._update_personal_best()
        self._update_neighborhood_best_batch()
        self._update_global_best()
        self._track_best_history()
        
        # Main optimization loop
        while not stopping_condition():
            # Generate trials
            trials, op_idx = self._generate_trials_batch()
            
            # Check stopping condition before evaluation
            if stopping_condition():
                break
                
            # Evaluate trials
            trials, trial_fitness = self._evaluate_population_batch(trials)
            
            if self._handle_budget_exhaustion(trial_fitness, len(trials)):
                break
                
            # Select survivors
            self._select_survivors_batch(trials, trial_fitness, op_idx)
            
            # Check stopping after selection
            if stopping_condition():
                break
                
            # Update best positions
            self._update_personal_best()
            self._update_neighborhood_best_batch()
            self._update_global_best()
            self._track_best_history()
            
            # Adapt parameters
            self._adapt_step_size()
            self._adapt_acceleration_coefficients()
            
            # Restart if stagnant
            self._restart_if_stagnant()
            
            self.generation += 1
            
        return self.global_best_fit, self.global_best
```