```python
import numpy as np


class AdaptiveTopologySwarm:
    """
    Particle swarm optimizer with adaptive ring topology and multi-strategy position updates.
    Key features:
    - Adaptive neighborhood size based on swarm convergence
    - Multiple position update strategies (standard, temporal drift, fitness rank, Cauchy perturbation)
    - Inertia weight adaptation based on swarm performance gradient
    - Diversity-triggered restart for stuck particles
    - Batched operations throughout for efficiency
    """
    
    def __init__(self, dim, np_size=None, *args, **kwargs):
        self.dim = dim
        self.np = np_size if np_size else min(150, max(60, 4 * dim))
        self.lb = -100.0
        self.ub = 100.0
        
        # Initialize state variables
        self.population = None
        self.velocity = None
        self.fitness = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.neighborhood_best = None
        self.neighborhood_best_fitness = None
        
        # Topology parameters
        self.neighbor_count = max(3, self.np // 15)
        self.topology_adapt_rate = 0.02
        
        # Strategy control
        self.strategy_weights = np.ones(4) / 4.0
        self.strategy_scores = np.zeros(4)
        self.strategy_names = ['standard', 'temporal_drift', 'fitness_rank', 'cauchy']
        
        # Inertia and PSO parameters
        self.inertia = 0.729
        self.cognitive = 1.494
        self.social = 1.494
        self.inertia_min = 0.3
        self.inertia_max = 0.9
        
        # Tracking
        self.best_fitness_history = []
        self.stagnation_count = 0
        self.generation = 0
        self.fitness_evaluations = 0
        
    def __call__(self, func, stopping_condition):
        self._initialize_population()
        self._initialize_velocity()
        self._initialize_topology()
        self._initialize_personal_best()
        
        self.fitness = func(self.population)
        self.fitness_evaluations += len(self.fitness)
        self._handle_budget_truncation(func, self.population, self.fitness)
        
        self.personal_best_fitness = self.fitness.copy()
        self.personal_best = self.population.copy()
        
        self._update_neighborhood_best()
        self.best_fitness_history.append(np.min(self.fitness))
        
        while not stopping_condition():
            self.generation += 1
            
            # Build trial population using multiple strategies
            trial_population = self._build_trials_batched()
            
            # Handle potential budget truncation
            if len(trial_population) < self.np:
                trial_population = trial_population[:len(trial_population)]
            
            trial_fitness = func(trial_population)
            self.fitness_evaluations += len(trial_fitness)
            
            if len(trial_fitness) < len(trial_population):
                trial_population = trial_population[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
            
            if stopping_condition():
                break
            
            # Selection and update
            self._select_and_update_batched(trial_population, trial_fitness)
            
            # Adaptation methods
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            self._adapt_strategy_weights(trial_fitness)
            
            # Diversity maintenance
            diversity = self._compute_diversity()
            if diversity < 0.01 * self.dim:
                self._restart_diverse_particles()
            
            # Update tracking
            current_best = np.min(self.fitness)
            self.best_fitness_history.append(current_best)
            
            if current_best < self.best_fitness_history[-2] if len(self.best_fitness_history) > 1 else True:
                self.stagnation_count = 0
            else:
                self.stagnation_count += 1
            
            if self.stagnation_count > 50:
                self._adaptive_restart()
                
        best_idx = np.argmin(self.personal_best_fitness)
        return self.personal_best_fitness[best_idx], self.personal_best[best_idx].copy()
    
    def _initialize_population(self):
        """Initialize population within bounds using latin hypercube sampling."""
        self.population = np.random.uniform(self.lb, self.ub, (self.np, self.dim))
        # Ensure population is clipped to bounds
        self.population = np.clip(self.population, self.lb, self.ub)
        
    def _initialize_velocity(self):
        """Initialize velocity as fraction of search space."""
        vel_range = (self.ub - self.lb) * 0.1
        self.velocity = np.random.uniform(-vel_range, vel_range, (self.np, self.dim))
        
    def _initialize_topology(self):
        """Initialize ring topology neighborhood indices."""
        self._neighbor_indices = np.zeros((self.np, self.neighbor_count), dtype=int)
        for i in range(self.np):
            indices = [(i - j - 1) % self.np for j in range(self.neighbor_count)]
            indices += [(i + j + 1) % self.np for j in range(self.neighbor_count)]
            self._neighbor_indices[i] = indices[:self.neighbor_count]
            
    def _initialize_personal_best(self):
        """Initialize personal best tracking arrays."""
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.np, np.inf)
        
    def _build_trials_batched(self):
        """Build trial population using multiple strategies."""
        # Determine strategy counts based on weights
        n_strategies = len(self.strategy_weights)
        base_count = self.np // n_strategies
        remainder = self.np % n_strategies
        
        trial_parts = []
        start_idx = 0
        
        for s in range(n_strategies):
            count = base_count + (1 if s < remainder else 0)
            end_idx = start_idx + count
            
            if count > 0:
                indices = np.arange(start_idx, end_idx)
                sub_pop = self.population[indices]
                sub_vel = self.velocity[indices]
                sub_pbest = self.personal_best[indices]
                sub_nbset = self.neighborhood_best[indices]
                
                if self.strategy_names[s] == 'standard':
                    trials = self._position_update_batch(sub_pop, sub_vel, sub_pbest, sub_nbset)
                elif self.strategy_names[s] == 'temporal_drift':
                    trials = self._position_update_temporal_drift(sub_pop, sub_vel, sub_pbest)
                elif self.strategy_names[s] == 'fitness_rank':
                    trials = self._position_update_fitness_rank(sub_pop, sub_pbest, sub_nbset)
                else:  # cauchy
                    trials = self._position_update_cauchy_perturbation(sub_pop, sub_nbset)
                
                trial_parts.append(trials)
            start_idx = end_idx
            
        trials = np.concatenate(trial_parts, axis=0)
        trials = np.clip(trials, self.lb, self.ub)
        return trials
    
    def _position_update_batch(self, pop, vel, pbest, nbset):
        """Standard PSO position update with adapted parameters."""
        r1 = np.random.uniform(0, 1, pop.shape)
        r2 = np.random.uniform(0, 1, pop.shape)
        
        cognitive_term = self.cognitive * r1 * (pbest - pop)
        social_term = self.social * r2 * (nbset - pop)
        
        new_vel = self.inertia * vel + cognitive_term + social_term
        new_pos = pop + new_vel
        
        # Update velocity for particles using this strategy
        mask = np.zeros(self.np, dtype=bool)
        # This is handled in the calling context by tracking indices
        
        self.velocity[:len(new_vel)] = new_vel
        return new_pos
    
    def _position_update_temporal_drift(self, pop, vel, pbest):
        """Position update with temporal drift component for exploration."""
        # Compute temporal difference
        temporal_diff = pbest - pop
        
        # Drift factor decreases with generation
        drift_strength = self.inertia * (1 - self.generation / 1000)
        
        # Add drift to velocity
        drifted_vel = vel + drift_strength * temporal_diff * np.random.uniform(0, 1, pop.shape)
        
        # Clamp velocity
        vel_limit = (self.ub - self.lb) * 0.2
        drifted_vel = np.clip(drifted_vel, -vel_limit, vel_limit)
        
        new_pos = pop + drifted_vel
        return new_pos
    
    def _position_update_fitness_rank(self, pop, pbest, nbset):
        """Rank-based position update for robust selection pressure."""
        # Compute fitness ranks for personal and neighborhood bests
        n = len(pop)
        
        # Rank transform
        pbest_rank = np.argsort(np.argsort(self.personal_best_fitness[:n]))
        nbset_rank = np.argsort(np.argsort(self.neighborhood_best_fitness[:n]))
        
        # Normalize ranks to [0, 1]
        pbest_rank = pbest_rank / (n - 1) if n > 1 else np.ones(n) * 0.5
        nbset_rank = nbset_rank / (n - 1) if n > 1 else np.ones(n) * 0.5
        
        # Weighted combination with rank-based coefficients
        r1 = np.random.uniform(0, 1, pop.shape)
        r2 = np.random.uniform(0, 1, pop.shape)
        
        # Rank-weighted coefficients (higher rank = higher coefficient)
        cognitive_coef = self.cognitive * (0.5 + 0.5 * pbest_rank.reshape(-1, 1))
        social_coef = self.social * (0.5 + 0.5 * nbset_rank.reshape(-1, 1))
        
        new_pos = pop + cognitive_coef * r1 * (pbest - pop) + social_coef * r2 * (nbset - pop)
        return new_pos
    
    def _position_update_cauchy_perturbation(self, pop, nbset):
        """Cauchy perturbation around neighborhood best for escaping local optima."""
        # Small step toward neighborhood best
        step_size = 0.1 * (self.ub - self.lb)
        
        # Direction to neighborhood best
        direction = nbset - pop
        direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction_norm = np.maximum(direction_norm, 1e-10)
        direction_unit = direction / direction_norm
        
        # Cauchy perturbation
        cauchy_perturb = self._cauchy_sample((self.dim,))
        cauchy_perturb = cauchy_perturb / (1 + np.abs(cauchy_perturb))
        
        # Combine deterministic step with Cauchy noise
        new_pos = pop + step_size * 0.1 * direction + step_size * cauchy_perturb
        return new_pos
    
    def _cauchy_sample(self, shape):
        """Generate Cauchy samples using inverse CDF method."""
        uniform = np.random.uniform(0.001, 0.999, shape)
        return np.tan(np.pi * (uniform - 0.5))
    
    def _update_neighborhood_best(self):
        """Update neighborhood best for each particle based on ring topology."""
        self.neighborhood_best = np.empty((self.np, self.dim))
        self.neighborhood_best_fitness = np.full(self.np, np.inf)
        
        for i in range(self.np):
            neighbors = self._neighbor_indices[i]
            neighbor_fitness = self.personal_best_fitness[neighbors]
            best_neighbor = np.argmin(neighbor_fitness)
            
            if neighbor_fitness[best_neighbor] < self.personal_best_fitness[i]:
                self.neighborhood_best[i] = self.personal_best[neighbors[best_neighbor]]
                self.neighborhood_best_fitness[i] = neighbor_fitness[best_neighbor]
            else:
                self.neighborhood_best[i] = self.personal_best[i]
                self.neighborhood_best_fitness[i] = self.personal_best_fitness[i]
                
    def _select_and_update_batched(self, trials, trial_fitness):
        """Batch selection and update of personal bests and positions."""
        # Selection: keep trial if better
        improve_mask = trial_fitness < self.fitness
        
        # Update population where improvement found
        self.population = np.where(improve_mask[:, np.newaxis], trials, self.population)
        self.fitness = np.where(improve_mask, trial_fitness, self.fitness)
        
        # Update personal bests
        pbest_improve = trial_fitness < self.personal_best_fitness
        self.personal_best = np.where(pbest_improve[:, np.newaxis], self.population, self.personal_best)
        self.personal_best_fitness = np.where(pbest_improve, trial_fitness, self.personal_best_fitness)
        
        # Update neighborhood bests
        self._update_neighborhood_best()
        
    def _adapt_inertia_weight(self):
        """Adapt inertia weight based on recent fitness improvement gradient."""
        if len(self.best_fitness_history) >= 5:
            recent = self.best_fitness_history[-5:]
            gradient = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-10)
            
            # If improving: reduce inertia for fine-tuning
            # If stagnating: increase inertia for exploration
            if gradient < -0.01:
                self.inertia = max(self.inertia_min, self.inertia * 0.95)
            elif gradient > 0.001:
                self.inertia = min(self.inertia_max, self.inertia * 1.05)
                
    def _adapt_neighborhood_size(self):
        """Adapt neighborhood size based on convergence status."""
        if len(self.best_fitness_history) >= 10:
            recent_best = np.min(self.best_fitness_history[-10:])
            older_best = np.min(self.best_fitness_history[-20:-10]) if len(self.best_fitness_history) >= 20 else recent_best
            
            # If not improving, increase neighborhood for stronger social influence
            if recent_best >= older_best:
                self.neighbor_count = min(self.np // 3, self.neighbor_count + 1)
            else:
                self.neighbor_count = max(3, self.neighbor_count - 1)
                
            # Rebuild topology if size changed
            self._initialize_topology()
            
    def _adapt_strategy_weights(self, trial_fitness):
        """Adapt strategy selection weights based on recent success."""
        # Compute success rate for each strategy
        n_strategies = len(self.strategy_weights)
        base_count = self.np // n_strategies
        
        success_counts = np.zeros(n_strategies)
        start_idx = 0
        
        for s in range(n_strategies):
            count = base_count + (1 if s < self.np % n_strategies else 0)
            end_idx = start_idx + count
            
            if count > 0:
                # Compare trial fitness to current fitness
                trial_part = trial_fitness[start_idx:end_idx]
                current_part = self.fitness[start_idx:end_idx]
                success_counts[s] = np.sum(trial_part < current_part) / count
                
            start_idx = end_idx
            
        # Update scores with exponential moving average
        self.strategy_scores = 0.7 * self.strategy_scores + 0.3 * success_counts
        
        # Convert scores to weights via softmax
        exp_scores = np.exp(self.strategy_scores * 10)
        self.strategy_weights = exp_scores / np.sum(exp_scores)
        
    def _compute_diversity(self):
        """Compute population diversity as average pairwise distance."""
        if self.np < 2:
            return 0.0
        
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _restart_diverse_particles(self):
        """Restart low-diversity particles with random perturbations."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        
        threshold = np.percentile(distances, 25)
        restart_mask = distances < threshold
        
        n_restart = np.sum(restart_mask)
        if n_restart > 0:
            # Generate new random positions
            new_positions = np.random.uniform(self.lb, self.ub, (n_restart, self.dim))
            
            # Blend with centroid for some diversity
            blend = np.random.uniform(0, 0.3, (n_restart, 1))
            new_positions = new_positions * (1 - blend) + centroid * blend
            
            self.population[restart_mask] = new_positions
            
            # Reset velocities for restarted particles
            vel_range = (self.ub - self.lb) * 0.1
            self.velocity[restart_mask] = np.random.uniform(-vel_range, vel_range, (n_restart, self.dim))
            
    def _adaptive_restart(self):
        """Full adaptive restart when stuck in local optima."""
        # Keep best particles
        n_keep = max(5, self.np // 10)
        best_indices = np.argsort(self.fitness)[:n_keep]
        
        # Generate new random population
        new_pop = np.random.uniform(self.lb, self.ub, (self.np - n_keep, self.dim))
        
        # Blend some best solutions into new population
        blend_count = min(n_keep, self.np // 5)
        for i in range(blend_count):
            blend_factor = 0.2 * (i + 1) / blend_count
            new_pop[i] = (1 - blend_factor) * new_pop[i] + blend_factor * self.personal_best[best_indices[i]]
        
        # Combine and update
        self.population = np.vstack([self.population[best_indices], new_pop])
        self.velocity = np.random.uniform(-1, 1, (self.np, self.dim)) * (self.ub - self.lb) * 0.05
        
        # Update personal bests for new particles
        for i in range(n_keep, self.np):
            self.personal_best[i] = self.population[i]
            self.personal_best_fitness[i] = np.inf
            
        self.stagnation_count = 0
        self._initialize_topology()
        
    def _handle_budget_truncation(self, func, population, fitness):
        """Handle case where budget is exhausted mid-evaluation."""
        if len(fitness) < len(population):
            self.population = population[:len(fitness)]
            self.velocity = self.velocity[:len(fitness)]
            self.np = len(fitness)
```