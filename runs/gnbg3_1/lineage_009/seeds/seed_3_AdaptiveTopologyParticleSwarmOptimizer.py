import numpy as np


class AdaptiveTopologyParticleSwarmOptimizer:
    """
    Particle Swarm Optimizer with adaptive topology and niching.
    Uses topology-aware neighborhoods (ring, von Neumann, fully connected)
    with adaptive parameter control and diversity-driven restart.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower = -100.0
        self.upper = 100.0
        
        # Population size: 6*dim is a good balance
        self.np = min(max(6 * dim, 30), 300)
        
        # PSO parameters with adaptive control
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1_max = 2.5
        self.c1_min = 1.5
        self.c2_max = 2.5
        self.c2_min = 1.5
        self.v_max = 0.2 * (self.upper - self.lower)
        
        # Topology selection (0=ring, 1=vonNeumann, 2=fullyConnected, 3=random)
        self.topology_type = 1
        self.topology_change_freq = 50
        
        # Diversity and convergence tracking
        self.diversity_threshold = 1e-6
        self.stagnation_counter = 0
        self.stagnation_limit = 100
        
        # Adaptation rewards for parameter tuning
        self.param_rewards = {
            'w': 0.0,
            'c1': 0.0,
            'c2': 0.0,
            'topology': np.zeros(4)
        }
        
        # Success tracking for adaptations
        self.recent_improvements = []
        
    def __call__(self, func, stopping_condition):
        """Main optimization loop orchestrating batch operations."""
        population = self._initialize_population()
        velocities = self._initialize_velocities(population)
        
        fitness = self._evaluate_batch(func, population)
        personal_best_pos, personal_best_fit = self._update_personal_best_batch(
            population, fitness
        )
        global_best_pos, global_best_fit = self._update_global_best_batch(
            population, personal_best_pos, personal_best_fit
        )
        
        generation = 0
        best_history = [global_best_fit]
        
        while not stopping_condition():
            # Adapt parameters based on recent performance
            self._adapt_parameters(generation, best_history)
            
            # Change topology periodically
            if generation > 0 and generation % self.topology_change_freq == 0:
                self._select_topology(global_best_fit, best_history)
            
            # Compute neighborhood bests for current topology
            neighborhood_best = self._compute_neighborhood_best_batch(
                personal_best_pos, personal_best_fit
            )
            
            # Build trial batch: velocity update + position update + clip
            trials = self._build_trial_batch(
                population, velocities, personal_best_pos, 
                neighborhood_best, generation
            )
            
            # Early exit check after batched trial construction
            if stopping_condition():
                break
            
            # Evaluate trial batch
            trial_fitness = self._evaluate_batch(func, trials)
            
            # Handle budget exhaustion
            if len(trial_fitness) < len(trials):
                valid_len = len(trial_fitness)
                trials = trials[:valid_len]
                trial_fitness = trial_fitness[:valid_len]
            
            if stopping_condition():
                break
            
            # Selection: keep better of trial vs current
            population, velocities, personal_best_pos, personal_best_fit = \
                self._select_survivors_batch(
                    population, velocities, trials,
                    personal_best_pos, personal_best_fit, trial_fitness
                )
            
            # Update global best
            g_best_pos, g_best_fit = self._update_global_best_batch(
                population, personal_best_pos, personal_best_fit
            )
            
            if g_best_fit < global_best_fit:
                global_best_fit = g_best_fit
                global_best_pos = g_best_pos.copy()
                self.stagnation_counter = 0
                self._record_improvement(g_best_fit)
            else:
                self.stagnation_counter += 1
            
            best_history.append(global_best_fit)
            
            # Diversity monitoring and restart
            if self._should_restart(generation):
                population, velocities, personal_best_pos, personal_best_fit = \
                    self._restart_diversified(population, personal_best_pos)
            
            generation += 1
        
        return global_best_fit, global_best_pos
    
    def _initialize_population(self):
        """Initialize swarm positions uniformly in bounds."""
        return np.random.uniform(
            self.lower, self.upper, size=(self.np, self.dim)
        )
    
    def _initialize_velocities(self, population):
        """Initialize velocities with scaled random values."""
        range_val = 0.1 * (self.upper - self.lower)
        return np.random.uniform(-range_val, range_val, size=(self.np, self.dim))
    
    def _clip_to_bounds(self, candidates):
        """Clip all candidates to search bounds."""
        return np.clip(candidates, self.lower, self.upper)
    
    def _evaluate_batch(self, func, population):
        """Evaluate fitness for entire population in one batch call."""
        clipped = self._clip_to_bounds(population)
        fitness = func(clipped)
        
        # Handle NaN values by assigning worst fitness
        nan_mask = np.isnan(fitness)
        if np.any(nan_mask):
            fitness = np.where(nan_mask, 1e20, fitness)
        
        return fitness
    
    def _update_personal_best_batch(self, population, fitness):
        """Track personal best positions and fitness for all particles."""
        personal_best_pos = population.copy()
        personal_best_fit = fitness.copy()
        return personal_best_pos, personal_best_fit
    
    def _update_global_best_batch(self, population, personal_best_pos, personal_best_fit):
        """Find global best across all personal bests."""
        best_idx = np.argmin(personal_best_fit)
        return personal_best_pos[best_idx].copy(), personal_best_fit[best_idx]
    
    def _build_trial_batch(self, population, velocities, personal_best_pos,
                          neighborhood_best, generation):
        """Build trial positions using PSO update with adaptive parameters."""
        # Compute adaptive inertia weight
        progress = generation / max(generation, 1)
        w = self.w_max - (self.w_max - self.w_min) * progress
        
        # Compute adaptive cognitive and social coefficients
        c1 = self.c1_max - (self.c1_max - self.c1_min) * progress
        c2 = self.c2_min + (self.c2_max - self.c2_min) * progress
        
        # Random factors for each particle
        r1 = np.random.uniform(0, 1, size=(self.np, self.dim))
        r2 = np.random.uniform(0, 1, size=(self.np, self.dim))
        
        # Velocity update components
        cognitive = c1 * r1 * (personal_best_pos - population)
        social = c2 * r2 * (neighborhood_best - population)
        
        # Update velocity with inertia
        velocities = w * velocities + cognitive + social
        
        # Velocity clamping to prevent explosion
        velocities = np.clip(velocities, -self.v_max, self.v_max)
        
        # Position update
        trials = population + velocities
        
        # Store updated velocities back (needed for next iteration)
        self._current_velocities = velocities
        
        return self._clip_to_bounds(trials)
    
    def _select_survivors_batch(self, population, velocities, trials,
                                personal_best_pos, personal_best_fit, trial_fitness):
        """Select survivors by comparing trial fitness to current fitness."""
        # Determine which trials improve
        improves = trial_fitness < personal_best_fit[-len(trial_fitness):] \
            if len(trial_fitness) == len(population) else \
            trial_fitness < personal_best_fit[:len(trial_fitness)]
        
        # Handle length mismatch
        if len(improves) < len(population):
            improves = np.pad(improves, (0, len(population) - len(improves)),
                            constant_values=False)
        
        # Update population with improving trials
        new_population = population.copy()
        new_population[improves] = trials[improves[:len(trials)]]
        
        # Update velocities for improved particles
        new_velocities = velocities.copy()
        new_velocities[improves] = self._current_velocities[improves[:len(self._current_velocities)]]
        
        # Update personal bests
        new_personal_best_pos = personal_best_pos.copy()
        new_personal_best_fit = personal_best_fit.copy()
        
        update_mask = improves[:len(trials)]
        new_personal_best_pos[update_mask] = trials[update_mask]
        new_personal_best_fit[update_mask] = trial_fitness[update_mask]
        
        return new_population, new_velocities, new_personal_best_pos, new_personal_best_fit
    
    def _compute_neighborhood_best_batch(self, personal_best_pos, personal_best_fit):
        """Compute neighborhood best for each particle based on current topology."""
        if self.topology_type == 0:
            return self._ring_topology_best(personal_best_pos, personal_best_fit)
        elif self.topology_type == 1:
            return self._von_neumann_topology_best(personal_best_pos, personal_best_fit)
        elif self.topology_type == 2:
            return self._fully_connected_topology_best(personal_best_pos, personal_best_fit)
        else:
            return self._random_topology_best(personal_best_pos, personal_best_fit)
    
    def _ring_topology_best(self, personal_best_pos, personal_best_fit):
        """Ring topology: each particle connects to two neighbors."""
        neighborhood_best = np.zeros_like(personal_best_pos)
        for i in range(self.np):
            left = (i - 1) % self.np
            right = (i + 1) % self.np
            candidates = [personal_best_pos[i], personal_best_pos[left], 
                         personal_best_pos[right]]
            fits = [personal_best_fit[i], personal_best_fit[left], 
                   personal_best_fit[right]]
            neighborhood_best[i] = candidates[np.argmin(fits)]
        return neighborhood_best
    
    def _von_neumann_topology_best(self, personal_best_pos, personal_best_fit):
        """Von Neumann topology: 4-connected grid neighbors."""
        neighborhood_best = np.zeros_like(personal_best_pos)
        rows = int(np.ceil(np.sqrt(self.np)))
        cols = rows
        
        for i in range(self.np):
            row = i // cols
            col = i % cols
            neighbors = [
                (row, (col - 1) % cols),
                (row, (col + 1) % cols),
                ((row - 1) % rows, col),
                ((row + 1) % rows, col)
            ]
            neighbor_indices = [r * cols + c for r, c in neighbors]
            neighbor_indices = [j for j in neighbor_indices if j < self.np]
            neighbor_indices.append(i)
            
            candidates = [personal_best_pos[j] for j in neighbor_indices]
            fits = [personal_best_fit[j] for j in neighbor_indices]
            neighborhood_best[i] = candidates[np.argmin(fits)]
        return neighborhood_best
    
    def _fully_connected_topology_best(self, personal_best_pos, personal_best_fit):
        """Fully connected: global best influences all particles."""
        best_idx = np.argmin(personal_best_fit)
        return np.tile(personal_best_pos[best_idx], (self.np, 1))
    
    def _random_topology_best(self, personal_best_pos, personal_best_fit):
        """Random topology: random subset of particles influence each particle."""
        neighborhood_best = np.zeros_like(personal_best_pos)
        k = max(3, self.np // 10)
        
        for i in range(self.np):
            indices = np.random.choice(self.np, min(k, self.np), replace=False)
            neighbor_fits = personal_best_fit[indices]
            neighborhood_best[i] = personal_best_pos[indices[np.argmin(neighbor_fits)]]
        return neighborhood_best
    
    def _select_topology(self, current_best, best_history):
        """Adaptively select topology based on performance history."""
        if len(best_history) < 10:
            return
        
        recent = best_history[-10:]
        improvement_rate = (recent[0] - recent[-1]) / max(abs(recent[0]), 1e-10)
        
        # Score each topology based on recent performance
        scores = np.zeros(4)
        
        # Favor ring for exploration, fully connected for exploitation
        if improvement_rate < 0.01:
            # Stagnation: favor exploration
            scores[0] = 1.5  # Ring
            scores[1] = 1.2  # Von Neumann
            scores[3] = 1.0  # Random
            scores[2] = 0.5  # Fully connected
        else:
            # Good progress: favor exploitation
            scores[2] = 1.5  # Fully connected
            scores[1] = 1.2  # Von Neumann
            scores[0] = 0.8  # Ring
            scores[3] = 1.0  # Random
        
        # Add some randomness to topology selection
        scores += np.random.uniform(-0.2, 0.2, 4)
        
        self.topology_type = np.argmax(scores)
        self.param_rewards['topology'][self.topology_type] += 1.0
    
    def _adapt_parameters(self, generation, best_history):
        """Adapt PSO parameters based on recent performance."""
        if len(best_history) < 20:
            return
        
        recent = best_history[-20:]
        improvement = recent[0] - recent[-1]
        relative_improvement = improvement / max(abs(recent[0]), 1e-10)
        
        # Adapt inertia weight based on diversity
        diversity = self._compute_diversity()
        
        if diversity < 0.1:
            # Low diversity: increase exploration
            self.w_max = min(0.95, self.w_max + 0.01)
            self.w_min = min(0.5, self.w_min + 0.01)
            self.param_rewards['w'] += 0.5
        elif diversity > 0.5:
            # High diversity: increase exploitation
            self.w_max = max(0.7, self.w_max - 0.01)
            self.w_min = max(0.3, self.w_min - 0.01)
        
        # Adapt cognitive/social based on performance
        if relative_improvement > 0.1:
            self.param_rewards['c1'] += 0.3
            self.param_rewards['c2'] += 0.5
        elif relative_improvement < 0.01:
            self.param_rewards['c1'] += 0.5
            self.param_rewards['c2'] += 0.2
    
    def _compute_diversity(self):
        """Compute population diversity as average distance to centroid."""
        if not hasattr(self, '_last_population'):
            return 0.5
        
        centroid = np.mean(self._last_population, axis=0)
        distances = np.linalg.norm(self._last_population - centroid, axis=1)
        avg_distance = np.mean(distances)
        
        # Normalize by search space diagonal
        search_space_diag = np.sqrt(self.dim) * (self.upper - self.lower)
        return avg_distance / search_space_diag
    
    def _should_restart(self, generation):
        """Determine if restart is needed due to stagnation or low diversity."""
        diversity = self._compute_diversity()
        
        # Restart if diversity is too low
        if diversity < self.diversity_threshold:
            return True
        
        # Restart if stagnant for too long
        if self.stagnation_counter > self.stagnation_limit:
            return True
        
        # Periodic restart to maintain diversity
        if generation > 0 and generation % 200 == 0 and diversity < 0.2:
            return True
        
        return False
    
    def _restart_diversified(self, current_pop, personal_best_pos):
        """Restart with diversified population mixing good solutions."""
        # Keep best 20% of current population
        keep_count = max(1, self.np // 5)
        
        fitness = np.array([personal_best_pos[i].dot(personal_best_pos[i]) 
                          for i in range(self.np)])
        keep_indices = np.argsort(fitness)[:keep_count]
        
        new_population = np.zeros((self.np, self.dim))
        new_velocities = np.zeros((self.np, self.dim))
        new_personal_best = np.zeros((self.np, self.dim))
        
        # Copy kept solutions
        for idx, keep_idx in enumerate(keep_indices):
            new_population[idx] = current_pop[keep_idx].copy()
            new_personal_best[idx] = personal_best_pos[keep_idx].copy()
        
        # Generate new random solutions for rest
        for i in range(keep_count, self.np):
            # Bias towards unexplored regions
            if np.random.random() < 0.7:
                # Random exploration
                new_population[i] = np.random.uniform(
                    self.lower, self.upper, self.dim
                )
            else:
                # Recombination from good solutions
                parents = np.random.choice(keep_count, 2, replace=False)
                new_population[i] = 0.5 * (new_population[parents[0]] + 
                                          new_population[parents[1]]) + \
                    np.random.uniform(-10, 10, self.dim)
        
        # Initialize velocities for new particles
        range_val = 0.1 * (self.upper - self.lower)
        new_velocities = np.random.uniform(-range_val, range_val, size=(self.np, self.dim))
        
        # Reset stagnation counter
        self.stagnation_counter = 0
        
        return self._clip_to_bounds(new_population), new_velocities, \
               new_personal_best, np.full(self.np, 1e20)
    
    def _record_improvement(self, current_best):
        """Record improvement for tracking adaptation success."""
        self.recent_improvements.append(current_best)
        if len(self.recent_improvements) > 50:
            self.recent_improvements.pop(0)
