import numpy as np


class AdaptiveVelocityGuidedDE:
    """
    Hybrid Differential Evolution with Velocity-Guided Mutation and
    Adaptive Inertia Weight. Combines DE's global search strength with
    particle-swarm-like velocity dynamics and fitness-ranked position updates.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 60), 300)  # Population size
        self.lower = -100.0
        self.upper = 100.0
        self.F_init = 0.5
        self.CR_init = 0.9
        self.inertia = 0.7
        self.cognitive = 1.5
        self.social = 1.5
        self.p_best_fraction = 0.1
        self.fitness_rank_weight = 0.3
        self.temporal_drift_rate = 0.05
        self.min_diversity = 1e-6
        self.stagnation_threshold = 50
        self.diversity_threshold = 0.1
        
        # State variables
        self.population = None
        self.velocity = None
        self.fitness = None
        self.p_best = None
        self.p_best_fitness = None
        self.g_best = None
        self.g_best_fitness = np.inf
        self.generation = 0
        self.stagnation_counter = 0
        self.best_history = []
        self.temporal_drift = np.zeros(dim)
        
    def __call__(self, func, stopping_condition):
        self._initialize_population()
        self._initialize_velocity()
        self._initialize_best_tracking()
        
        fitness = self._evaluate_batch(func, self.population)
        self.fitness = fitness.copy()
        self._update_best_batch(fitness)
        
        while not stopping_condition():
            self.generation += 1
            
            # Build trial population using multiple strategies
            trials = self._build_trial_batch()
            trials = self._clip_to_bounds(trial_batch=trials)
            
            if stopping_condition():
                break
            
            trial_fitness = self._evaluate_batch(func, trials)
            trials, trial_fitness = self._handle_budget_truncation(trials, trial_fitness)
            
            if len(trial_fitness) == 0:
                break
            
            # Selection and adaptation
            self.population, self.fitness = self._select_survivors_batch(
                self.population, self.fitness, trials, trial_fitness
            )
            self._update_best_batch(self.fitness)
            
            # Adaptation methods (responsive to tuning per feedback)
            self._adapt_inertia_weight()
            self._update_velocity_batch()
            self._position_update_fitness_rank()
            self._position_update_temporal_drift()
            
            # Diversity monitoring and restart
            if self._compute_diversity() < self.diversity_threshold:
                self._restart_if_stagnant()
                
        return self.g_best_fitness, self.g_best.copy()
    
    def _initialize_population(self):
        """Initialize population using Latin Hypercube Sampling for better coverage."""
        self.population = np.random.uniform(
            self.lower, self.upper, size=(self.np, self.dim)
        )
        # LHS-style scrambling for better spread
        for d in range(self.dim):
            perm = np.random.permutation(self.np)
            self.population[:, d] = self.lower + (self.upper - self.lower) * (
                perm + np.random.uniform(0, 1, self.np)
            ) / self.np
    
    def _initialize_velocity(self):
        """Initialize velocity based on search space range."""
        range_val = self.upper - self.lower
        self.velocity = np.random.uniform(
            -0.1 * range_val, 0.1 * range_val, size=(self.np, self.dim)
        )
    
    def _initialize_best_tracking(self):
        """Initialize personal and global best tracking structures."""
        self.p_best = self.population.copy()
        self.p_best_fitness = np.full(self.np, np.inf)
        self.g_best = None
        self.g_best_fitness = np.inf
        self.generation = 0
        self.stagnation_counter = 0
        self.best_history = []
        self.temporal_drift = np.zeros(self.dim)
    
    def _evaluate_batch(self, func, population_batch):
        """Evaluate fitness for entire population batch at once."""
        fitness = func(population_batch)
        if not isinstance(fitness, np.ndarray):
            fitness = np.array(fitness)
        return fitness
    
    def _handle_budget_truncation(self, trials, trial_fitness):
        """Handle case where func returns fewer values than trials passed."""
        if len(trial_fitness) < len(trials):
            trials = trials[:len(trial_fitness)]
        return trials, trial_fitness
    
    def _clip_to_bounds(self, trial_batch):
        """Clip trial candidates to search bounds."""
        return np.clip(trial_batch, self.lower, self.upper)
    
    def _build_trial_batch(self):
        """Build trial population using hybrid DE/PSO strategies."""
        # Strategy 1: Current-to-pbest/1 with velocity perturbation
        trials_pbest = self._mutate_current_to_pbest_batch()
        
        # Strategy 2: Velocity-guided mutation
        trials_velocity = self._mutate_velocity_guided_batch()
        
        # Strategy 3: Random target with weighted combination
        trials_random = self._mutate_random_weighted_batch()
        
        # Combine strategies based on generation and diversity
        alpha = self._compute_strategy_mix()
        
        trials = (alpha[0] * trials_pbest + 
                  alpha[1] * trials_velocity + 
                  alpha[2] * trials_random)
        
        # Apply crossover
        trials = self._crossover_batch(trials)
        
        return trials
    
    def _compute_strategy_mix(self):
        """Compute mixing coefficients for different mutation strategies."""
        diversity = self._compute_diversity()
        # Early: more exploration (pbest, random). Late: more exploitation (velocity).
        exploration_weight = max(0.2, 1.0 - self.generation / 200)
        velocity_weight = 1.0 - exploration_weight
        random_weight = exploration_weight * 0.3
        
        alpha = np.array([exploration_weight - random_weight, velocity_weight, random_weight])
        return alpha / alpha.sum()
    
    def _mutate_current_to_pbest_batch(self):
        """Current-to-pbest/1 mutation strategy with adaptive F."""
        n_pbest = max(2, int(self.np * self.p_best_fraction))
        pbest_indices = np.argsort(self.p_best_fitness)[:n_pbest]
        
        F = self.F_init * (0.9 + 0.2 * np.random.random())
        F = np.clip(F, 0.3, 1.5)
        
        # Select 3 distinct indices different from target
        indices = np.arange(self.np)
        r1 = np.random.choice(indices, size=self.np, replace=True)
        r2 = np.random.choice(indices, size=self.np, replace=True)
        r3 = np.random.choice(indices, size=self.np, replace=True)
        
        # Ensure different from target
        for i in range(self.np):
            while r1[i] == i:
                r1[i] = np.random.randint(0, self.np)
            while r2[i] == i or r2[i] == r1[i]:
                r2[i] = np.random.randint(0, self.np)
            while r3[i] == i or r3[i] == r1[i] or r3[i] == r2[i]:
                r3[i] = np.random.randint(0, self.np)
        
        pbest_idx = np.random.choice(pbest_indices, size=self.np)
        
        base = self.population
        mutant = (base + F * (self.p_best[pbest_idx] - base) +
                  F * (self.population[r1] - self.population[r2]) +
                  F * (self.population[r3] - base) * 0.5)
        
        return mutant
    
    def _mutate_velocity_guided_batch(self):
        """Mutation using velocity information for direction-guided search."""
        F_velocity = 0.4 * self.inertia
        r = np.random.random(self.np)[:, np.newaxis]
        
        # Blend current position with velocity-guided direction
        velocity_direction = self.velocity * F_velocity
        mutant = self.population + velocity_direction + r * (
            self.g_best - self.population + np.random.randn(self.np, self.dim) * 0.1
        )
        
        return mutant
    
    def _mutate_random_weighted_batch(self):
        """Random target mutation with weighted difference vectors."""
        indices = np.arange(self.np)
        r1 = np.random.choice(indices, size=self.np, replace=True)
        r2 = np.random.choice(indices, size=self.np, replace=True)
        r3 = np.random.choice(indices, size=self.np, replace=True)
        r4 = np.random.choice(indices, size=self.np, replace=True)
        
        F = self.F_init * (0.8 + 0.4 * np.random.random(self.np))[:, np.newaxis]
        
        mutant = (self.population[r1] + 
                  F * (self.population[r2] - self.population[r3]) +
                  F * 0.5 * (self.g_best - self.population[r4]))
        
        return mutant
    
    def _crossover_batch(self, mutant):
        """Binomial crossover with adaptive CR."""
        CR = self.CR_init * (0.8 + 0.4 * np.random.random(self.np))[:, np.newaxis]
        
        # Random dimension for guaranteed change
        j_rand = np.random.randint(0, self.dim, size=self.np)
        
        # Generate crossover mask
        mask = np.random.random((self.np, self.dim)) < CR
        
        # Ensure at least one dimension from mutant
        rows = np.arange(self.np)
        mask[rows, j_rand] = True
        
        trials = np.where(mask, mutant, self.population)
        
        return trials
    
    def _select_survivors_batch(self, pop, fit, trials, trial_fit):
        """Greedy selection between current and trial individuals."""
        better_mask = trial_fit < fit
        
        new_pop = pop.copy()
        new_fit = fit.copy()
        
        new_pop[better_mask] = trials[better_mask]
        new_fit[better_mask] = trial_fit[better_mask]
        
        # Update personal bests
        improved_mask = trial_fit < self.p_best_fitness
        self.p_best[improved_mask] = trials[improved_mask]
        self.p_best_fitness[improved_mask] = trial_fit[improved_mask]
        
        return new_pop, new_fit
    
    def _update_best_batch(self, fitness):
        """Update global best and track stagnation."""
        current_best_idx = np.argmin(fitness)
        current_best_fit = fitness[current_best_idx]
        
        if current_best_fit < self.g_best_fitness:
            improvement = self.g_best_fitness - current_best_fit
            self.g_best_fitness = current_best_fit
            self.g_best = self.population[current_best_idx].copy()
            self.stagnation_counter = 0
            self.best_history.append((self.generation, self.g_best_fitness))
        else:
            self.stagnation_counter += 1
        
        # Keep history bounded
        if len(self.best_history) > 100:
            self.best_history = self.best_history[-50:]
    
    def _compute_diversity(self):
        """Compute population diversity using average pairwise distance."""
        if self.population is None or len(self.population) < 2:
            return 1.0
        
        # Sample for efficiency
        sample_size = min(50, len(self.population))
        indices = np.random.choice(len(self.population), sample_size, replace=False)
        sample = self.population[indices]
        
        # Compute pairwise distances
        diff = sample[:, np.newaxis, :] - sample[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Average of upper triangle
        n = len(sample)
        mask = np.triu_indices(n, k=1)
        avg_distance = np.mean(distances[mask])
        
        # Normalize by search space diagonal
        space_diagonal = np.sqrt(self.dim) * (self.upper - self.lower)
        
        return avg_distance / space_diagonal
    
    def _adapt_inertia_weight(self):
        """Adapt inertia weight based on fitness improvement and diversity."""
        if len(self.best_history) < 2:
            return
        
        recent_improvement = self.best_history[-1][1] - self.best_history[-2][1]
        diversity = self._compute_diversity()
        
        if self.stagnation_counter > 10:
            # Increase exploration when stuck
            self.inertia = min(0.95, self.inertia * 1.05)
        elif recent_improvement > 0 and self.stagnation_counter < 5:
            # Decrease inertia to refine solution when improving
            self.inertia = max(0.3, self.inertia * 0.95)
        else:
            # Moderate adjustment based on diversity
            if diversity < 0.2:
                self.inertia = min(0.9, self.inertia * 1.02)
            elif diversity > 0.5:
                self.inertia = max(0.4, self.inertia * 0.98)
    
    def _update_velocity_batch(self):
        """Update velocity using PSO-like formula with adaptive coefficients."""
        cognitive = self.cognitive * (0.9 + 0.2 * np.random.random())
        social = self.social * (0.9 + 0.2 * np.random.random())
        
        # Velocity update: inertia * v + cognitive * r1 * (pbest - x) + social * r2 * (gbest - x)
        r1 = np.random.random((self.np, self.dim))
        r2 = np.random.random((self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.p_best - self.population)
        social_component = social * r2 * (self.g_best - self.population)
        
        self.velocity = (self.inertia * self.velocity + 
                        cognitive_component + social_component)
        
        # Velocity clamping
        max_velocity = 0.2 * (self.upper - self.lower)
        vel_magnitude = np.sqrt(np.sum(self.velocity ** 2, axis=1, keepdims=True))
        vel_magnitude = np.maximum(vel_magnitude, 1e-10)
        scale = np.minimum(1.0, max_velocity / vel_magnitude)
        self.velocity *= scale
    
    def _position_update_fitness_rank(self):
        """Apply position update based on fitness ranking (responsive method)."""
        if np.random.random() > self.fitness_rank_weight:
            return
        
        # Rank individuals by fitness
        ranks = np.argsort(np.argsort(self.fitness))
        rank_normalized = ranks / self.np
        
        # Blend toward better-ranked individuals' positions
        for i in range(self.np):
            better_mask = ranks < ranks[i]
            if np.any(better_mask):
                # Move slightly toward better individuals
                better_positions = self.population[better_mask]
                centroid = np.mean(better_positions, axis=0)
                blend = 0.05 * rank_normalized[i]
                self.population[i] = (1 - blend) * self.population[i] + blend * centroid
    
    def _position_update_temporal_drift(self):
        """Apply temporal drift to population (responsive method)."""
        if self.generation < 10:
            return
        
        # Update temporal drift based on global best movement
        if len(self.best_history) >= 2:
            drift_direction = self.best_history[-1][1] - self.best_history[-2][1]
            if drift_direction < 0:
                # Improving: reduce drift
                self.temporal_drift *= (1 - self.temporal_drift_rate)
            else:
                # Stagnating: increase drift
                self.temporal_drift += np.random.randn(self.dim) * 0.1 * self.temporal_drift_rate
        
        # Apply small temporal perturbation
        if np.random.random() < 0.2:
            noise = np.random.randn(self.np, self.dim) * np.abs(self.temporal_drift)
            self.population += noise * 0.01
            self.population = self._clip_to_bounds(self.population)
    
    def _restart_if_stagnant(self):
        """Restart population when stagnating to maintain diversity."""
        if self.stagnation_counter < self.stagnation_threshold:
            return
        
        print(f"Restarting at generation {self.generation}, stagnation={self.stagnation_counter}")
        
        # Keep best individual
        best_idx = np.argmin(self.fitness)
        best_individual = self.population[best_idx].copy()
        
        # Reinitialize population around best
        spread = 0.5 * (self.upper - self.lower)
        self.population = np.random.uniform(
            best_individual - spread, best_individual + spread, 
            size=(self.np, self.dim)
        )
        self.population = self._clip_to_bounds(self.population)
        
        # Reset velocity
        self._initialize_velocity()
        
        # Inject best at random location
        inject_idx = np.random.randint(0, self.np)
        self.population[inject_idx] = best_individual
        
        # Reset tracking
        self.stagnation_counter = 0
        self.inertia = 0.7
        self.temporal_drift *= 0.1
