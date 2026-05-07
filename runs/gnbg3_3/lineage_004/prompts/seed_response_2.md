```python
import numpy as np


class CovarianceRankParticleSwarm:
    """
    Hybrid optimizer combining CMA-ES-style covariance adaptation with
    particle swarm dynamics and fitness-ranked position updates.
    Operates entirely in vectorized batch mode.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower = -100.0
        self.upper = 100.0
        self.NP = min(8 * dim, 300)  # Population size
        
        # Evolution paths for covariance adaptation
        self.p_c = np.zeros(dim)
        self.p_s = np.zeros(dim)
        
        # Covariance matrix and step size
        self.C = np.eye(dim)
        self.sigma = 0.5 * (self.upper - self.lower) / 3.0
        
        # Particle swarm components
        self.velocity = np.zeros((self.NP, dim))
        self.inertia_weight = 0.729
        self.cognitive_coeff = 1.49445
        self.social_coeff = 1.49445
        
        # Best positions
        self.personal_best_fitness = np.full(self.NP, np.inf)
        self.personal_best_pos = np.zeros((self.NP, dim))
        
        # Rank-based adaptation
        self.rank_decay = 0.99
        
        # Temporal drift parameters
        self.drift_strength = 0.1
        self.drift_direction = np.zeros(dim)
        
        # Strategy parameters
        self.cc = 1.0 / (dim + 1)
        self.cs = (dim + 2) / (dim + 2 + dim)
        self.c1 = 2.0 / (dim ** 0.5)
        self.cmu = 2.0 / (dim ** 0.5) ** 2
        
        # Diversity tracking
        self.diversity_threshold = 1e-6
        self.stagnation_counter = 0
        self.max_stagnation = 50
        
        # Success history for rank adaptation
        self.success_history = np.zeros(10)
        self.success_idx = 0

    def _initialize_population(self):
        """Initialize population within bounds."""
        pop = np.random.uniform(
            self.lower, self.upper, size=(self.NP, self.dim)
        )
        return pop

    def _clip_to_bounds(self, population):
        """Clip population to search bounds."""
        return np.clip(population, self.lower, self.upper)

    def _compute_fitness_ranks(self, fitness):
        """Compute rank-based weights from fitness values."""
        ranks = np.argsort(np.argsort(fitness))
        ranks = ranks / (len(ranks) - 1)  # Normalize to [0, 1]
        return ranks

    def _update_personal_best(self, population, fitness):
        """Update personal best positions and fitness for each particle."""
        improved = fitness < self.personal_best_fitness
        self.personal_best_fitness[improved] = fitness[improved]
        self.personal_best_pos[improved] = population[improved]
        return self.personal_best_pos.copy()

    def _velocity_update_batch(self, population, global_best, personal_best):
        """Update velocities using PSO formula with covariance guidance."""
        r1 = np.random.uniform(0, 1, size=(self.NP, self.dim))
        r2 = np.random.uniform(0, 1, size=(self.NP, self.dim))
        
        # Cognitive component
        cognitive = self.cognitive_coeff * r1 * (personal_best - population)
        
        # Social component
        social = self.social_coeff * r2 * (global_best - population)
        
        # Covariance-guided component
        cov_guide = self.sigma * np.sqrt(self.C) @ np.random.randn(self.dim)
        cov_guide = np.tile(cov_guide, (self.NP, 1)) * np.random.uniform(0, 1, size=(self.NP, self.dim))
        
        # Update velocity
        self.velocity = self.inertia_weight * self.velocity + cognitive + social + 0.1 * cov_guide
        
        # Velocity clamping
        v_max = 0.2 * (self.upper - self.lower)
        self.velocity = np.clip(self.velocity, -v_max, v_max)
        
        return self.velocity

    def _position_update_batch(self, population, velocity):
        """Update positions using velocity."""
        new_pos = population + velocity
        return self._clip_to_bounds(new_pos)

    def _position_update_temporal_drift(self, population, fitness):
        """Apply temporal drift based on fitness landscape estimation."""
        # Compute weighted centroid
        ranks = self._compute_fitness_ranks(fitness)
        weights = 1.0 / (1.0 + fitness - fitness.min() + 1e-10)
        weights = weights / weights.sum()
        centroid = (population * weights[:, np.newaxis]).sum(axis=0)
        
        # Update drift direction with momentum
        self.drift_direction = 0.9 * self.drift_direction + 0.1 * (centroid - population.mean(axis=0))
        
        # Apply drift perturbation
        drift_matrix = np.eye(self.dim) + 0.05 * np.outer(self.drift_direction, self.drift_direction)
        drifted = (drift_matrix @ population.T).T
        
        return self._clip_to_bounds(drifted)

    def _position_update_fitness_rank(self, population, fitness):
        """Update positions based on fitness ranking with rank-based mutation."""
        ranks = self._compute_fitness_ranks(fitness)
        
        # Higher ranked individuals get more exploration
        exploration_scale = 0.5 * (1.0 - ranks[:, np.newaxis])
        
        # Generate rank-based perturbations
        rank_mutations = np.random.randn(self.NP, self.dim) * exploration_scale * self.sigma
        rank_mutations = rank_mutations @ np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
        
        # Apply mutation scaled by rank
        mutated = population + rank_mutations * self.rank_decay
        
        return self._clip_to_bounds(mutated)

    def _adapt_inertia_weight(self, improvement_rate):
        """Adapt inertia weight based on recent improvement."""
        if improvement_rate > 0.2:
            self.inertia_weight = min(0.95, self.inertia_weight * 1.05)
        elif improvement_rate < 0.05:
            self.inertia_weight = max(0.4, self.inertia_weight * 0.95)
        return self.inertia_weight

    def _adapt_covariance_batch(self, selected_pop, parent_pop, fitness):
        """Adapt covariance matrix using weighted rank-μ update."""
        # Compute weights based on fitness
        weights = self._compute_fitness_ranks(fitness)
        weights = np.maximum(weights, 0.1)  # Ensure minimum weight
        
        # Weighted mean shift
        mean_old = parent_pop.mean(axis=0)
        mean_new = selected_pop.mean(axis=0)
        mean_diff = mean_new - mean_old
        
        # Update evolution paths
        self.p_s = (1 - self.cs) * self.p_s + np.sqrt(self.cs * (2 - self.cs)) * mean_diff / self.sigma
        
        # Cumulation for covariance
        self.p_c = (1 - self.cc) * self.p_c + np.sqrt(self.cc * (2 - self.cc)) * mean_diff / self.sigma
        
        # Rank-μ update for covariance
        y = (selected_pop - mean_old) / self.sigma
        rank_term = np.zeros((self.dim, self.dim))
        for i in range(len(selected_pop)):
            y_i = y[i].reshape(-1, 1)
            rank_term += weights[i] * (y_i @ y_i.T)
        
        # Update covariance matrix
        self.C = (1 - self.cmu) * self.C + self.cmu * rank_term + 1e-8 * np.eye(self.dim)
        
        # Ensure positive definiteness
        self.C = 0.5 * (self.C + self.C.T)
        
        # Adapt step size
        cn = self.cs / self.dim
        self.sigma *= np.exp((np.linalg.norm(self.p_s) / np.sqrt(self.dim) - 1) * cn)

    def _generate_cma_mutations(self, population):
        """Generate mutations using current covariance matrix."""
        L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
        noise = np.random.randn(self.NP, self.dim) @ L.T
        mutations = population + self.sigma * noise
        return self._clip_to_bounds(mutations), noise

    def _crossover_batch(self, population, mutations, cr=0.9):
        """Binomial crossover between parent and mutant."""
        mask = np.random.uniform(0, 1, size=(self.NP, self.dim)) < cr
        # Ensure at least one dimension from mutant
        mask[np.arange(self.NP), np.random.randint(0, self.dim, size=self.NP)] = True
        trial = np.where(mask, mutations, population)
        return trial

    def _select_survivors_batch(self, population, fitness, trial_pop, trial_fitness):
        """Select survivors based on fitness comparison."""
        better = trial_fitness < fitness
        new_population = population.copy()
        new_population[better] = trial_pop[better]
        new_fitness = fitness.copy()
        new_fitness[better] = trial_fitness[better]
        return new_population, new_fitness

    def _compute_diversity(self, population):
        """Compute population diversity as average pairwise distance."""
        mean_pos = population.mean(axis=0)
        distances = np.sqrt(((population - mean_pos) ** 2).sum(axis=1))
        return distances.mean()

    def _restart_if_stagnant(self, iteration, best_fitness_history):
        """Check for stagnation and trigger restart if needed."""
        if len(best_fitness_history) > self.max_stagnation:
            recent_improvements = np.diff(best_fitness_history[-self.max_stagnation:])
            if np.abs(recent_improvements).max() < 1e-8:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        
        if self.stagnation_counter >= 2:
            return True
        return False

    def _full_restart(self):
        """Perform a full restart of the algorithm state."""
        self.C = np.eye(self.dim)
        self.sigma = 0.5 * (self.upper - self.lower) / 3.0
        self.p_c = np.zeros(self.dim)
        self.p_s = np.zeros(self.dim)
        self.velocity = np.zeros((self.NP, self.dim))
        self.personal_best_fitness = np.full(self.NP, np.inf)
        self.personal_best_pos = np.zeros((self.NP, self.dim))
        self.stagnation_counter = 0

    def __call__(self, func, stopping_condition):
        """Execute the optimization."""
        # Initialize
        population = self._initialize_population()
        fitness = func(population)
        
        # Handle budget exhaustion
        if len(fitness) < self.NP:
            valid_count = len(fitness)
            population = population[:valid_count]
            fitness = fitness[:valid_count]
            if len(fitness) == 0:
                return np.inf, population[0] if len(population) > 0 else np.zeros(self.dim)
        
        # Track global best
        best_idx = np.argmin(fitness)
        f_opt = fitness[best_idx]
        x_opt = population[best_idx].copy()
        
        # Personal best initialization
        self.personal_best_fitness[:len(fitness)] = fitness.copy()
        self.personal_best_pos[:len(fitness)] = population.copy()
        
        best_fitness_history = [f_opt]
        iteration = 0
        improvement_count = 0
        recent_improvements = []
        
        # Main loop
        while not stopping_condition():
            # Update personal bests
            personal_best = self._update_personal_best(population, fitness)
            global_best_pos = population[np.argmin(fitness)].copy()
            
            # Velocity update
            velocity = self._velocity_update_batch(population, global_best_pos, personal_best)
            
            # Position update with multiple strategies
            pos_velocity = self._position_update_batch(population, velocity)
            pos_drift = self._position_update_temporal_drift(pos_velocity, fitness)
            pos_rank = self._position_update_fitness_rank(pos_drift, fitness)
            
            # CMA-based mutations
            cma_mutations, _ = self._generate_cma_mutations(pos_rank)
            trial_pop = self._crossover_batch(pos_rank, cma_mutations)
            
            # Evaluate trial population
            trial_fitness = func(trial_pop)
            
            # Handle budget exhaustion
            if len(trial_fitness) < len(trial_pop):
                trial_pop = trial_pop[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
            
            if len(trial_fitness) == 0:
                break
            
            # Check stopping condition after evaluation
            if stopping_condition():
                break
            
            # Selection
            population, fitness = self._select_survivors_batch(population, fitness, trial_pop, trial_fitness)
            
            # Update global best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < f_opt:
                old_f_opt = f_opt
                f_opt = fitness[current_best_idx]
                x_opt = population[current_best_idx].copy()
                improvement_count += 1
                recent_improvements.append(f_opt - old_f_opt)
            else:
                recent_improvements.append(0.0)
            
            # Keep recent improvements history bounded
            if len(recent_improvements) > 10:
                recent_improvements.pop(0)
            
            # Adaptation
            improvement_rate = improvement_count / max(1, iteration + 1)
            self._adapt_inertia_weight(improvement_rate)
            
            # Adapt covariance
            self._adapt_covariance_batch(population, pos_rank, fitness)
            
            # Diversity check
            diversity = self._compute_diversity(population)
            if diversity < self.diversity_threshold:
                self.sigma *= 1.5  # Increase exploration
            
            # Restart check
            best_fitness_history.append(f_opt)
            if self._restart_if_stagnant(iteration, best_fitness_history):
                population = self._initialize_population()
                fitness = func(population)
                if len(fitness) < self.NP:
                    population = population[:len(fitness)]
                    fitness = fitness[:len(fitness)]
                if len(fitness) == 0:
                    break
                self._full_restart()
                self.personal_best_fitness[:len(fitness)] = fitness.copy()
                self.personal_best_pos[:len(fitness)] = population.copy()
                best_fitness_history = [f_opt]
            
            iteration += 1
            
            # Final budget check
            if len(func(np.zeros((1, self.dim)))) == 0:
                break
        
        return f_opt, x_opt
```