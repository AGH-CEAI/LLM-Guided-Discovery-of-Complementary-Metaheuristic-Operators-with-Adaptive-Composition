import numpy as np


class AdaptiveHybridDEWithTemporalDrift:
    """
    Hybrid Differential Evolution with:
    - Temporal drift mutation incorporating momentum from past improvements
    - Adaptive inertia weight for exploration/exploitation balance
    - Adaptive crossover rate based on success history
    - Diversity-guided restart mechanism
    - Neighborhood-based information sharing
    
    Designed to handle multi-modal, deceptive, and sharp-ridge functions.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(150, max(60, 5 * dim))  # Population size: 5*dim, capped at 150
        self.f_opt = np.inf
        self.x_opt = None
        
        # Bounds
        self.lb = -100.0
        self.ub = 100.0
        
        # DE parameters (will be adapted)
        self.f_scale = 0.5  # Differential weight
        self.cr = 0.7       # Crossover rate
        
        # Temporal/momentum parameters
        self.inertia = 0.7
        self.momentum = np.zeros(dim)
        self.momentum_decay = 0.95
        
        # Adaptation tracking
        self.success_history = []
        self.improvement_history = []
        self.max_history_len = 5
        
        # Stagnation detection
        self.stagnation_counter = 0
        self.stagnation_threshold = 30
        self.last_best_fitness = np.inf
        
        # Diversity tracking
        self.diversity_threshold = 1e-6
        
        # Tracking arrays
        self.population_best_fitness = np.full(self.np, np.inf)
        self.population_best_positions = np.zeros((self.np, dim))
    
    def __call__(self, func, stopping_condition):
        return self._optimize(func, stopping_condition)
    
    def _initialize_population(self):
        """Initialize population uniformly in bounds."""
        pop = np.random.uniform(self.lb, self.ub, size=(self.np, self.dim))
        pop = np.clip(pop, self.lb, self.ub)
        return pop
    
    def _evaluate_batch(self, population, func):
        """
        Evaluate fitness for entire population.
        Handles budget exhaustion by checking returned array length.
        """
        fitness = func(population)
        if len(fitness) < len(population):
            # Budget exhausted - truncate
            fitness = np.array(fitness)
            if len(fitness) == 0:
                return None, None
            population = population[:len(fitness)]
        return fitness, population
    
    def _mutate_temporal_drift(self, population, fitness):
        """
        Temporal drift mutation: combines DE-style differential with 
        momentum from past improvements and adaptive neighborhood.
        """
        np_pop = len(population)
        
        # Get global best
        gbest_idx = np.argmin(fitness)
        gbest_pos = population[gbest_idx].copy()
        
        # Update population bests (PSO-like personal best tracking)
        improved = fitness < self.population_best_fitness
        self.population_best_fitness[improved] = fitness[improved]
        self.population_best_positions[improved] = population[improved]
        
        # Compute momentum from recent improvements
        current_improvement = self.population_best_fitness.mean() - fitness.mean()
        self.improvement_history.append(current_improvement)
        if len(self.improvement_history) > self.max_history_len:
            self.improvement_history.pop(0)
        
        # Update momentum with temporal drift
        self._update_momentum_batch(current_improvement)
        
        # Build mutation targets
        targets = np.zeros((np_pop, self.dim))
        
        for i in range(np_pop):
            # Personal best contribution
            pbest = self.population_best_positions[i]
            
            # Select 2 random indices different from current
            indices = np.random.choice([j for j in range(np_pop) if j != i], 
                                       size=3, replace=False)
            r1, r2, r3 = indices
            
            # DE-style differential with inertia
            de_component = (self.population_best_positions[r1] - 
                          self.population_best_positions[r2])
            
            # Temporal drift component: direction from momentum + gbest attraction
            drift_strength = self.inertia * 0.5
            drift_component = (self.momentum + 
                             0.3 * (gbest_pos - pbest))
            
            # Combine: personal best + scaled DE + temporal drift
            targets[i] = (pbest + 
                         self.f_scale * de_component + 
                         drift_strength * drift_component)
        
        return targets
    
    def _crossover_batch(self, population, mutants):
        """
        Binomial crossover with adaptive rate.
        """
        # Generate random mask
        mask = np.random.random((self.np, self.dim)) < self.cr
        
        # Ensure at least one dimension comes from mutant
        enforce_idx = np.random.randint(0, self.dim, size=self.np)
        mask[np.arange(self.np), enforce_idx] = True
        
        # Create trial vectors
        trials = np.where(mask, mutants, population)
        return trials
    
    def _clip_to_bounds(self, population):
        """Ensure all candidates within search bounds."""
        return np.clip(population, self.lb, self.ub)
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """
        Greedy selection: keep better of target and trial.
        """
        better_mask = trial_fitness < fitness
        new_population = np.where(better_mask[:, np.newaxis], trials, population)
        new_fitness = np.where(better_mask, trial_fitness, fitness)
        
        # Track successes for adaptation
        success_rate = better_mask.mean()
        self.success_history.append(success_rate)
        if len(self.success_history) > self.max_history_len:
            self.success_history.pop(0)
        
        return new_population, new_fitness
    
    def _adapt_inertia_weight(self):
        """
        Adapt inertia weight based on success rate.
        High success -> increase inertia for more exploration.
        Low success -> decrease inertia for more exploitation.
        """
        if len(self.success_history) < 2:
            return
        
        avg_success = np.mean(self.success_history)
        
        if avg_success > 0.3:
            # High acceptance: explore more
            self.inertia = min(0.9, self.inertia * 1.1)
        elif avg_success < 0.1:
            # Low acceptance: exploit more
            self.inertia = max(0.3, self.inertia * 0.9)
    
    def _adapt_crossover_rate(self):
        """
        Adapt crossover rate based on success history.
        High success -> increase CR for more diversity.
        Low success -> decrease CR to preserve good solutions.
        """
        if len(self.success_history) < 2:
            return
        
        avg_success = np.mean(self.success_history)
        
        if avg_success > 0.25:
            self.cr = min(0.95, self.cr * 1.05)
        elif avg_success < 0.1:
            self.cr = max(0.3, self.cr * 0.95)
    
    def _adapt_differential_scale(self):
        """
        Adapt F scale based on recent improvements.
        """
        if len(self.improvement_history) < 2:
            return
        
        recent_improvement = self.improvement_history[-1]
        if recent_improvement > np.mean(self.improvement_history):
            # Good improvement: try larger steps for diversity
            self.f_scale = min(1.0, self.f_scale * 1.05)
        else:
            # Stagnation: reduce scale for fine-tuning
            self.f_scale = max(0.3, self.f_scale * 0.95)
    
    def _update_momentum_batch(self, current_improvement):
        """
        Update momentum vector based on improvement direction.
        """
        if len(self.improvement_history) >= 2:
            prev_improvement = self.improvement_history[-2]
            improvement_delta = current_improvement - prev_improvement
            
            if improvement_delta > 0:
                # Improving: boost momentum in current direction
                self.momentum *= 1.1
            else:
                # Not improving: decay momentum
                self.momentum *= self.momentum_decay
            
            # Clamp momentum magnitude
            momentum_norm = np.linalg.norm(self.momentum)
            if momentum_norm > 10.0:
                self.momentum *= 10.0 / momentum_norm
    
    def _compute_diversity(self, population):
        """Measure population diversity via average pairwise distance."""
        centroid = population.mean(axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return distances.mean()
    
    def _check_stagnation(self, fitness):
        """Detect if algorithm is stagnating."""
        current_best = fitness.min()
        
        if self.last_best_fitness - current_best < 1e-6 * max(1, abs(self.last_best_fitness)):
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        self.last_best_fitness = current_best
        return self.stagnation_counter >= self.stagnation_threshold
    
    def _restart_if_stagnant(self, population, fitness):
        """
        Diversity-guided restart: reinitialize portion of population
        while preserving best individuals.
        """
        if not self._check_stagnation(fitness):
            return population, fitness
        
        diversity = self._compute_diversity(population)
        
        # Determine how many to reinitialize
        if diversity < self.diversity_threshold:
            # Low diversity: reinitialize more aggressively
            n_replace = self.np // 2
        else:
            # Moderate diversity: reinitialize some
            n_replace = self.np // 4
        
        # Sort by fitness, keep best
        sorted_indices = np.argsort(fitness)
        keep_indices = sorted_indices[:self.np - n_replace]
        
        # Generate new random individuals
        new_individuals = np.random.uniform(
            self.lb, self.ub, size=(n_replace, self.dim)
        )
        
        # Combine best with new
        population[:] = np.concatenate([
            population[keep_indices],
            new_individuals
        ])
        
        # Re-evaluate new individuals
        fitness[:] = np.concatenate([
            fitness[keep_indices],
            np.full(n_replace, np.inf)
        ])
        
        # Reset momentum and counters
        self.momentum *= 0.5
        self.stagnation_counter = 0
        
        return population, fitness
    
    def _update_global_best(self, fitness, population):
        """Track and return best solution found."""
        current_best_idx = np.argmin(fitness)
        current_best_fitness = fitness[current_best_idx]
        
        if current_best_fitness < self.f_opt:
            self.f_opt = current_best_fitness
            self.x_opt = population[current_best_idx].copy()
        
        return self.f_opt, self.x_opt
    
    def _optimize(self, func, stopping_condition):
        """
        Main optimization loop with batched operations.
        """
        # Initialize
        population = self._initialize_population()
        fitness, population = self._evaluate_batch(population, func)
        
        if fitness is None:
            return self.f_opt, self.x_opt if self.x_opt is not None else np.zeros(self.dim)
        
        # Initialize tracking arrays
        self.population_best_fitness = fitness.copy()
        self.population_best_positions = population.copy()
        
        # Update best
        self._update_global_best(fitness, population)
        
        # Main loop
        while not stopping_condition():
            # Mutation with temporal drift
            mutants = self._mutate_temporal_drift(population, fitness)
            
            # Crossover
            trials = self._crossover_batch(population, mutants)
            
            # Clip to bounds BEFORE evaluation
            trials = self._clip_to_bounds(trials)
            
            # Check stopping before evaluation
            if stopping_condition():
                break
            
            # Evaluate trials
            trial_fitness, trials = self._evaluate_batch(trials, func)
            
            if trial_fitness is None:
                break
            
            # Selection
            population, fitness = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Update global best
            self._update_global_best(fitness, population)
            
            # Adapt parameters
            self._adapt_inertia_weight()
            self._adapt_crossover_rate()
            self._adapt_differential_scale()
            
            # Restart if stagnant
            population, fitness = self._restart_if_stagnant(population, fitness)
            
            # Re-evaluate if population size changed
            if len(fitness) < self.np:
                self.np = len(fitness)
        
        return self.f_opt, self.x_opt if self.x_opt is not None else population[np.argmin(fitness)]
