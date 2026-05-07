```python
import numpy as np


class AdaptiveCompositeDE:
    """
    Adaptive Composite Differential Evolution with Ring Topology and Strategy Pool
    
    Key design elements:
    - Ring topology for neighborhood information exchange
    - Multiple mutation strategies with weighted selection
    - Per-dimension F adaptation inspired by EPSDE
    - Success-based strategy score updates
    - Adaptive population size reduction (like L-SHADE)
    - Diversity-triggered restart mechanism
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np_init = min(10 * dim, 200)  # Initial population size
        self.np_min = 4 * dim              # Minimum population size
        self.bounds = (-100.0, 100.0)
        
        # DE parameters
        self.F_base = 0.5
        self.Cr_base = 0.5
        
        # Strategy pool with different mutation schemes
        self.strategies = [
            'composite_ring',
            'best_deviation', 
            'composite_global',
            'neighbor_best_ring'
        ]
        self.strategy_scores = np.ones(len(self.strategies))
        self.strategy_weights = np.ones(len(self.strategies)) / len(self.strategies)
        self.current_strategy_idx = 0
        
        # Per-dimension F values
        self.F_per_dim = np.full(dim, self.F_base)
        
        # Success history for adaptation
        self.success_F_history = []
        self.success_Cr_history = []
        self.history_size = 20
        
        # Stagnation tracking
        self.stagnation_counter = 0
        self.max_stagnation = 15
        self.best_fitness_history = []
        
        # Diversity management
        self.min_diversity = 1e-4 * dim
        
        # Current state
        self.population = None
        self.fitness = None
        self.np_current = self.np_init
        self.current_F = self.F_base
        self.current_Cr = self.Cr_base
        self.best_idx = None
        self.best_fitness = np.inf
        self.best_position = None
        self.iteration = 0
        
        # Ring topology cache
        self._build_ring_topology()
        
        # Archive for successful solutions (used in some strategies)
        self.archive = []
        
    def _build_ring_topology(self):
        """Pre-compute ring topology indices for efficiency"""
        n = self.np_init
        self.ring_left = np.roll(np.arange(n), 1)
        self.ring_right = np.roll(np.arange(n), -1)
        self.ring_left2 = np.roll(np.arange(n), 2)
        self.ring_right2 = np.roll(np.arange(n), -2)
        
    def _initialize_population(self, func):
        """Initialize population with Latin Hypercube Sampling for better coverage"""
        low, high = self.bounds
        n, d = self.np_init, self.dim
        
        # Simple uniform initialization with clipping
        self.population = np.random.uniform(low, high, (n, d))
        self.population = np.clip(self.population, low, high)
        
        # Batch evaluation
        self.fitness = func(self.population)
        self.np_current = n
        
        # Handle budget exhaustion
        if len(self.fitness) < n:
            actual_n = len(self.fitness)
            self.population = self.population[:actual_n]
            self.fitness = self.fitness[:actual_n]
            self.np_current = actual_n
            self._build_ring_topology()
        
        # Track best
        self.best_idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[self.best_idx]
        self.best_position = self.population[self.best_idx].copy()
        self.best_fitness_history = [self.best_fitness]
        
        return self.population, self.fitness
    
    def _select_strategy(self):
        """Roulette wheel selection based on strategy scores"""
        weights = self.strategy_weights / self.strategy_weights.sum()
        self.current_strategy_idx = np.random.choice(len(self.strategies), p=weights)
        return self.strategies[self.current_strategy_idx]
    
    def _mutate_composite_ring(self):
        """Composite mutation using ring topology: best + F*(r1 - r2) + F*(r3 - r4)"""
        n = self.np_current
        # Select 4 distinct ring neighbors for each target
        r1 = self.ring_left
        r2 = self.ring_right
        r3 = self.ring_left2
        r4 = self.ring_right2
        
        # Two differential terms
        d1 = self.population[r1] - self.population[r2]
        d2 = self.population[r3] - self.population[r4]
        
        # Best-based composite
        mutants = self.population[self.best_idx] + self.current_F * d1 + 0.5 * self.current_F * d2
        return mutants
    
    def _mutate_best_deviation(self):
        """Best + F * (p_best - p_curr) + F * (neighbor - neighbor2)"""
        n = self.np_current
        r1 = self.ring_left
        r2 = self.ring_right
        
        # Deviation from current position
        deviation = self.population[self.best_idx] - self.population
        
        # Ring differential
        ring_diff = self.population[r1] - self.population[r2]
        
        mutants = self.population + self.current_F * deviation + 0.5 * self.current_F * ring_diff
        return mutants
    
    def _mutate_composite_global(self):
        """Composite of global information: centroid + best + differential"""
        n = self.np_current
        centroid = np.mean(self.population, axis=0)
        
        # Random selection for differential
        indices = np.random.permutation(n)
        r1, r2 = indices[:n//2], indices[n//2:]
        
        # Pad if odd
        if len(r1) < len(r2):
            r2 = r2[:-1]
        elif len(r2) < len(r1):
            r1 = r1[:-1]
        
        mutants = centroid + self.current_F * (self.population[r1] - self.population[r2])
        
        # Add best influence for some
        mask = np.random.rand(n) < 0.3
        mutants[mask] = self.population[mask] + self.current_F * (self.best_position - self.population[mask])
        
        return mutants
    
    def _mutate_neighbor_best_ring(self):
        """Each individual uses its ring neighbor's best-so-far information"""
        n = self.np_current
        mutants = np.empty((n, self.dim))
        
        # For each position, find the best in its neighborhood
        for i in range(n):
            neighborhood = [i, self.ring_left[i], self.ring_right[i]]
            neighborhood_fitness = self.fitness[neighborhood]
            best_neighbor = neighborhood[np.argmin(neighborhood_fitness)]
            
            r1, r2 = self.ring_left[i], self.ring_right[i]
            
            # Mutation using neighborhood best
            mutants[i] = self.population[best_neighbor] + self.current_F * (
                self.population[r1] - self.population[r2]
            )
        
        return mutants
    
    def _mutate_batch(self):
        """Select and apply mutation strategy"""
        strategy = self._select_strategy()
        
        if strategy == 'composite_ring':
            mutants = self._mutate_composite_ring()
        elif strategy == 'best_deviation':
            mutants = self._mutate_best_deviation()
        elif strategy == 'composite_global':
            mutants = self._mutate_composite_global()
        else:  # neighbor_best_ring
            mutants = self._mutate_neighbor_best_ring()
        
        # Apply per-dimension F adaptation
        mutants = self.population + self.F_per_dim * (mutants - self.population)
        
        return mutants, strategy
    
    def _crossover_batch(self, population, mutants):
        """Binomial crossover with adaptive Cr"""
        n, d = self.np_current, self.dim
        
        # Generate crossover mask
        Cr = np.random.uniform(0.3, 0.9, n)  # Randomize Cr per individual
        cr_mask = np.random.rand(n, d) < Cr[:, np.newaxis]
        
        # Ensure at least one dimension comes from mutant
        j_rand = np.random.randint(0, d, n)
        cr_mask[np.arange(n), j_rand] = True
        
        # Create trial population
        trials = np.where(cr_mask, mutants, population)
        
        # Clip to bounds
        trials = np.clip(trials, self.bounds[0], self.bounds[1])
        
        return trials, Cr
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness, Cr_used):
        """One-to-one selection based on fitness improvement"""
        n = self.np_current
        
        # Find improvements
        improvements = trial_fitness < fitness
        
        # Create new population
        new_population = np.where(improvements[:, np.newaxis], trials, population)
        new_fitness = np.where(improvements, trial_fitness, fitness)
        
        # Record successful parameters
        if np.any(improvements):
            success_mask = improvements
            self.success_F_history.extend([self.current_F] * np.sum(success_mask))
            self.success_Cr_history.extend(Cr_used[success_mask])
            
            # Trim history
            if len(self.success_F_history) > self.history_size * self.np_current:
                self.success_F_history = self.success_F_history[-self.history_size * self.np_current:]
                self.success_Cr_history = self.success_Cr_history[-self.history_size * self.np_current:]
        
        return new_population, new_fitness, improvements
    
    def _adapt_parameters(self):
        """Adapt F and Cr based on success history"""
        if len(self.success_F_history) > 0:
            # Weighted average of successful parameters
            self.current_F = 0.1 * self.F_base + 0.9 * np.mean(self.success_F_history)
            self.current_Cr = 0.1 * self.Cr_base + 0.9 * np.mean(self.success_Cr_history)
            
            # Update per-dimension F values
            self.F_per_dim = np.clip(
                self.F_per_dim * 0.9 + 0.1 * np.random.uniform(0.4, 0.9, self.dim),
                0.1, 1.0
            )
        else:
            # Reset to base values
            self.current_F = self.F_base
            self.current_Cr = self.Cr_base
    
    def _update_strategy_scores(self, strategy, improved_count, total_count):
        """Update strategy selection scores based on success rate"""
        idx = self.strategies.index(strategy)
        success_rate = improved_count / max(total_count, 1)
        
        # Exponential moving average update
        self.strategy_scores[idx] = 0.8 * self.strategy_scores[idx] + 0.2 * success_rate
        
        # Recompute weights
        self.strategy_weights = self.strategy_scores / self.strategy_scores.sum()
    
    def _check_stagnation(self):
        """Check if best fitness has stagnated"""
        self.best_fitness_history.append(self.best_fitness)
        history_len = len(self.best_fitness_history)
        
        if history_len > self.max_stagnation:
            recent_best = min(self.best_fitness_history[-self.max_stagnation:])
            if recent_best >= self.best_fitness_history[0]:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
    
    def _compute_diversity(self):
        """Compute population diversity using average pairwise distance"""
        if self.np_current < 2:
            return 0.0
        
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_population_size(self):
        """Linear population size reduction (inspired by L-SHADE)"""
        # Only adapt if population is larger than minimum
        if self.np_current > self.np_min:
            reduction = max(1, self.np_init // 200)
            new_np = max(self.np_min, self.np_current - reduction)
            
            if new_np < self.np_current:
                # Remove worst individuals
                sorted_indices = np.argsort(self.fitness)
                keep_indices = sorted_indices[:new_np]
                
                self.population = self.population[keep_indices]
                self.fitness = self.fitness[keep_indices]
                self.np_current = new_np
                self._build_ring_topology()
    
    def _restart_if_needed(self, func):
        """Restart population if stagnant or low diversity"""
        should_restart = False
        reason = ""
        
        # Check stagnation
        if self.stagnation_counter >= 2:
            should_restart = True
            reason = "stagnation"
        # Check diversity
        elif self._compute_diversity() < self.min_diversity:
            should_restart = True
            reason = "low_diversity"
        
        if should_restart:
            # Keep best solution
            best_pos = self.best_position.copy()
            best_fit = self.best_fitness
            
            # Reinitialize population
            self.population = np.random.uniform(
                self.bounds[0], self.bounds[1], 
                (self.np_init, self.dim)
            )
            self.population = np.clip(self.population, self.bounds[0], self.bounds[1])
            
            # Evaluate
            self.fitness = func(self.population)
            
            if len(self.fitness) < self.np_init:
                actual_n = len(self.fitness)
                self.population = self.population[:actual_n]
                self.fitness = self.fitness[:actual_n]
            
            self.np_current = len(self.population)
            self._build_ring_topology()
            
            # Reset tracking
            self.stagnation_counter = 0
            self.best_fitness_history = []
            self.success_F_history = []
            self.success_Cr_history = []
            
            # Inject best solution
            if self.np_current > 0:
                self.population[0] = best_pos
                self.fitness[0] = best_fit
                
                # Re-evaluate to ensure consistency
                new_fitness = func(self.population[:1])[0]
                if not np.isnan(new_fitness):
                    self.fitness[0] = new_fitness
                    if new_fitness < self.best_fitness:
                        self.best_fitness = new_fitness
                        self.best_position = best_pos.copy()
            
            # Update best tracking
            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[current_best_idx]
                self.best_position = self.population[current_best_idx].copy()
        
        return should_restart, reason
    
    def _update_best_tracking(self, fitness):
        """Update best solution if improvement found"""
        current_best_idx = np.argmin(fitness)
        current_best_fitness = fitness[current_best_idx]
        
        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_position = self.population[current_best_idx].copy()
            self.best_idx = current_best_idx
            return True
        return False
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop"""
        # Initialize
        self.population, self.fitness = self._initialize_population(func)
        self.iteration = 0
        
        # Main loop
        while not stopping_condition():
            self.iteration += 1
            
            # Mutation phase
            mutants, strategy_used = self._mutate_batch()
            
            # Crossover phase
            trials, Cr_used = self._crossover_batch(self.population, mutants)
            
            # Check budget before evaluation
            if stopping_condition():
                break
            
            # Evaluate trials (batch evaluation)
            trial_fitness = func(trials)
            
            # Handle budget exhaustion
            if len(trial_fitness) < len(trials):
                trials = trials[:len(trial_fitness)]
                trial_fitness = trial_fitness[:len(trial_fitness)]
                if len(trial_fitness) == 0:
                    break
            
            # Check stopping condition after evaluation
            if stopping_condition():
                break
            
            # Selection phase
            self.population, self.fitness, improvements = self._select_survivors_batch(
                self.population, self.fitness, trials, trial_fitness, Cr_used
            )
            
            # Update best tracking
            improved = self._update_best_tracking(self.fitness)
            if improved:
                self.stagnation_counter = 0
            
            # Update strategy scores
            improved_count = np.sum(improvements)
            self._update_strategy_scores(strategy_used, improved_count, self.np_current)
            
            # Adaptation phases
            self._adapt_parameters()
            self._check_stagnation()
            
            # Restart if needed
            self._restart_if_needed(func)
            
            # Adapt population size (every 25 iterations)
            if self.iteration % 25 == 0:
                self._adapt_population_size()
        
        return self.best_fitness, self.best_position
```