import numpy as np


class AdaptiveDirectionalDE:
    """
    Adaptive Directional Differential Evolution (ADDE)
    
    A DE variant with:
    - Ring topology for mutation parent selection
    - Directional memory that biases mutation toward successful directions
    - Adaptive F and Cr based on historical success rates
    - Composite mutation: directional + rand/1 blended adaptively
    - Diversity-triggered restart mechanism
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 60), 300)  # Population size
        self.lower = -100.0
        self.upper = 100.0
        
        # Control parameters
        self.F_base = 0.5
        self.Cr_base = 0.3
        self.F = self.F_base
        self.Cr = self.Cr_base
        
        # Directional memory
        self.direction_memory_size = 5
        self.successful_directions = []
        self.direction_decay = 0.95
        
        # Adaptation tracking
        self.adaptation_window = 10
        self.F_success_history = []
        self.Cr_success_history = []
        
        # Restart control
        self.stagnation_threshold = 50
        self.min_diversity_threshold = 1e-6
        self.generation = 0
        
    def __call__(self, func, stopping_condition):
        return self._optimize(func, stopping_condition)
    
    def _initialize_population(self, func):
        """Initialize using Latin Hypercube Sampling for better spread."""
        sampler = np.linspace(self.lower, self.upper, self.np)
        population = np.zeros((self.np, self.dim))
        
        for d in range(self.dim):
            indices = np.random.permutation(self.np)
            offsets = np.random.uniform(0, 1, self.np)
            population[:, d] = sampler[indices] + offsets * (sampler[1] - sampler[0])
        
        population = self._clip_to_bounds(population)
        fitness = func(population)
        
        if len(fitness) < self.np:
            fitness = np.resize(fitness, self.np)
            fitness[np.isnan(fitness)] = np.inf
        
        best_idx = np.argmin(fitness)
        return population, fitness, fitness[best_idx], population[best_idx].copy()
    
    def _clip_to_bounds(self, population):
        """Hard clip all candidates to search bounds."""
        return np.clip(population, self.lower, self.upper)
    
    def _build_ring_topology(self):
        """Create ring neighbor indices for each population member."""
        indices = np.arange(self.np)
        ring_prev = np.roll(indices, 1)
        ring_next = np.roll(indices, -1)
        return ring_prev, ring_next
    
    def _compute_population_centroid(self, population, fitness):
        """Compute fitness-weighted centroid for directional bias."""
        weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
        weights = weights / np.sum(weights)
        centroid = np.sum(population * weights[:, np.newaxis], axis=0)
        return centroid
    
    def _update_direction_memory(self, successful_mutations, successful_mask):
        """Track successful mutation directions for adaptive biasing."""
        if not np.any(successful_mask):
            self.successful_directions = [
                d * self.direction_decay for d in self.successful_directions
            ]
            return
        
        for mutation_vec in successful_mutations[successful_mask]:
            self.successful_directions.append(mutation_vec.copy())
        
        if len(self.successful_directions) > self.direction_memory_size:
            self.successful_directions = self.successful_directions[-self.direction_memory_size:]
    
    def _compute_directional_bias(self, target_indices):
        """Compute directional bias from historical successful mutations."""
        if len(self.successful_directions) == 0:
            return np.zeros((len(target_indices), self.dim))
        
        direction_stack = np.array(self.successful_directions)
        weights = np.exp(-0.5 * np.arange(len(direction_stack))[::-1])
        weights = weights / np.sum(weights)
        
        weighted_direction = np.sum(direction_stack * weights[:, np.newaxis], axis=0)
        directional_bias = np.tile(weighted_direction, (len(target_indices), 1))
        
        return 0.3 * directional_bias
    
    def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Generate mutation vectors using composite strategy:
        - Ring-based rand/1 component
        - Directional component from historical memory
        """
        np_pop = len(population)
        trials = np.zeros((np_pop, self.dim))
        
        centroid = self._compute_population_centroid(population, fitness)
        directional_bias = self._compute_directional_bias(np.arange(np_pop))
        
        # Select ring-based parents
        r1 = ring_prev
        r2 = ring_next
        
        # Standard DE rand/1 mutation
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        # Adaptive F with perturbation based on diversity
        current_F = self.F * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)
        
        # Base mutation: ring-based differential
        base_mutation = rand1_vectors + current_F * (rand2_vectors - population)
        
        # Directional component: bias toward successful regions
        directional_component = directional_bias
        
        # Blend based on adaptation success
        blend_weight = 0.2 + 0.3 * (1.0 - self.Cr)
        trials = (1 - blend_weight) * base_mutation + blend_weight * (centroid + directional_component)
        
        return trials, current_F
    
    def _crossover_batch(self, population, mutant_batch):
        """Binomial crossover with adaptive Cr."""
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        # Per-individual Cr with bounded distribution
        Cr_individual = np.clip(
            self.Cr + 0.1 * np.random.randn(np_pop),
            0.1, 0.9
        )
        
        # Ensure at least one dimension is crossed
        j_rand = np.random.randint(0, dim, size=np_pop)
        
        for i in range(np_pop):
            mask = np.random.rand(dim) < Cr_individual[i]
            mask[j_rand[i]] = True
            trials[i] = np.where(mask, mutant_batch[i], population[i])
        
        return trials, Cr_individual
    
    def _evaluate_batch(self, batch, func):
        """Evaluate batch, handle budget exhaustion gracefully."""
        batch = self._clip_to_bounds(batch)
        fitness = func(batch)
        
        if len(fitness) < len(batch):
            valid_len = len(fitness)
            fitness = np.resize(fitness, len(batch))
            fitness[valid_len:] = np.inf
        
        fitness = np.where(np.isnan(fitness), np.inf, fitness)
        return fitness
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Temporal regime-adaptive selection with EMA-based stagnation detection."""
        improved_mask = trial_fitness < fitness

        # Initialize temporal state on first call
        if not hasattr(self, 'gen_improvement_ema'):
            self.gen_improvement_ema = np.zeros(self.np)  # Per-individual EMA of improvement
            self.pop_improvement_ema_short = 0.0
            self.pop_improvement_long = 0.0
            self.temporal_regime = 'neutral'  # 'improving', 'stagnant', 'declining'
            self.regime_patience = 0
            self.stagnation_window = []

        # Compute per-individual improvement magnitude (negative = improvement)
        improvement_mag = trial_fitness - fitness
        improvement_mag = np.where(improved_mask, improvement_mag, 0.0)

        # Update per-individual EMA of improvement magnitude
        alpha_indiv = 0.3
        self.gen_improvement_ema = alpha_indiv * improvement_mag + (1 - alpha_indiv) * self.gen_improvement_ema

        # Compute population-level improvement signal
        pop_improvement = np.sum(np.where(improved_mask, fitness - trial_fitness, 0.0))

        # Update population EMAs with different time constants
        alpha_short = 0.4
        alpha_long = 0.1
        self.pop_improvement_ema_short = alpha_short * pop_improvement + (1 - alpha_short) * self.pop_improvement_ema_short
        self.pop_improvement_long = alpha_long * pop_improvement + (1 - alpha_long) * self.pop_improvement_long

        # Maintain rolling window of improvement signals
        self.stagnation_window.append(np.mean(improved_mask))
        if len(self.stagnation_window) > 12:
            self.stagnation_window.pop(0)

        # Temporal regime detection via EMA divergence and window statistics
        window_mean = np.mean(self.stagnation_window)
        ema_trend = self.pop_improvement_ema_short - self.pop_improvement_long

        if ema_trend > 0.01 and window_mean > 0.25:
            self.temporal_regime = 'improving'
            self.regime_patience = 0
        elif ema_trend < -0.005 or window_mean < 0.15:
            self.regime_patience += 1
            if self.regime_patience > 5:
                self.temporal_regime = 'stagnant'
        else:
            self.temporal_regime = 'neutral'
            self.regime_patience = max(0, self.regime_patience - 1)

        # Compute temporal fitness: blend current fitness with EMA-based history
        # Individuals that consistently improve get a "temporal bonus" (lower effective fitness)
        temporal_bonus = 0.1 * self.gen_improvement_ema  # Positive EMA = consistent improver
        effective_fitness = fitness - temporal_bonus
        effective_trial_fitness = trial_fitness.copy()

        # Determine selection pressure based on temporal regime
        if self.temporal_regime == 'improving':
            # High selection pressure: trust current generation's greedy signal
            selection_mask = trial_fitness < fitness
        elif self.temporal_regime == 'stagnant':
            # Low selection pressure: use temporal fitness, favor consistent improvers
            # Allow some non-improving trials if they have good temporal history
            temporal_improvement_mask = effective_trial_fitness < effective_fitness
            # Also accept trials that improve individuals with poor temporal history
            poor_temporal_mask = self.gen_improvement_ema < np.percentile(self.gen_improvement_ema, 30)
            selection_mask = temporal_improvement_mask | (improved_mask & poor_temporal_mask)
        else:
            # Neutral: standard greedy with slight temporal weighting
            selection_mask = trial_fitness < fitness

        # Apply selection
        new_population = population.copy()
        new_fitness = fitness.copy()

        new_population[selection_mask] = trials[selection_mask]
        new_fitness[selection_mask] = trial_fitness[selection_mask]

        return new_population, new_fitness, improved_mask
    
    def _compute_diversity(self, population):
        """Compute population diversity via average pairwise distance."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Adapt F and Cr using temporal EMA dynamics and parameter drift detection."""
        success_rate = np.mean(improved_mask)

        # Initialize EMA state on first call
        if not hasattr(self, 'success_ewma_short'):
            self.success_ewma_short = success_rate
            self.success_ewma_long = success_rate
            self.stagnation_counter = 0
            self.F_history = [self.F]
            self.Cr_history = [self.Cr]

        # Update EMAs with different smoothing constants
        alpha_short = 0.3   # Fast EMA for short-term dynamics
        alpha_long = 0.1    # Slow EMA for trend detection
        self.success_ewma_short = alpha_short * success_rate + (1 - alpha_short) * self.success_ewma_short
        self.success_ewma_long = alpha_long * success_rate + (1 - alpha_long) * self.success_ewma_long

        # Track parameter history for drift detection
        self.F_history.append(F_used)
        self.Cr_history.append(F_used)
        if len(self.F_history) > 15:
            self.F_history.pop(0)
            self.Cr_history.pop(0)

        # Compute parameter drift from historical mean
        F_mean = np.mean(self.F_history)
        Cr_mean = np.mean(self.Cr_history)
        F_drift = self.F / (F_mean + 1e-10)
        Cr_drift = self.Cr / (Cr_mean + 1e-10)

        # Regime detection via EMA divergence
        short_trending_up = self.success_ewma_short > self.success_ewma_long
        short_trending_down = self.success_ewma_short < self.success_ewma_long
        both_stagnant = self.success_ewma_short < 0.1 and self.success_ewma_long < 0.15

        # Compute momentum-adjusted base adjustments
        momentum_factor = np.clip(F_drift, 0.7, 1.4)

        if short_trending_up and self.success_ewma_short > 0.25:
            # Exploitation phase: success accelerating, increase convergence pressure
            F_adjustment = 1.05 * momentum_factor
            Cr_adjustment = 1.08
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
        elif short_trending_down and self.success_ewma_short < 0.2:
            # Exploration phase: success declining, inject diversity
            F_adjustment = 0.88 / momentum_factor
            Cr_adjustment = 0.92
            self.stagnation_counter += 1
        elif both_stagnant:
            # Deep stagnation: large perturbation to escape local traps
            F_adjustment = 0.75
            Cr_adjustment = 1.15
            self.stagnation_counter += 2
        else:
            # Neutral regime: gentle correction toward baseline
            delta = self.success_ewma_short - self.success_ewma_long
            F_adjustment = 1.0 + 0.08 * delta
            Cr_adjustment = 1.0 - 0.05 * delta

        # Apply adjustments with drift correction
        self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
    
    def _restart_if_stagnant(self, population, fitness, best_fitness, best_solution):
        """Reinitialize portion of population if stagnation detected."""
        current_best = np.min(fitness)
        
        if current_best >= best_fitness:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            best_fitness = current_best
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx].copy()
        
        diversity = self._compute_diversity(population)
        
        if (self.stagnation_counter > self.stagnation_threshold or 
            diversity < self.min_diversity_threshold):
            
            n_replace = max(self.np // 5, 2)
            replace_indices = np.random.choice(self.np, n_replace, replace=False)
            
            for idx in replace_indices:
                population[idx] = np.random.uniform(
                    self.lower, self.upper, self.dim
                )
            
            fitness[replace_indices] = np.inf
            self.stagnation_counter = 0
            self.successful_directions = []
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop orchestrating all batched operations."""
        population, fitness, best_fitness, best_solution = self._initialize_population(func)
        
        self.stagnation_counter = 0
        self.generation = 0
        
        ring_prev, ring_next = self._build_ring_topology()
        
        while not stopping_condition():
            self.generation += 1
            
            # Build mutation batch
            mutant_batch, F_used = self._mutate_batch(
                population, fitness, ring_prev, ring_next
            )
            
            # Build crossover batch
            trial_batch, Cr_used = self._crossover_batch(population, mutant_batch)
            
            # Check budget before evaluation
            if stopping_condition():
                break
            
            # Evaluate trial batch
            trial_fitness = self._evaluate_batch(trial_batch, func)
            
            if stopping_condition():
                break
            
            # Select survivors
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trial_batch, trial_fitness
            )
            
            # Update best solution
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
            
            # Compute successful mutations for direction memory
            successful_mutations = trial_batch - population
            self._update_direction_memory(successful_mutations, improved_mask)
            
            # Adapt control parameters
            self._adapt_parameters(improved_mask, F_used, Cr_used)
            
            # Check for stagnation and restart if needed
            population, fitness, best_fitness, best_solution = self._restart_if_stagnant(
                population, fitness, best_fitness, best_solution
            )
            
            # Rebuild topology after potential restart
            ring_prev, ring_next = self._build_ring_topology()
        
        return best_fitness, best_solution
