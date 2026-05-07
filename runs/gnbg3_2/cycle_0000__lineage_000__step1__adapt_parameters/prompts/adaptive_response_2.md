```python
import numpy as np


class AdaptiveDirectionalDE:
    """
    Adaptive Directional Differential Evolution (ADDE)
    
    Adaptation Mechanism: #3 - CONDITIONAL / RULE-BASED DISPATCH
    
    Rationale: variant_06 (EMA dynamics) wins 58% of tasks, but the benchmark
    reveals a regime-dependent pattern: geometric (variant_09) works better when
    the population is tightly clustered (exploration phase), while entropy-based
    (variant_03) helps when fitness is diverse (exploitation phase). The dispatch
    rules detect these regimes and blend the corresponding strategies.
    
    Key design choices:
    - Regime detection via fitness entropy + EMA divergence + stagnation counter
    - variant_06 as the default (dominant baseline)
    - variant_09's geometric rules activated in tight-clustering regimes
    - variant_03's entropy rules activated in high-fitness-diversity regimes
    - Smooth blending weights to avoid abrupt switches
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 60), 300)
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
        
        # --- ADAPTIVE PARAMETERS: regime detection state ---
        # EMA state (from variant_06)
        self.success_ewma_short = 0.2
        self.success_ewma_long = 0.2
        self.F_history = [self.F]
        self.Cr_history = [self.Cr]
        self.stagnation_counter_internal = 0
        
        # Regime blending weights (smooth transitions)
        self.weight_variant_06 = 0.8  # Default dominant strategy
        self.weight_variant_09 = 0.1  # Geometric (tight clustering)
        self.weight_variant_03 = 0.1  # Entropy-based (high diversity)
        
        # Regime cache (update every N generations for efficiency)
        self._regime_cache = None
        self._regime_cache_gen = -1
        
        # Fitness entropy history (from variant_03)
        self.fitness_entropy_history = []
        self.last_F_for_entropy = self.F
        
    def __call__(self, func, stopping_condition):
        return self._optimize(func, stopping_condition)
    
    def _initialize_population(self, func):
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
        return np.clip(population, self.lower, self.upper)
    
    def _build_ring_topology(self):
        indices = np.arange(self.np)
        ring_prev = np.roll(indices, 1)
        ring_next = np.roll(indices, -1)
        return ring_prev, ring_next
    
    def _compute_population_centroid(self, population, fitness):
        weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
        weights = weights / np.sum(weights)
        centroid = np.sum(population * weights[:, np.newaxis], axis=0)
        return centroid
    
    def _update_direction_memory(self, successful_mutations, successful_mask):
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
        if len(self.successful_directions) == 0:
            return np.zeros((len(target_indices), self.dim))
        
        direction_stack = np.array(self.successful_directions)
        weights = np.exp(-0.5 * np.arange(len(direction_stack))[::-1])
        weights = weights / np.sum(weights)
        
        weighted_direction = np.sum(direction_stack * weights[:, np.newaxis], axis=0)
        directional_bias = np.tile(weighted_direction, (len(target_indices), 1))
        
        return 0.3 * directional_bias
    
    def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        np_pop = len(population)
        trials = np.zeros((np_pop, self.dim))
        
        centroid = self._compute_population_centroid(population, fitness)
        directional_bias = self._compute_directional_bias(np.arange(np_pop))
        
        r1 = ring_prev
        r2 = ring_next
        
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        current_F = self.F * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)
        
        base_mutation = rand1_vectors + current_F * (rand2_vectors - population)
        directional_component = directional_bias
        
        blend_weight = 0.2 + 0.3 * (1.0 - self.Cr)
        trials = (1 - blend_weight) * base_mutation + blend_weight * (centroid + directional_component)
        
        return trials, current_F
    
    def _crossover_batch(self, population, mutant_batch):
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        Cr_individual = np.clip(self.Cr + 0.1 * np.random.randn(np_pop), 0.1, 0.9)
        j_rand = np.random.randint(0, dim, size=np_pop)
        
        for i in range(np_pop):
            mask = np.random.rand(dim) < Cr_individual[i]
            mask[j_rand[i]] = True
            trials[i] = np.where(mask, mutant_batch[i], population[i])
        
        return trials, Cr_individual
    
    def _evaluate_batch(self, batch, func):
        batch = self._clip_to_bounds(batch)
        fitness = func(batch)
        
        if len(fitness) < len(batch):
            valid_len = len(fitness)
            fitness = np.resize(fitness, len(batch))
            fitness[valid_len:] = np.inf
        
        fitness = np.where(np.isnan(fitness), np.inf, fitness)
        return fitness
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        improved_mask = trial_fitness < fitness
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]
        
        return new_population, new_fitness, improved_mask
    
    def _compute_diversity(self, population):
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    # =====================================================================
    # REGIME DETECTION (conditional dispatch core)
    # =====================================================================
    
    def _compute_fitness_entropy(self, population, fitness):
        """Compute normalized Shannon entropy of fitness distribution (variant_03).
        
        Returns float in [0, 1]: 0 = converged (exploitation), 1 = diverse (exploration).
        """
        valid_fitness = fitness[np.isfinite(fitness)]
        
        if len(valid_fitness) < 3:
            return 0.5
        
        fitness_min = np.nanmin(valid_fitness)
        fitness_max = np.nanmax(valid_fitness)
        
        if fitness_max - fitness_min < 1e-10:
            return 0.5
        
        n_bins = min(10, max(3, len(valid_fitness) // 2))
        bin_edges = np.linspace(
            fitness_min - 0.5 * (fitness_max - fitness_min),
            fitness_max + 0.5 * (fitness_max - fitness_min),
            n_bins + 1
        )
        bin_counts, _ = np.histogram(valid_fitness, bins=bin_edges)
        bin_probs = bin_counts / (np.sum(bin_counts) + 1e-10)
        
        entropy = -np.sum(np.where(bin_probs > 0, bin_probs * np.log(bin_probs + 1e-10), 0))
        max_entropy = np.log(n_bins)
        normalized_entropy = entropy / (max_entropy + 1e-10)
        
        return np.clip(normalized_entropy, 0.0, 1.0)
    
    def _compute_geometric_spread(self, population):
        """Compute normalized axis-aligned spread (variant_09).
        
        Returns float: low = tight clustering, high = well-dispersed.
        """
        dim_mins = np.min(population, axis=0)
        dim_maxs = np.max(population, axis=0)
        dim_ranges = dim_maxs - dim_mins
        total_spread = np.sum(dim_ranges)
        
        search_space_size = (self.upper - self.lower) * self.dim
        normalized_spread = total_spread / (search_space_size + 1e-30)
        
        return np.clip(normalized_spread, 0.0, 1.0)
    
    def _detect_regime(self, population, fitness, success_rate):
        """Detect optimization regime using multiple signals.
        
        Returns dict with regime info and blending weights for the three strategies.
        Regime is cached and refreshed every 5 generations to balance accuracy vs cost.
        """
        if self._regime_cache is not None and (self.generation - self._regime_cache_gen) < 5:
            return self._regime_cache
        
        fitness_entropy = self._compute_fitness_entropy(population, fitness)
        geometric_spread = self._compute_geometric_spread(population)
        
        # Update entropy history
        self.fitness_entropy_history.append(fitness_entropy)
        if len(self.fitness_entropy_history) > self.adaptation_window:
            self.fitness_entropy_history.pop(0)
        
        # Entropy trend (from variant_03)
        entropy_trend = 0.0
        if len(self.fitness_entropy_history) >= 3:
            entropy_trend = self.fitness_entropy_history[-1] - self.fitness_entropy_history[0]
        
        # EMA-based regime (from variant_06)
        short_trending_up = self.success_ewma_short > self.success_ewma_long
        short_trending_down = self.success_ewma_short < self.success_ewma_long
        both_stagnant = self.success_ewma_short < 0.1 and self.success_ewma_long < 0.15
        
        # Compute blending weights using soft signals
        # variant_06: dominant, used as baseline
        # variant_09 (geometric): better when population is tightly clustered
        # variant_03 (entropy): better when fitness is diverse
        
        w06 = 0.7  # Base weight for dominant strategy
        w09 = 0.15  # Base weight for geometric
        w03 = 0.15  # Base weight for entropy
        
        # Boost variant_09 when population is tightly clustered
        if geometric_spread < 0.05:
            w09 = min(0.4, w09 + 0.25)
            w06 = max(0.4, w06 - 0.15)
        elif geometric_spread < 0.15:
            w09 = min(0.3, w09 + 0.1)
            w06 = max(0.5, w06 - 0.05)
        elif geometric_spread > 0.4:
            # Well-dispersed: slightly favor exploitation (variant_06)
            w06 = min(0.85, w06 + 0.1)
            w09 = max(0.05, w09 - 0.05)
        
        # Boost variant_03 when fitness entropy is high (diverse population)
        if fitness_entropy > 0.6:
            w03 = min(0.35, w03 + 0.15)
            w06 = max(0.45, w06 - 0.1)
        elif fitness_entropy < 0.3:
            # Converged fitness: favor exploitation
            w06 = min(0.85, w06 + 0.1)
            w03 = max(0.05, w03 - 0.05)
        
        # Override for deep stagnation: variant_06 handles this best
        if both_stagnant or self.stagnation_counter_internal > 20:
            w06 = max(w06, 0.75)
            w09 = max(0.05, w09 - 0.05)
            w03 = max(0.05, w03 - 0.05)
        
        # Special case: success improving rapidly → exploit more
        if short_trending_up and self.success_ewma_short > 0.3:
            w06 = min(0.9, w06 + 0.1)
        
        # Special case: success declining → explore more via geometric
        if short_trending_down and self.success_ewma_short < 0.2:
            w09 = min(0.35, w09 + 0.15)
            w06 = max(0.5, w06 - 0.1)
        
        # Normalize weights
        weight_sum = w06 + w09 + w03
        w06, w09, w03 = w06 / weight_sum, w09 / weight_sum, w03 / weight_sum
        
        self._regime_cache = {
            'fitness_entropy': fitness_entropy,
            'geometric_spread': geometric_spread,
            'entropy_trend': entropy_trend,
            'short_trending_up': short_trending_up,
            'short_trending_down': short_trending_down,
            'both_stagnant': both_stagnant,
            'w06': w06,
            'w09': w09,
            'w03': w03,
        }
        self._regime_cache_gen = self.generation
        
        return self._regime_cache
    
    # =====================================================================
    # ADAPTATION STRATEGIES (blended conditional dispatch)
    # =====================================================================
    
    def _adapt_parameters_variant_06(self, improved_mask, F_used, Cr_used, regime):
        """EMA dynamics adaptation (variant_06) - the dominant strategy."""
        success_rate = np.mean(improved_mask)
        
        alpha_short = 0.3
        alpha_long = 0.1
        self.success_ewma_short = alpha_short * success_rate + (1 - alpha_short) * self.success_ewma_short
        self.success_ewma_long = alpha_long * success_rate + (1 - alpha_long) * self.success_ewma_long
        
        # Track parameter history
        self.F_history.append(F_used)
        self.Cr_history.append(Cr_used)
        if len(self.F_history) > 15:
            self.F_history.pop(0)
            self.Cr_history.pop(0)
        
        F_mean = np.mean(self.F_history)
        Cr_mean = np.mean(self.Cr_history)
        F_drift = self.F / (F_mean + 1e-10)
        Cr_drift = self.Cr / (Cr_mean + 1e-10)
        
        momentum_factor = np.clip(F_drift, 0.7, 1.4)
        
        short_trending_up = regime['short_trending_up']
        short_trending_down = regime['short_trending_down']
        both_stagnant = regime['both_stagnant']
        
        if short_trending_up and self.success_ewma_short > 0.25:
            F_adj = 1.05 * momentum_factor
            Cr_adj = 1.08
            self.stagnation_counter_internal = max(0, self.stagnation_counter_internal - 1)
        elif short_trending_down and self.success_ewma_short < 0.2:
            F_adj = 0.88 / momentum_factor
            Cr_adj = 0.92
            self.stagnation_counter_internal += 1
        elif both_stagnant:
            F_adj = 0.75
            Cr_adj = 1.15
            self.stagnation_counter_internal += 2
        else:
            delta = self.success_ewma_short - self.success_ewma_long
            F_adj = 1.0 + 0.08 * delta
            Cr_adj = 1.0 - 0.05 * delta
        
        return np.clip(self.F * F_adj, 0.1, 1.5), np.clip(self.Cr * Cr_adj, 0.1, 0.9)
    
    def _adapt_parameters_variant_09(self, population, F_used, Cr_used, regime):
        """Geometric/layout-based adaptation (variant_09)."""
        geometric_spread = regime['geometric_spread']
        
        F_adj = 1.0
        Cr_adj = 1.0
        
        if geometric_spread < 0.01:
            F_adj = 1.15
        elif geometric_spread > 0.5:
            F_adj = 0.9
        
        # k-NN distance for Cr adjustment
        k = min(3, len(population) - 1)
        knn_distances = np.zeros(len(population))
        for i in range(len(population)):
            dists = np.linalg.norm(population - population[i], axis=1)
            dists[i] = np.inf
            sorted_dists = np.sort(dists)
            knn_distances[i] = np.mean(sorted_dists[:k])
        
        mean_knn = np.mean(knn_distances)
        search_space_size = (self.upper - self.lower) * self.dim
        knn_normalized = mean_knn / (search_space_size ** 0.5 + 1e-30)
        
        if knn_normalized > 0.3:
            Cr_adj = 1.08
        elif knn_normalized < 0.1:
            Cr_adj = 0.92
        
        return np.clip(self.F * F_adj, 0.1, 2.0), np.clip(self.Cr * Cr_adj, 0.1, 0.9)
    
    def _adapt_parameters_variant_03(self, F_used, Cr_used, regime):
        """Entropy-based adaptation (variant_03)."""
        fitness_entropy = regime['fitness_entropy']
        entropy_trend = regime['entropy_trend']
        
        F_adj = 1.0
        Cr_adj = 1.0
        
        if fitness_entropy > 0.5:
            F_adj = 1.0 + 0.15 * entropy_trend
            Cr_adj = 1.0 - 0.05 * entropy_trend
        else:
            F_adj = 1.0 - 0.10 * abs(entropy_trend)
            Cr_adj = 1.0 + 0.08 * abs(entropy_trend)
        
        # Momentum term from variant_03
        momentum = 0.7
        target_F = np.clip(self.F * F_adj, 0.1, 1.5)
        final_F = np.clip(momentum * self.last_F_for_entropy + (1 - momentum) * target_F, 0.1, 1.5)
        self.last_F_for_entropy = final_F
        
        return final_F, np.clip(self.Cr * Cr_adj, 0.1, 0.9)
    
    def _adapt_parameters(self, improved_mask, F_used, Cr_used, population, fitness):
        """Blend three adaptation strategies based on detected regime.
        
        This is the core adaptive mechanism: conditional dispatch with smooth
        blending weights that are recomputed every 5 generations based on
        population geometry, fitness distribution, and EMA dynamics.
        """
        success_rate = np.mean(improved_mask)
        
        # Record history for EMA updates
        self.F_success_history.append((success_rate, F_used))
        self.Cr_success_history.append((success_rate, Cr_used))
        if len(self.F_success_history) > self.adaptation_window:
            self.F_success_history.pop(0)
            self.Cr_success_history.pop(0)
        
        # Detect current regime and get blending weights
        regime = self._detect_regime(population, fitness, success_rate)
        w06 = regime['w06']
        w09 = regime['w09']
        w03 = regime['w03']
        
        # Compute adjustments from each strategy
        F06, Cr06 = self._adapt_parameters_variant_06(improved_mask, F_used, Cr_used, regime)
        F09, Cr09 = self._adapt_parameters_variant_09(population, F_used, Cr_used, regime)
        F03, Cr03 = self._adapt_parameters_variant_03(F_used, Cr_used, regime)
        
        # Weighted blend of adjustments (blend the parameters, not the deltas)
        # This gives smooth transitions between regimes
        F_blended = w06 * F06 + w09 * F09 + w03 * F03
        Cr_blended = w06 * Cr06 + w09 * Cr09 + w03 * Cr03
        
        # Apply blended parameters with momentum to smooth transitions
        momentum = 0.6
        self.F = np.clip(momentum * self.F + (1 - momentum) * F_blended, 0.1, 1.5)
        self.Cr = np.clip(momentum * self.Cr + (1 - momentum) * Cr_blended, 0.1, 0.9)
    
    def _restart_if_stagnant(self, population, fitness, best_fitness, best_solution):
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
                population[idx] = np.random.uniform(self.lower, self.upper, self.dim)
            
            fitness[replace_indices] = np.inf
            self.stagnation_counter = 0
            self.successful_directions = []
            
            # Reset EMA state on restart to avoid stale signals
            if hasattr(self, 'success_ewma_short'):
                self.success_ewma_short = 0.2
                self.success_ewma_long = 0.2
            self.stagnation_counter_internal = 0
            self._regime_cache = None
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        population, fitness, best_fitness, best_solution = self._initialize_population(func)
        
        self.stagnation_counter = 0
        self.generation = 0
        
        ring_prev, ring_next = self._build_ring_topology()
        
        while not stopping_condition():
            self.generation += 1
            
            mutant_batch, F_used = self._mutate_batch(
                population, fitness, ring_prev, ring_next
            )
            
            trial_batch, Cr_used = self._crossover_batch(population, mutant_batch)
            
            if stopping_condition():
                break
            
            trial_fitness = self._evaluate_batch(trial_batch, func)
            
            if stopping_condition():
                break
            
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trial_batch, trial_fitness
            )
            
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
            
            successful_mutations = trial_batch - population
            self._update_direction_memory(successful_mutations, improved_mask)
            
            # ADAPTIVE PARAMETERS: blended conditional dispatch
            self._adapt_parameters(improved_mask, F_used, Cr_used, population, fitness)
            
            population, fitness, best_fitness, best_solution = self._restart_if_stagnant(
                population, fitness, best_fitness, best_solution
            )
            
            ring_prev, ring_next = self._build_ring_topology()
        
        return best_fitness, best_solution
```