import numpy as np


class AdaptiveDirectionalDE:
    """
    META-CONTROLLED ADAPTIVE (Mechanism #9 + #3)
    
    Uses EMA-based adaptation (variant_06, 14/24 wins) as primary strategy,
    with regime detection via entropy (catC) and geometric (catA) signals
    to decide when to blend in alternative strategies. The three signals
    capture complementary aspects: temporal dynamics (EMA), fitness
    distribution shape (entropy), and spatial structure (geometric).
    No scale-dependent constants; all rewards are rank-based.
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
        
        # ── EMA state (variant_06 strategy) ──
        self.ema_initialized = False
        self.success_ewma_short = 0.2
        self.success_ewma_long = 0.2
        self.stagnation_counter_ema = 0
        self.F_history = []
        self.Cr_history = []
        
        # ── Entropy state (variant_03 strategy) ──
        self.entropy_initialized = False
        self.fitness_entropy_history = []
        self.last_F_for_entropy = self.F
        
        # ── Geometric state (variant_09 strategy) ──
        self.geometric_initialized = False
        self.prev_normalized_spread = 0.5
        self.prev_knn_normalized = 0.2
        
        # ── Meta-controller state (mechanism #9) ──
        self.meta_window = 15
        self.success_rate_history = []
        self.current_mode = 'ema'  # primary: ema, fallback: entropy, geometric
        self.mode_switch_counter = 0
        self.mode_performance = {'ema': [], 'entropy': [], 'geometric': []}
        self.entropy_blend_weight = 0.0
        self.geometric_blend_weight = 0.0
        
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
        
        # Adaptive F with perturbation based on diversity
        current_F = self.F * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)
        
        base_mutation = rand1_vectors + current_F * (rand2_vectors - population)
        directional_component = directional_bias
        
        # Blend based on adaptation success and meta-controller signals
        blend_weight = 0.2 + 0.3 * (1.0 - self.Cr)
        
        # Blend entropy and geometric signals into directional component
        if self.entropy_blend_weight > 0.0 or self.geometric_blend_weight > 0.0:
            # Perturb F/Cr based on alternative strategies
            entropy_F_factor = 1.0 + 0.05 * (self.entropy_blend_weight - 0.5)
            entropy_Cr_factor = 1.0 - 0.03 * (self.entropy_blend_weight - 0.5)
            geo_F_factor = 1.0 + 0.05 * (self.geometric_blend_weight - 0.5)
            geo_Cr_factor = 1.0 - 0.03 * (self.geometric_blend_weight - 0.5)
            
            current_F = np.clip(current_F * entropy_F_factor * geo_F_factor, 0.1, 2.0)
            blend_weight = np.clip(blend_weight * entropy_Cr_factor * geo_Cr_factor, 0.1, 0.7)
        
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
    
    # ─────────────────────────────────────────────────────────────────
    # ADAPTATION STRATEGIES (from winning variants)
    # ─────────────────────────────────────────────────────────────────
    
    def _adapt_ema(self, improved_mask, F_used, Cr_used):
        """Variant_06: EMA dynamics with regime detection."""
        success_rate = np.mean(improved_mask)
        
        if not self.ema_initialized:
            self.success_ewma_short = success_rate
            self.success_ewma_long = success_rate
            self.stagnation_counter_ema = 0
            self.F_history = [self.F]
            self.Cr_history = [self.Cr]
            self.ema_initialized = True
        
        alpha_short = 0.3
        alpha_long = 0.1
        self.success_ewma_short = alpha_short * success_rate + (1 - alpha_short) * self.success_ewma_short
        self.success_ewma_long = alpha_long * success_rate + (1 - alpha_long) * self.success_ewma_long
        
        self.F_history.append(F_used)
        self.Cr_history.append(Cr_used)
        if len(self.F_history) > 15:
            self.F_history.pop(0)
            self.Cr_history.pop(0)
        
        F_mean = np.mean(self.F_history)
        Cr_mean = np.mean(self.Cr_history)
        F_drift = self.F / (F_mean + 1e-10)
        Cr_drift = self.Cr / (Cr_mean + 1e-10)
        
        short_trending_up = self.success_ewma_short > self.success_ewma_long
        short_trending_down = self.success_ewma_short < self.success_ewma_long
        both_stagnant = self.success_ewma_short < 0.1 and self.success_ewma_long < 0.15
        
        momentum_factor = np.clip(F_drift, 0.7, 1.4)
        
        if short_trending_up and self.success_ewma_short > 0.25:
            F_adjustment = 1.05 * momentum_factor
            Cr_adjustment = 1.08
            self.stagnation_counter_ema = max(0, self.stagnation_counter_ema - 1)
        elif short_trending_down and self.success_ewma_short < 0.2:
            F_adjustment = 0.88 / momentum_factor
            Cr_adjustment = 0.92
            self.stagnation_counter_ema += 1
        elif both_stagnant:
            F_adjustment = 0.75
            Cr_adjustment = 1.15
            self.stagnation_counter_ema += 2
        else:
            delta = self.success_ewma_short - self.success_ewma_long
            F_adjustment = 1.0 + 0.08 * delta
            Cr_adjustment = 1.0 - 0.05 * delta
        
        self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
    
    def _adapt_entropy(self, improved_mask, F_used, Cr_used):
        """Variant_03: Information-theoretic / entropy-based."""
        fitness_values = [f for f, _ in self.F_success_history[-self.adaptation_window:]]
        if len(fitness_values) > 0:
            fitness_values.append(np.mean(fitness_values))
        else:
            fitness_values = [0.0]
        
        fitness = np.array(fitness_values)
        
        if len(fitness) < 2 or np.all(np.isinf(fitness)):
            return
        
        fitness_min = np.nanmin(fitness)
        fitness_max = np.nanmax(fitness)
        if fitness_max - fitness_min < 1e-10:
            return
        
        n_bins = min(10, max(3, len(fitness) // 2))
        bin_edges = np.linspace(
            fitness_min - 0.5 * (fitness_max - fitness_min),
            fitness_max + 0.5 * (fitness_max - fitness_min),
            n_bins + 1
        )
        bin_counts, _ = np.histogram(fitness, bins=bin_edges)
        bin_probs = bin_counts / (np.sum(bin_counts) + 1e-10)
        
        entropy = -np.sum(np.where(bin_probs > 0, bin_probs * np.log(bin_probs + 1e-10), 0))
        max_entropy = np.log(n_bins)
        normalized_entropy = entropy / (max_entropy + 1e-10)
        
        if not self.entropy_initialized:
            self.fitness_entropy_history = []
            self.last_F_for_entropy = self.F
            self.entropy_initialized = True
        
        self.fitness_entropy_history.append(normalized_entropy)
        if len(self.fitness_entropy_history) > self.adaptation_window:
            self.fitness_entropy_history.pop(0)
        
        if len(self.fitness_entropy_history) >= 3:
            recent_entropies = self.fitness_entropy_history[-3:]
            entropy_change = recent_entropies[-1] - recent_entropies[0]
            
            if recent_entropies[-1] > 0.5:
                self.F = np.clip(self.F * (1.0 + 0.15 * entropy_change), 0.1, 1.5)
                self.Cr = np.clip(self.Cr * (1.0 - 0.05 * entropy_change), 0.1, 0.9)
            else:
                self.F = np.clip(self.F * (1.0 - 0.10 * abs(entropy_change)), 0.1, 1.5)
                self.Cr = np.clip(self.Cr * (1.0 + 0.08 * abs(entropy_change)), 0.1, 0.9)
            
            momentum = 0.7
            target_F = self.F
            self.F = np.clip(momentum * self.last_F_for_entropy + (1 - momentum) * target_F, 0.1, 1.5)
            self.last_F_for_entropy = self.F
    
    def _adapt_geometric(self, population, improved_mask, F_used, Cr_used):
        """Variant_09: Geometric / spatial structure-based."""
        pop = population if hasattr(self, 'population') else None
        if pop is None or len(pop) < 3:
            return
        
        dim_mins = np.min(pop, axis=0)
        dim_maxs = np.max(pop, axis=0)
        dim_ranges = dim_maxs - dim_mins
        total_spread = np.sum(dim_ranges)
        
        search_space_size = (self.upper - self.lower) * self.dim
        normalized_spread = total_spread / (search_space_size + 1e-30)
        
        k = min(3, len(pop) - 1)
        knn_distances = np.zeros(len(pop))
        for i in range(len(pop)):
            dists = np.linalg.norm(pop - pop[i], axis=1)
            dists[i] = np.inf
            sorted_dists = np.sort(dists)
            knn_distances[i] = np.mean(sorted_dists[:k])
        
        mean_knn = np.mean(knn_distances)
        knn_normalized = mean_knn / (search_space_size ** 0.5 + 1e-30)
        
        if not self.geometric_initialized:
            self.prev_normalized_spread = normalized_spread
            self.prev_knn_normalized = knn_normalized
            self.geometric_initialized = True
        
        spread_change = normalized_spread - self.prev_normalized_spread
        knn_change = knn_normalized - self.prev_knn_normalized
        
        self.prev_normalized_spread = normalized_spread
        self.prev_knn_normalized = knn_normalized
        
        if normalized_spread < 0.01:
            self.F = np.clip(self.F * 1.15, 0.1, 2.0)
        elif normalized_spread > 0.5:
            self.F = np.clip(self.F * 0.9, 0.1, 2.0)
        
        if knn_normalized > 0.3:
            self.Cr = np.clip(self.Cr * 1.08, 0.1, 0.9)
        elif knn_normalized < 0.1:
            self.Cr = np.clip(self.Cr * 0.92, 0.1, 0.9)
        
        # Detect regime changes for meta-controller
        return spread_change, knn_change, normalized_spread, knn_normalized
    
    # ─────────────────────────────────────────────────────────────────
    # META-CONTROLLER (mechanism #9)
    # ─────────────────────────────────────────────────────────────────
    
    def _update_meta_controller(self, population, improved_mask, F_used, Cr_used):
        """Meta-controlled adaptation blending all three strategies."""
        success_rate = np.mean(improved_mask)
        
        # Record success rate for regime detection
        self.success_rate_history.append(success_rate)
        if len(self.success_rate_history) > self.meta_window:
            self.success_rate_history.pop(0)
        
        # Run all three adaptation strategies
        F_before_ema = self.F
        Cr_before_ema = self.Cr
        self._adapt_ema(improved_mask, F_used, Cr_used)
        F_after_ema = self.F
        Cr_after_ema = self.Cr
        
        F_before_entropy = self.F
        Cr_before_entropy = self.Cr
        self._adapt_entropy(improved_mask, F_used, Cr_used)
        F_after_entropy = self.F
        Cr_after_entropy = self.Cr
        
        F_before_geo = self.F
        Cr_before_geo = self.Cr
        geo_info = self._adapt_geometric(population, improved_mask, F_used, Cr_used)
        F_after_geo = self.F
        Cr_after_geo = self.Cr
        
        # Compute regime indicators (rank-based, scale-free)
        regime_indicators = self._compute_regime_indicators(population, improved_mask)
        
        # Determine blend weights based on regime detection
        # If success rate is low → exploration regime → boost entropy/geo signals
        # If success rate is high → exploitation regime → trust EMA
        # If stagnation detected → diversity regime → boost geometric signal
        
        if len(self.success_rate_history) >= 5:
            recent_success = self.success_rate_history[-5:]
            avg_success = np.mean(recent_success)
            success_trend = recent_success[-1] - recent_success[0]
        else:
            avg_success = 0.2
            success_trend = 0.0
        
        # Stagnation detection
        is_stagnant = (self.stagnation_counter_ema > 10 or 
                      avg_success < 0.1 or 
                      (regime_indicators['spread'] < 0.02 and regime_indicators['knn'] < 0.1))
        
        # Compute blend weights (no scale-dependent constants)
        if is_stagnant:
            # Deep stagnation: boost geometric (detects clustering)
            self.entropy_blend_weight = 0.3
            self.geometric_blend_weight = 0.4
        elif avg_success > 0.3:
            # Good progress: trust EMA (wins 14/24 tasks)
            self.entropy_blend_weight = 0.1
            self.geometric_blend_weight = 0.1
        elif avg_success < 0.15:
            # Low success: balance entropy and geometric
            self.entropy_blend_weight = 0.25
            self.geometric_blend_weight = 0.25
        else:
            # Moderate: gentle blending
            self.entropy_blend_weight = 0.15
            self.geometric_blend_weight = 0.15
        
        # Apply entropy regime detection for additional F/Cr adjustment
        if len(self.fitness_entropy_history) >= 3:
            entropy_regime = self.fitness_entropy_history[-1]
            if entropy_regime > 0.6:
                # High entropy = exploration regime
                self.entropy_blend_weight = min(0.5, self.entropy_blend_weight * 1.3)
            elif entropy_regime < 0.3:
                # Low entropy = exploitation regime
                self.entropy_blend_weight = max(0.05, self.entropy_blend_weight * 0.7)
        
        # Blend the three strategies using computed weights
        # Use EMA as baseline, blend in entropy and geometric adjustments
        total_blend = self.entropy_blend_weight + self.geometric_blend_weight
        
        if total_blend > 0.01:
            # Compute the adjustments from each strategy
            ema_F_adj = F_after_ema / (F_before_ema + 1e-10)
            ema_Cr_adj = Cr_after_ema / (Cr_before_ema + 1e-10)
            
            entropy_F_adj = F_after_entropy / (F_before_entropy + 1e-10)
            entropy_Cr_adj = Cr_after_entropy / (Cr_before_entropy + 1e-10)
            
            geo_F_adj = F_after_geo / (F_before_geo + 1e-10)
            geo_Cr_adj = Cr_after_geo / (Cr_before_geo + 1e-10)
            
            # Weighted geometric mean of adjustments
            w_ema = 1.0 - total_blend
            w_entropy = self.entropy_blend_weight
            w_geo = self.geometric_blend_weight
            
            blended_F_adj = (ema_F_adj ** w_ema * 
                            entropy_F_adj ** w_entropy * 
                            geo_F_adj ** w_geo)
            blended_Cr_adj = (ema_Cr_adj ** w_ema * 
                             entropy_Cr_adj ** w_entropy * 
                             geo_Cr_adj ** w_geo)
            
            # Apply blended adjustment
            self.F = np.clip(F_before_ema * blended_F_adj, 0.1, 1.5)
            self.Cr = np.clip(Cr_before_ema * blended_Cr_adj, 0.1, 0.9)
        else:
            # Just use EMA
            self.F = F_after_ema
            self.Cr = Cr_after_ema
        
        # Clamp to valid ranges
        self.F = np.clip(self.F, 0.1, 1.5)
        self.Cr = np.clip(self.Cr, 0.1, 0.9)
    
    def _compute_regime_indicators(self, population, improved_mask):
        """Compute regime indicators from population state (scale-free)."""
        pop = population
        
        # Spread indicator
        dim_mins = np.min(pop, axis=0)
        dim_maxs = np.max(pop, axis=0)
        dim_ranges = dim_maxs - dim_mins
        total_spread = np.sum(dim_ranges)
        search_space_size = (self.upper - self.lower) * self.dim
        normalized_spread = total_spread / (search_space_size + 1e-30)
        
        # k-NN density indicator
        k = min(3, len(pop) - 1)
        knn_distances = np.zeros(len(pop))
        for i in range(len(pop)):
            dists = np.linalg.norm(pop - pop[i], axis=1)
            dists[i] = np.inf
            knn_distances[i] = np.mean(np.sort(dists)[:k])
        
        mean_knn = np.mean(knn_distances)
        knn_normalized = mean_knn / (search_space_size ** 0.5 + 1e-30)
        
        # Success rate in recent window
        if len(self.success_rate_history) > 0:
            recent_success = np.mean(self.success_rate_history[-5:])
        else:
            recent_success = 0.2
        
        # Diversity indicator
        diversity = self._compute_diversity(pop)
        diversity_normalized = diversity / (search_space_size ** 0.5 + 1e-30)
        
        return {
            'spread': normalized_spread,
            'knn': knn_normalized,
            'success': recent_success,
            'diversity': diversity_normalized
        }
    
    # ─────────────────────────────────────────────────────────────────
    # UNIFIED ADAPTATION (replaces original _adapt_parameters)
    # ─────────────────────────────────────────────────────────────────
    
    def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Meta-controlled adaptation combining all three strategies."""
        self._update_meta_controller(self.population, improved_mask, F_used, Cr_used)
    
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
                population[idx] = np.random.uniform(
                    self.lower, self.upper, self.dim
                )
            
            fitness[replace_indices] = np.inf
            self.stagnation_counter = 0
            self.successful_directions = []
            
            # Reset adaptation state on restart
            self.ema_initialized = False
            self.entropy_initialized = False
            self.geometric_initialized = False
            self.success_rate_history = []
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        population, fitness, best_fitness, best_solution = self._initialize_population(func)
        
        self.stagnation_counter = 0
        self.generation = 0
        self.population = population  # For geometric adaptation
        
        ring_prev, ring_next = self._build_ring_topology()
        
        while not stopping_condition():
            self.generation += 1
            self.population = population  # Keep updated for adaptation
            
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
            
            # Record F/Cr success for entropy adaptation
            success_rate = np.mean(improved_mask)
            self.F_success_history.append((success_rate, F_used))
            self.Cr_success_history.append((success_rate, Cr_used))
            if len(self.F_success_history) > self.adaptation_window:
                self.F_success_history.pop(0)
            if len(self.Cr_success_history) > self.adaptation_window:
                self.Cr_success_history.pop(0)
            
            # Meta-controlled adaptation
            self._adapt_parameters(improved_mask, F_used, Cr_used)
            
            population, fitness, best_fitness, best_solution = self._restart_if_stagnant(
                population, fitness, best_fitness, best_solution
            )
            
            ring_prev, ring_next = self._build_ring_topology()
        
        return best_fitness, best_solution
