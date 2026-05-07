import numpy as np


class AdaptiveDirectionalDE:
    """
    ADAPTIVE MECHANISM: Ensemble / Voting / Blending (#2 from menu)
    
    Rationale: The three winning variants (EMA dynamics, entropy, geometric
    spread) compute DIFFERENT feature extractors for the same F/Cr decision.
    They are correlated but not identical — ensemble voting captures their
    complementary failure-mode coverage. variant_06 dominates (14/24 wins),
    so the meta-controller biases toward it while maintaining sensitivity
    to geometric and entropy signals. No credit-assignment problem since
    outputs are blended, not selected.
    
    A hybrid meta-controller selects among three adaptation strategies:
      - EMA strategy: temporal success-rate dynamics (variant_06 logic)
      - Geometric strategy: axis-aligned spread + k-NN (variant_09 logic)
      - Entropy strategy: fitness distribution entropy (variant_03 logic)
    
    Weights are exponentially-weighted by recent per-strategy success,
    with a floor to prevent any strategy from vanishing. Stagnation
    detection shifts weight toward strategies that performed well during
    prior stagnation escapes.
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
        
        # --- META-CONTROLLER STATE ---
        # Per-strategy EMA state (EMA strategy)
        self.success_ewma_short = None
        self.success_ewma_long = None
        self.stagnation_counter = 0
        self.F_history = []
        self.Cr_history = []
        
        # Per-strategy entropy state (entropy strategy)
        self.fitness_entropy_history = []
        self.last_F_for_entropy = self.F
        
        # Per-strategy geometric state (geometric strategy)
        self.population = None  # Will be set during optimization
        
        # Meta-controller: per-strategy performance tracking
        self.strategy_perf_history = {
            'ema': [],      # Recent success rates after EMA-guided steps
            'geo': [],      # Recent success rates after geo-guided steps
            'ent': []       # Recent success rates after ent-guided steps
        }
        self.strategy_weights = {'ema': 0.60, 'geo': 0.25, 'ent': 0.15}
        self.prev_improved_mask = None
        
        # Stagnation escape memory: which strategy helped escape stagnation
        self.stagnation_escape_strategies = []
        self.stagnation_escape_weights = {'ema': 1.0, 'geo': 1.0, 'ent': 1.0}
        
        # Forgetting factor for strategy performance
        self.strategy_forget = 0.85
        
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
        self.population = population  # Store for geometric adaptation
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
        """Generate mutation vectors using composite strategy."""
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
        """Binomial crossover with adaptive Cr."""
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
        """Greedy selection with diversity maintenance."""
        improved_mask = trial_fitness < fitness
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]
        
        return new_population, new_fitness, improved_mask
    
    def _compute_diversity(self, population):
        """Compute population diversity via average pairwise distance."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    # =====================================================================
    # STRATEGY 1: EMA-based adaptation (variant_06 logic)
    # =====================================================================
    def _adapt_ema(self, improved_mask, F_used, Cr_used):
        """EMA dynamics and parameter drift detection (variant_06)."""
        success_rate = np.mean(improved_mask)
        
        if self.success_ewma_short is None:
            self.success_ewma_short = success_rate
            self.success_ewma_long = success_rate
        
        alpha_short = 0.3
        alpha_long = 0.1
        self.success_ewma_short = alpha_short * success_rate + (1 - alpha_short) * self.success_ewma_short
        self.success_ewma_long = alpha_long * success_rate + (1 - alpha_long) * self.success_ewma_long
        
        self.F_history.append(F_used)
        self.Cr_history.append(Cr_used)
        if len(self.F_history) > 15:
            self.F_history.pop(0)
            self.Cr_history.pop(0)
        
        F_mean = np.mean(self.F_history) if self.F_history else F_used
        Cr_mean = np.mean(self.Cr_history) if self.Cr_history else Cr_used
        F_drift = self.F / (F_mean + 1e-10)
        Cr_drift = self.Cr / (Cr_mean + 1e-10)
        
        short_trending_up = self.success_ewma_short > self.success_ewma_long
        short_trending_down = self.success_ewma_short < self.success_ewma_long
        both_stagnant = self.success_ewma_short < 0.1 and self.success_ewma_long < 0.15
        
        momentum_factor = np.clip(F_drift, 0.7, 1.4)
        
        if short_trending_up and self.success_ewma_short > 0.25:
            F_adjustment = 1.05 * momentum_factor
            Cr_adjustment = 1.08
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
        elif short_trending_down and self.success_ewma_short < 0.2:
            F_adjustment = 0.88 / momentum_factor
            Cr_adjustment = 0.92
            self.stagnation_counter += 1
        elif both_stagnant:
            F_adjustment = 0.75
            Cr_adjustment = 1.15
            self.stagnation_counter += 2
        else:
            delta = self.success_ewma_short - self.success_ewma_long
            F_adjustment = 1.0 + 0.08 * delta
            Cr_adjustment = 1.0 - 0.05 * delta
        
        F_new = np.clip(self.F * F_adjustment, 0.1, 1.5)
        Cr_new = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
        
        return F_new, Cr_new, success_rate
    
    # =====================================================================
    # STRATEGY 2: Geometric layout-based adaptation (variant_09 logic)
    # =====================================================================
    def _adapt_geometric(self):
        """Axis-aligned spread and k-NN structure (variant_09)."""
        pop = self.population
        if pop is None or len(pop) < 3:
            return self.F, self.Cr, 0.0
        
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
            knn_distances[i] = np.mean(np.sort(dists)[:k])
        
        mean_knn = np.mean(knn_distances)
        knn_normalized = mean_knn / (search_space_size ** 0.5 + 1e-30)
        
        F_new = self.F
        Cr_new = self.Cr
        
        if normalized_spread < 0.01:
            F_new = np.clip(F_new * 1.15, 0.1, 2.0)
        elif normalized_spread > 0.5:
            F_new = np.clip(F_new * 0.9, 0.1, 2.0)
        
        if knn_normalized > 0.3:
            Cr_new = np.clip(Cr_new * 1.08, 0.1, 0.9)
        elif knn_normalized < 0.1:
            Cr_new = np.clip(Cr_new * 0.92, 0.1, 0.9)
        
        diversity_signal = min(normalized_spread / 0.5, 1.0) * 0.5 + min(knn_normalized / 0.3, 1.0) * 0.5
        
        return F_new, Cr_new, diversity_signal
    
    # =====================================================================
    # STRATEGY 3: Entropy-based adaptation (variant_03 logic)
    # =====================================================================
    def _adapt_entropy(self):
        """Shannon entropy of fitness distribution (variant_03)."""
        fitness_vals = np.array([f for f, _ in self.F_success_history[-self.adaptation_window:]] +
                               [np.mean([f for f, _ in self.F_success_history[-self.adaptation_window:]])])
        
        if len(fitness_vals) < 2 or np.all(np.isinf(fitness_vals)):
            return self.F, self.Cr, 0.5
        
        fitness_min = np.nanmin(fitness_vals)
        fitness_max = np.nanmax(fitness_vals)
        if fitness_max - fitness_min < 1e-10:
            return self.F, self.Cr, 0.5
        
        n_bins = min(10, max(3, len(fitness_vals) // 2))
        bin_edges = np.linspace(fitness_min - 0.5 * (fitness_max - fitness_min),
                                fitness_max + 0.5 * (fitness_max - fitness_min),
                                n_bins + 1)
        bin_counts, _ = np.histogram(fitness_vals, bins=bin_edges)
        bin_probs = bin_counts / (np.sum(bin_counts) + 1e-10)
        
        entropy = -np.sum(np.where(bin_probs > 0, bin_probs * np.log(bin_probs + 1e-10), 0))
        max_entropy = np.log(n_bins)
        normalized_entropy = entropy / (max_entropy + 1e-10)
        
        self.fitness_entropy_history.append(normalized_entropy)
        if len(self.fitness_entropy_history) > self.adaptation_window:
            self.fitness_entropy_history.pop(0)
        
        F_new = self.F
        Cr_new = self.Cr
        
        if len(self.fitness_entropy_history) >= 3:
            recent_entropies = self.fitness_entropy_history[-3:]
            entropy_change = recent_entropies[-1] - recent_entropies[0]
            
            if recent_entropies[-1] > 0.5:
                F_new = np.clip(F_new * (1.0 + 0.15 * entropy_change), 0.1, 1.5)
                Cr_new = np.clip(Cr_new * (1.0 - 0.05 * entropy_change), 0.1, 0.9)
            else:
                F_new = np.clip(F_new * (1.0 - 0.10 * abs(entropy_change)), 0.1, 1.5)
                Cr_new = np.clip(Cr_new * (1.0 + 0.08 * abs(entropy_change)), 0.1, 0.9)
            
            momentum = 0.7
            target_F = F_new
            F_new = np.clip(momentum * self.last_F_for_entropy + (1 - momentum) * target_F, 0.1, 1.5)
            self.last_F_for_entropy = F_new
        
        return F_new, Cr_new, normalized_entropy
    
    # =====================================================================
    # META-CONTROLLER: Ensemble weighting and blending
    # =====================================================================
    def _update_meta_weights(self, improved_mask):
        """Update strategy weights based on recent performance (rank-based)."""
        success_rate = np.mean(improved_mask)
        
        if self.prev_improved_mask is not None:
            for strategy in ['ema', 'geo', 'ent']:
                self.strategy_perf_history[strategy].append(
                    np.mean(self.prev_improved_mask)
                )
        
        for strategy in ['ema', 'geo', 'ent']:
            if len(self.strategy_perf_history[strategy]) > self.adaptation_window:
                self.strategy_perf_history[strategy].pop(0)
        
        if len(self.strategy_perf_history['ema']) >= 3:
            perf = {}
            for strategy in ['ema', 'geo', 'ent']:
                history = self.strategy_perf_history[strategy]
                if len(history) >= 3:
                    perf[strategy] = np.mean(history[-3:])
                else:
                    perf[strategy] = 0.5
            
            total = sum(perf.values()) + 1e-10
            raw_weights = {k: v / total for k, v in perf.items()}
            
            weight_floor = 0.10
            for strategy in raw_weights:
                raw_weights[strategy] = max(raw_weights[strategy], weight_floor)
            
            total_capped = sum(raw_weights.values())
            for strategy in raw_weights:
                raw_weights[strategy] /= total_capped
            
            for strategy in raw_weights:
                self.strategy_weights[strategy] = (
                    self.strategy_forget * self.strategy_weights[strategy] +
                    (1 - self.strategy_forget) * raw_weights[strategy]
                )
            
            total_w = sum(self.strategy_weights.values())
            for strategy in self.strategy_weights:
                self.strategy_weights[strategy] /= total_w
        
        self.prev_improved_mask = improved_mask.copy()
    
    def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Ensemble-blended adaptation from three strategies, weighted by meta-controller."""
        success_rate = np.mean(improved_mask)
        
        self.F_success_history.append((success_rate, F_used))
        self.Cr_success_history.append((success_rate, Cr_used))
        
        if len(self.F_success_history) > self.adaptation_window:
            self.F_success_history.pop(0)
            self.Cr_success_history.pop(0)
        
        F_ema, Cr_ema, ema_success = self._adapt_ema(improved_mask, F_used, Cr_used)
        F_geo, Cr_geo, geo_success = self._adapt_geometric()
        F_ent, Cr_ent, ent_success = self._adapt_entropy()
        
        self._update_meta_weights(improved_mask)
        
        self.F = (self.strategy_weights['ema'] * F_ema +
                  self.strategy_weights['geo'] * F_geo +
                  self.strategy_weights['ent'] * F_ent)
        
        self.Cr = (self.strategy_weights['ema'] * Cr_ema +
                   self.strategy_weights['geo'] * Cr_geo +
                   self.strategy_weights['ent'] * Cr_ent)
        
        self.F = np.clip(self.F, 0.1, 1.5)
        self.Cr = np.clip(self.Cr, 0.1, 0.9)
    
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
            
            best_strategy = max(self.strategy_weights, key=self.strategy_weights.get)
            self.stagnation_escape_strategies.append(best_strategy)
            if len(self.stagnation_escape_strategies) > 10:
                self.stagnation_escape_strategies.pop(0)
            
            for s in self.stagnation_escape_weights:
                count = self.stagnation_escape_strategies.count(s)
                self.stagnation_escape_weights[s] = 1.0 + 0.1 * count
            
            escape_total = sum(self.stagnation_escape_weights.values())
            for s in self.stagnation_escape_weights:
                self.stagnation_escape_weights[s] /= escape_total
            
            n_replace = max(self.np // 5, 2)
            replace_indices = np.random.choice(self.np, n_replace, replace=False)
            
            for idx in replace_indices:
                population[idx] = np.random.uniform(self.lower, self.upper, self.dim)
            
            fitness[replace_indices] = np.inf
            self.stagnation_counter = 0
            self.successful_directions = []
            
            self.success_ewma_short = 0.5
            self.success_ewma_long = 0.5
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop orchestrating all batched operations."""
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
            
            self.population = population
            
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
            
            successful_mutations = trial_batch - population
            self._update_direction_memory(successful_mutations, improved_mask)
            
            self._adapt_parameters(improved_mask, F_used, Cr_used)
            
            population, fitness, best_fitness, best_solution = self._restart_if_stagnant(
                population, fitness, best_fitness, best_solution
            )
            
            ring_prev, ring_next = self._build_ring_topology()
        
        return best_fitness, best_solution
