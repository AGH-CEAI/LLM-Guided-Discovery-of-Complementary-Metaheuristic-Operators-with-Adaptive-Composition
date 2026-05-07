import numpy as np


class AdaptiveDirectionalDE:
    """
    // ENSEMBLE/VOTING (Mechanism #2)
    // The four winning variants compute related but distinct spatial/fitness signals
    // (k-NN connectivity, eigendecomposition, hybrid weighting). They are correlated
    // but capture different failure modes. A bandit cannot reliably credit-assign
    // between near-identical arms, but a weighted ensemble blends their strengths
    // and provides a smoother, more robust signal.
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
        
        # ============================================================
        # ENSEMBLE ADAPTATION STATE
        # Four mutation strategies: original ring-based, k-NN graph,
        # hybrid k-NN+fitness, eigendecomposition-based
        # ============================================================
        self.num_strategies = 4
        self.ensemble_weights = np.ones(self.num_strategies) / self.num_strategies
        
        # Rank-based reward history (sliding window)
        # Window size calibrated: K >= 3 * num_strategies
        self.reward_window_size = max(12, 3 * self.num_strategies)
        self.reward_history = [[] for _ in range(self.num_strategies)]
        
        # Track per-strategy fitness for rank-based rewards
        self.strategy_fitness_history = [[] for _ in range(self.num_strategies)]
        
        # EMA smoothing for weight updates
        self.ema_alpha = 0.3  # Forgetting factor for strategy quality
        
        # Ensemble diversity floor: minimum weight per strategy
        self.min_strategy_weight = 0.05
        
        # Hybrid state tracking (used by strategy 2)
        self.hybrid_success_ewma = 0.5
        self.hybrid_diversity_ewma = 1.0
        self.prev_best_fitness = None
    
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
        self.prev_best_fitness = fitness[best_idx]
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
    
    # ================================================================
    # STRATEGY 0: Original ring-based mutation with directional memory
    # ================================================================
    def _strategy_original(self, population, fitness, ring_prev, ring_next):
        """Original ring-based composite mutation with directional bias."""
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
    
    # ================================================================
    # STRATEGY 1: k-NN graph-based mutation (from variant_05)
    # ================================================================
    def _strategy_knn_graph(self, population, fitness, ring_prev, ring_next):
        """k-NN graph connectivity-based mutation with sparse region detection."""
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        k = max(3, min(10, np_pop // 4))
        
        diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        knn_indices = np.argsort(dists_sq, axis=1)[:, :k]
        
        local_density = np.zeros(np_pop)
        for i in range(np_pop):
            local_density[i] = np.mean(dists_sq[i, knn_indices[i]])
        local_density = np.clip(local_density, 1e-10, None)
        connectivity = 1.0 / local_density
        
        conn_min, conn_max = np.min(connectivity), np.max(connectivity)
        if conn_max > conn_min:
            conn_norm = (connectivity - conn_min) / (conn_max - conn_min)
        else:
            conn_norm = np.ones(np_pop) * 0.5
        
        sparse_mask = conn_norm < 0.3
        
        best_idx = np.argmin(fitness)
        sorted_idx = np.argsort(fitness)
        second_best_idx = sorted_idx[1] if np_pop > 1 else best_idx
        
        graph_parent1 = np.array([knn_indices[i, np.random.randint(k)] for i in range(np_pop)])
        far_indices = np.argsort(dists_sq, axis=1)[:, -1]
        graph_parent2 = far_indices[graph_parent1]
        
        current_F = self.F * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)
        
        p1 = population[graph_parent1]
        p2 = population[graph_parent2]
        
        graph_mutation = p1 + current_F * (p2 - population)
        
        best_vector = population[best_idx]
        bias_strength = np.maximum(0, 0.3 - conn_norm) * 2.0
        
        for i in range(np_pop):
            if sparse_mask[i]:
                trials[i] = (1 - bias_strength[i]) * graph_mutation[i] + \
                            bias_strength[i] * (population[i] + current_F * (best_vector - population[i]))
            else:
                trials[i] = graph_mutation[i]
        
        trials = np.clip(trials, self.lower, self.upper)
        return trials, current_F
    
    # ================================================================
    # STRATEGY 2: Hybrid k-NN + fitness improvement (from variant_08)
    # ================================================================
    def _strategy_hybrid_knn_fitness(self, population, fitness, ring_prev, ring_next):
        """Hybrid mutation combining k-NN spatial and fitness improvement mechanisms."""
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        k = min(5, np_pop - 1)
        sorted_indices = np.argsort(fitness)
        knn_centroids = np.zeros((np_pop, dim))
        
        for i in range(np_pop):
            better_indices = sorted_indices[:min(k + 1, np_pop)]
            better_indices = better_indices[better_indices != i]
            
            if len(better_indices) > 0:
                knn_neighbors = population[better_indices]
                knn_centroids[i] = np.mean(knn_neighbors, axis=0)
            else:
                knn_centroids[i] = np.mean(population, axis=0)
        
        current_best = np.min(fitness)
        if self.prev_best_fitness is not None:
            improvement_occurred = current_best < self.prev_best_fitness
            improvement_signal = 1.0 if improvement_occurred else 0.0
            self.hybrid_success_ewma = 0.7 * self.hybrid_success_ewma + 0.3 * improvement_signal
        self.prev_best_fitness = current_best
        
        centroid = np.mean(population, axis=0)
        avg_dist = np.mean(np.linalg.norm(population - centroid, axis=1))
        self.hybrid_diversity_ewma = 0.9 * self.hybrid_diversity_ewma + 0.1 * avg_dist
        
        search_space_scale = self.upper - self.lower
        diversity_ratio = avg_dist / (search_space_scale + 1e-10)
        diversity_threshold = 0.05
        
        low_diversity = diversity_ratio < diversity_threshold
        low_improvement = self.hybrid_success_ewma < 0.3
        
        if low_diversity:
            spatial_weight = 0.8
        elif low_improvement:
            spatial_weight = 0.65
        else:
            spatial_weight = 0.5
        
        r1 = ring_prev
        r2 = ring_next
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        diversity_factor = np.clip(avg_dist / 50.0, 0.5, 2.0)
        current_F = self.F * diversity_factor * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)
        
        spatial_mutation = knn_centroids + current_F * (rand1_vectors - rand2_vectors)
        
        weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
        weights = weights / np.sum(weights)
        fitness_centroid = np.sum(population * weights[:, np.newaxis], axis=0)
        
        to_centroid = fitness_centroid - population
        fitness_mutation = population + current_F * to_centroid + 0.5 * (rand1_vectors - rand2_vectors)
        
        trials = spatial_weight * spatial_mutation + (1 - spatial_weight) * fitness_mutation
        
        return trials, current_F
    
    # ================================================================
    # STRATEGY 3: Eigendecomposition-based (from variant_10)
    # ================================================================
    def _strategy_eigen(self, population, fitness, ring_prev, ring_next):
        """Eigendecomposition-based mutation modulating along principal axes."""
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        centroid = np.mean(population, axis=0)
        centered = population - centroid
        cov = (centered.T @ centered) / max(np_pop - 1, 1)
        
        cov_reg = cov + 1e-6 * np.eye(dim)
        
        try:
            eigvals, eigvecs = np.linalg.eigh(cov_reg)
            idx = np.argsort(eigvals)
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
        except np.linalg.LinAlgError:
            r1 = ring_prev
            r2 = ring_next
            current_F = np.clip(self.F * (1.0 + 0.1 * np.random.randn()), 0.1, 2.0)
            trials = population[r1] + current_F * (population[r2] - population)
            return trials, current_F
        
        eigvals_safe = np.clip(eigvals, 1e-12, None)
        cond_num = np.max(eigvals_safe) / np.min(eigvals_safe)
        
        eigvals_norm = eigvals_safe / (np.sum(eigvals_safe) + 1e-12)
        spectral_weights = 1.0 / (eigvals_norm + 0.01)
        spectral_weights = spectral_weights / np.sum(spectral_weights)
        
        anisotropy_factor = np.log1p(cond_num) / np.log1p(dim * dim)
        anisotropy_factor = np.clip(anisotropy_factor, 0.0, 1.0)
        
        r1 = ring_prev
        r2 = ring_next
        current_F = np.clip(self.F * (1.0 + 0.1 * np.random.randn()), 0.1, 2.0)
        base_mutation = population[r1] + current_F * (population[r2] - population)
        
        base_mutation_centered = base_mutation - centroid
        proj_coeffs = base_mutation_centered @ eigvecs
        
        modulation_strength = 0.3 + 0.4 * anisotropy_factor
        modulation = 1.0 + modulation_strength * (spectral_weights - np.mean(spectral_weights)) / (np.std(spectral_weights) + 1e-8)
        modulation = np.clip(modulation, 0.3, 2.5)
        
        modulated_coeffs = proj_coeffs * modulation
        modulated_mutation = modulated_coeffs @ eigvecs.T + centroid
        
        blend_weight = 0.3 + 0.4 * anisotropy_factor
        blend_weight = np.clip(blend_weight, 0.2, 0.7)
        trials = (1 - blend_weight) * base_mutation + blend_weight * modulated_mutation
        
        weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
        weights = weights / np.sum(weights)
        centroid_weighted = np.sum(population * weights[:, np.newaxis], axis=0)
        
        directional_component = centroid_weighted - centroid
        trials = trials + 0.15 * directional_component
        
        return trials, current_F
    
    # ================================================================
    # ENSEMBLE MUTATION: Run all strategies, blend by learned weights
    # ================================================================
    def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Generate mutation vectors using adaptive ensemble of four strategies.
        Each strategy is evaluated, and weights are updated based on rank-based
        rewards within a sliding window.
        """
        np_pop = len(population)
        
        # Run all strategies
        strategy_mutations = []
        strategy_Fs = []
        
        strat0, F0 = self._strategy_original(population, fitness, ring_prev, ring_next)
        strategy_mutations.append(strat0)
        strategy_Fs.append(F0)
        
        strat1, F1 = self._strategy_knn_graph(population, fitness, ring_prev, ring_next)
        strategy_mutations.append(strat1)
        strategy_Fs.append(F1)
        
        strat2, F2 = self._strategy_hybrid_knn_fitness(population, fitness, ring_prev, ring_next)
        strategy_mutations.append(strat2)
        strategy_Fs.append(F2)
        
        strat3, F3 = self._strategy_eigen(population, fitness, ring_prev, ring_next)
        strategy_mutations.append(strat3)
        strategy_Fs.append(F3)
        
        # Stack mutations: (num_strategies, np_pop, dim)
        all_mutations = np.stack(strategy_mutations, axis=0)
        
        # Evaluate each strategy's trial fitness (quick proxy evaluation)
        # Use distance to centroid as a proxy for exploration quality
        centroid = np.mean(population, axis=0)
        strategy_scores = np.zeros(self.num_strategies)
        
        for s in range(self.num_strategies):
            # Score = mean distance of mutation from centroid (exploration)
            # + mean improvement potential (negative distance to best)
            mutation_dist = np.mean(np.linalg.norm(all_mutations[s] - centroid, axis=1))
            
            # Direction quality: does mutation go toward better fitness regions?
            best_idx = np.argmin(fitness)
            direction_to_best = np.mean(np.linalg.norm(
                all_mutations[s] - population[best_idx], axis=1
            ))
            
            # Combined score: prefer mutations that explore and head toward good regions
            strategy_scores[s] = mutation_dist - 0.5 * direction_to_best
        
        # Update reward history with rank-based rewards
        current_best_fitness = np.min(fitness)
        
        for s in range(self.num_strategies):
            # Rank-based reward: convert score to percentile rank
            rank = (self.num_strategies - np.argsort(strategy_scores)[::-1].tolist().index(s)) / self.num_strategies
            
            self.reward_history[s].append(rank)
            
            # Also record fitness proxy
            self.strategy_fitness_history[s].append(strategy_scores[s])
            
            # Maintain sliding window
            if len(self.reward_history[s]) > self.reward_window_size:
                self.reward_history[s].pop(0)
            if len(self.strategy_fitness_history[s]) > self.reward_window_size:
                self.strategy_fitness_history[s].pop(0)
        
        # Update ensemble weights using EMA of recent rewards
        for s in range(self.num_strategies):
            if len(self.reward_history[s]) > 0:
                # Compute mean reward in sliding window
                mean_reward = np.mean(self.reward_history[s])
                
                # Z-score normalize rewards within window for stable updates
                if len(self.reward_history[s]) > 1:
                    std_reward = np.std(self.reward_history[s])
                    if std_reward > 1e-10:
                        normalized_reward = (mean_reward - np.mean(self.reward_history[s])) / std_reward
                    else:
                        normalized_reward = 0.0
                else:
                    normalized_reward = mean_reward - 0.5  # Center around 0.5
                
                # Update weight with EMA
                # Map normalized reward to weight change: reward > 0.5 -> increase weight
                weight_delta = self.ema_alpha * (normalized_reward)
                
                # Apply update with momentum
                current_weight = self.ensemble_weights[s]
                new_weight = current_weight + weight_delta
                
                # Apply diversity floor
                self.ensemble_weights[s] = max(self.min_strategy_weight, new_weight)
        
        # Renormalize weights to sum to 1
        weight_sum = np.sum(self.ensemble_weights)
        if weight_sum > 0:
            self.ensemble_weights = self.ensemble_weights / weight_sum
        else:
            self.ensemble_weights = np.ones(self.num_strategies) / self.num_strategies
        
        # Weighted blend of all strategy mutations
        # Shape: (np_pop, dim) = sum over strategies of weight * mutation
        blended_trials = np.zeros((np_pop, self.dim))
        
        for s in range(self.num_strategies):
            blended_trials += self.ensemble_weights[s] * all_mutations[s]
        
        # Average F from all strategies (weighted)
        avg_F = np.mean(strategy_Fs)
        
        return blended_trials, avg_F
    
    def _crossover_batch(self, population, mutant_batch):
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        Cr_individual = np.clip(
            self.Cr + 0.1 * np.random.randn(np_pop),
            0.1, 0.9
        )
        
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
    
    def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        success_rate = np.mean(improved_mask)

        if not hasattr(self, 'success_ewma_short'):
            self.success_ewma_short = success_rate
            self.success_ewma_long = success_rate
            self.stagnation_counter = 0
            self.F_history = [self.F]
            self.Cr_history = [self.Cr]

        alpha_short = 0.3
        alpha_long = 0.1
        self.success_ewma_short = alpha_short * success_rate + (1 - alpha_short) * self.success_ewma_short
        self.success_ewma_long = alpha_long * success_rate + (1 - alpha_long) * self.success_ewma_long

        self.F_history.append(F_used)
        self.Cr_history.append(F_used)
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

        self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
    
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
            
            # Reset ensemble weights slightly toward uniform on restart
            # (search regime may have changed)
            self.ensemble_weights = 0.7 * self.ensemble_weights + 0.3 * (np.ones(self.num_strategies) / self.num_strategies)
        
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
            
            self._adapt_parameters(improved_mask, F_used, Cr_used)
            
            population, fitness, best_fitness, best_solution = self._restart_if_stagnant(
                population, fitness, best_fitness, best_solution
            )
            
            ring_prev, ring_next = self._build_ring_topology()
        
        return best_fitness, best_solution
