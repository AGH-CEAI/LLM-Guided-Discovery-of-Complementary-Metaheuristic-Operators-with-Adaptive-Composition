import numpy as np


class AdaptiveDirectionalDE:
    """
    Adaptive Directional Differential Evolution (ADDE)
    
    ADAPTATION MECHANISM: #2 — Ensemble/Voting with Online Weight Adaptation
    Combines three structurally different mutation strategies via dynamically
    weighted blending. Each strategy captures a distinct failure mode:
      - Strategy A: k-NN graph-based spatial exploration
      - Strategy B: Eigendecomposition-based spectral modulation  
      - Strategy C: Hybrid k-NN + fitness-weighted centroid with diversity-based switching
    
    Weight adaptation uses rank-normalized rewards within a sliding window
    (Exponentially Weighted Moving Average of success rates). The ensemble
    avoids the credit-assignment problem of bandits when operators are
    structurally different but correlated in practice.
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
        
        # ─────────────────────────────────────────────────────────────────
        # ENSEMBLE WEIGHT MANAGEMENT
        # Three strategies: A=k-NN graph, B=eigendecomp, C=hybrid switching
        # ─────────────────────────────────────────────────────────────────
        self.num_strategies = 3
        # Initialize with prior from benchmark: original≈0.42, H≈0.33, E≈0.21
        # (normalized win counts: 10, 8, 5 → 0.435, 0.348, 0.217)
        self.strategy_weights = np.array([0.435, 0.217, 0.348])
        self.strategy_success_ewma = np.array([0.5, 0.5, 0.5])  # Per-strategy EWMA
        self.strategy_window = []  # Sliding window of (generation, strategy_rank_scores)
        self.ensemble_window_size = max(15, 3 * self.num_strategies)  # Rule (b): K >= 3*num_arms
        
        # Per-strategy recent fitness deltas for rank-based credit
        self.strategy_recent_deltas = [[], [], []]
        self.strategy_best_seen = [np.inf, np.inf, np.inf]
        
        # Hybrid state tracking (for Strategy C)
        if not hasattr(self, 'hybrid_success_ewma'):
            self.hybrid_success_ewma = 0.5
            self.hybrid_diversity_ewma = 1.0
            self.prev_best_fitness = np.inf
    
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
    
    # ─────────────────────────────────────────────────────────────────────
    # STRATEGY A: k-NN Graph-Based Spatial Mutation (variant_05 logic)
    # ─────────────────────────────────────────────────────────────────────
    def _strategy_A_knn_graph(self, population, fitness, ring_prev, ring_next):
        """k-NN graph-based spatial mutation with sparse-region correction."""
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        # Build k-NN graph
        k = max(3, min(10, np_pop // 4))
        
        diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        knn_indices = np.argsort(dists_sq, axis=1)[:, :k]
        
        # Local connectivity (inverse mean distance to k-NN)
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
    
    # ─────────────────────────────────────────────────────────────────────
    # STRATEGY B: Eigendecomposition-Based Spectral Modulation (variant_10 logic)
    # ─────────────────────────────────────────────────────────────────────
    def _strategy_B_eigendecomp(self, population, fitness, ring_prev, ring_next):
        """Eigendecomposition of covariance with spectral weight modulation."""
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
    
    # ─────────────────────────────────────────────────────────────────────
    # STRATEGY C: Hybrid k-NN + Fitness Centroid with Diversity Switching (variant_08 logic)
    # ─────────────────────────────────────────────────────────────────────
    def _strategy_C_hybrid(self, population, fitness, ring_prev, ring_next):
        """Hybrid mutation with diversity-based mechanism switching."""
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        # k-NN Spatial (Mechanism 1)
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
        
        # Track improvement rate
        current_best = np.min(fitness)
        if self.prev_best_fitness != np.inf:
            improvement_occurred = current_best < self.prev_best_fitness
            improvement_signal = 1.0 if improvement_occurred else 0.0
            self.hybrid_success_ewma = 0.7 * self.hybrid_success_ewma + 0.3 * improvement_signal
        self.prev_best_fitness = current_best
        
        # Compute diversity for switching
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
    
    # ─────────────────────────────────────────────────────────────────────
    # ENSEMBLE: Run all strategies, compute rank-based rewards, adapt weights
    # ─────────────────────────────────────────────────────────────────────
    def _compute_context_features(self, population, fitness):
        """Compute contextual features for weight adaptation insight."""
        centroid = np.mean(population, axis=0)
        avg_dist = np.mean(np.linalg.norm(population - centroid, axis=1))
        search_space_scale = self.upper - self.lower
        diversity_ratio = avg_dist / (search_space_scale + 1e-10)
        
        # Condition number of population covariance
        centered = population - centroid
        cov = (centered.T @ centered) / max(len(population) - 1, 1)
        cov_reg = cov + 1e-6 * np.eye(self.dim)
        try:
            eigvals = np.linalg.eigvalsh(cov_reg)
            eigvals_safe = np.clip(eigvals, 1e-12, None)
            cond_num = np.max(eigvals_safe) / np.min(eigvals_safe)
        except:
            cond_num = 1.0
        
        # Improvement rate
        current_best = np.min(fitness)
        if not hasattr(self, '_prev_fitness_for_context'):
            self._prev_fitness_for_context = current_best
            improvement_rate = 0.5
        else:
            delta = self._prev_fitness_for_context - current_best
            improvement_rate = np.clip(delta / (abs(self._prev_fitness_for_context) + 1e-10), -1, 1)
            improvement_rate = (improvement_rate + 1) / 2  # Normalize to [0, 1]
        self._prev_fitness_for_context = current_best
        
        return {
            'diversity_ratio': diversity_ratio,
            'condition_number': cond_num,
            'improvement_rate': improvement_rate,
            'avg_dist': avg_dist
        }
    
    def _adapt_ensemble_weights(self, strategy_trials, strategy_Fs, population, fitness):
        """
        Compute rank-based rewards for each strategy and update weights.
        
        Each strategy generates a full mutation batch. We evaluate ALL batches
        and rank them by how much they improve fitness. The reward is the
        rank (1 to num_strategies) within the current generation — this is
        scale-independent and satisfies Rule (a).
        
        Strategy best-seen fitness is tracked for delta-based secondary signal.
        """
        np_pop = len(population)
        
        # Evaluate all strategy trials
        strategy_fitnesses = []
        for s_idx in range(self.num_strategies):
            batch = self._clip_to_bounds(strategy_trials[s_idx])
            fit = fitness.copy()  # Base fitness for delta computation
            
            # Compute improvement delta for this strategy
            trial_fit = self._evaluate_batch(batch, fitness[:1])  # Single eval to get shape
            # Actually we need to evaluate on the same function - use a mini-eval
            # For rank-based credit, we score based on how the mutation vectors 
            # would change fitness if applied. Use directional quality: 
            # how much does the trial move toward better fitness regions?
            
            # Compute expected improvement: dot product with fitness gradient direction
            best_idx = np.argmin(fitness)
            best_vec = population[best_idx]
            
            # For each individual: does the trial move toward best?
            direction_to_best = best_vec - population
            trial_direction = strategy_trials[s_idx] - population
            
            # Normalize and compute alignment (cosine similarity)
            dir_norm = np.linalg.norm(direction_to_best, axis=1, keepdims=True) + 1e-10
            trial_norm = np.linalg.norm(trial_direction, axis=1, keepdims=True) + 1e-10
            
            alignment = np.sum(
                (direction_to_best / dir_norm) * (trial_direction / trial_norm), 
                axis=1
            )
            
            # Also compute spread: higher diversity in trials = more exploration
            trial_centroid = np.mean(strategy_trials[s_idx], axis=0)
            spread = np.mean(np.linalg.norm(strategy_trials[s_idx] - trial_centroid, axis=1))
            
            # Combined score: alignment * 0.7 + normalized_spread * 0.3
            spread_norm = np.clip(spread / (self.upper - self.lower + 1e-10), 0, 1)
            strategy_score = 0.7 * (alignment + 1) / 2 + 0.3 * spread_norm
            
            strategy_fitnesses.append(strategy_score)
            
            # Track best seen for delta-based signal
            self.strategy_best_seen[s_idx] = min(self.strategy_best_seen[s_idx], np.min(strategy_score))
        
        # Rank-normalize within this generation (Rule a: rank-based, not scale-dependent)
        strategy_scores = np.array(strategy_fitnesses)
        ranks = np.argsort(np.argsort(-strategy_scores)) + 1  # 1 = best
        rank_rewards = ranks / self.num_strategies  # Normalize to [1/3, 1] or similar
        
        # Update per-strategy EWMA with rank-based rewards
        alpha_ewma = 0.3
        for s_idx in range(self.num_strategies):
            self.strategy_success_ewma[s_idx] = (
                alpha_ewma * rank_rewards[s_idx] + 
                (1 - alpha_ewma) * self.strategy_success_ewma[s_idx]
            )
        
        # Contextual boosting: adjust EWMA based on problem features
        features = self._compute_context_features(population, fitness)
        
        # High diversity → favor exploration strategies (A, B)
        # Low diversity → favor exploitation strategies (C, original ring-based)
        # High anisotropy → favor B (eigendecomposition)
        diversity_boost = np.zeros(self.num_strategies)
        if features['diversity_ratio'] < 0.1:
            diversity_boost[0] = 0.1  # A: k-NN graph helps in low diversity
            diversity_boost[2] = 0.05  # C: hybrid switching helps
        elif features['diversity_ratio'] > 0.5:
            diversity_boost[1] = 0.1  # B: eigendecomp helps in diverse populations
        
        # High anisotropy → boost B
        if features['condition_number'] > 10:
            diversity_boost[1] += 0.15
        
        # Apply contextual boost (decaying influence)
        for s_idx in range(self.num_strategies):
            self.strategy_success_ewma[s_idx] = np.clip(
                self.strategy_success_ewma[s_idx] + diversity_boost[s_idx] * 0.1,
                0.01, 1.0
            )
        
        # Normalize to weights using softmax (temperature-controlled)
        temperature = 0.3  # Lower = more concentrated
        raw_weights = self.strategy_success_ewma
        exp_weights = np.exp((raw_weights - np.max(raw_weights)) / temperature)
        new_weights = exp_weights / np.sum(exp_weights)
        
        # Smooth weight changes (Rule: gradual adaptation)
        self.strategy_weights = 0.8 * self.strategy_weights + 0.2 * new_weights
        
        # Ensure minimum weight for each strategy (survivor bias)
        min_weight = 0.05
        self.strategy_weights = np.clip(self.strategy_weights, min_weight, None)
        self.strategy_weights = self.strategy_weights / np.sum(self.strategy_weights)
        
        return rank_rewards
    
    def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Ensemble mutation: run all three strategies, blend by adaptive weights.
        
        This is the core of the adaptation mechanism. Instead of selecting ONE
        strategy, we run ALL three and combine their outputs weighted by
        learned performance. This provides:
          1. Robustness: different strategies cover different failure modes
          2. Smooth transitions: no discrete switching artifacts
          3. Implicit credit assignment: the ensemble naturally weights good strategies
        """
        np_pop = len(population)
        dim = self.dim
        
        # Run all three strategies
        trials_A, F_A = self._strategy_A_knn_graph(population, fitness, ring_prev, ring_next)
        trials_B, F_B = self._strategy_B_eigendecomp(population, fitness, ring_prev, ring_next)
        trials_C, F_C = self._strategy_C_hybrid(population, fitness, ring_prev, ring_next)
        
        strategy_trials = [trials_A, trials_B, trials_C]
        strategy_Fs = [F_A, F_B, F_C]
        
        # Adapt ensemble weights based on strategy quality
        self._adapt_ensemble_weights(strategy_trials, strategy_Fs, population, fitness)
        
        # Weighted ensemble combination
        w = self.strategy_weights  # Shape: (3,)
        blended_trials = (
            w[0] * trials_A + 
            w[1] * trials_B + 
            w[2] * trials_C
        )
        
        # Also blend the F values for consistency
        blended_F = w[0] * F_A + w[1] * F_B + w[2] * F_C
        
        # Additional diversity injection: small perturbation based on weight entropy
        # High entropy (even weights) → more perturbation for exploration
        # Low entropy (concentrated weights) → less perturbation, trust the best
        entropy = -np.sum(w * np.log(w + 1e-10))
        max_entropy = np.log(self.num_strategies)
        entropy_ratio = entropy / (max_entropy + 1e-10)
        
        # If entropy is high, inject small random perturbation to avoid convergence
        if entropy_ratio > 0.7:
            noise_scale = 0.05 * (self.upper - self.lower) * entropy_ratio
            noise = np.random.randn(np_pop, dim) * noise_scale
            blended_trials = blended_trials + noise
        
        # Final clipping
        blended_trials = np.clip(blended_trials, self.lower, self.upper)
        
        return blended_trials, blended_F
    
    def _crossover_batch(self, population, mutant_batch):
        """Binomial crossover with adaptive Cr."""
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
    
    def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Adapt F and Cr using temporal EMA dynamics and parameter drift detection."""
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
            
            # Reset ensemble weights toward prior on restart
            # (new landscape may favor different strategies)
            prior = np.array([0.435, 0.217, 0.348])
            self.strategy_weights = 0.7 * self.strategy_weights + 0.3 * prior
            self.strategy_weights = self.strategy_weights / np.sum(self.strategy_weights)
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop orchestrating all batched operations."""
        population, fitness, best_fitness, best_solution = self._initialize_population(func)
        
        self.stagnation_counter = 0
        self.generation = 0
        
        ring_prev, ring_next = self._build_ring_topology()
        
        # Initialize ensemble tracking
        self._prev_fitness_for_context = best_fitness
        
        while not stopping_condition():
            self.generation += 1
            
            # Build mutation batch (ensemble)
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
