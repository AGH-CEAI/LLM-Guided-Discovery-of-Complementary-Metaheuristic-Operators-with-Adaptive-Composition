```python
import numpy as np


class AdaptiveDirectionalDE:
    """
    MECHANISM #2 (ENSEMBLE/VOTING): Combining three mutation strategies with
    exponentially-weighted averaging based on rank-normalized reward signals.
    
    Why ensemble fits the benchmark data:
    - Wins are spread across 4 variants (no single dominant strategy >50%)
    - All three strategies compute correlated population-structure metrics
    - Ensemble provides robustness without costly credit-assignment in a bandit
    - Rank-based rewards avoid scale-dependent magic constants (Rule a)
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
        
        # --- ENSEMBLE ADAPTATION STATE ---
        # Three mutation strategies: original, k-NN graph, eigendecomposition
        self.num_strategies = 3
        self.ensemble_weights = np.ones(self.num_strategies) / self.num_strategies
        
        # Sliding window for rank-based rewards (Rule b: K >= 3 * num_arms = 9)
        self.reward_window_size = 12
        self.strategy_best_history = [[] for _ in range(self.num_strategies)]
        
        # Softmax learning rate for weight updates
        self.ensemble_lr = 0.1
        
        # Fallback threshold to prevent weight collapse
        self.min_weight_threshold = 0.05
        
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # STRATEGY 0: Original ring-based mutation (baseline from original.py)
    # ─────────────────────────────────────────────────────────────────────────
    def _mutate_original(self, population, fitness, ring_prev, ring_next):
        """Original ring-based mutation strategy."""
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # STRATEGY 1: k-NN graph-based mutation (from variant_05_catE_idea_0.py)
    # ─────────────────────────────────────────────────────────────────────────
    def _mutate_knn_graph(self, population, fitness, ring_prev, ring_next):
        """k-NN graph-based mutation exploiting spatial connectivity."""
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        # Build k-NN graph based on Euclidean distance
        k = max(3, min(10, np_pop // 4))
        
        diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        
        knn_indices = np.argsort(dists_sq, axis=1)[:, :k]
        
        # Graph-based property: local connectivity (inverse mean distance to k-NN)
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
    
    # ─────────────────────────────────────────────────────────────────────────
    # STRATEGY 2: Eigendecomposition-based mutation (from variant_10_catB_idea_0.py)
    # ─────────────────────────────────────────────────────────────────────────
    def _mutate_eigen(self, population, fitness, ring_prev, ring_next):
        """Eigendecomposition-based mutation modulating along population covariance axes."""
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
            # Fallback: ring-based rand/1
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
        
        # Fitness-weighted directional component
        weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
        weights = weights / np.sum(weights)
        centroid_weighted = np.sum(population * weights[:, np.newaxis], axis=0)
        
        directional_component = centroid_weighted - centroid
        trials = trials + 0.15 * directional_component
        
        return trials, current_F
    
    # ─────────────────────────────────────────────────────────────────────────
    # ENSEMBLE WEIGHT UPDATE (rank-based, scale-independent per Rule a)
    # ─────────────────────────────────────────────────────────────────────────
    def _update_ensemble_weights(self, strategy_best_fitness):
        """
        Update ensemble weights using rank-normalized rewards within sliding window.
        
        For each strategy, compute mean best-fitness over the reward window,
        rank them (1 = best = lowest fitness), then update via softmax on ranks.
        
        This is scale-independent: we only use relative ordering, not magnitudes.
        """
        # Record current best fitness for each strategy
        for s, bf in enumerate(strategy_best_fitness):
            self.strategy_best_history[s].append(bf)
            if len(self.strategy_best_history[s]) > self.reward_window_size:
                self.strategy_best_history[s].pop(0)
        
        # Need at least 2 entries per strategy to compute meaningful rank
        min_entries = 2
        if all(len(h) >= min_entries for h in self.strategy_best_history):
            # Compute mean best-fitness for each strategy over the window
            mean_fits = np.array([np.mean(h) for h in self.strategy_best_history])
            
            # Rank: 1 = best (lowest), 3 = worst (highest) — only finite values
            valid_mask = np.isfinite(mean_fits)
            if np.sum(valid_mask) >= 2:
                ranks = np.zeros(self.num_strategies)
                valid_fits = mean_fits[valid_mask]
                sorted_order = np.argsort(valid_fits)
                ranks[valid_mask] = sorted_order + 1  # 1-indexed ranks
                ranks[~valid_mask] = (self.num_strategies + 1) / 2  # middle rank for invalid
                
                # Softmax update: higher rank (better) → higher weight
                # softmax(-lr * rank) gives higher weight to smaller rank
                logits = -self.ensemble_lr * ranks
                logits -= np.max(logits)  # numerical stability
                exp_logits = np.exp(logits)
                new_weights = exp_logits / np.sum(exp_logits)
                
                # Fallback: prevent weight collapse (Rule d)
                if np.min(new_weights) < self.min_weight_threshold:
                    self.ensemble_weights = np.ones(self.num_strategies) / self.num_strategies
                else:
                    # EMA-like smoothing for stability
                    self.ensemble_weights = 0.7 * self.ensemble_weights + 0.3 * new_weights
                    self.ensemble_weights /= np.sum(self.ensemble_weights)
    
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
        """Compute population diversity via average distance to centroid."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Adapt F and Cr using temporal EMA dynamics."""
        success_rate = np.mean(improved_mask)
        
        if not hasattr(self, 'success_ewma_short'):
            self.success_ewma_short = success_rate
            self.success_ewma_long = success_rate
            self.stagnation_counter = 0
            self.F_history = [self.F]
            self.Cr_history = [self.Cr]
        
        alpha_short, alpha_long = 0.3, 0.1
        self.success_ewma_short = alpha_short * success_rate + (1 - alpha_short) * self.success_ewma_short
        self.success_ewma_long = alpha_long * success_rate + (1 - alpha_long) * self.success_ewma_long
        
        self.F_history.append(F_used)
        self.Cr_history.append(Cr_used)
        if len(self.F_history) > 15:
            self.F_history.pop(0)
            self.Cr_history.pop(0)
        
        F_mean = np.mean(self.F_history)
        F_drift = self.F / (F_mean + 1e-10)
        
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
                population[idx] = np.random.uniform(self.lower, self.upper, self.dim)
            
            fitness[replace_indices] = np.inf
            self.stagnation_counter = 0
            self.successful_directions = []
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop with ensemble mutation strategy."""
        population, fitness, best_fitness, best_solution = self._initialize_population(func)
        
        self.stagnation_counter = 0
        self.generation = 0
        
        ring_prev, ring_next = self._build_ring_topology()
        
        while not stopping_condition():
            self.generation += 1
            
            # ── ENSEMBLE: Compute all three mutation strategies ──
            t0, F0 = self._mutate_original(population, fitness, ring_prev, ring_next)
            t1, F1 = self._mutate_knn_graph(population, fitness, ring_prev, ring_next)
            t2, F2 = self._mutate_eigen(population, fitness, ring_prev, ring_next)
            
            trials_list = [t0, t1, t2]
            F_list = [F0, F1, F2]
            
            # ── BLEND using current ensemble weights ──
            mutant_batch = np.zeros_like(population)
            for t, w in zip(trials_list, self.ensemble_weights):
                mutant_batch += w * t
            
            F_used = np.mean(F_list)
            
            # Crossover on blended mutant batch
            trial_batch, Cr_used = self._crossover_batch(population, mutant_batch)
            
            if stopping_condition():
                break
            
            # Evaluate ensemble's trial population
            trial_fitness = self._evaluate_batch(trial_batch, func)
            
            # ── ENSEMBLE REWARD: Evaluate each strategy separately ──
            # Apply same crossover to each strategy's mutations to get comparable fitness
            strategy_best_fitness = []
            for strat_trials in trials_list:
                strat_trial = self._crossover_batch(population, strat_trials)[0]
                strat_fit = self._evaluate_batch(strat_trial, func)
                strategy_best_fitness.append(np.min(strat_fit))
            
            # Update ensemble weights using rank-based rewards
            self._update_ensemble_weights(strategy_best_fitness)
            
            if stopping_condition():
                break
            
            # ── Standard DE selection and adaptation ──
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
```