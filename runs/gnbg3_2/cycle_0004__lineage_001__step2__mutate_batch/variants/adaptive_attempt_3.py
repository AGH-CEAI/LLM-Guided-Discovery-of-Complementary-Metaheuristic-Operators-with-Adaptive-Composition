import numpy as np


class AdaptiveDirectionalDE:
    """
    Mechanism #2: ENSEMBLE/VOTING/BLENDING
    The winning variants compute correlated quantities (distances, centroids,
    differential vectors) but capture different failure modes. Running all three
    and blending by recent success rate avoids credit assignment problems while
    providing robustness across different problem types and optimization phases.
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
        
        # --- ENSEMBLE STATE ---
        self.num_variants = 3
        self.variant_names = ['original', 'variant_08', 'variant_10']
        # Start with prior favoring original (15/24 wins in benchmark)
        self.variant_success_ema = np.array([0.45, 0.35, 0.20])
        self.variant_temperature = 0.3  # Lower = more concentrated on best
        self.variant_weights = np.array([0.45, 0.35, 0.20])
        # Per-variant improvement tracking
        self.variant_improvement_counts = np.zeros(self.num_variants)
        # Variant-specific state for stateful variants
        self.variant_08_exploration_ema = 0.8
        self.variant_08_recent_success = 0.5
        
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
    
    def _crossover_single(self, population, mutant, Cr_override=None):
        """Crossover for single individual."""
        dim = self.dim
        Cr = Cr_override if Cr_override is not None else self.Cr
        Cr = np.clip(Cr + 0.1 * np.random.randn(), 0.1, 0.9)
        
        j_rand = np.random.randint(0, dim)
        mask = np.random.rand(dim) < Cr
        mask[j_rand] = True
        
        trial = np.where(mask, mutant, population)
        return trial, Cr
    
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
        batch = self._clip_to_bounds(batch)
        fitness = func(batch)
        
        if len(fitness) < len(batch):
            valid_len = len(fitness)
            fitness = np.resize(fitness, len(batch))
            fitness[valid_len:] = np.inf
        
        fitness = np.where(np.isnan(fitness), np.inf, fitness)
        return fitness
    
    # ========== VARIANT 1: Original (ring-based + directional bias) ==========
    def _mutate_variant_original(self, population, fitness, ring_prev, ring_next):
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
    
    # ========== VARIANT 2: variant_08 (exploration/exploitation switching) ==========
    def _mutate_variant_08(self, population, fitness, ring_prev, ring_next):
        np_pop = len(population)
        trials = np.zeros((np_pop, self.dim))
        
        current_spread = np.std(population, axis=0)
        initial_spread = (self.upper - self.lower) / 3.0
        diversity_ratio = np.mean(current_spread / (initial_spread + 1e-10))
        diversity_ratio = np.clip(diversity_ratio, 0.01, 1.0)
        
        current_success = self.variant_improvement_counts.sum() / (self.num_variants * np_pop + 1e-10)
        self.variant_08_recent_success = 0.3 * current_success + 0.7 * self.variant_08_recent_success
        
        diversity_score = diversity_ratio
        success_score = self.variant_08_recent_success
        
        exploration_weight = 0.5 + 0.4 * (1.0 - diversity_score) + 0.2 * (0.5 - success_score)
        exploration_weight = np.clip(exploration_weight, 0.1, 0.9)
        
        self.variant_08_exploration_ema = 0.7 * self.variant_08_exploration_ema + 0.3 * exploration_weight
        final_exploration_weight = self.variant_08_exploration_ema
        
        r1 = ring_prev
        r2 = ring_next
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        F_explore = np.clip(self.F * (1.0 + 0.15 * np.random.randn()), 0.3, 2.0)
        explore_mutation = rand1_vectors + F_explore * (rand2_vectors - population)
        
        best_idx = np.argmin(fitness)
        best_vector = population[best_idx]
        
        centroid = self._compute_population_centroid(population, fitness)
        centroid_direction = centroid - best_vector
        
        F_exploit = np.clip(0.5 * self.F, 0.1, 1.0)
        exploit_mutation = best_vector + F_exploit * centroid_direction
        
        exploit_mutation = np.tile(exploit_mutation, (np_pop, 1))
        
        trials = final_exploration_weight * explore_mutation + (1.0 - final_exploration_weight) * exploit_mutation
        
        perturbation = 0.01 * (np.random.rand(np_pop, self.dim) - 0.5) * (self.upper - self.lower)
        trials = trials + perturbation
        
        return trials, F_explore
    
    # ========== VARIANT 3: variant_10 (eigendecomposition-driven) ==========
    def _mutate_variant_10(self, population, fitness, ring_prev, ring_next):
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        centroid = self._compute_population_centroid(population, fitness)
        
        r1 = ring_prev
        r2 = ring_next
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        current_F = self.F * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)
        
        base_mutation = rand1_vectors + current_F * (rand2_vectors - population)
        
        # Eigendecomposition from successful direction memory
        if len(self.successful_directions) >= dim + 1:
            D = np.array(self.successful_directions)
            D_centered = D - np.mean(D, axis=0)
            cov_mutations = np.cov(D_centered, rowvar=False)
            
            cov_mutations += 0.01 * np.eye(dim)
            
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(cov_mutations)
                eigenvalues = np.maximum(eigenvalues, 1e-10)
                
                inv_sqrt_eig = 1.0 / np.sqrt(eigenvalues)
                inv_sqrt_eig = np.clip(inv_sqrt_eig, 0.1, 10.0)
                inv_sqrt_eig = inv_sqrt_eig / np.mean(inv_sqrt_eig)
                
                scaling_matrix = eigenvectors @ np.diag(inv_sqrt_eig) @ eigenvectors.T
                
                diff_vectors = rand2_vectors - population
                scaled_diff = diff_vectors @ scaling_matrix
                
                eigen_scaled = rand1_vectors + current_F * scaled_diff
                
                blend_weight = 0.3 + 0.2 * (1.0 - self.Cr)
                trials = (1 - blend_weight) * base_mutation + blend_weight * eigen_scaled
            except np.linalg.LinAlgError:
                trials = base_mutation
        else:
            trials = base_mutation
        
        return trials, current_F
    
    def _run_variant_trial(self, population, fitness, ring_prev, ring_next, variant_idx):
        """Run a single variant through mutation, crossover, and evaluation."""
        if variant_idx == 0:
            mutant, F_used = self._mutate_variant_original(population, fitness, ring_prev, ring_next)
        elif variant_idx == 1:
            mutant, F_used = self._mutate_variant_08(population, fitness, ring_prev, ring_next)
        else:
            mutant, F_used = self._mutate_variant_10(population, fitness, ring_prev, ring_next)
        
        trial_batch = np.zeros_like(mutant)
        Cr_used = np.zeros(len(population))
        for i in range(len(population)):
            trial_batch[i], Cr_used[i] = self._crossover_single(population[i], mutant[i])
        
        trial_fitness = self._evaluate_batch(trial_batch, lambda x: x)
        # Re-evaluate properly
        trial_batch_clipped = self._clip_to_bounds(trial_batch)
        trial_fitness = self._evaluate_batch(trial_batch_clipped, lambda x: x)
        
        improved_mask = trial_fitness < fitness
        return trial_batch, trial_fitness, improved_mask, F_used
    
    def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Ensemble mutation: run all variants and blend based on recent success.
        Each variant goes through full mutation+crossover+eval cycle to measure
        its actual contribution to improvements.
        """
        np_pop = len(population)
        
        # Run all three variants and track improvements
        self.variant_improvement_counts = np.zeros(self.num_variants)
        variant_trials = []
        
        for v in range(self.num_variants):
            trials, trial_fitness, improved_mask, F_used = self._run_variant_trial(
                population, fitness, ring_prev, ring_next, v
            )
            variant_trials.append(trials)
            self.variant_improvement_counts[v] = np.sum(improved_mask)
        
        # Update variant weights using rank-based EMA
        total_improvements = np.sum(self.variant_improvement_counts)
        
        if total_improvements > 0:
            # Rank-based improvement rate (avoid scale dependence)
            improvement_rates = self.variant_improvement_counts / np_pop
            
            # Update EMA with momentum
            alpha = 0.3
            self.variant_success_ema = (
                alpha * improvement_rates + (1 - alpha) * self.variant_success_ema
            )
        
        # Compute weights via softmax with temperature
        ema_scaled = self.variant_success_ema / self.variant_temperature
        ema_scaled = ema_scaled - np.max(ema_scaled)  # Numerical stability
        exp_ema = np.exp(ema_scaled)
        raw_weights = exp_ema / np.sum(exp_ema)
        
        # Enforce minimum weight (keep all variants partially active)
        min_weight = 0.05
        self.variant_weights = np.maximum(raw_weights, min_weight)
        self.variant_weights = self.variant_weights / np.sum(self.variant_weights)
        
        # Blend all variant trials
        blended_trials = np.zeros_like(population)
        for v in range(self.num_variants):
            blended_trials += self.variant_weights[v] * variant_trials[v]
        
        # Average F from variants
        avg_F = self.F
        
        return blended_trials, avg_F
    
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
            
            # Soft reset variant weights toward prior (problem may have changed)
            prior = np.array([0.45, 0.35, 0.20])
            self.variant_weights = 0.7 * self.variant_weights + 0.3 * prior
            self.variant_weights = self.variant_weights / np.sum(self.variant_weights)
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        population, fitness, best_fitness, best_solution = self._initialize_population(func)
        
        self.stagnation_counter = 0
        self.generation = 0
        
        ring_prev, ring_next = self._build_ring_topology()
        
        while not stopping_condition():
            self.generation += 1
            
            # Ensemble mutation with variant evaluation
            mutant_batch, F_used = self._mutate_batch(
                population, fitness, ring_prev, ring_next
            )
            
            # Crossover
            trial_batch, Cr_used = self._crossover_batch(population, mutant_batch)
            
            if stopping_condition():
                break
            
            # Evaluate
            trial_fitness = self._evaluate_batch(trial_batch, func)
            
            if stopping_condition():
                break
            
            # Select survivors
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trial_batch, trial_fitness
            )
            
            # Update best
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
            
            # Direction memory from actual improvements
            successful_mutations = trial_batch - population
            self._update_direction_memory(successful_mutations, improved_mask)
            
            # Adapt parameters
            self._adapt_parameters(improved_mask, F_used, Cr_used)
            
            # Restart check
            population, fitness, best_fitness, best_solution = self._restart_if_stagnant(
                population, fitness, best_fitness, best_solution
            )
            
            ring_prev, ring_next = self._build_ring_topology()
        
        return best_fitness, best_solution
