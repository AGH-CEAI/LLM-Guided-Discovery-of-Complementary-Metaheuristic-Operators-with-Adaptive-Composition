```python
import numpy as np


class AdaptiveDirectionalDE:
    """
    MECHANISM #3 (CONDITIONAL/RULE-BASED DISPATCH) — chosen because:
    1) original.py dominates (15/24 wins), making it the clear default.
    2) Winning variants target specific, diagnosable regimes: high-dimensional
       ill-conditioned tasks (eigendecomp), boundary/deceptive tasks (geometric),
       and stagnation phases (hybrid). These are detectable via diversity and
       improvement signals at runtime.
    3) The variants are highly correlated (all DE-based), so a bandit cannot
       reliably credit-assign. Conditional dispatch avoids this pitfall.
    
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
        
        # --- DISPATCH STATE ---
        # Regime signals for conditional operator selection
        self._regime_init_done = False
        
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
    
    def _detect_regime(self, population, fitness):
        """
        Detect current search regime to dispatch the appropriate mutation strategy.
        Returns a dict of regime signals used for conditional branching.
        """
        # --- Diversity ratio ---
        current_spread = np.std(population, axis=0)
        initial_spread = (self.upper - self.lower) / 3.0
        diversity_ratio = np.clip(np.mean(current_spread / (initial_spread + 1e-10)), 0.01, 1.0)
        
        # --- Improvement rate ---
        if hasattr(self, 'last_improved_count'):
            improvement_rate = self.last_improved_count / self.np
        else:
            improvement_rate = 0.5
        
        # --- Stagnation severity ---
        stagnation_severity = min(getattr(self, 'stagnation_counter', 0) / max(1, self.stagnation_threshold), 2.0)
        
        # --- Population condition number (crude proxy for ill-conditioning) ---
        centered = population - np.mean(population, axis=0)
        cov = np.cov(centered, rowvar=False) + 1e-10 * np.eye(self.dim)
        try:
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]
            cond_num = eigvals[0] / max(eigvals[-1], 1e-10)
        except:
            cond_num = 1.0
        
        # --- Boundary proximity (fraction of individuals near bounds) ---
        eps = 0.05 * (self.upper - self.lower)
        at_lower = np.all(population < self.lower + eps, axis=1)
        at_upper = np.all(population > self.upper - eps, axis=1)
        boundary_fraction = np.mean(at_lower | at_upper)
        
        return {
            'diversity_ratio': diversity_ratio,
            'improvement_rate': improvement_rate,
            'stagnation_severity': stagnation_severity,
            'condition_number': cond_num,
            'boundary_fraction': boundary_fraction,
        }
    
    def _should_use_eigendecomp(self, regime):
        """
        Dispatch condition for eigendecomposition strategy (variant_10).
        Triggers on: ill-conditioned landscapes (high condition number) where
        the principal axes of successful mutations matter most.
        """
        # High condition number = elongated valley = eigendecomp helps
        cond_trigger = regime['condition_number'] > 10.0
        # Also trigger if diversity is collapsing (converged to valley)
        collapse_trigger = regime['diversity_ratio'] < 0.3 and regime['improvement_rate'] < 0.3
        return cond_trigger or collapse_trigger
    
    def _should_use_geometric(self, regime):
        """
        Dispatch condition for geometric strategy (variant_01).
        Triggers on: boundary-heavy or deceptive landscapes where geometric
        structure (k-NN directions, hull detection) helps exploration.
        """
        # Many points near boundary = geometric hull detection helps
        boundary_trigger = regime['boundary_fraction'] > 0.3
        # Also helps when diversity is moderate but stuck (deceptive traps)
        deceptive_trigger = (regime['diversity_ratio'] > 0.3 and 
                            regime['diversity_ratio'] < 0.7 and
                            regime['stagnation_severity'] > 0.5)
        return boundary_trigger or deceptive_trigger
    
    def _should_use_hybrid(self, regime):
        """
        Dispatch condition for hybrid exploration/exploitation (variant_08).
        Triggers on: stagnation phases where switching between exploration
        and exploitation regimes is beneficial.
        """
        # Stagnation = need regime switching
        stagnation_trigger = regime['stagnation_severity'] > 0.8
        # Or when diversity is moderate and improvement is low (stuck in medium basin)
        medium_stuck = (regime['diversity_ratio'] > 0.2 and 
                       regime['diversity_ratio'] < 0.6 and
                       regime['improvement_rate'] < 0.2)
        return stagnation_trigger or medium_stuck
    
    # ─── VARIANT 10: Eigendecomposition-driven covariance adaptation ───────────
    def _mutate_eigendecomp(self, population, fitness, ring_prev, ring_next):
        """Eigendecomposition strategy: scale differential vectors along PCA axes."""
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
        
        if len(self.successful_directions) >= dim + 1:
            D = np.array(self.successful_directions)
            D_centered = D - np.mean(D, axis=0)
            cov_mutations = np.cov(D_centered, rowvar=False)
            cov_mutations += 0.01 * np.eye(dim)
            
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(cov_mutations)
                eigenvalues = np.maximum(eigenvalues, 1e-10)
                inv_sqrt_eig = np.clip(1.0 / np.sqrt(eigenvalues), 0.1, 10.0)
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
    
    # ─── VARIANT 01: Geometric structure (k-NN + hull detection) ───────────────
    def _mutate_geometric(self, population, fitness, ring_prev, ring_next):
        """Geometric strategy: k-NN local directions + convex hull boundary detection."""
        np_pop, dim = population.shape
        
        expanded = population[:, np.newaxis, :]
        diff_matrix = expanded - population[np.newaxis, :, :]
        pairwise_dist = np.linalg.norm(diff_matrix, axis=2)
        np.fill_diagonal(pairwise_dist, np.inf)
        
        k = min(5, np_pop - 1)
        knn_indices = np.argsort(pairwise_dist, axis=1)[:, :k]
        
        neighbor_weights = 1.0 / (pairwise_dist[np.arange(np_pop)[:, None], knn_indices] + 1e-10)
        neighbor_weights = neighbor_weights / (neighbor_weights.sum(axis=1, keepdims=True) + 1e-10)
        
        local_directions = np.zeros((np_pop, dim))
        for i in range(np_pop):
            neighbors = population[knn_indices[i]]
            offsets = neighbors - population[i]
            local_directions[i] = np.sum(offsets * neighbor_weights[i, :, np.newaxis], axis=0)
        
        centroid = np.mean(population, axis=0)
        centroid_distances = np.linalg.norm(population - centroid, axis=1)
        max_centroid_dist = np.max(centroid_distances) + 1e-10
        norm_centroid_dist = 0.1 + 0.9 * (centroid_distances / max_centroid_dist)
        
        mins = np.min(population, axis=0)
        maxs = np.max(population, axis=0)
        boundary_score = np.zeros(np_pop)
        for d in range(dim):
            boundary_score += (population[:, d] - mins[d]) / (maxs[d] - mins[d] + 1e-10)
            boundary_score += (maxs[d] - population[:, d]) / (maxs[d] - mins[d] + 1e-10)
        boundary_score = boundary_score / (2 * dim)
        
        is_hull_point = np.any([
            np.allclose(population, mins, atol=1e-5),
            np.allclose(population, maxs, atol=1e-5)
        ], axis=0)
        
        r1 = ring_prev
        r2 = ring_next
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        current_F = self.F * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)
        base_mutation = rand1_vectors + current_F * (rand2_vectors - population)
        
        exploration_factor = 1.0 / (norm_centroid_dist + 0.2)
        hull_boost = np.where(is_hull_point, 1.3, 1.0)
        total_scale = np.clip(exploration_factor * hull_boost, 0.5, 2.5)
        geometric_component = centroid + total_scale[:, np.newaxis] * local_directions
        
        blend_weight = np.clip(0.15 + 0.35 * boundary_score, 0.1, 0.5)
        trials = (1 - blend_weight[:, np.newaxis]) * base_mutation + blend_weight[:, np.newaxis] * geometric_component
        
        return trials, current_F
    
    # ─── VARIANT 08: Hybrid exploration/exploitation regime switching ───────────
    def _mutate_hybrid(self, population, fitness, ring_prev, ring_next):
        """Hybrid strategy: switch between DE/rand/1 (exploration) and best+centroid (exploitation)."""
        np_pop = len(population)
        trials = np.zeros((np_pop, self.dim))
        
        if not hasattr(self, 'exploration_ema'):
            self.exploration_ema = 0.8
            self.recent_success_rate = 0.5
        
        current_spread = np.std(population, axis=0)
        initial_spread = (self.upper - self.lower) / 3.0
        diversity_ratio = np.clip(np.mean(current_spread / (initial_spread + 1e-10)), 0.01, 1.0)
        
        if hasattr(self, 'last_improved_count'):
            current_success = self.last_improved_count / np_pop
            self.recent_success_rate = 0.3 * current_success + 0.7 * self.recent_success_rate
        
        diversity_score = diversity_ratio
        success_score = self.recent_success_rate
        
        exploration_weight = 0.5 + 0.4 * (1.0 - diversity_score) + 0.2 * (0.5 - success_score)
        exploration_weight = np.clip(exploration_weight, 0.1, 0.9)
        self.exploration_ema = 0.7 * self.exploration_ema + 0.3 * exploration_weight
        final_exploration_weight = self.exploration_ema
        
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
        centroid_dist = np.linalg.norm(centroid_direction) + 1e-10
        F_exploit = np.clip(0.5 * self.F, 0.1, 1.0)
        exploit_mutation = best_vector + F_exploit * centroid_direction
        exploit_mutation = np.tile(exploit_mutation, (np_pop, 1))
        
        trials = final_exploration_weight * explore_mutation + (1.0 - final_exploration_weight) * exploit_mutation
        perturbation = 0.01 * (np.random.rand(np_pop, self.dim) - 0.5) * (self.upper - self.lower)
        trials = trials + perturbation
        
        return trials, F_explore
    
    # ─── DEFAULT: Original composite mutation ───────────────────────────────────
    def _mutate_default(self, population, fitness, ring_prev, ring_next):
        """Default strategy: composite of ring-based rand/1 + directional memory bias."""
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
    
    def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Dispatch to the appropriate mutation strategy based on detected regime.
        Falls back to default (original) unless a regime condition strongly triggers.
        """
        regime = self._detect_regime(population, fitness)
        
        if self._should_use_eigendecomp(regime):
            return self._mutate_eigendecomp(population, fitness, ring_prev, ring_next)
        elif self._should_use_geometric(regime):
            return self._mutate_geometric(population, fitness, ring_prev, ring_next)
        elif self._should_use_hybrid(regime):
            return self._mutate_hybrid(population, fitness, ring_prev, ring_next)
        else:
            return self._mutate_default(population, fitness, ring_prev, ring_next)
    
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
            # Reset hybrid EMA on restart to avoid stale regime signal
            if hasattr(self, 'exploration_ema'):
                self.exploration_ema = 0.8
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop orchestrating all batched operations."""
        population, fitness, best_fitness, best_solution = self._initialize_population(func)
        
        self.stagnation_counter = 0
        self.generation = 0
        self._regime_init_done = False
        
        ring_prev, ring_next = self._build_ring_topology()
        
        while not stopping_condition():
            self.generation += 1
            
            # Build mutation batch (dispatched based on regime)
            mutant_batch, F_used = self._mutate_batch(
                population, fitness, ring_prev, ring_next
            )
            
            # Build crossover batch
            trial_batch, Cr_used = self._crossover_batch(population, mutant_batch)
            
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
            
            # Track improvement count for regime detection
            self.last_improved_count = int(np.sum(improved_mask))
            
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
```