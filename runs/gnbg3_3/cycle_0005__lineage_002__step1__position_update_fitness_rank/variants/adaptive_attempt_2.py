import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: SOFT ENSEMBLE WITH ADAPTIVE WEIGHTS (mechanism #2)
    
    The gap table is BLEND-VIABLE (all gaps < 3×):
      - Maximum gap is 2.8× (Task 5: variant_05 beats variant_08)
      - All other gaps are ≤ 1.8× (Tasks 10, 1, 3, 17)
      - A 50/50 blend of 9.7 and 26.7 yields ~18.2, which is only ~1.9×
        worse than the winner — acceptable for robustness.
    
    Why ensemble over bandit/probe-and-commit:
      - With gaps < 3×, blending CAN preserve most of the per-task advantage.
      - 5 distinct winners with roughly even win counts (7/7/6/3/1) means
        no single operator dominates. An ensemble captures all of them.
      - No probing budget wasted: all 5 methods run every generation.
      - Smoothly adapts to which methods are working on the current landscape.
      - More robust than probe-and-commit (which wastes ~15 generations on
        probing and can mis-identify the best operator on noisy landscapes).
    
    Weight computation:
      - Rank-based scores within a sliding window (scale-independent).
      - Softmax normalization → weights sum to 1.0.
      - Temperature controls exploration vs exploitation of the ensemble.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5 * dim, capped at reasonable range
        self.np = min(max(5 * dim, 20), 300)
        
        # Ring topology neighborhood size (will adapt)
        self.neighborhood_size = max(3, dim // 5)
        
        # Acceleration coefficients (will adapt)
        self.cognitive_base = 1.496
        self.social_base = 1.496
        
        # Velocity bounds
        self.v_max = 0.2 * (self.upper_bound - self.lower_bound)
        self.v_min = -self.v_max
        
        # Diversity thresholds for adaptation and restart
        self.diversity_threshold_low = 1e-6
        self.diversity_threshold_high = 10.0
        
        # Internal state
        self.population = None
        self.velocity = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.local_best = None
        self.local_best_fitness = None
        self.global_best = None
        self.global_best_fitness = None
        self.inertia_weight = 0.729
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        self.current_fitness = None
        
        # --- ENSEMBLE STATE ---
        # Map method index -> list of recent rank-based scores
        self._method_scores = {i: [] for i in range(5)}
        self._method_total_score = {i: 0.0 for i in range(5)}
        self._window_size = 10  # Sliding window for rank-based scoring
        self._softmax_temp = 0.5  # Temperature for softmax weighting
        
        # Candidate populations from each method (for ensemble)
        self._candidate_populations = None
    
    def _initialize_population(self):
        """Initialize swarm with uniform random distribution."""
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.np, self.dim)
        )
        self.velocity = np.random.uniform(
            self.v_min * 0.1, self.v_max * 0.1, (self.np, self.dim)
        )
        
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(self.np, np.inf)
        
        self.local_best = self.population.copy()
        self.local_best_fitness = np.full(self.np, np.inf)
        
        self.global_best = None
        self.global_best_fitness = np.inf
    
    def _clip_to_bounds(self, population):
        """Clip all individuals to search bounds."""
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def _evaluate_batch(self, population, func):
        """Evaluate fitness for entire population in one batch call."""
        clipped = self._clip_to_bounds(population)
        fitness = func(clipped)
        
        if len(fitness) < len(population):
            actual_len = len(fitness)
            return fitness[:actual_len], actual_len
        return fitness, len(population)
    
    def _update_personal_best_batch(self):
        """Update personal best positions where current fitness is better."""
        improved = self.personal_best_fitness > self.current_fitness
        self.personal_best[improved] = self.population[improved]
        self.personal_best_fitness[improved] = self.current_fitness[improved]
    
    def _compute_local_best_batch(self):
        """Compute ring topology local bests using vectorized operations."""
        for i in range(self.np):
            left = (i - self.neighborhood_size) % self.np
            right = (i + self.neighborhood_size + 1) % self.np
            
            if left < right:
                neighborhood_indices = np.arange(left, right)
            else:
                neighborhood_indices = np.concatenate([
                    np.arange(left, self.np), np.arange(0, right)
                ])
            
            all_indices = np.concatenate([neighborhood_indices, [i]])
            fitness_in_neighborhood = self.personal_best_fitness[all_indices]
            best_idx_in_neighborhood = np.argmin(fitness_in_neighborhood)
            
            self.local_best[i] = self.personal_best[all_indices[best_idx_in_neighborhood]]
            self.local_best_fitness[i] = fitness_in_neighborhood[best_idx_in_neighborhood]
    
    def _compute_global_best(self):
        """Compute global best from personal bests."""
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best = self.personal_best[best_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            self.stagnation_counter = 0
            self.last_improvement_gen = self.generation
        else:
            self.stagnation_counter += 1
    
    def _compute_diversity(self):
        """Compute swarm diversity as average Euclidean distance to centroid."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_inertia_weight(self):
        """Adapt inertia weight using condition number of population covariance."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 10000.0)

            spectral_signal = np.log1p(cond) / np.log1p(10000.0)
            spectral_signal = np.clip(spectral_signal, 0.0, 1.0)

            target_inertia = 0.4 + 0.55 * spectral_signal
            decay = 0.729 - 0.15 * (self.generation / 1000)
            target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)

            self.inertia_weight = self.inertia_weight * 0.8 + target_inertia * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
        except np.linalg.LinAlgError:
            decay = 0.729 - 0.15 * (self.generation / 1000)
            self.inertia_weight = self.inertia_weight * 0.8 + decay * 0.2
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _adapt_neighborhood_size(self):
        """Adapt ring topology neighborhood size based on diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif diversity > self.diversity_threshold_high:
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    def _adaptive_coefficients(self):
        """Compute adaptive cognitive and social coefficients."""
        improvement_ratio = 1.0 / (1.0 + self.stagnation_counter)
        cognitive = self.cognitive_base * (1.0 + 0.5 * improvement_ratio)
        social = self.social_base * (2.0 - improvement_ratio)
        return cognitive, social
    
    # -------------------------------------------------------------------------
    # VELOCITY UPDATE (shared across all position update strategies)
    # -------------------------------------------------------------------------
    
    def _velocity_update_base(self):
        """Base velocity update: cognitive + social + DE mutation."""
        cognitive, social = self._adaptive_coefficients()
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.1 * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold
        
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False
            )
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        mutation_component = np.where(
            mutation_active,
            0.3 * (mutation_vectors - self.population),
            0.0
        )
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )
        
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    # -------------------------------------------------------------------------
    # POSITION UPDATE STRATEGIES (the 5 benchmark winners)
    # -------------------------------------------------------------------------
    
    def _position_update_spectral(self):
        """Spectral: eigenvalue-weighted velocity projection (variant_02).
        
        1 win on Task 0. Key insight: project velocity onto principal
        components of population covariance, then scale each component
        INVERSELY to its eigenvalue. High-eigenvalue directions get
        dampened (already well-explored), low-eigenvalue directions
        get amplified (collapsed/neglected subspace).
        """
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)

            cond = eigenvalues[-1] / (eigenvalues[0] + 1e-10)
            cond = np.clip(cond, 1.0, 10000.0)

            total_var = np.sum(eigenvalues)
            if total_var > 1e-10:
                sorted_desc = eigenvalues[::-1]
                cumvar = np.cumsum(sorted_desc) / total_var
                eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            else:
                eff_dim = self.dim

            spectral_signal = np.clip((cond - 10.0) / 90.0, 0.0, 1.0)
            exploration_scale = 1.0 + 0.7 * spectral_signal

            max_eig = eigenvalues[-1] + 1e-10
            inv_eig_weights = max_eig / (eigenvalues + 1e-10)
            inv_eig_weights = inv_eig_weights / (np.max(inv_eig_weights) + 1e-10)

            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * inv_eig_weights

            new_population = self.population + exploration_scale * (vel_scaled @ eigenvectors.T)

        except np.linalg.LinAlgError:
            new_population = self.population + 1.2 * self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_svd_whitening(self):
        """SVD-whitening: inverse-square spectral modulation (variant_02).
        
        1 win on Task 0. Key insight: counteracts anisotropy by dampening
        high-spread directions and amplifying collapsed directions using
        inverse-square-law spectral modulation.
        """
        centered = self.population - np.mean(self.population, axis=0)

        try:
            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)

            singular_values = np.clip(singular_values, 1e-10, None)

            cond = singular_values[0] / (singular_values[-1] + 1e-10)
            total_var = np.sum(singular_values ** 2) + 1e-10

            sv_norm = singular_values / (singular_values[0] + 1e-10)

            spectral_modulation = 1.0 / (sv_norm ** 2 + 0.01)
            spectral_modulation = spectral_modulation / (np.max(spectral_modulation) + 1e-10)

            blend_weight = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
            uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
            per_component_scale = uniform_scale * (1.0 - blend_weight) + spectral_modulation * blend_weight

            vel_proj = self.velocity @ Vt.T
            vel_scaled = vel_proj * per_component_scale
            new_population = self.population + (vel_scaled @ Vt)

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_fitness_rank(self):
        """Hybrid: FDC exploration/exploitation + temporal centroid-drift (variant_08).
        
        7 wins on Tasks 2,3,11,12,13,14,20. Key insight: FDC-based modulation
        combined with EMA centroid dynamics for convergence sensing. Principled
        switching: stagnation drives drift_weight.
        """
        # --- MECHANISM 1: FDC-based exploration/exploitation ---
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        if self.global_best is not None:
            dists = np.linalg.norm(self.population - self.global_best, axis=1)
            dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
            if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                fdc_corr = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
            else:
                fdc_corr = 0.0
        else:
            fdc_corr = 0.0

        exploration_factor = np.clip(1.0 - fdc_corr, 0.3, 2.0)

        # --- MECHANISM 2: Temporal centroid-drift ---
        centroid = np.mean(self.population, axis=0)

        if not hasattr(self, '_ema_centroid'):
            self._ema_centroid = centroid.copy()
            self._ema_centroid_velocity = np.zeros(self.dim)

        alpha_pos = 0.1
        self._ema_centroid = alpha_pos * centroid + (1 - alpha_pos) * self._ema_centroid

        current_centroid_vel = centroid - self._ema_centroid
        alpha_vel = 0.2
        self._ema_centroid_velocity = alpha_vel * current_centroid_vel + (1 - alpha_vel) * self._ema_centroid_velocity

        # --- PRINCIPLED SWITCHING: stagnation drives drift_weight ---
        stagnation_threshold = 20
        stagnation_normalized = np.clip(self.stagnation_counter / max(stagnation_threshold, 1), 0.0, 1.0)
        drift_weight = stagnation_normalized ** 0.5
        fdc_weight = 1.0 - drift_weight

        # --- FDC-based velocity scaling ---
        if not hasattr(self, '_success_count'):
            self._success_count = np.zeros(self.np)
        improved = self.personal_best_fitness >= self.current_fitness
        self._success_count[improved] += 1
        self._success_count[~improved] *= 0.9
        success_norm = self._success_count / (np.max(self._success_count) + 1.0)

        vel_scale = exploration_factor * (1.0 + 0.3 * success_norm)
        vel_scale = np.clip(vel_scale, 0.5, 2.0)

        # --- TEMPORAL-DRIFT correction ---
        to_ema_dir = self._ema_centroid - self.population
        to_ema_dist = np.linalg.norm(to_ema_dir, axis=1, keepdims=True) + 1e-10
        drift_correction = (to_ema_dir / to_ema_dist) * np.linalg.norm(self._ema_centroid_velocity)
        drift_correction_scaled = drift_correction * drift_weight

        # FDC-based random perturbation
        fitness_perturb = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
        exploration_perturb = exploration_factor * fitness_perturb
        random_perturb = exploration_perturb * np.random.uniform(-0.5, 0.5, (self.np, self.dim))
        fdc_perturb = random_perturb * fdc_weight

        # Combined position update
        new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + drift_correction_scaled + fdc_perturb

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_graph_laplacian(self):
        """Graph-Laplacian eigenvalue-based velocity modulation (variant_05).
        
        6 wins on Tasks 4,5,6,8,17,22. Key insight: k-NN graph over population;
        Laplacian eigenvalues detect clustering. Small spectral gap = clustered
        → more exploration. Topology-aware approach orthogonal to spectral-matrix
        and fitness-landscape approaches.
        """
        # Build k-NN graph over population
        k = min(5, self.np - 1)
        sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

        # Connected component analysis using graph traversal
        visited = np.zeros(self.np, dtype=bool)
        components = []
        for start in range(self.np):
            if visited[start]:
                continue
            component = []
            stack = [start]
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                component.append(node)
                for neighbor in knn_indices[node]:
                    if not visited[neighbor]:
                        stack.append(neighbor)
            components.append(component)

        component_sizes = np.array([len(c) for c in components])
        particle_component_size = np.array([component_sizes[np.argmax([i in c for c in components])] for i in range(self.np)])

        # Graph Laplacian spectral analysis
        try:
            row_indices = np.repeat(np.arange(self.np), k)
            col_indices = knn_indices.ravel()
            data = np.ones(len(row_indices))
            adj = np.zeros((self.np, self.np))
            adj[row_indices, col_indices] = 1.0
            adj = adj + adj.T
            adj = np.clip(adj, 0, 1)

            degrees = np.sum(adj, axis=1) + 1e-10
            d_inv_sqrt = 1.0 / np.sqrt(degrees)
            d_inv_sqrt_diag = np.diag(d_inv_sqrt)
            lap = np.eye(self.np) - d_inv_sqrt_diag @ adj @ d_inv_sqrt_diag

            eigenvalues = np.linalg.eigvalsh(lap)
            eigenvalues = np.sort(eigenvalues)

            spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 1.0
            spectral_gap = np.clip(spectral_gap, 0.0, 1.0)

            modularity_signal = 1.0 - eigenvalues[0] if len(eigenvalues) > 0 else 0.0
        except:
            spectral_gap = 1.0
            modularity_signal = 0.0

        # Fitness rank signal
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        # Graph-based exploration factor
        clustering_explore = 1.0 + 0.5 * (1.0 - spectral_gap)
        size_factor = np.clip(particle_component_size / max(1, self.np), 0.3, 1.0)
        graph_explore = clustering_explore * (2.0 - size_factor)
        graph_explore = np.clip(graph_explore, 0.5, 2.0)

        # Success history
        if not hasattr(self, '_success_count'):
            self._success_count = np.zeros(self.np)
        improved = self.personal_best_fitness >= self.current_fitness
        self._success_count[improved] += 1
        self._success_count[~improved] *= 0.9
        success_norm = self._success_count / (np.max(self._success_count) + 1.0)

        # Velocity scaling
        vel_scale = graph_explore * (1.0 + 0.3 * success_norm) * (1.0 + 0.3 * (1.0 - fitness_ranks))
        vel_scale = np.clip(vel_scale, 0.5, 2.5)

        # Fitness-based directional perturbation
        fitness_perturb = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            to_best_dir = to_best / to_best_norm
            directional = fitness_perturb * to_best_dir
        else:
            directional = fitness_perturb * np.random.uniform(-1, 1, (self.np, self.dim))

        random_perturb = graph_explore[:, np.newaxis] * np.random.uniform(-0.5, 0.5, (self.np, self.dim))

        new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + directional + random_perturb
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_eigenvalue_entropy(self):
        """Spectral: eigenvalue entropy + fitness-rank velocity modulation (variant_10).
        
        7 wins on Tasks 9,10,15,16,18,19,23. Key insight: eigenvalue spread
        (normalized variance of eigenspectrum) captures full distribution shape.
        Unlike condition-number approaches, this captures the full distribution.
        """
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)

            eigenvalues_sorted = np.sort(eigenvalues)[::-1]
            total_var = np.sum(eigenvalues_sorted)

            if total_var > 1e-10:
                eigenvalues_normalized = eigenvalues_sorted / total_var
                eigenvalue_entropy = -np.sum(eigenvalues_normalized * np.log(eigenvalues_normalized + 1e-10))
                max_entropy = np.log(self.dim + 1e-10)
                eigenvalue_spread = eigenvalue_entropy / (max_entropy + 1e-10)
            else:
                eigenvalue_spread = 1.0

            spectral_scale = 0.5 + 1.5 * eigenvalue_spread
            spectral_scale = np.clip(spectral_scale, 0.3, 2.5)
        except np.linalg.LinAlgError:
            spectral_scale = 1.0

        fitness_scale = 1.0 - 0.4 * fitness_ranks
        vel_scale = spectral_scale * fitness_scale
        vel_scale = np.clip(vel_scale, 0.1, 3.0)

        new_population = self.population + self.velocity * vel_scale[:, np.newaxis]
        self.population = self._clip_to_bounds(new_population)
    
    # -------------------------------------------------------------------------
    # ENSEMBLE: run all methods, weight by rank-based performance
    # -------------------------------------------------------------------------
    
    def _run_method_and_evaluate(self, method_idx, func):
        """Run a position update method and return (candidate_pop, best_fitness).
        
        Saves and restores state so each method is evaluated on the same
        starting point. Rank-based scoring is scale-independent.
        """
        # Save current state
        saved_pop = self.population.copy()
        saved_vel = self.velocity.copy()
        saved_pbest = self.personal_best.copy()
        saved_pbest_fit = self.personal_best_fitness.copy()
        saved_lbest = self.local_best.copy()
        saved_lbest_fit = self.local_best_fitness.copy()
        saved_curr_fit = self.current_fitness.copy()
        
        try:
            # Run the method
            if method_idx == 0:
                self._position_update_spectral()
            elif method_idx == 1:
                self._position_update_svd_whitening()
            elif method_idx == 2:
                self._position_update_fitness_rank()
            elif method_idx == 3:
                self._position_update_graph_laplacian()
            else:
                self._position_update_eigenvalue_entropy()
            
            # Evaluate the candidate population
            cand_pop = self.population.copy()
            cand_fit, _ = self._evaluate_batch(cand_pop, func)
            
            # Replace NaN/Inf with large values
            cand_fit = np.where(np.isfinite(cand_fit), cand_fit, 1e20)
            
            best_fitness = float(np.min(cand_fit))
            
        finally:
            # Restore state
            self.population = saved_pop
            self.velocity = saved_vel
            self.personal_best = saved_pbest
            self.personal_best_fitness = saved_pbest_fit
            self.local_best = saved_lbest
            self.local_best_fitness = saved_lbest_fit
            self.current_fitness = saved_curr_fit
        
        return cand_pop, best_fitness
    
    def _compute_ensemble_weights(self):
        """Compute softmax-normalized weights from rank-based scores.
        
        Uses a sliding window of recent scores. Temperature controls
        exploration vs exploitation of the ensemble. Rank-based scoring
        ensures scale-independence (no magic constants).
        """
        scores = []
        for i in range(5):
            if len(self._method_scores[i]) == 0:
                scores.append(0.0)
            else:
                # Use mean of recent scores (within window)
                recent = self._method_scores[i][-self._window_size:]
                scores.append(np.mean(recent))
        
        scores = np.array(scores)
        
        # Replace NaN with 0
        scores = np.where(np.isfinite(scores), scores, 0.0)
        
        # Softmax with temperature
        scores_shifted = scores - np.max(scores)  # Numerical stability
        exp_scores = np.exp(scores_shifted / self._softmax_temp)
        weights = exp_scores / (np.sum(exp_scores) + 1e-10)
        
        return weights
    
    def _update_method_scores(self, candidate_fitnesses):
        """Update rank-based scores for each method.
        
        Score = rank of method's best_fitness among all 5 methods (0-1).
        Lower fitness = better = higher score. This is scale-independent.
        """
        # Rank the methods (0 = worst, 4 = best)
        fitness_array = np.array(candidate_fitnesses)
        ranks = np.argsort(np.argsort(fitness_array)) / 4.0  # Normalize to [0, 1]
        
        for i in range(5):
            self._method_scores[i].append(ranks[i])
            self._method_total_score[i] += ranks[i]
            
            # Keep only window size
            if len(self._method_scores[i]) > self._window_size:
                self._method_scores[i].pop(0)
    
    def _ensemble_position_update(self, func):
        """Run all 5 position update methods and combine with adaptive weights.
        
        Each method is evaluated on the current population state, producing
        a candidate population. The ensemble combines them with softmax-
        normalized weights based on rank-based performance history.
        """
        candidate_fitnesses = []
        candidate_populations = []
        
        for i in range(5):
            cand_pop, best_fit = self._run_method_and_evaluate(i, func)
            candidate_populations.append(cand_pop)
            candidate_fitnesses.append(best_fit)
        
        # Update rank-based scores
        self._update_method_scores(candidate_fitnesses)
        
        # Compute adaptive weights
        weights = self._compute_ensemble_weights()
        
        # Weighted average of candidate populations
        new_population = np.zeros_like(self.population)
        for i in range(5):
            new_population += weights[i] * candidate_populations[i]
        
        self.population = new_population
    
    # -------------------------------------------------------------------------
    # RESTART & LOCAL BEST
    # -------------------------------------------------------------------------
    
    def _restart_if_stagnant(self):
        """Reinitialize part of the population if stagnation detected."""
        stagnation_threshold = 50 + self.dim // 2
        
        if self.stagnation_counter > stagnation_threshold:
            n_replace = max(1, int(0.3 * self.np))
            worst_indices = np.argsort(self.personal_best_fitness)[-n_replace:]
            
            self.population[worst_indices] = np.random.uniform(
                self.lower_bound, self.upper_bound, (n_replace, self.dim)
            )
            self.velocity[worst_indices] = np.random.uniform(
                self.v_min * 0.1, self.v_max * 0.1, (n_replace, self.dim)
            )
            
            self.personal_best_fitness[worst_indices] = np.inf
            self.personal_best[worst_indices] = self.population[worst_indices]
            
            self.stagnation_counter = 0
            
            if self.global_best is not None:
                perturbation = np.random.uniform(-5, 5, self.dim)
                self.population[0] = self._clip_to_bounds(self.global_best + perturbation)
                self.personal_best[0] = self.population[0]
                self.personal_best_fitness[0] = np.inf
    
    def _update_local_best_from_personal(self):
        """Update local best when personal best improves."""
        improved = self.personal_best_fitness < self.local_best_fitness
        self.local_best[improved] = self.personal_best[improved]
        self.local_best_fitness[improved] = self.personal_best_fitness[improved]
    
    # -------------------------------------------------------------------------
    # MAIN OPTIMIZATION LOOP
    # -------------------------------------------------------------------------
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer with SOFT ENSEMBLE position-update selection.
        
        Args:
            func: Objective function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        # Initialize
        self._initialize_population()
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        
        # Reset ensemble state
        for i in range(5):
            self._method_scores[i] = []
            self._method_total_score[i] = 0.0
        
        # Reset EMA states
        if hasattr(self, '_ema_centroid'):
            del self._ema_centroid
        if hasattr(self, '_ema_centroid_velocity'):
            del self._ema_centroid_velocity
        if hasattr(self, '_success_count'):
            self._success_count = np.zeros(self.np)
        
        # Initial evaluation
        self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
        
        # Replace NaN/Inf with large values
        self.current_fitness = np.where(np.isfinite(self.current_fitness), self.current_fitness, 1e20)
        
        if actual_n < self.np:
            self.np = actual_n
            self.population = self.population[:actual_n]
            self.velocity = self.velocity[:actual_n]
            self.personal_best = self.personal_best[:actual_n]
            self.personal_best_fitness = self.personal_best_fitness[:actual_n]
            self.local_best = self.local_best[:actual_n]
            self.local_best_fitness = self.local_best_fitness[:actual_n]
            self.current_fitness = self.current_fitness[:actual_n]
            if hasattr(self, '_success_count'):
                self._success_count = np.zeros(self.np)
        
        if stopping_condition():
            if self.global_best is None or np.isnan(self.global_best_fitness) or np.isinf(self.global_best_fitness):
                best_idx = np.nanargmin(self.current_fitness)
                return self.current_fitness[best_idx], self.population[best_idx].copy()
            return self.global_best_fitness, self.global_best.copy()
        
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # Adaptation
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            self._update_local_best_from_personal()
            
            # Velocity update (shared base)
            self._velocity_update_base()
            
            # Ensemble position update: run all 5 methods, weight by rank-based performance
            self._ensemble_position_update(func)
            
            self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
            
            # Replace NaN/Inf with large values
            self.current_fitness = np.where(np.isfinite(self.current_fitness), self.current_fitness, 1e20)
            
            if actual_n < self.np:
                self.np = actual_n
                self.population = self.population[:actual_n]
                self.velocity = self.velocity[:actual_n]
                self.current_fitness = self.current_fitness[:actual_n]
                break
            
            if stopping_condition():
                break
            
            self._update_personal_best_batch()
            self._compute_local_best_batch()
            self._compute_global_best()
            self._restart_if_stagnant()
        
        # Final result with NaN/Inf safety
        if self.global_best is None or np.isnan(self.global_best_fitness) or np.isinf(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
