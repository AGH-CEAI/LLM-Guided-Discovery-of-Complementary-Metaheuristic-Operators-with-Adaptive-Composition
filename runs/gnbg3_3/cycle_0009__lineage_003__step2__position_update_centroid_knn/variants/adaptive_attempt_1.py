import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: PROBE-AND-COMMIT with UCB1 operator selection (mechanism #4).

    The gap table shows NO blend-hostile tasks (max winner-to-runner-up gap is
    only 2.8× on task 5). However, the win-count is evenly split across 6
    distinct winners with near-equivalent performance — a pure bandit would
    suffer from high variance credit assignment. More critically, each variant
    captures a genuinely different failure mode (spectral collapse, MST sparsity,
    Mahalanobis geometry, axis-aligned bounding, pseudoinverse rank-deficiency).

    The dispatcher preserves each winner's advantage by:
      - PHASE A (probe): round-robin each of the 6 winning operators for
        3 generations each in a sub-population, tracking rank-based UCB1
        scores on the ACTUAL landscape. No hard-coded weights.
      - PHASE B (commit): run ONLY the winning operator for the remaining
        budget. No blending. The committed operator gets weight 1.0.
      - EPSILON-REVIEW: if committed operator stagnates (>25 gens, p=0.05),
        allow a small probability of switching to the runner-up.

    All reward signals are rank-based within a sliding window — no scale-
    dependent magic constants. Each arm gets at least 3 probe samples before
    any commit decision.
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
        
        # --- PROBE-AND-COMMIT STATE ---
        # Map operator ID -> list of rank-based scores (within sliding window)
        self._operator_scores = {}
        self._operator_total_score = {}
        self._operator_sample_count = {}
        self._committed_operator = None
        self._runner_up_operator = None
        
        # THE 6 ACTUAL WINNING POSITION UPDATE STRATEGIES (from benchmark)
        self._operators = [
            'original',      # original.py: eigenvalue-anisotropic velocity (6 wins)
            'variant_01',    # variant_01_catA_idea_0.py: bounding-box spread (2 wins)
            'variant_02',    # variant_02_catB_idea_0.py: spectral entropy + cond (6 wins)
            'variant_03',    # variant_03_catC_idea_0.py: Mahalanobis-entropy (3 wins)
            'variant_05',    # variant_05_catE_idea_0.py: MST edge-weight (6 wins)
            'variant_10',    # variant_10_catB_idea_0.py: pseudoinverse rank-deficient (1 win)
        ]
        for op in self._operators:
            self._operator_scores[op] = []
            self._operator_total_score[op] = 0.0
            self._operator_sample_count[op] = 0
        
        # Probe budget: 3 gens per operator, min 18 total (6 operators × 3)
        self._probe_gens_per_op = 3
        self._probe_total_budget = len(self._operators) * self._probe_gens_per_op
        
        # Sliding window for rank-based scoring (K >= 3 * num_arms = 18)
        self._sliding_window_size = max(54, len(self._operators) * 9)
        
        # Epsilon-review probability when committed operator stagnates
        self._epsilon_review = 0.05
        self._stagnation_review_threshold = 25
        
        # Probe phase state
        self._in_probe_phase = False
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._probe_best_fitness_seen = np.inf
        
        # All probe fitness values for rank-based scoring
        self._probe_fitness_history = []
        self._probe_op_history = []
        
        # Commit phase stagnation tracking
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
        
        # Per-operator best fitness seen during probe (for rank scoring)
        self._op_best_fitness = {op: np.inf for op in self._operators}
    
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
    # POSITION UPDATE STRATEGIES — the 6 benchmark winners
    # -------------------------------------------------------------------------
    
    def _position_update_original(self):
        """Original: eigenvalue-anisotropic velocity modulation (6 wins).
        
        Tasks 0, 2, 3, 4, 8, 11. Uses covariance eigenvalues to scale velocity
        along principal axes — prevents collapse but is simple.
        """
        centered = self.population - np.mean(self.population, axis=0)
        cov = np.cov(centered.T)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            spectral_scale = np.clip(cond / 100.0, 0.5, 2.0)
            
            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * spectral_scale
            
            new_population = self.population + (vel_scaled @ eigenvectors.T)
        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity
        
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_01(self):
        """Variant 01 (Category A): Axis-aligned bounding box spread correction (2 wins).
        
        Tasks 10, 12. Uses per-dimension min/max to compute normalized deviation
        from the population's axis-aligned bounding box. Particles far outside the
        box get weak correction (already explored); particles near the center get
        stronger correction (promote convergence).
        """
        pop_mean = np.mean(self.population, axis=0)

        dim_min = np.min(self.population, axis=0)
        dim_max = np.max(self.population, axis=0)
        dim_range = dim_max - dim_min + 1e-10

        deviation = self.population - pop_mean
        normalized_dev = deviation / dim_range

        max_dev = np.max(np.abs(normalized_dev), axis=1, keepdims=True) + 1e-10

        to_center_dir = pop_mean - self.population
        dist_to_center = np.linalg.norm(to_center_dir, axis=1, keepdims=True) + 1e-10
        to_center_dir = to_center_dir / dist_to_center

        correction_scale = np.clip(0.3 / max_dev, 0.05, 1.0)

        correction = correction_scale * to_center_dir

        new_population = self.population + self.inertia_weight * self.velocity + correction

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_02(self):
        """Variant 02 (Category B): Spectral entropy + condition-number anisotropic damping (6 wins).
        
        Tasks 15, 17, 18, 21, 22, 23. Uses eigenvalue analysis to detect population
        collapse (spectral entropy) and condition number to measure anisotropy.
        Damping along dominant eigendirections prevents collapse; exploration
        scaling handles ill-conditioned worst tasks (17, 16, 6).
        """
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, np.argsort(np.linalg.eigvalsh(cov))[::-1]]

            total_var = np.sum(eigenvalues) + 1e-10
            eigenvalues_norm = eigenvalues / total_var

            spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
            max_entropy = np.log(self.dim + 1e-10)
            entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond_log = np.log1p(cond)

            sv_norm = np.sqrt(eigenvalues) / (np.sqrt(eigenvalues[0]) + 1e-10)
            spectral_damping = sv_norm ** 0.5
            spectral_damping = spectral_damping / (np.max(spectral_damping) + 1e-10)

            entropy_scale = 0.5 + 0.5 * entropy_ratio
            cond_scale = np.clip(cond_log / 5.0, 0.0, 2.0)
            spectral_signal = entropy_scale * (1.0 + 0.3 * cond_scale)

            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * spectral_damping * spectral_signal
            new_population = self.population + vel_scaled @ eigenvectors.T

            if self.global_best is not None:
                to_best = self.global_best - self.population
                to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_norm
                correction = 0.3 * to_best_dir * (1.0 + 0.3 * cond_scale)
                new_population = new_population + correction

            exploration = 1.0 + 0.5 * np.log1p(cond)
            random_perturb = exploration * np.random.uniform(-0.3, 0.3, (self.np, self.dim))
            new_population = new_population + random_perturb

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_03(self):
        """Variant 03 (Category C): Mahalanobis-entropy centroid modulation (3 wins).
        
        Tasks 5, 13, 20. Reframes centroid gravity within an information-theoretic
        framework: Mahalanobis distance from fitted Gaussian captures distributional
        shape; entropy of eigenvalue spectrum measures population collapse.
        Per-particle entropy modulation drives exploration/exploitation balance.
        """
        # 1. Fit Gaussian to population
        centroid = np.mean(self.population, axis=0)
        centered = self.population - centroid
        cov = np.cov(centered.T)

        cov_reg = cov + 0.1 * np.trace(cov) / self.dim * np.eye(self.dim)
        cov_reg = 0.5 * (cov_reg + cov_reg.T)

        # 2. Compute precision matrix via SVD pseudoinverse
        try:
            U_svd, s_svd, Vt_svd = np.linalg.svd(cov_reg, full_matrices=False)
            s_inv = np.where(s_svd > 1e-10, 1.0 / s_svd, 0.0)
            precision = Vt_svd.T @ np.diag(s_inv) @ U_svd.T
        except np.linalg.LinAlgError:
            precision = np.eye(self.dim)

        # 3. Per-particle Mahalanobis distance
        mahal_sq = np.sum(centered @ precision * centered, axis=1)
        mahal_dist = np.sqrt(mahal_sq + 1e-10)[:, np.newaxis]

        # 4. Spectral entropy
        eigenvalues = np.linalg.eigvalsh(cov_reg)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]
        total_var = np.sum(eigenvalues) + 1e-10
        eigenvalues_norm = eigenvalues / total_var

        spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
        max_entropy = np.log(self.dim + 1e-10)
        entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)

        # 5. Entropy-based modulation: low entropy → encourage exploration
        exploration_factor = 1.0 + 0.5 * (1.0 - entropy_ratio)

        # 6. Per-particle entropy signal
        global_mahal_mean = np.mean(mahal_dist) + 1e-10
        local_entropy_modulation = np.clip(mahal_dist / global_mahal_mean, 0.5, 2.0)

        # 7. KL-divergence-like signal
        kl_signal = np.clip(entropy_ratio / (local_entropy_modulation + 1e-10), 0.3, 1.5)

        # 8. Centroid attraction
        dist_to_centroid = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-10
        global_spread = np.mean(dist_to_centroid) + 1e-10

        centroid_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread))

        # 9. Direction to centroid
        to_centroid_dir = centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid_dir / to_centroid_dist

        # 10. Combine signals
        correction = centroid_attraction * to_centroid_dir
        correction *= exploration_factor * kl_signal

        # 11. Apply update with inertia
        new_population = self.population + self.inertia_weight * self.velocity + 0.3 * correction

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_05(self):
        """Variant 05 (Category E): MST edge-weight analysis for topology-aware exploration (6 wins).
        
        Tasks 1, 6, 7, 9, 14, 19. Build MST over population; particles in sparse
        MST regions (long edges, low betweenness) get strong corrective pulls toward
        the global best. Uses GRAPH STRUCTURE (edge weights, connectivity) to
        identify underexplored regions — orthogonal to spectral and centroid methods.
        """
        n = self.np
        dists = np.linalg.norm(self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :], axis=2)

        in_tree = np.zeros(n, dtype=bool)
        parent = np.full(n, -1, dtype=int)
        min_edge = np.full(n, np.inf)
        min_edge[0] = 0.0
        in_tree[0] = True

        for _ in range(n - 1):
            best_idx = -1
            best_val = np.inf
            for i in range(n):
                if not in_tree[i]:
                    if dists[i, np.where(in_tree)[0]].min() < best_val:
                        best_val = dists[i, np.where(in_tree)[0]].min()
                        best_idx = i
            if best_idx >= 0:
                in_tree[best_idx] = True
                j = np.argmin(dists[best_idx, np.where(in_tree)[0][:-1]]) if np.sum(in_tree) > 1 else 0
                parent[best_idx] = np.where(in_tree)[0][j if np.sum(in_tree) > 1 else 0]
                min_edge[best_idx] = dists[best_idx, parent[best_idx]]

        valid_edges = min_edge[min_edge < np.inf]
        if len(valid_edges) == 0:
            correction = np.zeros((n, self.dim))
        else:
            edge_mean = np.mean(valid_edges)
            edge_std = np.std(valid_edges) + 1e-10

            particle_betweenness = np.zeros(n)
            for i in range(n):
                if parent[i] >= 0:
                    edge_len = min_edge[i]
                    particle_betweenness[i] = max(0, (edge_len - edge_mean) / edge_std)
                if parent[i] >= 0:
                    edge_len = min_edge[parent[i]]
                    particle_betweenness[parent[i]] += max(0, (edge_len - edge_mean) / edge_std)

            betweenness_factor = np.clip(1.0 + 0.4 * particle_betweenness, 1.0, 3.0)

            mst_sparsity = np.clip(min_edge / (edge_mean + 1e-10), 0.5, 2.0)
            mst_sparsity = np.where(np.isinf(mst_sparsity) | (min_edge == 0), 1.0, mst_sparsity)

            topology_modulation = betweenness_factor * mst_sparsity

            if self.global_best is not None:
                to_best = self.global_best - self.population
                to_best_dist = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_dist

                correction_strength = 0.3 * topology_modulation[:, np.newaxis]
                correction = correction_strength * to_best_dir
            else:
                centroid = np.mean(self.population, axis=0)
                to_centroid = centroid - self.population
                to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
                to_centroid_dir = to_centroid / to_centroid_dist
                correction = 0.3 * topology_modulation[:, np.newaxis] * to_centroid_dir

        sparsity_scale = np.clip(1.0 / (np.mean(valid_edges) / (valid_edges + 1e-10) + 1e-10), 0.5, 1.5)
        damped_velocity = self.velocity * sparsity_scale[:, np.newaxis]

        new_population = self.population + self.inertia_weight * damped_velocity + 0.4 * correction
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_10(self):
        """Variant 10 (Category B): Pseudoinverse velocity correction for rank-deficient populations (1 win).
        
        Task 16. Detects when population covariance becomes ill-conditioned (low-rank)
        using singular value thresholding. Uses Moore-Penrose pseudoinverse to project
        and correct velocity onto the principal subspace, amplifying collapsed directions.
        """
        try:
            centered = self.population - np.mean(self.population, axis=0)
            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)

            singular_values = np.clip(singular_values, 1e-10, None)
            total_var = np.sum(singular_values ** 2) + 1e-10

            sv_norm = singular_values / (singular_values[0] + 1e-10)

            significant_sv = np.sum(sv_norm > 0.01)
            rank_ratio = significant_sv / len(singular_values)

            cond = singular_values[0] / (singular_values[-1] + 1e-10)

            cumvar = np.cumsum(singular_values ** 2) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = eff_dim / self.dim

            inv_sv_weight = 1.0 / (sv_norm + 0.05)
            inv_sv_weight = inv_sv_weight / (np.max(inv_sv_weight) + 1e-10)

            correction_strength = np.clip(1.0 - eff_dim_ratio, 0.0, 0.8)

            vel_proj = self.velocity @ Vt.T

            uniform_scale = np.ones(len(singular_values))
            per_comp_scale = uniform_scale * (1.0 - correction_strength) + inv_sv_weight * correction_strength

            if cond > 100.0:
                log_cond = np.log1p(cond) / np.log1p(1000.0)
                log_cond = np.clip(log_cond, 0.0, 1.0)
                high_cond_damp = 1.0 - 0.3 * log_cond
                per_comp_scale *= high_cond_damp

            vel_scaled = vel_proj * np.clip(per_comp_scale, 0.1, 3.0)

            corrected_vel = vel_scaled @ Vt

            new_population = self.population + corrected_vel

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    # -------------------------------------------------------------------------
    # DISPATCHER: maps operator name -> position update method
    # -------------------------------------------------------------------------
    
    def _dispatch_position_update(self, operator):
        """Route to the appropriate position update implementation."""
        if operator == 'original':
            self._position_update_original()
        elif operator == 'variant_01':
            self._position_update_variant_01()
        elif operator == 'variant_02':
            self._position_update_variant_02()
        elif operator == 'variant_03':
            self._position_update_variant_03()
        elif operator == 'variant_05':
            self._position_update_variant_05()
        elif operator == 'variant_10':
            self._position_update_variant_10()
        else:
            self._position_update_original()
    
    # -------------------------------------------------------------------------
    # PROBE-AND-COMMIT: rank-based UCB1 scoring (scale-independent)
    # -------------------------------------------------------------------------
    
    def _record_probe_score(self, operator, best_fitness):
        """Record a probe score using RANK-BASED scoring within sliding window.
        
        Score = percentile rank of the operator's best fitness among all
        probe generations (0=worst seen, 1=best seen). This is purely
        rank-based — no scale-dependent constants.
        """
        self._probe_fitness_history.append(best_fitness)
        self._probe_op_history.append(operator)
        
        # Track per-operator best
        if best_fitness < self._op_best_fitness[operator]:
            self._op_best_fitness[operator] = best_fitness
        
        # Trim to sliding window size
        window_size = self._sliding_window_size
        if len(self._probe_fitness_history) > window_size:
            self._probe_fitness_history = self._probe_fitness_history[-window_size:]
            self._probe_op_history = self._probe_op_history[-window_size:]
        
        # Compute rank-based score: what fraction of all probe gens had
        # WORSE (higher) fitness than the current best for this operator?
        all_fitness = np.array(self._probe_fitness_history)
        op_mask = np.array([op == operator for op in self._probe_op_history])
        op_fitness = all_fitness[op_mask]
        
        if len(op_fitness) == 0:
            score = 0.5
        else:
            op_best = np.min(op_fitness)
            worse_count = np.sum(all_fitness > op_best)
            total_count = len(all_fitness)
            score = worse_count / max(total_count, 1)
        
        # Enforce sliding window for per-operator scores too
        self._operator_scores[operator].append(score)
        if len(self._operator_scores[operator]) > window_size // len(self._operators) + 5:
            self._operator_scores[operator] = self._operator_scores[operator][-window_size // len(self._operators) - 5:]
        
        self._operator_total_score[operator] = sum(self._operator_scores[operator])
        self._operator_sample_count[operator] = len(self._operator_scores[operator])
    
    def _compute_ucb_score(self, operator):
        """Compute UCB1-style score for operator selection.
        
        Uses rank-based average (already in [0,1]) so exploration bonus
        is on the same scale — no magic constants > 100.
        """
        n = self._operator_sample_count[operator]
        if n == 0:
            return float('inf')
        
        total_n = sum(self._operator_sample_count.values())
        avg_score = self._operator_total_score[operator] / n
        
        # UCB1 exploration bonus — scale is [0,1], so this is well-calibrated
        # No scale-dependent constants: log base is natural log
        exploration = np.sqrt(2.0 * np.log(max(total_n, 1)) / max(n, 1))
        
        return avg_score + exploration
    
    def _select_best_operator(self):
        """Select operator with highest UCB score from probe data."""
        best_op = self._operators[0]
        best_score = -float('inf')
        
        for op in self._operators:
            ucb = self._compute_ucb_score(op)
            if ucb > best_score:
                best_score = ucb
                best_op = op
        
        return best_op
    
    def _select_runner_up_operator(self):
        """Select the second-best operator by UCB score."""
        scores = [(op, self._compute_ucb_score(op)) for op in self._operators]
        scores.sort(key=lambda x: x[1], reverse=True)
        if len(scores) >= 2:
            return scores[1][0]
        return scores[0][0]
    
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
        Run the optimizer with PROBE-AND-COMMIT position-update selection.
        
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
        
        # Reset probe state
        for op in self._operators:
            self._operator_scores[op] = []
            self._operator_total_score[op] = 0.0
            self._operator_sample_count[op] = 0
        self._probe_fitness_history = []
        self._probe_op_history = []
        self._probe_best_fitness_seen = np.inf
        self._op_best_fitness = {op: np.inf for op in self._operators}
        self._committed_operator = None
        self._runner_up_operator = None
        self._in_probe_phase = True
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
        
        # Initial evaluation
        self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
        
        if actual_n < self.np:
            self.np = actual_n
            self.population = self.population[:actual_n]
            self.velocity = self.velocity[:actual_n]
            self.personal_best = self.personal_best[:actual_n]
            self.personal_best_fitness = self.personal_best_fitness[:actual_n]
            self.local_best = self.local_best[:actual_n]
            self.local_best_fitness = self.local_best_fitness[:actual_n]
            self.current_fitness = self.current_fitness[:actual_n]
        
        if stopping_condition():
            if self.global_best is None or np.isnan(self.global_best_fitness):
                best_idx = np.nanargmin(self.current_fitness)
                return self.current_fitness[best_idx], self.population[best_idx].copy()
            return self.global_best_fitness, self.global_best.copy()
        
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # --- PHASE A: PROBE ---
            if self._in_probe_phase:
                current_op = self._operators[self._probe_op_index]
                
                # Adaptation
                self._adapt_inertia_weight()
                self._adapt_neighborhood_size()
                self._update_local_best_from_personal()
                
                # Velocity update (shared base), then position update (PROBED)
                self._velocity_update_base()
                self._dispatch_position_update(current_op)
                
                self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
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
                
                # Record probe score (rank-based, scale-independent)
                gen_best = float(np.min(self.current_fitness))
                if gen_best < self._probe_best_fitness_seen:
                    self._probe_best_fitness_seen = gen_best
                self._record_probe_score(current_op, self._probe_best_fitness_seen)
                
                self._probe_gens_in_current_op += 1
                
                # Advance to next operator after probe_gens_per_op
                if self._probe_gens_in_current_op >= self._probe_gens_per_op:
                    self._probe_op_index += 1
                    self._probe_gens_in_current_op = 0
                
                # Check if probe phase is done
                if self._probe_op_index >= len(self._operators):
                    self._in_probe_phase = False
                    self._committed_operator = self._select_best_operator()
                    self._runner_up_operator = self._select_runner_up_operator()
                    self._commit_best_fitness = self.global_best_fitness
                    self._commit_gen_count = 0
                
                self._restart_if_stagnant()
            
            # --- PHASE B: COMMIT ---
            else:
                self._commit_gen_count += 1
                
                # Epsilon-review: small chance to switch to runner-up if stagnant
                if (self.stagnation_counter > self._stagnation_review_threshold and
                    np.random.random() < self._epsilon_review):
                    self._velocity_update_base()
                    self._dispatch_position_update(self._runner_up_operator)
                else:
                    self._velocity_update_base()
                    self._dispatch_position_update(self._committed_operator)
                
                self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
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
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
