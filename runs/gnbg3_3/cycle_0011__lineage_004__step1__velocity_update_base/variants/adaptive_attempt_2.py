import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: PROBE-AND-COMMIT with per-operator velocity-update selection.
    
    The gap table is BLEND-HOSTILE:
      - Task 5: original (2.63e+01) vs variant_03 (5.56e+00) = 4.7× runner-up gap,
        4764× median gap. A 50/50 blend of 25 and 5.6 yields ~15, which is ~2.7×
        WORSE than the winner. Linear blending mathematically cannot preserve
        per-task advantages when gaps exceed ~3–5×.
      - 7 distinct winners across 24 tasks confirm that no single operator is
        universally best. Ensemble blending is therefore excluded.
    
    The dispatcher preserves each winner's advantage by:
      - PHASE A (probe): round-robin across all 7 winning velocity-update
        strategies, each for 3 generations using the full population. Rank-
        based scoring (fraction of probe generations with improvement) ensures
        scale-independence — critical since task errors span 1e-1 to 1e+6.
      - PHASE B (commit): run ONLY the winning operator for the remaining
        budget. No blending. No hard-coded weights.
      - EPSILON-REVIEW: if committed operator stagnates for >20 generations,
        allow a 5% chance to switch to the runner-up.
    
    Why 7 arms when the top 2 account for 15/24 wins?
    The remaining 5 arms collectively win 9 tasks — each with meaningful gaps.
    Skipping them would sacrifice accuracy on those 9 tasks. The probe budget
    (21 generations ≈ 3% of typical 5000-gen budget) is a small price for
    correct routing on ~38% of tasks.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        self.np = min(max(5 * dim, 20), 300)
        self.neighborhood_size = max(3, dim // 5)
        self.cognitive_base = 1.496
        self.social_base = 1.496
        
        self.v_max = 0.2 * (self.upper_bound - self.lower_bound)
        self.v_min = -self.v_max
        
        self.diversity_threshold_low = 1e-6
        self.diversity_threshold_high = 10.0
        
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
        self._operator_scores = {}
        self._operator_total_score = {}
        self._operator_sample_count = {}
        self._committed_operator = None
        self._runner_up_operator = None
        
        # The 7 winning velocity-update strategies from the benchmark
        self._operators = [
            'original',      # 4 wins: base PSO + DE mutation
            'catA',          # 1 win: axis-aligned spread + centroid geometry
            'catB',          # 1 win: spectral velocity modulation (eigenvalue)
            'catC',          # 2 wins: information-theoretic entropy modulation
            'catE',          # 6 wins: k-NN graph centrality + topology correction
            'catF',          # 9 wins: temporal dynamics tracking (centroid drift,
                             #           spread change rate, velocity autocorrelation)
            'catA_knn',      # 1 win: k-NN density + axis-aligned spread
        ]
        for op in self._operators:
            self._operator_scores[op] = []
            self._operator_total_score[op] = 0.0
            self._operator_sample_count[op] = 0
        
        # Probe budget: 3 gens per operator
        self._probe_gens_per_op = 3
        
        # Epsilon-review probability
        self._epsilon_review = 0.05
        
        # Probe phase state
        self._in_probe_phase = False
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._probe_best_fitness_seen = np.inf
        
        # Running history for rank-based scoring
        self._probe_fitness_history = []
        self._probe_op_history = []
        
        # Commit phase stagnation tracking
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
    
    def _initialize_population(self):
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
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def _evaluate_batch(self, population, func):
        clipped = self._clip_to_bounds(population)
        fitness = func(clipped)
        if len(fitness) < len(population):
            actual_len = len(fitness)
            return fitness[:actual_len], actual_len
        return fitness, len(population)
    
    def _update_personal_best_batch(self):
        improved = self.personal_best_fitness > self.current_fitness
        self.personal_best[improved] = self.population[improved]
        self.personal_best_fitness[improved] = self.current_fitness[improved]
    
    def _compute_local_best_batch(self):
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
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best = self.personal_best[best_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[best_idx]
            self.stagnation_counter = 0
            self.last_improvement_gen = self.generation
        else:
            self.stagnation_counter += 1
    
    def _compute_diversity(self):
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_inertia_weight(self):
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
        diversity = self._compute_diversity()
        if diversity < self.diversity_threshold_low:
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif diversity > self.diversity_threshold_high:
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    def _adaptive_coefficients(self):
        improvement_ratio = 1.0 / (1.0 + self.stagnation_counter)
        cognitive = self.cognitive_base * (1.0 + 0.5 * improvement_ratio)
        social = self.social_base * (2.0 - improvement_ratio)
        return cognitive, social
    
    def _restart_if_stagnant(self):
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
        improved = self.personal_best_fitness < self.local_best_fitness
        self.local_best[improved] = self.personal_best[improved]
        self.local_best_fitness[improved] = self.personal_best_fitness[improved]
    
    # -------------------------------------------------------------------------
    # VELOCITY UPDATE — 7 BENCHMARK-WINNING STRATEGIES
    # Each replaces or wraps _velocity_update_base() and is dispatched by name.
    # -------------------------------------------------------------------------
    
    def _velocity_update_original(self):
        """Base velocity update: cognitive + social + DE mutation (original.py, 4 wins)."""
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
        mutation_component = np.where(mutation_active, 0.3 * (mutation_vectors - self.population), 0.0)
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component + social_component + mutation_component
        )
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_catA(self):
        """Geometry-aware velocity update: axis-aligned spread + centroid geometry (1 win, task 14)."""
        cognitive, social = self._adaptive_coefficients()
        centroid = np.mean(self.population, axis=0)
        dim_min = np.min(self.population, axis=0)
        dim_max = np.max(self.population, axis=0)
        dim_range = dim_max - dim_min + 1e-10
        max_range = np.max(dim_range)
        normalized_spread = np.clip(dim_range / (max_range + 1e-10), 0.01, 1.0)
        spread_modulation = 1.0 / normalized_spread
        spread_modulation = np.clip(spread_modulation, 0.3, 3.0)
        spread_modulation = spread_modulation / (np.max(spread_modulation) + 1e-10)
        
        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True) + 1e-10
        global_spread = np.mean(dist_to_centroid) + 1e-10
        centroid_attraction_strength = np.clip(dist_to_centroid / (2.0 * global_spread), 0.0, 1.5)
        to_centroid = centroid - self.population
        to_centroid_norm = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid / to_centroid_norm
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        centroid_correction = 0.2 * centroid_attraction_strength * to_centroid_dir
        
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
            mutation_active, 0.3 * spread_modulation * (mutation_vectors - self.population), 0.0
        )
        
        new_velocity = (
            self.inertia_weight * self.velocity * spread_modulation +
            cognitive_component * spread_modulation +
            social_component * spread_modulation +
            mutation_component + centroid_correction
        )
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_catB(self):
        """Spectral velocity modulation: dampen high-variance directions (1 win, task 23)."""
        cognitive, social = self._adaptive_coefficients()
        centered = self.population - np.mean(self.population, axis=0)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered.T))
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            cond = eigenvalues[0] / eigenvalues[-1]
            total_var = np.sum(eigenvalues) + 1e-10
            var_ratio = eigenvalues / total_var
            spectral_scale = np.sqrt(var_ratio[-1] / (var_ratio + 1e-10))
            spectral_scale = np.clip(spectral_scale, 0.3, 3.0)
            blend = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
            effective_scale = 1.0 * (1.0 - blend) + spectral_scale * blend
            vel_proj = self.velocity @ eigenvectors
            vel_modulated = vel_proj * effective_scale
            modulated_velocity = vel_modulated @ eigenvectors.T
        except np.linalg.LinAlgError:
            modulated_velocity = self.velocity.copy()
        
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
            mutation_active, 0.3 * (mutation_vectors - self.population), 0.0
        )
        
        new_velocity = (
            self.inertia_weight * modulated_velocity +
            cognitive_component + social_component + mutation_component
        )
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_catC(self):
        """Information-theoretic velocity update: entropy + KL distribution matching (2 wins, tasks 0, 5)."""
        cognitive, social = self._adaptive_coefficients()
        centered = self.population - np.mean(self.population, axis=0)
        try:
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            dim = self.dim
            var_range = (self.upper_bound - self.lower_bound) ** 2
            det_cov = np.prod(eigenvalues) + 1e-10
            entropy = 0.5 * (dim * np.log(2 * np.pi * np.e * var_range / dim) + np.log(det_cov))
            max_entropy = 0.5 * dim * np.log(2 * np.pi * np.e * var_range / dim)
            min_entropy = 0.5 * dim * np.log(2 * np.pi * np.e * 1e-4)
            normalized_entropy = np.clip((entropy - min_entropy) / (max_entropy - min_entropy + 1e-10), 0.0, 1.0)
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            anisotropy = np.clip(np.log1p(cond) / np.log1p(1e4), 0.0, 1.0)
        except:
            normalized_entropy = 0.5
            anisotropy = 0.5
        
        exploration_boost = 1.0 + 0.6 * (1.0 - normalized_entropy)
        exploitation_damp = 1.0 - 0.3 * normalized_entropy
        
        try:
            hist, _ = np.histogram(self.current_fitness, bins=min(10, self.np), density=True)
            hist = hist + 1e-10
            hist = hist / np.sum(hist)
            fitness_entropy = -np.sum(hist * np.log(hist + 1e-10))
            max_fitness_entropy = np.log(min(10, self.np))
            normalized_fitness_entropy = np.clip(fitness_entropy / (max_fitness_entropy + 1e-10), 0.0, 1.0)
        except:
            normalized_fitness_entropy = 0.5
        
        combined_entropy = 0.6 * normalized_entropy + 0.4 * normalized_fitness_entropy
        
        try:
            eigenvalues_normalized = eigenvalues / (eigenvalues[0] + 1e-10)
            kl_like = np.mean(np.log(eigenvalues_normalized + 1e-10))
            kl_modulation = np.clip(1.0 + 0.2 * kl_like, 0.5, 1.5)
        except:
            kl_modulation = 1.0
        
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
            mutation_active, 0.3 * (mutation_vectors - self.population), 0.0
        )
        mutation_scale = 1.0 - 0.3 * combined_entropy
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            exploration_boost * cognitive_component +
            exploitation_damp * social_component +
            kl_modulation * mutation_scale * mutation_component
        )
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_catE(self):
        """k-NN graph centrality velocity update (6 wins, tasks 6, 10, 11, 12, 15, 18)."""
        cognitive, social = self._adaptive_coefficients()
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        k = min(5, self.np - 1)
        sq_dists = np.sum(
            (self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2
        )
        np.fill_diagonal(sq_dists, np.inf)
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
        knn_dists = np.sqrt(np.sort(sq_dists, axis=1)[:, :k])
        avg_knn_dist = np.mean(knn_dists, axis=1) + 1e-10
        global_avg_dist = np.mean(avg_knn_dist) + 1e-10
        centrality = global_avg_dist / avg_knn_dist
        centrality = np.clip(centrality / (np.max(centrality) + 1e-10) * 1.5, 0.5, 2.0)
        
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
        n_components = len(components)
        component_sizes = np.array([len(c) for c in components])
        particle_component_size = np.array([
            component_sizes[next((i for i, c in enumerate(components) if p in c), 0)]
            for p in range(self.np)
        ])
        size_factor = np.clip(particle_component_size / max(1, self.np), 0.1, 1.0)
        topology_correction_strength = (2.0 - centrality) * (2.0 - size_factor)
        topology_correction_strength = np.clip(topology_correction_strength, 0.0, 2.0)
        
        if self.global_best is not None:
            to_global = self.global_best - self.population
            to_global_dist = np.linalg.norm(to_global, axis=1, keepdims=True) + 1e-10
            to_global_dir = to_global / to_global_dist
            topology_correction = topology_correction_strength[:, np.newaxis] * to_global_dir
        else:
            topology_correction = topology_correction_strength[:, np.newaxis] * np.random.uniform(
                -1, 1, (self.np, self.dim)
            )
        
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
            mutation_active, 0.3 * (mutation_vectors - self.population), 0.0
        )
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component + social_component +
            0.5 * topology_correction + mutation_component
        )
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_catF(self):
        """Temporal dynamics velocity update: centroid drift + spread change + velocity autocorrelation (9 wins)."""
        if not hasattr(self, '_ema_centroid_drift'):
            self._ema_centroid_drift = 0.0
            self._ema_spread_change = 0.0
            self._prev_centroid = np.mean(self.population, axis=0)
            self._prev_spread = np.std(self.population)
            self._prev_velocity_mag = np.mean(np.linalg.norm(self.velocity, axis=1))
            self._velocity_history = []
        
        current_centroid = np.mean(self.population, axis=0)
        centroid_drift = np.linalg.norm(current_centroid - self._prev_centroid)
        self._prev_centroid = current_centroid.copy()
        self._ema_centroid_drift = 0.2 * centroid_drift + 0.8 * self._ema_centroid_drift
        
        current_spread = np.std(self.population)
        spread_delta = abs(current_spread - self._prev_spread)
        self._prev_spread = current_spread
        self._ema_spread_change = 0.2 * spread_delta + 0.8 * self._ema_spread_change
        
        current_velocity_mag = np.mean(np.linalg.norm(self.velocity, axis=1))
        if self._prev_velocity_mag > 1e-10:
            autocorr = current_velocity_mag / (self._prev_velocity_mag + 1e-10)
        else:
            autocorr = 1.0
        self._prev_velocity_mag = current_velocity_mag
        self._velocity_history.append(autocorr)
        if len(self._velocity_history) > 10:
            self._velocity_history.pop(0)
        
        normalized_drift = self._ema_centroid_drift / (current_spread + 1e-10)
        normalized_spread_change = self._ema_spread_change / (current_spread + 1e-10)
        stagnation_signal = 1.0 - min(1.0, np.mean(self._velocity_history[-5:]) if len(self._velocity_history) >= 5 else 1.0)
        
        if stagnation_signal > 0.7:
            temporal_inertia = min(0.95, self.inertia_weight + 0.1)
        elif normalized_drift < 0.02:
            temporal_inertia = max(0.4, self.inertia_weight - 0.05)
        else:
            temporal_inertia = max(0.4, 0.729 - 0.15 * (self.generation / 3000))
        
        cognitive, social = self._adaptive_coefficients()
        if stagnation_signal > 0.5:
            temporal_cog_mult = 1.3
            temporal_social_mult = 0.7
        elif normalized_drift > 0.1 and normalized_spread_change > 0.05:
            temporal_cog_mult = 1.0
            temporal_social_mult = 1.0
        elif normalized_drift < 0.02:
            temporal_cog_mult = 0.8
            temporal_social_mult = 1.2
        else:
            temporal_cog_mult = 1.0
            temporal_social_mult = 1.0
        cognitive = cognitive * temporal_cog_mult
        social = social * temporal_social_mult
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_base = 0.1 * (1.0 - self.generation / 5000)
        mutation_boost = 0.05 * stagnation_signal + 0.03 * (1.0 - min(1.0, normalized_spread_change))
        mutation_threshold = max(0.01, mutation_base - mutation_boost)
        mutation_active = mutation_mask < mutation_threshold
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False
            )
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        mutation_scale = 0.3 * (1.0 + 0.5 * stagnation_signal)
        mutation_component = np.where(
            mutation_active, mutation_scale * (mutation_vectors - self.population), 0.0
        )
        
        new_velocity = (
            temporal_inertia * self.velocity +
            cognitive_component + social_component + mutation_component
        )
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_catA_knn(self):
        """k-NN density + axis-aligned spread modulation (1 win, task 17)."""
        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_dists = np.sort(sq_dists, axis=1)[:, :k]
        mean_knn_dist = np.mean(knn_dists, axis=1) + 1e-10
        density_factor = mean_knn_dist / (np.mean(mean_knn_dist) + 1e-10)
        density_factor = np.clip(density_factor, 0.3, 3.0)
        
        per_dim_std = np.std(self.population, axis=0) + 1e-10
        mean_spread = np.mean(per_dim_std) + 1e-10
        spread_ratio = per_dim_std / mean_spread
        spread_modulation = 1.0 / (spread_ratio + 0.5)
        spread_modulation = np.clip(spread_modulation, 0.5, 2.5)
        
        density_scale = density_factor[:, np.newaxis]
        spread_scale = spread_modulation[np.newaxis, :]
        geometric_scale = density_scale * spread_scale
        geometric_scale = np.clip(geometric_scale, 0.2, 4.0)
        
        cognitive, social = self._adaptive_coefficients()
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        cognitive_component = cognitive_component * geometric_scale
        social_component = social_component * geometric_scale
        
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
            mutation_active, 0.3 * (mutation_vectors - self.population), 0.0
        )
        mutation_component = mutation_component * geometric_scale
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component + social_component + mutation_component
        )
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    # -------------------------------------------------------------------------
    # DISPATCHER: maps operator name -> velocity update method
    # -------------------------------------------------------------------------
    
    def _dispatch_velocity_update(self, operator):
        if operator == 'original':
            self._velocity_update_original()
        elif operator == 'catA':
            self._velocity_update_catA()
        elif operator == 'catB':
            self._velocity_update_catB()
        elif operator == 'catC':
            self._velocity_update_catC()
        elif operator == 'catE':
            self._velocity_update_catE()
        elif operator == 'catF':
            self._velocity_update_catF()
        elif operator == 'catA_knn':
            self._velocity_update_catA_knn()
        else:
            self._velocity_update_original()
    
    # -------------------------------------------------------------------------
    # PROBE-AND-COMMIT: rank-based scoring (scale-independent)
    # -------------------------------------------------------------------------
    
    def _record_probe_score(self, operator, best_fitness):
        """Record probe score using rank-based (scale-independent) scoring.
        
        Score = normalized rank of best_fitness within the probe history.
        All operators scored on the same [0,1] scale — no magic constants.
        """
        self._probe_fitness_history.append(best_fitness)
        self._probe_op_history.append(operator)
        
        if len(self._probe_fitness_history) < 2:
            score = 1.0
        else:
            current_best = min(self._probe_fitness_history)
            current_worst = max(self._probe_fitness_history)
            if current_worst > current_best:
                normalized = (current_worst - best_fitness) / (current_worst - current_best + 1e-10)
                score = np.clip(normalized, 0.0, 1.0)
            else:
                score = 1.0
        
        self._operator_scores[operator].append(score)
        self._operator_total_score[operator] += score
        self._operator_sample_count[operator] += 1
    
    def _compute_ucb_score(self, operator):
        """UCB1-style score for operator selection.
        
        Uses rank-based average (already in [0,1]) so the exploration
        bonus is on the same scale — no scale-dependent constants.
        """
        n = self._operator_sample_count[operator]
        if n == 0:
            return float('inf')
        total_n = sum(self._operator_sample_count.values())
        avg_score = self._operator_total_score[operator] / n
        exploration = np.sqrt(2.0 * np.log(max(total_n, 1)) / max(n, 1))
        return avg_score + exploration
    
    def _select_best_operator(self):
        best_op = self._operators[0]
        best_score = -float('inf')
        for op in self._operators:
            ucb = self._compute_ucb_score(op)
            if ucb > best_score:
                best_score = ucb
                best_op = op
        return best_op
    
    def _select_runner_up_operator(self):
        scores = [(op, self._compute_ucb_score(op)) for op in self._operators]
        scores.sort(key=lambda x: x[1], reverse=True)
        if len(scores) >= 2:
            return scores[1][0]
        return scores[0][0]
    
    # -------------------------------------------------------------------------
    # MAIN OPTIMIZATION LOOP
    # -------------------------------------------------------------------------
    
    def __call__(self, func, stopping_condition):
        self._initialize_population()
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        
        for op in self._operators:
            self._operator_scores[op] = []
            self._operator_total_score[op] = 0.0
            self._operator_sample_count[op] = 0
        self._probe_fitness_history = []
        self._probe_op_history = []
        self._probe_best_fitness_seen = np.inf
        self._committed_operator = None
        self._in_probe_phase = True
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
        
        # Reset temporal state for catF
        for attr in ['_ema_centroid_drift', '_ema_spread_change', '_velocity_history',
                     '_prev_centroid', '_prev_spread', '_prev_velocity_mag']:
            if hasattr(self, attr):
                delattr(self, attr)
        
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
        
        while not stopping_condition():
            self.generation += 1
            
            # --- PHASE A: PROBE ---
            if self._in_probe_phase:
                current_op = self._operators[self._probe_op_index]
                self._adapt_inertia_weight()
                self._adapt_neighborhood_size()
                self._update_local_best_from_personal()
                
                self._dispatch_velocity_update(current_op)
                self.population = self._clip_to_bounds(self.population + self.velocity)
                
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
                
                gen_best = float(np.min(self.current_fitness))
                if gen_best < self._probe_best_fitness_seen:
                    self._probe_best_fitness_seen = gen_best
                self._record_probe_score(current_op, self._probe_best_fitness_seen)
                
                self._probe_gens_in_current_op += 1
                if self._probe_gens_in_current_op >= self._probe_gens_per_op:
                    self._probe_op_index += 1
                    self._probe_gens_in_current_op = 0
                
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
                if (self.stagnation_counter > 20 and
                    np.random.random() < self._epsilon_review):
                    self._dispatch_velocity_update(self._runner_up_operator)
                else:
                    self._dispatch_velocity_update(self._committed_operator)
                
                self.population = self._clip_to_bounds(self.population + self.velocity)
                
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
        
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        return self.global_best_fitness, self.global_best.copy()
