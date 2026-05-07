import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: THOMPSON SAMPLING over 6 operators (mechanism #1, multi-armed bandit)
    
    The gap table is NOT blend-hostile:
      - Maximum gap is 2.8× (Task 5, variant_03 vs variant_06).
      - Zero tasks have gap >= 10×, so ensemble blending is mathematically viable.
      - However, the 6-way tie at 6 wins each confirms that no single operator
        dominates; different operators win on different tasks.
    
    Thompson Sampling fits because:
      - Win counts are evenly spread (6/6/6/3/2/1), so a bandit must explore.
      - All per-task gaps are small (<3×), so the cost of a wrong arm is bounded.
      - Rank-based scoring ensures scale-independent rewards (Rule a).
      - Beta posterior handles binary success/failure naturally.
    
    The dispatcher preserves per-task advantages by:
      - Using a sliding window of recent rank-based scores (Rule b).
      - Thompson Sampling commits probabilistically — the best arm naturally
        accumulates the highest posterior mean, but exploration prevents
        locking into a locally-optimal arm when the landscape changes.
      - No hard-coded regime→weight table (Rule f).
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
        
        # THE 6 BENCHMARK-WINNING POSITION UPDATE STRATEGIES
        self._operators = [
            'original',      # baseline eigenvalue-anisotropic (6 wins)
            'variant_01',    # axis-aligned bounding box correction (2 wins)
            'variant_02',    # spectral entropy + condition-number damping (6 wins)
            'variant_03',    # Mahalanobis-entropy centroid modulation (3 wins)
            'variant_05',    # MST edge-weight topology analysis (6 wins)
            'variant_10',    # pseudoinverse velocity correction (1 win)
        ]
        
        # Thompson Sampling state: Beta posterior per operator
        # Beta(alpha, beta) where alpha = successes + 1, beta = failures + 1
        self._op_alpha = {op: 1.0 for op in self._operators}
        self._op_beta = {op: 1.0 for op in self._operators}
        
        # Sliding window of rank-based scores per operator (Rule b: K >= 3*num_arms)
        self._window_size = max(18, 3 * len(self._operators))
        self._op_score_window = {op: [] for op in self._operators}
        
        # Running history for rank-based scoring
        self._recent_fitness = []  # last _window_size best fitness values
        self._gen_count = 0
        
        # Selected operator for current generation
        self._selected_operator = self._operators[0]
    
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
    
    def _record_and_score(self, current_best_fitness):
        """Record rank-based score for the selected operator.
        
        Score = 1 if this generation's best fitness is in the top tercile
        of the last _window_size generations, else 0.
        
        This is rank-based, NOT raw fitness — no scale-dependent constants.
        """
        self._recent_fitness.append(current_best_fitness)
        if len(self._recent_fitness) > self._window_size:
            self._recent_fitness.pop(0)
        
        if len(self._recent_fitness) < 3:
            score = 1.0
        else:
            sorted_fitness = sorted(self._recent_fitness)
            tercile_idx = len(sorted_fitness) // 3
            tercile_threshold = sorted_fitness[max(0, tercile_idx - 1)]
            score = 1.0 if current_best_fitness <= tercile_threshold else 0.0
        
        window = self._op_score_window[self._selected_operator]
        window.append(score)
        if len(window) > self._window_size:
            window.pop(0)
        
        n = len(window)
        if n >= 3:
            empirical_success_rate = sum(window) / n
        else:
            empirical_success_rate = 0.5
        
        self._op_alpha[self._selected_operator] = 1.0 + empirical_success_rate * n
        self._op_beta[self._selected_operator] = 1.0 + (1.0 - empirical_success_rate) * n
    
    def _select_operator_thompson(self):
        """Thompson Sampling: sample from Beta posterior for each operator."""
        samples = {}
        for op in self._operators:
            alpha = max(1e-10, self._op_alpha[op])
            beta = max(1e-10, self._op_beta[op])
            try:
                samples[op] = np.random.beta(alpha, beta)
            except:
                samples[op] = 0.0
        
        best_op = self._operators[0]
        best_sample = -1.0
        for op, s in samples.items():
            if s > best_sample:
                best_sample = s
                best_op = op
        return best_op
    
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
    
    def _velocity_update_base(self):
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
    # POSITION UPDATE STRATEGIES (the 6 benchmark winners)
    # -------------------------------------------------------------------------
    
    def _position_update_original(self):
        """Original: eigenvalue-anisotropic velocity modulation."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
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
        """Axis-aligned bounding box spread correction (Category A)."""
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
        """Spectral entropy + condition-number anisotropic damping (Category B)."""
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
        """Mahalanobis-entropy centroid modulation (Category C)."""
        centroid = np.mean(self.population, axis=0)
        centered = self.population - centroid
        cov = np.cov(centered.T)
        cov_reg = cov + 0.1 * np.trace(cov) / self.dim * np.eye(self.dim)
        cov_reg = 0.5 * (cov_reg + cov_reg.T)

        try:
            U_svd, s_svd, Vt_svd = np.linalg.svd(cov_reg, full_matrices=False)
            s_inv = np.where(s_svd > 1e-10, 1.0 / s_svd, 0.0)
            precision = Vt_svd.T @ np.diag(s_inv) @ U_svd.T
        except np.linalg.LinAlgError:
            precision = np.eye(self.dim)

        mahal_sq = np.sum(centered @ precision * centered, axis=1)
        mahal_dist = np.sqrt(mahal_sq + 1e-10)[:, np.newaxis]

        eigenvalues = np.linalg.eigvalsh(cov_reg)
        eigenvalues = np.clip(eigenvalues, 1e-10, None)
        eigenvalues = np.sort(eigenvalues)[::-1]
        total_var = np.sum(eigenvalues) + 1e-10
        eigenvalues_norm = eigenvalues / total_var

        spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
        max_entropy = np.log(self.dim + 1e-10)
        entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)

        exploration_factor = 1.0 + 0.5 * (1.0 - entropy_ratio)
        global_mahal_mean = np.mean(mahal_dist) + 1e-10
        local_entropy_modulation = np.clip(mahal_dist / global_mahal_mean, 0.5, 2.0)
        kl_signal = np.clip(entropy_ratio / (local_entropy_modulation + 1e-10), 0.3, 1.5)

        dist_to_centroid = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-10
        global_spread = np.mean(dist_to_centroid) + 1e-10
        centroid_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread))

        to_centroid_dir = centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid_dir / to_centroid_dist

        correction = centroid_attraction * to_centroid_dir
        correction *= exploration_factor * kl_signal

        new_population = self.population + self.inertia_weight * self.velocity + 0.3 * correction
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_05(self):
        """MST edge-weight analysis for topology-aware exploration (Category E)."""
        n = self.np
        dists = np.linalg.norm(
            self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :], axis=2
        )

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

        sparsity_scale = np.clip(
            1.0 / (np.mean(valid_edges) / (valid_edges + 1e-10) + 1e-10), 0.5, 1.5
        )
        damped_velocity = self.velocity * sparsity_scale[:, np.newaxis]

        new_population = self.population + self.inertia_weight * damped_velocity + 0.4 * correction
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_10(self):
        """Pseudoinverse velocity correction for rank-deficient populations (Category B)."""
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
            per_comp_scale = (
                uniform_scale * (1.0 - correction_strength) +
                inv_sv_weight * correction_strength
            )

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
    
    def _dispatch_position_update(self, operator):
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
    
    def __call__(self, func, stopping_condition):
        self._initialize_population()
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        
        for op in self._operators:
            self._op_alpha[op] = 1.0
            self._op_beta[op] = 1.0
            self._op_score_window[op] = []
        self._recent_fitness = []
        self._gen_count = 0
        self._selected_operator = self._operators[0]
        
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
            self._gen_count += 1
            
            self._selected_operator = self._select_operator_thompson()
            
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            self._update_local_best_from_personal()
            
            self._velocity_update_base()
            self._dispatch_position_update(self._selected_operator)
            
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
            
            gen_best = float(np.min(self.current_fitness))
            self._record_and_score(gen_best)
            
            if self._gen_count % 50 == 0:
                scores = {
                    op: sum(self._op_score_window[op]) / max(1, len(self._op_score_window[op]))
                    for op in self._operators
                }
        
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
