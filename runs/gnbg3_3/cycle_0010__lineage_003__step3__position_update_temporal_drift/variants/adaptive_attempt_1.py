import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: SLIDING-WINDOW THOMPSON SAMPLING (Multi-Armed Bandit, mechanism #1)
    
    Gap table analysis: ALL gaps are < 2× (max is 1.5× on task 16). No task has
    gap >= 10×, so ensemble blending is mathematically viable. However, the 7
    winning operators are complementary (geometric, fitness-rank, graph-topology,
    temporal-momentum, spectral-alignment, hybrid) and win on different tasks.
    Since task identity is unknown at runtime, a bandit learns which operators
    work well in which regimes via context features.
    
    Reward = rank-based improvement within sliding window (scale-independent).
    Each arm has a reachable path to weight 1.0 (exclusive selection).
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
        
        # --- BANDIT STATE: 7 winning operators ---
        self._operators = [
            'original',    # 9 wins: eigenvalue-anisotropic
            'catA',        # 1 win:  geometric centroid/hull
            'catD',        # 3 wins: fitness-rank/FDC
            'catE',        # 1 win:  graph Laplacian/topology
            'catF',        # 1 win:  centroid momentum/velocity autocorr
            'catB',        # 3 wins: directional alignment/subspace
            'catH',        # 6 wins: hybrid temporal+topology
        ]
        self._num_operators = len(self._operators)
        
        # Sliding window for rank-based rewards (K >= 3 * num_arms)
        self._window_size = max(21, 3 * self._num_operators)
        self._operator_rewards = {op: [] for op in self._operators}
        self._operator_selection_count = {op: 0 for op in self._operators}
        self._total_selections = 0
        
        # Beta distribution parameters for Thompson Sampling
        self._alpha = {op: 1.0 for op in self._operators}
        self._beta = {op: 1.0 for op in self._operators}
        
        # Generation-level fitness history for rank-based reward
        self._gen_fitness_history = []
        self._window_gen_fitness = []
        
        # Context features for contextual bandit
        self._context_features = np.zeros(4)
        
        # Per-operator context tracking (for credit assignment)
        self._op_context_history = {op: [] for op in self._operators}
        self._op_reward_history = {op: [] for op in self._operators}
    
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
            best_idx = np.argmin(fitness_in_neighborhood)
            self.local_best[i] = self.personal_best[all_indices[best_idx]]
            self.local_best_fitness[i] = fitness_in_neighborhood[best_idx]
    
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
    
    def _compute_context_features(self):
        """Compute 4 context features from current population state (scale-independent)."""
        features = np.zeros(4)
        
        # 1. Condition number of population covariance (convergence signal)
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            features[0] = np.clip(np.log1p(cond) / 10.0, 0.0, 1.0)
        except:
            features[0] = 0.5
        
        # 2. Spectral entropy (effective dimensionality proxy)
        try:
            total_var = np.sum(eigenvalues) + 1e-10
            eigenvalues_norm = eigenvalues / total_var
            spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
            max_entropy = np.log(self.dim + 1e-10)
            features[1] = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)
        except:
            features[1] = 0.5
        
        # 3. Diversity relative to search range (exploration signal)
        div = self._compute_diversity()
        search_range = self.upper_bound - self.lower_bound
        features[2] = np.clip(div / (search_range * 0.1), 0.0, 1.0)
        
        # 4. Improvement rate EMA (progress signal)
        if self.global_best is not None and hasattr(self, '_prev_gb_fitness'):
            improvement = max(0.0, self._prev_gb_fitness - self.global_best_fitness)
            if not hasattr(self, '_improvement_ema'):
                self._improvement_ema = 0.0
            self._improvement_ema = 0.3 * improvement + 0.7 * self._improvement_ema
            features[3] = np.clip(self._improvement_ema / 1.0, 0.0, 1.0)
        else:
            features[3] = 0.5
        self._prev_gb_fitness = self.global_best_fitness if self.global_best is not None else np.inf
        
        self._context_features = features
        return features
    
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
        except:
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
    # POSITION UPDATE STRATEGIES (7 benchmark winners)
    # -------------------------------------------------------------------------
    
    def _position_update_original(self):
        """Eigenvalue-anisotropic velocity modulation (9 wins)."""
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
        except:
            new_population = self.population + self.velocity
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catA(self):
        """Geometry-based temporal drift (1 win)."""
        current_centroid = np.mean(self.population, axis=0)
        if not hasattr(self, '_prev_centroid_A'):
            self._prev_centroid_A = current_centroid.copy()
            self._centroid_ema_A = 0.0
            self._hull_vol_ema_A = None
        else:
            raw_drift = np.linalg.norm(current_centroid - self._prev_centroid_A)
            self._centroid_ema_A = 0.7 * self._centroid_ema_A + 0.3 * raw_drift
            self._prev_centroid_A = current_centroid.copy()
        
        sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        pairwise_dists = np.sqrt(sq_dists)
        valid_dists = pairwise_dists[pairwise_dists < np.inf]
        mean_dist = np.mean(valid_dists) if len(valid_dists) > 0 else 1.0
        std_dist = np.std(valid_dists) + 1e-10 if len(valid_dists) > 0 else 1.0
        dist_cv = std_dist / mean_dist
        
        mins = np.min(self.population, axis=0)
        maxs = np.max(self.population, axis=0)
        axis_spreads = maxs - mins + 1e-10
        max_spread = np.max(axis_spreads)
        min_spread = np.min(axis_spreads)
        spread_ratio = min_spread / max_spread
        
        k = min(5, self.np - 1)
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
        visited = np.zeros(self.np, dtype=bool)
        n_components = 0
        component_sizes = []
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
            n_components += 1
            component_sizes.append(len(component))
        
        component_sizes = np.array(component_sizes)
        largest_ratio = np.max(component_sizes) / self.np if len(component_sizes) > 0 else 1.0
        fragmentation = 1.0 - largest_ratio
        
        hull_volume = np.prod(axis_spreads)
        hull_volume_log = np.log(hull_volume + 1e-10)
        if self._hull_vol_ema_A is None:
            self._hull_vol_ema_A = hull_volume_log
        else:
            self._hull_vol_ema_A = 0.9 * self._hull_vol_ema_A + 0.1 * hull_volume_log
        hull_shrink_rate = hull_volume_log - self._hull_vol_ema_A
        
        drift_signal = np.clip(self._centroid_ema_A / (mean_dist + 1e-10), 0.0, 2.0)
        spread_signal = np.clip(spread_ratio, 0.0, 1.0)
        component_signal = 1.0 - fragmentation
        hull_signal = np.clip(-hull_shrink_rate / 10.0, 0.0, 1.0)
        geo_signal = (drift_signal * 0.25 + spread_signal * 0.25 + 
                      component_signal * 0.25 + hull_signal * 0.25)
        
        if self._centroid_ema_A > mean_dist * 0.05:
            base_scale = 1.2
        elif hull_shrink_rate < -0.01:
            base_scale = 0.7
        else:
            base_scale = 1.0
        temporal_scale = base_scale * (0.8 + 0.4 * geo_signal)
        
        knn_avg_dist = np.mean(pairwise_dists[np.arange(self.np)[:, None], knn_indices], axis=1, keepdims=True) + 1e-10
        density_factor = np.clip(knn_avg_dist / (mean_dist + 1e-10), 0.5, 2.0)
        
        to_centroid = current_centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid / to_centroid_dist
        centroid_pull = 0.2 * to_centroid_dir * density_factor
        
        new_population = (self.population + 
                          self.inertia_weight * self.velocity * temporal_scale * density_factor + 
                          centroid_pull)
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catD(self):
        """Fitness-landscape rank-based velocity modulation (3 wins)."""
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
        if not hasattr(self, '_success_count_D'):
            self._success_count_D = np.zeros(self.np)
        improved = self.personal_best_fitness > self.current_fitness
        self._success_count_D[improved] += 1
        self._success_count_D[~improved] *= 0.9
        success_norm = self._success_count_D / (np.max(self._success_count_D) + 1.0)
        vel_scale = 0.5 + 1.5 * fitness_ranks
        vel_scale = np.clip(vel_scale, 0.5, 2.0)
        
        if self.global_best is not None and self.np > 2:
            dists = np.linalg.norm(self.population - self.global_best, axis=1)
            dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
            if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                fdc = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
            else:
                fdc = 0.0
        else:
            fdc = 0.0
        
        fdc_explore_factor = np.clip(2.0 - fdc, 0.5, 2.5)
        success_boost = 1.0 + 0.5 * success_norm
        success_boost = np.clip(success_boost, 1.0, 2.0)
        combined_scale = vel_scale * fdc_explore_factor * success_boost
        combined_scale = np.clip(combined_scale, 0.3, 3.0)
        
        top_k = max(1, self.np // 5)
        top_indices = np.argsort(self.current_fitness)[:top_k]
        top_centroid = np.mean(self.population[top_indices], axis=0)
        to_top = top_centroid - self.population
        to_top_dist = np.linalg.norm(to_top, axis=1, keepdims=True) + 1e-10
        to_top_dir = to_top / to_top_dist
        gradient_strength = 0.5 * fitness_ranks[:, np.newaxis]
        
        scaled_velocity = self.velocity * combined_scale[:, np.newaxis]
        gradient_push = gradient_strength * to_top_dir
        random_perturb = fdc_explore_factor * np.random.uniform(-0.5, 0.5, (self.np, self.dim))
        
        new_population = self.population + scaled_velocity + gradient_push + random_perturb
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catE(self):
        """Graph Laplacian escape via topology-aware velocity modulation (1 win)."""
        try:
            k = min(5, self.np - 1)
            sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
            
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
            particle_component_size = np.array([
                component_sizes[np.argmax([i in c for c in components])]
                for i in range(self.np)
            ])
            
            try:
                row_idx = np.repeat(np.arange(self.np), k)
                col_idx = knn_indices.ravel()
                data = np.ones(len(row_idx))
                adj = np.zeros((self.np, self.np))
                adj[row_idx, col_idx] = 1.0
                adj = np.clip(adj + adj.T, 0.0, 1.0)
                np.fill_diagonal(adj, 0.0)
                degrees = np.sum(adj, axis=1) + 1e-10
                d_inv_sqrt = 1.0 / np.sqrt(degrees)
                d_mat = np.diag(d_inv_sqrt)
                lap = np.eye(self.np) - d_mat @ adj @ d_mat
                eigenvalues = np.linalg.eigvalsh(lap)
                eigenvalues = np.sort(eigenvalues)
                spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 1.0
                spectral_gap = np.clip(spectral_gap, 0.0, 1.0)
            except:
                spectral_gap = 1.0
            
            centrality = np.zeros(self.np)
            for i in range(self.np):
                neighbors = knn_indices[i]
                neighbor_degrees = component_sizes[[np.argmax([n in c for c in components]) for n in neighbors]]
                centrality[i] = np.sum(1.0 / (neighbor_degrees + 1e-10))
            centrality = centrality / (np.max(centrality) + 1e-10)
            
            explore_factor = 1.0 + 0.6 * (1.0 - spectral_gap)
            size_factor = np.clip(particle_component_size / max(1, self.np), 0.2, 1.0)
            isolation_boost = 1.5 * (1.0 - size_factor) + 1.0
            bridge_damp = 1.0 - 0.3 * centrality
            vel_scale = explore_factor * isolation_boost * bridge_damp
            vel_scale = np.clip(vel_scale, 0.3, 2.5)
            
            if self.global_best is not None and hasattr(self, '_prev_gb_fitness_E'):
                improvement = max(0.0, self._prev_gb_fitness_E - self.global_best_fitness)
            else:
                improvement = 0.0
            self._prev_gb_fitness_E = self.global_best_fitness
            
            if not hasattr(self, '_graph_ema_improvement'):
                self._graph_ema_improvement = 0.0
            self._graph_ema_improvement = 0.3 * improvement + 0.7 * self._graph_ema_improvement
            
            if self._graph_ema_improvement > 1e-6:
                temporal_scale = 1.2
            elif self._graph_ema_improvement > 1e-10:
                temporal_scale = 1.0
            else:
                temporal_scale = 0.7
            
            new_population = self.population + temporal_scale * self.velocity * vel_scale[:, np.newaxis]
            
            if spectral_gap < 0.15:
                isolated = particle_component_size <= 2
                if np.any(isolated):
                    random_perturb = np.random.uniform(-1.0, 1.0, (self.np, self.dim))
                    new_population[isolated] += 0.4 * random_perturb[isolated]
        except:
            new_population = self.population + self.velocity
        
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catF(self):
        """Centroid momentum with velocity autocorrelation stagnation detection (1 win)."""
        try:
            centroid = np.mean(self.population, axis=0)
            if not hasattr(self, '_temporal_centroid'):
                self._temporal_centroid = centroid.copy()
                self._temporal_centroid_history = []
                self._temporal_vel_history = []
            
            self._temporal_centroid = 0.7 * self._temporal_centroid + 0.3 * centroid
            
            if len(self._temporal_centroid_history) > 0:
                centroid_vel = self._temporal_centroid - self._temporal_centroid_history[-1]
            else:
                centroid_vel = np.zeros(self.dim)
            
            self._temporal_vel_history.append(centroid_vel)
            if len(self._temporal_vel_history) > 10:
                self._temporal_vel_history.pop(0)
            
            if len(self._temporal_vel_history) >= 3:
                v1 = np.array(self._temporal_vel_history[-2])
                v2 = np.array(self._temporal_vel_history[-1])
                v1_norm = np.linalg.norm(v1) + 1e-10
                v2_norm = np.linalg.norm(v2) + 1e-10
                velocity_autocorr = np.dot(v1, v2) / (v1_norm * v2_norm)
            else:
                velocity_autocorr = 0.0
            
            centroid_drift = centroid - self._temporal_centroid
            drift_norm = np.linalg.norm(centroid_drift) + 1e-10
            drift_dir = centroid_drift / drift_norm
            vel_alignment = np.dot(self.velocity, drift_dir) / (np.linalg.norm(self.velocity, axis=1) + 1e-10)
            cross_corr = np.mean(vel_alignment)
            
            self._temporal_centroid_history.append(centroid.copy())
            if len(self._temporal_centroid_history) > 10:
                self._temporal_centroid_history.pop(0)
            
            autocorr_signal = np.clip(velocity_autocorr, -1.0, 1.0)
            cross_signal = np.clip(cross_corr, -1.0, 1.0)
            stagnation_risk = 0.5 * abs(autocorr_signal) - 0.5 * abs(cross_signal)
            stagnation_risk = np.clip(stagnation_risk, -1.0, 1.0)
            
            if self.global_best is not None and hasattr(self, '_prev_gb_fitness_F'):
                improvement = max(0.0, self._prev_gb_fitness_F - self.global_best_fitness)
            else:
                improvement = 0.0
            self._prev_gb_fitness_F = self.global_best_fitness
            
            if not hasattr(self, '_temporal_ema_improvement'):
                self._temporal_ema_improvement = 0.0
            self._temporal_ema_improvement = 0.3 * improvement + 0.7 * self._temporal_ema_improvement
            
            if self._temporal_ema_improvement > 1e-6:
                base_scale = 1.2
            elif self._temporal_ema_improvement > 1e-10:
                base_scale = 1.0
            else:
                base_scale = 0.7
            
            temporal_modulation = 1.0 + 0.4 * (1.0 - stagnation_risk) + 0.3 * cross_signal
            temporal_scale = base_scale * np.clip(temporal_modulation, 0.5, 2.0)
            
            if stagnation_risk > 0.5:
                drift_rate = np.linalg.norm(centroid_drift)
                if drift_rate < 1e-4:
                    corrective_force = 0.3 * np.random.uniform(-1, 1, self.dim) * drift_norm
                else:
                    corrective_force = -0.2 * centroid_drift
            else:
                corrective_force = np.zeros(self.dim)
            
            new_population = self.population + temporal_scale * self.velocity + corrective_force
        except:
            new_population = self.population + self.velocity
        
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catB(self):
        """Directional alignment + subspace expansion (3 wins)."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond_log = np.log1p(cond)
            sv_scale = np.sqrt(eigenvalues + 1e-10)
            sv_norm = sv_scale / (sv_scale[0] + 1e-10)
            
            if self.global_best is not None:
                global_dir = self.global_best - np.mean(self.population, axis=0)
                global_dir_norm = np.linalg.norm(global_dir) + 1e-10
                global_dir = global_dir / global_dir_norm
                alignment = np.abs(eigenvectors.T @ global_dir)
                alignment_boost = 0.5 + 0.5 * alignment
                alignment_boost = np.clip(alignment_boost, 0.3, 1.5)
            else:
                alignment_boost = np.ones(self.dim)
            
            spectral_signal = np.clip(cond_log / 5.0, 0.0, 1.0)
            if spectral_signal > 0.3:
                minor_boost = 1.0 + 1.5 * spectral_signal * (1.0 - sv_norm)
                minor_boost = np.clip(minor_boost, 1.0, 3.0)
            else:
                minor_boost = np.ones(self.dim)
            
            per_comp_scale = alignment_boost * minor_boost
            per_comp_scale = np.clip(per_comp_scale, 0.3, 3.0)
            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * per_comp_scale
            new_population = self.population + (vel_scaled @ eigenvectors.T)
        except:
            new_population = self.population + self.velocity
        
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catH(self):
        """Hybrid: Temporal drift tracking + topology-based fragmentation detection (6 wins)."""
        try:
            centroid = np.mean(self.population, axis=0)
            if not hasattr(self, '_prev_centroid_H'):
                self._prev_centroid_H = centroid.copy()
                self._centroid_drift_history_H = []
            
            drift = np.linalg.norm(centroid - self._prev_centroid_H)
            self._prev_centroid_H = centroid.copy()
            
            if not hasattr(self, '_centroid_drift_history_H'):
                self._centroid_drift_history_H = []
            self._centroid_drift_history_H.append(drift)
            if len(self._centroid_drift_history_H) > 20:
                self._centroid_drift_history_H.pop(0)
            
            if not hasattr(self, '_ema_drift_H'):
                self._ema_drift_H = 0.0
            self._ema_drift_H = 0.3 * drift + 0.7 * self._ema_drift_H
            
            population_spread = np.std(self.population) + 1e-10
            normalized_drift = self._ema_drift_H / population_spread
            
            k = min(5, self.np - 1)
            sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
            
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
            max_component_size = np.max(component_sizes)
            fragmentation_ratio = 1.0 - (max_component_size / (self.np + 1e-10))
            
            particle_to_component = np.zeros(self.np, dtype=int)
            for idx, comp in enumerate(components):
                for particle_idx in comp:
                    particle_to_component[particle_idx] = idx
            
            is_converging = normalized_drift < 0.05
            is_fragmented = n_components > 1 and fragmentation_ratio > 0.3
            
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            total_var = np.sum(eigenvalues) + 1e-10
            eigenvalues_norm = eigenvalues / total_var
            spectral_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm + 1e-10))
            max_entropy = np.log(self.dim + 1e-10)
            entropy_ratio = np.clip(spectral_entropy / max_entropy, 0.0, 1.0)
            
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cumvar = np.cumsum(eigenvalues) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = eff_dim / self.dim
            spectral_signal = entropy_ratio * 0.5 + (1.0 - eff_dim_ratio) * 0.5
            
            if self.global_best is not None and hasattr(self, '_prev_gb_fitness_H'):
                improvement = max(0.0, self._prev_gb_fitness_H - self.global_best_fitness)
            else:
                improvement = 0.0
            self._prev_gb_fitness_H = self.global_best_fitness
            
            if not hasattr(self, '_spectral_ema_improvement_H'):
                self._spectral_ema_improvement_H = 0.0
            self._spectral_ema_improvement_H = 0.3 * improvement + 0.7 * self._spectral_ema_improvement_H
            
            if is_fragmented and is_converging:
                base_scale = 1.5
                topology_weight = 0.8
                spectral_weight = 0.2
            elif is_fragmented and not is_converging:
                base_scale = 1.3
                topology_weight = 0.6
                spectral_weight = 0.4
            elif not is_fragmented and is_converging:
                base_scale = 0.9
                topology_weight = 0.2
                spectral_weight = 0.8
            else:
                base_scale = 1.1
                topology_weight = 0.4
                spectral_weight = 0.6
            
            if self._spectral_ema_improvement_H > 1e-6:
                spectral_base = 1.2
            elif self._spectral_ema_improvement_H > 1e-10:
                spectral_base = 1.0
            else:
                spectral_base = 0.7
            
            spectral_scale = spectral_base * (1.0 + 0.4 * spectral_signal)
            if cond > 100.0:
                log_damp = np.log1p(cond) / np.log1p(1000.0)
                log_damp = np.clip(log_damp, 0.0, 1.0)
                spectral_scale *= (1.0 + 0.3 * log_damp)
            
            component_centroids = np.array([np.mean(self.population[c], axis=0) for c in components])
            largest_comp_idx = np.argmax(component_sizes)
            target_centroid = component_centroids[largest_comp_idx]
            
            topology_correction = np.zeros((self.np, self.dim))
            for i in range(self.np):
                comp_idx = particle_to_component[i]
                comp_size = component_sizes[comp_idx]
                if comp_idx != largest_comp_idx:
                    correction_strength = 0.5 * (1.0 - comp_size / self.np)
                    direction = target_centroid - self.population[i]
                    direction_norm = np.linalg.norm(direction) + 1e-10
                    topology_correction[i] = correction_strength * (direction / direction_norm)
            
            combined_scale = spectral_weight * spectral_scale + topology_weight * (1.0 + 0.5 * fragmentation_ratio)
            
            idx_eig = np.argsort(eigenvalues)[::-1]
            V = np.linalg.eigh(cov)[1][:, idx_eig]
            vel_proj = self.velocity @ V
            sv_scale_h = np.sqrt(eigenvalues + 1e-10)
            sv_norm_h = sv_scale_h / (sv_scale_h[0] + 1e-10)
            per_comp_scale_h = np.clip(sv_norm_h, 0.1, 2.0)
            vel_scaled = vel_proj * per_comp_scale_h
            
            new_population = self.population + combined_scale * (vel_scaled @ V.T) + topology_correction
        except:
            new_population = self.population + self.velocity
        
        self.population = self._clip_to_bounds(new_population)
    
    def _dispatch_position_update(self, operator):
        """Route to the appropriate position update implementation."""
        if operator == 'original':
            self._position_update_original()
        elif operator == 'catA':
            self._position_update_catA()
        elif operator == 'catD':
            self._position_update_catD()
        elif operator == 'catE':
            self._position_update_catE()
        elif operator == 'catF':
            self._position_update_catF()
        elif operator == 'catB':
            self._position_update_catB()
        elif operator == 'catH':
            self._position_update_catH()
        else:
            self._position_update_original()
    
    # -------------------------------------------------------------------------
    # BANDIT: Thompson Sampling with rank-based reward
    # -------------------------------------------------------------------------
    
    def _update_bandit_reward(self, operator, gen_best_fitness):
        """Update bandit reward using rank-based scoring within sliding window.
        
        Rank-based: score = fraction of window generations with WORSE fitness.
        Scale-independent — no magic constants.
        """
        self._gen_fitness_history.append(gen_best_fitness)
        
        if len(self._gen_fitness_history) > self._window_size:
            self._gen_fitness_history.pop(0)
        
        if len(self._gen_fitness_history) < 2:
            reward = 1.0
        else:
            current_best = min(self._gen_fitness_history)
            current_worst = max(self._gen_fitness_history)
            if current_worst > current_best:
                normalized = (current_worst - gen_best_fitness) / (current_worst - current_best + 1e-10)
                reward = np.clip(normalized, 0.0, 1.0)
            else:
                reward = 1.0
        
        self._operator_rewards[operator].append(reward)
        if len(self._operator_rewards[operator]) > self._window_size:
            self._operator_rewards[operator].pop(0)
        
        self._op_reward_history[operator].append(reward)
        if len(self._op_reward_history[operator]) > self._window_size:
            self._op_reward_history[operator].pop(0)
    
    def _thompson_sample(self):
        """Thompson Sampling: sample from Beta distribution per operator.
        
        Each operator has Beta(alpha, beta) where:
          - alpha = 1 + sum of positive rewards
          - beta = 1 + sum of negative rewards (inverse rank)
        
        Returns operator with highest sampled value.
        """
        samples = {}
        for op in self._operators:
            alpha = 1.0 + np.sum(self._operator_rewards[op])
            beta = 1.0 + np.sum([1.0 - r for r in self._operator_rewards[op]])
            samples[op] = np.random.beta(max(alpha, 1e-10), max(beta, 1e-10))
        
        return max(samples, key=samples.get)
    
    def _select_operator_ucb(self):
        """UCB1 selection as alternative to Thompson Sampling."""
        best_op = self._operators[0]
        best_score = -float('inf')
        
        for op in self._operators:
            n = self._operator_selection_count[op]
            if n == 0:
                return op
            
            avg_reward = np.mean(self._operator_rewards[op]) if len(self._operator_rewards[op]) > 0 else 0.0
            total_n = self._total_selections
            exploration = np.sqrt(2.0 * np.log(max(total_n, 1)) / max(n, 1))
            ucb = avg_reward + exploration
            
            if ucb > best_score:
                best_score = ucb
                best_op = op
        
        return best_op
    
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
        """
        Run the optimizer with THOMPSON SAMPLING position-update selection.
        
        Args:
            func: Objective function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        self._initialize_population()
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        
        # Reset bandit state
        for op in self._operators:
            self._operator_rewards[op] = []
            self._operator_selection_count[op] = 0
            self._op_reward_history[op] = []
        self._total_selections = 0
        self._gen_fitness_history = []
        
        # Reset operator-specific state
        for attr in ['_prev_centroid_A', '_hull_vol_ema_A', '_centroid_ema_A',
                     '_success_count_D', '_prev_gb_fitness_E', '_graph_ema_improvement',
                     '_prev_gb_fitness_F', '_temporal_centroid', '_temporal_centroid_history',
                     '_temporal_vel_history', '_temporal_ema_improvement', '_prev_gb_fitness_H',
                     '_spectral_ema_improvement_H', '_prev_centroid_H', '_centroid_drift_history_H',
                     '_ema_drift_H', '_prev_gb_fitness', '_improvement_ema']:
            if hasattr(self, attr):
                delattr(self, attr)
        
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
        
        # Track initial fitness for bandit
        init_best = float(np.min(self.current_fitness))
        self._gen_fitness_history.append(init_best)
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # Compute context features
            self._compute_context_features()
            
            # Select operator via Thompson Sampling
            selected_op = self._thompson_sample()
            self._operator_selection_count[selected_op] += 1
            self._total_selections += 1
            
            # Adaptation
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            self._update_local_best_from_personal()
            
            # Velocity update (shared base), then position update (SELECTED)
            self._velocity_update_base()
            self._dispatch_position_update(selected_op)
            
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
            
            # Record rank-based reward for selected operator
            gen_best = float(np.min(self.current_fitness))
            self._update_bandit_reward(selected_op, gen_best)
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
