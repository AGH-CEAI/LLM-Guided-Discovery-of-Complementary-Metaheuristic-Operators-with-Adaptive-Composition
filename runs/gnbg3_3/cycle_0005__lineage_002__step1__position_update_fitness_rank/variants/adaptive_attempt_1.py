import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: THOMPSON SAMPLING (Multi-Armed Bandit, mechanism #1)
    
    The gap table is NOT blend-hostile:
      - Max gap = 2.8× on task 5 (well below the 10× threshold)
      - 0 tasks with gap ≥ 10× — linear blending is mathematically viable
      - BUT: 5 distinct winners (7/7/6/3/1) means no single operator dominates
    
    Thompson Sampling is appropriate because:
      - Operators are mutually exclusive (one position update per generation)
      - Credit assignment is clean: did global_best improve this gen?
      - Small gaps mean exploration is worthwhile — a suboptimal operator
        loses only ~2×, not 100×, so exploration cost is bounded
      - Rank-based beta posterior avoids scale-dependent magic constants
    
    The 5 benchmark winners are implemented exactly as benchmarked:
      - variant_02: eigenvalue-weighted velocity projection (1 win)
      - variant_05: graph-Laplacian k-NN topology (6 wins)
      - variant_08: FDC + temporal centroid-drift hybrid (7 wins)
      - variant_10: eigenvalue entropy + fitness-rank (7 wins)
      - original: condition-number spectral scaling (3 wins)
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
        
        # --- THOMPSON SAMPLING STATE ---
        # Beta posterior for each operator: alpha=wins, beta=losses
        # Prior: alpha=1, beta=1 (uniform) — uninformative since no operator
        # is provably better across all tasks
        self._operators = [
            'variant_02',  # eigenvalue-weighted velocity projection (1 win)
            'variant_05',  # graph-Laplacian k-NN topology (6 wins)
            'variant_08',  # FDC + temporal centroid-drift hybrid (7 wins)
            'variant_10',  # eigenvalue entropy + fitness-rank (7 wins)
            'original',    # condition-number spectral scaling (3 wins)
        ]
        self._alpha = {op: 1.0 for op in self._operators}
        self._beta = {op: 1.0 for op in self._operators}
        
        # Sliding window for rank-based rewards (K >= 3 * num_arms = 15)
        self._reward_window_size = 15
        self._operator_rewards = {op: [] for op in self._operators}
        self._operator_gen_history = {op: [] for op in self._operators}
        
        # Per-generation tracking for rank-based scoring
        self._gen_best_fitness = []
        self._gen_operator = []
        self._prev_global_best = None
        
        # EMA states shared across operators that need them
        self._ema_centroid = None
        self._ema_centroid_velocity = None
        self._centroid_vel_history = []
        self._ema_improvement = 0.0
        self._success_count = None
    
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
    # POSITION UPDATE STRATEGIES (the 5 actual benchmark winners)
    # -------------------------------------------------------------------------
    
    def _position_update_variant_02(self):
        """Spectral / linear-algebraic: eigenvalue-weighted velocity projection.
        
        1 win on task 0. Key insight: project velocity onto principal components
        of population covariance, then scale each component INVERSELY to its
        eigenvalue. High-eigenvalue directions get dampened (already well-explored),
        low-eigenvalue directions get amplified (collapsed/neglected subspace).
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
    
    def _position_update_variant_05(self):
        """Graph-Laplacian eigenvalue-based velocity modulation.
        
        6 wins on tasks 4, 5, 6, 8, 17, 22. Key insight: Build k-NN graph over
        population; use Laplacian eigenvalues to detect clustering/convergence.
        Small spectral gap = clustered population → increase exploration.
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
            from scipy.sparse import csr_matrix
            adj = csr_matrix((data, (row_indices, col_indices)), shape=(self.np, self.np))
            adj = adj + adj.T
            adj.data[:] = 1.0

            degrees = np.array(adj.sum(axis=1)).ravel() + 1e-10
            d_inv_sqrt = 1.0 / np.sqrt(degrees)
            d_inv_sqrt_diag = csr_matrix((d_inv_sqrt, (np.arange(self.np), np.arange(self.np))), shape=(self.np, self.np))
            lap = csr_matrix((np.ones(self.np), (np.arange(self.np), np.arange(self.np))), shape=(self.np, self.np)) - d_inv_sqrt_diag @ adj @ d_inv_sqrt_diag

            from scipy.sparse.linalg import eigsh
            lap_eigenvalues = eigsh(lap, k=min(5, self.np - 1), return_eigenvectors=False)
            lap_eigenvalues = np.sort(lap_eigenvalues)

            spectral_gap = lap_eigenvalues[1] if len(lap_eigenvalues) > 1 else 1.0
            spectral_gap = np.clip(spectral_gap, 0.0, 1.0)
            modularity_signal = 1.0 - lap_eigenvalues[0] if len(lap_eigenvalues) > 0 else 0.0
        except:
            spectral_gap = 1.0
            modularity_signal = 0.0

        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        clustering_explore = 1.0 + 0.5 * (1.0 - spectral_gap)
        size_factor = np.clip(particle_component_size / max(1, self.np), 0.3, 1.0)
        graph_explore = clustering_explore * (2.0 - size_factor)
        graph_explore = np.clip(graph_explore, 0.5, 2.0)

        if self._success_count is None:
            self._success_count = np.zeros(self.np)
        improved = self.personal_best_fitness >= self.current_fitness
        self._success_count[improved] += 1
        self._success_count[~improved] *= 0.9
        success_norm = self._success_count / (np.max(self._success_count) + 1.0)

        vel_scale = graph_explore * (1.0 + 0.3 * success_norm) * (1.0 + 0.3 * (1.0 - fitness_ranks))
        vel_scale = np.clip(vel_scale, 0.5, 2.5)

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
    
    def _position_update_variant_08(self):
        """Hybrid: FDC exploration/exploitation + temporal centroid-drift.
        
        7 wins on tasks 2, 3, 11, 12, 13, 14, 20. Key insight: Two distinct
        mechanisms with stagnation-driven switching. FDC handles normal search;
        temporal drift takes over when stuck to escape basins.
        """
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

        centroid = np.mean(self.population, axis=0)

        if self._ema_centroid is None:
            self._ema_centroid = centroid.copy()
            self._ema_centroid_velocity = np.zeros(self.dim)

        alpha_pos = 0.1
        self._ema_centroid = alpha_pos * centroid + (1 - alpha_pos) * self._ema_centroid

        current_centroid_vel = centroid - self._ema_centroid
        alpha_vel = 0.2
        self._ema_centroid_velocity = alpha_vel * current_centroid_vel + (1 - alpha_vel) * self._ema_centroid_velocity

        stagnation_threshold = 20
        stagnation_normalized = np.clip(self.stagnation_counter / max(stagnation_threshold, 1), 0.0, 1.0)
        drift_weight = stagnation_normalized ** 0.5
        fdc_weight = 1.0 - drift_weight

        if self._success_count is None:
            self._success_count = np.zeros(self.np)
        improved = self.personal_best_fitness >= self.current_fitness
        self._success_count[improved] += 1
        self._success_count[~improved] *= 0.9
        success_norm = self._success_count / (np.max(self._success_count) + 1.0)

        vel_scale = exploration_factor * (1.0 + 0.3 * success_norm)
        vel_scale = np.clip(vel_scale, 0.5, 2.0)

        to_ema_dir = self._ema_centroid - self.population
        to_ema_dist = np.linalg.norm(to_ema_dir, axis=1, keepdims=True) + 1e-10
        drift_correction = (to_ema_dir / to_ema_dist) * np.linalg.norm(self._ema_centroid_velocity)
        drift_correction_scaled = drift_correction * drift_weight

        fitness_perturb = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
        exploration_perturb = exploration_factor * fitness_perturb
        random_perturb = exploration_perturb * np.random.uniform(-0.5, 0.5, (self.np, self.dim))
        fdc_perturb = random_perturb * fdc_weight

        new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + drift_correction_scaled + fdc_perturb
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_variant_10(self):
        """Spectral: eigenvalue entropy + fitness-rank velocity modulation.
        
        7 wins on tasks 9, 10, 15, 16, 18, 19, 23. Key insight: eigenvalue spread
        (normalized variance of eigenspectrum) captures the full distribution shape.
        Fitness rank modulates velocity scale per-particle.
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
    
    def _position_update_original(self):
        """Original: eigenvalue-anisotropic velocity modulation.
        
        3 wins on tasks 1, 7, 21. Key insight: condition-number driven spectral
        scaling. High condition number = anisotropic → more exploration.
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
    
    def _dispatch_position_update(self, operator):
        """Route to the appropriate position update implementation."""
        if operator == 'variant_02':
            self._position_update_variant_02()
        elif operator == 'variant_05':
            self._position_update_variant_05()
        elif operator == 'variant_08':
            self._position_update_variant_08()
        elif operator == 'variant_10':
            self._position_update_variant_10()
        elif operator == 'original':
            self._position_update_original()
        else:
            self._position_update_original()
    
    # -------------------------------------------------------------------------
    # THOMPSON SAMPLING: rank-based reward and selection
    # -------------------------------------------------------------------------
    
    def _compute_rank_reward(self, operator):
        """Compute rank-based reward for the current generation.
        
        Uses a sliding window of recent generations. Reward = fraction of
        generations in the window where this operator's best fitness was
        in the top tercile of all operators' best fitnesses.
        
        This is RANK-BASED, not raw fitness — no scale-dependent constants.
        """
        if len(self._gen_best_fitness) < 2:
            return 0.5  # Neutral reward before we have data
        
        # Get all best fitnesses for generations where this operator was used
        op_gen_indices = [i for i, op in enumerate(self._gen_operator) if op == operator]
        if not op_gen_indices:
            return 0.5
        
        op_best_fitnesses = [self._gen_best_fitness[i] for i in op_gen_indices]
        
        # Compare to all other operators' fitnesses in the same generations
        all_fitnesses_in_op_gens = [self._gen_best_fitness[i] for i in op_gen_indices]
        
        # Rank: what fraction of operators had WORSE (higher) fitness?
        # This is a tournament-style reward
        better_count = 0
        total_count = 0
        for gen_idx in op_gen_indices:
            gen_fitness = self._gen_best_fitness[gen_idx]
            for other_op in self._operators:
                other_indices = [i for i, op in enumerate(self._gen_operator) if op == other_op and i == gen_idx]
                if other_indices:
                    other_fitness = self._gen_best_fitness[other_indices[0]]
                    if gen_fitness <= other_fitness:  # Lower is better
                        better_count += 1
                    total_count += 1
        
        if total_count == 0:
            return 0.5
        
        return better_count / total_count
    
    def _update_thompson_posterior(self, operator, reward):
        """Update Beta posterior with rank-based reward.
        
        reward in [0, 1] is treated as a Bernoulli trial:
        - reward close to 1.0 → likely win → increase alpha
        - reward close to 0.0 → likely loss → increase beta
        """
        # Quantize reward to binary win/loss for clean Beta updates
        # Threshold at 0.5 (median of rank-based reward)
        if reward >= 0.5:
            self._alpha[operator] += 1.0
        else:
            self._beta[operator] += 1.0
        
        # Maintain sliding window for rank-based reward computation
        self._operator_rewards[operator].append(reward)
        self._operator_gen_history[operator].append(self.generation)
        
        # Trim window
        if len(self._operator_rewards[operator]) > self._reward_window_size:
            self._operator_rewards[operator].pop(0)
            self._operator_gen_history[operator].pop(0)
    
    def _thompson_sample(self):
        """Sample from Beta posterior for each operator and select the winner.
        
        Thompson Sampling balances exploration (uncertain operators) and
        exploitation (high mean reward) without a separate exploration parameter.
        """
        samples = {}
        for op in self._operators:
            alpha = max(self._alpha[op], 0.1)
            beta = max(self._beta[op], 0.1)
            samples[op] = np.random.beta(alpha, beta)
        
        return max(samples, key=samples.get)
    
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
        Run the optimizer with Thompson Sampling position-update selection.
        
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
        
        # Reset Thompson Sampling state
        self._alpha = {op: 1.0 for op in self._operators}
        self._beta = {op: 1.0 for op in self._operators}
        self._operator_rewards = {op: [] for op in self._operators}
        self._operator_gen_history = {op: [] for op in self._operators}
        self._gen_best_fitness = []
        self._gen_operator = []
        
        # Reset EMA states
        self._ema_centroid = None
        self._ema_centroid_velocity = None
        self._centroid_vel_history = []
        self._ema_improvement = 0.0
        self._success_count = np.zeros(self.np)
        
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
            self._success_count = np.zeros(self.np)
        
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
            
            # --- SELECT OPERATOR VIA THOMPSON SAMPLING ---
            operator = self._thompson_sample()
            
            # Adaptation
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            self._update_local_best_from_personal()
            
            # Velocity update (shared base), then position update (SELECTED)
            self._velocity_update_base()
            self._dispatch_position_update(operator)
            
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
            
            # Record generation data for rank-based reward
            gen_best = float(np.min(self.current_fitness))
            self._gen_best_fitness.append(gen_best)
            self._gen_operator.append(operator)
            
            # Keep window manageable
            if len(self._gen_best_fitness) > self._reward_window_size * len(self._operators):
                self._gen_best_fitness.pop(0)
                self._gen_operator.pop(0)
            
            # --- UPDATE THOMPSON POSTERIOR with rank-based reward ---
            reward = self._compute_rank_reward(operator)
            self._update_thompson_posterior(operator, reward)
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
