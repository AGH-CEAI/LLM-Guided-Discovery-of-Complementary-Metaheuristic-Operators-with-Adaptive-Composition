import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    MECHANISM: Probe-and-Commit (Specialized Contextual Bandit, #4)
    
    Per-task gap table analysis:
    - Task 5 has a 13917.6× gap to median competitor — BLEND-HOSTILE.
    - 6 distinct winners across 24 tasks with no single operator dominating.
    - Ensemble blending mathematically CANNOT preserve per-task advantages
      when winner beats median by 100×+.
    
    Therefore: a PROBE-AND-COMMIT dispatcher is used.
    
    PHASE A (PROBE, ~8% of budget):
      Run a short exploratory burst with each candidate operator using a
      mini-subpopulation. Extract CHEAP LANDSCAPE FINGERPRINTS:
        - Convergence rate (log-best slope over probe gens)
        - Effective dimensionality (eigenvalue analysis)
        - Rank-entropy (ruggedness proxy)
        - Stagnation fraction
      Rank-normalize these into a composite score per operator.
    
    PHASE B (COMMIT, ~92% of budget):
      Commit exclusively (weight 1.0) to the operator that won the probe.
      This preserves per-task advantages without blending degradation.
      Keep ε=0.05 probability of revisiting runner-up if committed operator
      stagnates for >20 generations.
    
    This is NOT a bandit-over-operators — it is a landscape fingerprinting
    and commit mechanism, appropriate when per-task gaps are blend-hostile.
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
        
        # --- PROBE-AND-COMMIT STATE ---
        # Candidates: original + 5 winning variants
        self._candidates = [
            'original',
            'variant_01_catA',  # geometric layout: centroid + k-NN repulsion
            'variant_03_catC',  # entropy-modulated distribution-guided exploration
            'variant_04_catD',  # fitness-rank-adaptive with mutation injection
            'variant_05_catE',  # k-NN graph topology + betweenness-based routing
            'variant_10_catB',  # spectral-condition-guided anisotropic mutation
        ]
        self._probe_budget_pct = 0.08  # 8% of budget for probing
        self._selected_candidate = None
        self._probe_scores = None
        self._probe_generations = 0
        self._in_probe_phase = True
        self._probe_candidate_idx = 0
        self._probe_results = {}  # candidate -> list of (best_fitness, diversity, stagnation)
        self._fallback_epsilon = 0.05  # small probability to revisit runner-up on stagnation
        self._stagnation_since_commit = 0
        
    def _initialize_population(self):
        """Initialize swarm with uniform random distribution."""
        range_width = self.upper_bound - self.lower_bound
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
        
        # Handle budget exhaustion gracefully
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
            # Ring neighborhood indices
            left = (i - self.neighborhood_size) % self.np
            right = (i + self.neighborhood_size + 1) % self.np
            
            if left < right:
                neighborhood_indices = np.arange(left, right)
            else:
                neighborhood_indices = np.concatenate([
                    np.arange(left, self.np), np.arange(0, right)
                ])
            
            # Include self in neighborhood comparison
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
            return True  # improvement occurred
        else:
            self.stagnation_counter += 1
            return False
    
    def _compute_diversity(self):
        """Compute swarm diversity as average Euclidean distance to centroid."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(distances)
    
    def _compute_effective_dimensionality(self):
        """Compute effective dimensionality from population covariance eigenvalues."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            total_var = np.sum(eigenvalues)
            if total_var < 1e-10:
                return 1.0
            cumvar = np.cumsum(eigenvalues) / total_var
            eff_dim = np.searchsorted(cumvar, 0.95) + 1
            return eff_dim / self.dim  # normalize to [0, 1]
        except:
            return 0.5  # fallback
    
    def _compute_rank_entropy(self):
        """Compute rank entropy as proxy for landscape ruggedness."""
        if self.global_best_fitness == np.inf:
            return 0.0
        try:
            # Compare current fitness ranks to previous ranks
            ranks = np.argsort(np.argsort(self.current_fitness))
            n = len(ranks)
            # Uniform distribution has max entropy
            max_entropy = np.log(n)
            # Compute empirical entropy of rank distribution
            hist, _ = np.histogram(ranks, bins=n, range=(0, n))
            hist = hist[hist > 0] / n
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            return entropy / (max_entropy + 1e-10)  # normalize
        except:
            return 0.5
    
    def _adapt_inertia_weight(self):
        """Adapt inertia weight based on swarm diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            # Low diversity: increase exploration
            self.inertia_weight = min(0.95, self.inertia_weight * 1.05)
        elif diversity > self.diversity_threshold_high:
            # High diversity: increase exploitation
            self.inertia_weight = max(0.4, self.inertia_weight * 0.95)
        else:
            # Normal range: gradual convergence
            self.inertia_weight = 0.729 - 0.1 * (self.generation / 1000)
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.95)
    
    def _adapt_neighborhood_size(self):
        """Adapt ring topology neighborhood size based on diversity."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            # Increase neighborhood size for more diversity
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif diversity > self.diversity_threshold_high:
            # Decrease neighborhood size for faster convergence
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    def _adaptive_coefficients(self):
        """Compute adaptive cognitive and social coefficients."""
        # Adapt cognitive coefficient based on personal best improvement
        improvement_ratio = 1.0 / (1.0 + self.stagnation_counter)
        cognitive = self.cognitive_base * (1.0 + 0.5 * improvement_ratio)
        
        # Adapt social coefficient inversely to cognitive
        social = self.social_base * (2.0 - improvement_ratio)
        
        return cognitive, social
    
    def _velocity_update_original(self):
        """Original velocity update with DE mutation."""
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
    
    def _velocity_update_variant_01_catA(self):
        """Update velocities using geometric layout: centroid, k-NN repulsion, spread."""
        centroid = np.mean(self.population, axis=0)
        
        min_pos = np.min(self.population, axis=0)
        max_pos = np.max(self.population, axis=0)
        spread = max_pos - min_pos + 1e-10
        
        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive = self.cognitive_base * (1.0 + 0.1 * np.exp(-dist_to_centroid / 50.0))
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        
        social = self.social_base * (1.0 + 0.1 * np.exp(-dist_to_centroid / 50.0))
        social_component = social * r2 * (self.local_best - self.population)
        
        centroid_direction = centroid - self.population
        centroid_strength = np.exp(-dist_to_centroid / 100.0)
        centroid_component = 0.3 * centroid_strength * centroid_direction
        
        k = min(5, self.np - 1)
        knn_repulsion = np.zeros((self.np, self.dim))
        for i in range(self.np):
            diffs = self.population - self.population[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf
            nn_indices = np.argpartition(dists, k)[:k]
            nn_dists = dists[nn_indices]
            for j, nn_idx in enumerate(nn_indices):
                if nn_dists[j] > 1e-10:
                    diff = self.population[i] - self.population[nn_idx]
                    knn_repulsion[i] += diff / (nn_dists[j] ** 2 + 1e-10)
        knn_component = 0.1 * knn_repulsion
        
        spread_modulation = np.mean(spread) / (spread + 1e-10)
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            centroid_component +
            knn_component
        )
        new_velocity = new_velocity * spread_modulation
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_variant_03_catC(self):
        """Update velocities with entropy-modulated distribution-guided exploration."""
        cognitive, social = self._adaptive_coefficients()
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        mean_pop = np.mean(self.population, axis=0)
        centered = self.population - mean_pop
        
        try:
            cov_pop = np.cov(centered.T)
            cov_pop += np.eye(self.dim) * 1e-8
            
            eigenvalues, eigenvectors = np.linalg.eigh(cov_pop)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            
            log_det = np.sum(np.log(eigenvalues))
            entropy = 0.5 * (self.dim * np.log(2 * np.pi * np.e) + log_det)
            
            max_entropy = self.dim * np.log(self.upper_bound - self.lower_bound + 1e-10)
            min_entropy = self.dim * np.log(1e-6)
            entropy_normalized = np.clip((entropy - min_entropy) / (max_entropy - min_entropy + 1e-10), 0, 1)
            
            dispersive_strength = 0.5 * (1.0 - entropy_normalized)
            
            inv_sqrt_eigen = 1.0 / np.sqrt(eigenvalues + 1e-10)
            scaled_diff = centered * inv_sqrt_eigen
            mahal_dist = np.linalg.norm(scaled_diff, axis=1, keepdims=True)
            
            max_mahal = np.max(mahal_dist) + 1e-10
            mahal_normalized = mahal_dist / max_mahal
            
            to_mean = mean_pop - self.population
            to_mean_norm = np.linalg.norm(to_mean, axis=1, keepdims=True) + 1e-10
            to_mean_dir = to_mean / to_mean_norm
            
            dispersive_component = (
                dispersive_strength *
                mahal_normalized *
                to_mean_dir *
                np.random.uniform(0, 1, (self.np, self.dim))
            )
            
            random_explore = 0.3 * (1.0 - entropy_normalized) * np.random.uniform(-1, 1, (self.np, self.dim))
            entropy_velocity = dispersive_component + random_explore
            
        except np.linalg.LinAlgError:
            entropy_velocity = 0.1 * np.random.uniform(-1, 1, (self.np, self.dim))
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            entropy_velocity
        )
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_variant_04_catD(self):
        """Update velocities with fitness-rank-adaptive components."""
        if self.global_best is not None:
            distances = np.linalg.norm(self.population - self.global_best, axis=1)
            
            if np.std(distances) > 1e-10 and np.std(self.current_fitness) > 1e-10:
                corr = np.corrcoef(
                    np.argsort(np.argsort(self.current_fitness)),
                    np.argsort(np.argsort(distances))
                )[0, 1]
            else:
                corr = 0.0
        else:
            corr = 0.0
        
        social = self.social_base * np.clip(1.0 + corr, 0.2, 1.5)
        cognitive = self.cognitive_base
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        mutation_strength = 0.3 * np.clip(1.0 - corr, 0.1, 1.0)
        
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
            mutation_strength * (mutation_vectors - self.population),
            0.0
        )
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_variant_05_catE(self):
        """Update velocities using k-NN graph topology and betweenness-based social routing."""
        cognitive, social = self._adaptive_coefficients()
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        
        k = min(max(3, self.neighborhood_size), self.np - 1)
        
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]
        
        adjacency = np.zeros((self.np, self.np), dtype=np.float64)
        for i in range(self.np):
            adjacency[i, knn_indices[i]] = 1.0
            adjacency[knn_indices[i], i] = 1.0
        
        degree = np.sum(adjacency, axis=1)
        degree_norm = degree / (degree.max() + 1e-10)
        
        d_sqrt_inv = np.diag(1.0 / (np.sqrt(degree) + 1e-10))
        laplacian = np.eye(self.np) - d_sqrt_inv @ adjacency @ d_sqrt_inv
        
        np.random.seed(42)
        v = np.random.randn(self.np)
        v = v / (np.linalg.norm(v) + 1e-10)
        for _ in range(20):
            v = laplacian @ v
            v = v / (np.linalg.norm(v) + 1e-10)
        
        fiedler = np.abs(v @ laplacian @ v) / (np.dot(v, v) + 1e-10)
        connectivity_strength = np.clip(fiedler * 5.0, 0.1, 2.0)
        
        betweenness = np.zeros(self.np)
        n_samples = min(50, self.np)
        sample_nodes = np.random.choice(self.np, n_samples, replace=False)
        
        for src in sample_nodes:
            dist = np.full(self.np, np.inf)
            pred = [[] for _ in range(self.np)]
            dist[src] = 0
            queue = [src]
            
            while queue:
                curr = queue.pop(0)
                for nb in knn_indices[curr]:
                    if dist[nb] == np.inf:
                        dist[nb] = dist[curr] + 1
                        queue.append(nb)
                    if dist[nb] == dist[curr] + 1:
                        pred[nb].append(curr)
            
            sigma = np.zeros(self.np)
            sigma[src] = 1
            for d in range(int(dist.max()) + 1) if dist.max() < np.inf else []:
                for node in np.where(dist == d)[0]:
                    for p in pred[node]:
                        sigma[node] += sigma[p]
            
            for node in range(self.np):
                if node != src and dist[node] < np.inf:
                    for p in pred[node]:
                        betweenness[node] += sigma[p] / (sigma[node] + 1e-10)
        
        betweenness_norm = betweenness / (betweenness.max() + 1e-10)
        
        social_component = np.zeros((self.np, self.dim))
        
        for i in range(self.np):
            neighbor_best_idx = knn_indices[i][np.argmin(self.personal_best_fitness[knn_indices[i]])]
            neighbor_best_pos = self.personal_best[neighbor_best_idx]
            
            influence_weight = 0.5 * (1.0 + degree_norm[i]) * (1.0 + betweenness_norm[i])
            social_scaled = social * connectivity_strength
            
            social_component[i] = social_scaled * r2[i] * influence_weight * (neighbor_best_pos - self.population[i])
        
        if fiedler < 0.2:
            rescue_strength = np.clip((0.2 - fiedler) * 3.0, 0.0, 0.8)
            if self.global_best is not None:
                rescue = rescue_strength * (self.global_best - self.population)
            else:
                best_third = np.argsort(self.personal_best_fitness)[:max(1, self.np // 3)]
                centroid = np.mean(self.population[best_third], axis=0)
                rescue = rescue_strength * (centroid - self.population)
            social_component += rescue
        
        mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mutation_threshold = 0.05 * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold
        
        mutation_vectors = np.zeros((self.np, self.dim))
        for i in range(self.np):
            indices = np.random.choice([j for j in range(self.np) if j != i], 3, replace=False)
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]]
            )
        
        mutation_component = np.where(mutation_active, 0.2 * (mutation_vectors - self.population), 0.0)
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            mutation_component
        )
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_variant_10_catB(self):
        """Update velocities with spectral-condition-guided anisotropic mutation."""
        cognitive, social = self._adaptive_coefficients()
        
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        
        cognitive_component = cognitive * r1 * (self.personal_best - self.population)
        social_component = social * r2 * (self.local_best - self.population)
        
        centered = self.population - np.mean(self.population, axis=0)
        
        try:
            _, singular_values, right_sv = np.linalg.svd(centered, full_matrices=False)
            
            cond = singular_values[0] / (singular_values[-1] + 1e-10)
            mutation_strength = np.clip(0.5 / (1.0 + 0.1 * np.log1p(cond)), 0.1, 0.5)
            
            total_variance = np.sum(singular_values ** 2) + 1e-10
            cumvar = np.cumsum(singular_values ** 2) / total_variance
            effective_dim = np.searchsorted(cumvar, 0.95) + 1
            
            if effective_dim < self.dim * 0.5:
                principal_axes = right_sv[:effective_dim].T
                parallel_component = self.population @ principal_axes @ principal_axes.T
                orthogonal_residual = self.population - parallel_component
                orthogonal_strength = 0.3 * (1.0 - effective_dim / self.dim)
                mutation_component = orthogonal_strength * orthogonal_residual
            else:
                mutation_component = np.zeros((self.np, self.dim))
            
            mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
            mutation_threshold = mutation_strength * (1.0 - self.generation / 5000)
            mutation_active = mutation_mask < mutation_threshold
            
            mutation_vectors = np.zeros((self.np, self.dim))
            for i in range(self.np):
                indices = np.random.choice(
                    [j for j in range(self.np) if j != i], 3, replace=False
                )
                mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                    self.population[indices[1]] - self.population[indices[2]]
                )
            
            de_component = np.where(
                mutation_active,
                mutation_strength * (mutation_vectors - self.population),
                0.0
            )
            
            total_mutation = mutation_component + de_component
            
        except np.linalg.LinAlgError:
            mutation_mask = np.random.uniform(0, 1, (self.np, self.dim))
            mutation_threshold = 0.3 * (1.0 - self.generation / 5000)
            mutation_active = mutation_mask < mutation_threshold
            
            mutation_vectors = np.zeros((self.np, self.dim))
            for i in range(self.np):
                indices = np.random.choice(
                    [j for j in range(self.np) if j != i], 3, replace=False
                )
                mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                    self.population[indices[1]] - self.population[indices[2]]
                )
            
            total_mutation = np.where(
                mutation_active,
                0.3 * (mutation_vectors - self.population),
                0.0
            )
        
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component +
            social_component +
            total_mutation
        )
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    def _velocity_update_batch(self):
        """Dispatch to the selected velocity update strategy."""
        if self._selected_candidate is None:
            self._velocity_update_original()
        elif self._selected_candidate == 'original':
            self._velocity_update_original()
        elif self._selected_candidate == 'variant_01_catA':
            self._velocity_update_variant_01_catA()
        elif self._selected_candidate == 'variant_03_catC':
            self._velocity_update_variant_03_catC()
        elif self._selected_candidate == 'variant_04_catD':
            self._velocity_update_variant_04_catD()
        elif self._selected_candidate == 'variant_05_catE':
            self._velocity_update_variant_05_catE()
        elif self._selected_candidate == 'variant_10_catB':
            self._velocity_update_variant_10_catB()
        else:
            self._velocity_update_original()
    
    def _position_update_batch(self):
        """Update positions with eigenvalue-anisotropic velocity modulation."""
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
    
    def _compute_probe_fingerprints(self):
        """Compute landscape fingerprints from probe phase data.
        
        Returns a dict of fingerprints for the committed operator selection.
        """
        fingerprints = {}
        
        for cand, results in self._probe_results.items():
            if len(results) < 3:
                continue
            
            best_fitnesses = [r[0] for r in results if not np.isinf(r[0]) and not np.isnan(r[0])]
            diversities = [r[1] for r in results if not np.isnan(r[1])]
            stagnations = [r[2] for r in results]
            
            # Convergence rate: negative slope of log(best_fitness)
            convergence_score = 0.0
            if len(best_fitnesses) >= 3:
                valid_fits = [f for f in best_fitnesses if f > 0]
                if len(valid_fits) >= 3:
                    log_fits = np.log(np.array(valid_fits) + 1e-10)
                    if len(log_fits) >= 2:
                        slopes = np.diff(log_fits)
                        convergence_score = -np.mean(slopes)  # negative = improving
            
            # Diversity score: higher diversity = better exploration
            diversity_score = np.mean(diversities) if diversities else 0.0
            
            # Stagnation score: lower is better
            stagnation_score = np.mean(stagnations) if stagnations else 1.0
            
            # Effective dimensionality preservation
            eff_dim_score = np.mean([r[3] for r in results if len(r) > 3 and not np.isnan(r[3])]) if any(len(r) > 3 for r in results) else 0.5
            
            fingerprints[cand] = {
                'convergence': convergence_score,
                'diversity': diversity_score,
                'stagnation': stagnation_score,
                'effective_dim': eff_dim_score,
            }
        
        return fingerprints
    
    def _rank_normalize_scores(self, fingerprints):
        """Rank-normalize fingerprint scores for fair comparison.
        
        Each metric is ranked 0..(n-1) and then combined. This avoids
        scale-dependent magic constants.
        """
        if not fingerprints:
            return {}
        
        metrics = ['convergence', 'diversity', 'stagnation', 'effective_dim']
        ranked = {cand: {} for cand in fingerprints}
        
        for metric in metrics:
            values = [(cand, fingerprints[cand][metric]) for cand in fingerprints]
            
            # For stagnation, lower is better; for others, higher is better
            if metric == 'stagnation':
                values.sort(key=lambda x: x[1])  # ascending for lower=better
            else:
                values.sort(key=lambda x: -x[1])  # descending for higher=better
            
            for rank, (cand, _) in enumerate(values):
                ranked[cand][metric] = rank / max(1, len(values) - 1)  # normalize to [0, 1]
        
        # Composite score: weighted combination of rank-normalized metrics
        weights = {'convergence': 0.35, 'diversity': 0.25, 'stagnation': 0.25, 'effective_dim': 0.15}
        composite = {}
        for cand in fingerprints:
            composite[cand] = sum(ranked[cand][m] * weights[m] for m in metrics)
        
        return composite
    
    def _select_best_candidate(self, scores):
        """Select the candidate with the highest composite score."""
        if not scores:
            return 'original'
        
        best_cand = max(scores, key=lambda c: scores[c])
        return best_cand
    
    def _should_try_alternative(self):
        """Determine if we should try a runner-up candidate on stagnation."""
        if self._selected_candidate is None or len(self._probe_results) < 2:
            return False
        
        if self._stagnation_since_commit > 20 and np.random.random() < self._fallback_epsilon:
            return True
        return False
    
    def _get_alternative_candidate(self):
        """Get a runner-up candidate to try."""
        if not self._probe_scores:
            return self._selected_candidate
        
        sorted_cands = sorted(self._probe_scores.keys(), key=lambda c: -self._probe_scores[c])
        if len(sorted_cands) < 2:
            return self._selected_candidate
        
        # Return the second-best candidate
        return sorted_cands[1]
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer with Probe-and-Commit adaptive operator selection.
        
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
        self._probe_results = {}
        self._probe_scores = None
        self._selected_candidate = None
        self._in_probe_phase = True
        self._probe_candidate_idx = 0
        self._stagnation_since_commit = 0
        
        # Initial evaluation
        self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
        
        # Handle truncated initial evaluation
        if actual_n < self.np:
            self.np = actual_n
            self.population = self.population[:actual_n]
            self.velocity = self.velocity[:actual_n]
            self.personal_best = self.personal_best[:actual_n]
            self.personal_best_fitness = self.personal_best_fitness[:actual_n]
            self.local_best = self.local_best[:actual_n]
            self.local_best_fitness = self.local_best_fitness[:actual_n]
            self.current_fitness = self.current_fitness[:actual_n]
        
        # Check initial stopping condition
        if stopping_condition():
            return self.global_best_fitness, self.global_best
        
        # Initialize bests from initial evaluation
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # Track probe budget
        max_generations_estimate = max(1000, 50 * self.dim)  # estimate
        probe_gens_per_candidate = max(3, int(max_generations_estimate * self._probe_budget_pct / len(self._candidates)))
        total_probe_gens = probe_gens_per_candidate * len(self._candidates)
        
        # PHASE A: PROBE
        probe_gen_count = 0
        
        while self._in_probe_phase and not stopping_condition():
            self.generation += 1
            probe_gen_count += 1
            
            # Determine current candidate for this generation
            current_cand = self._candidates[self._probe_candidate_idx]
            
            # Track which operator we're using
            old_selected = self._selected_candidate
            self._selected_candidate = current_cand
            
            # Record fingerprint before update
            best_fit_before = self.global_best_fitness
            diversity_before = self._compute_diversity()
            stagnation_before = self.stagnation_counter
            eff_dim_before = self._compute_effective_dimensionality()
            
            # Adaptation phase
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            
            # Update local bests before velocity update
            self._update_local_best_from_personal()
            
            # Velocity and position update (uses the current candidate)
            self._velocity_update_batch()
            self._position_update_batch()
            
            # Evaluate new population
            self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
            
            if actual_n < self.np:
                self.np = actual_n
                self.population = self.population[:actual_n]
                self.velocity = self.velocity[:actual_n]
                self.current_fitness = self.current_fitness[:actual_n]
                break
            
            if stopping_condition():
                break
            
            # Update bests
            self._update_personal_best_batch()
            self._compute_local_best_batch()
            improved = self._compute_global_best()
            
            # Record probe result
            best_fit_after = self.global_best_fitness
            diversity_after = self._compute_diversity()
            
            if current_cand not in self._probe_results:
                self._probe_results[current_cand] = []
            
            self._probe_results[current_cand].append((
                best_fit_after,
                diversity_after,
                self.stagnation_counter,
                eff_dim_before
            ))
            
            # Restart check
            self._restart_if_stagnant()
            
            # Move to next candidate after probe_gens_per_candidate generations
            if probe_gen_count >= probe_gens_per_candidate:
                probe_gen_count = 0
                self._probe_candidate_idx += 1
                
                # Reinitialize population between candidates to avoid bias
                self._initialize_population()
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
                self._update_personal_best_batch()
                self._compute_local_best_batch()
                self._compute_global_best()
                self.generation = 0
                self.stagnation_counter = 0
            
            # Check if all candidates have been probed
            if self._probe_candidate_idx >= len(self._candidates):
                self._in_probe_phase = False
        
        # PHASE B: COMMIT
        if not self._in_probe_phase and self._selected_candidate is None:
            # Compute fingerprints and select best candidate
            fingerprints = self._compute_probe_fingerprints()
            self._probe_scores = self._rank_normalize_scores(fingerprints)
            self._selected_candidate = self._select_best_candidate(self._probe_scores)
            self._stagnation_since_commit = 0
        
        # Main optimization loop (commit phase)
        while not stopping_condition():
            self.generation += 1
            
            # Check if we should try an alternative on stagnation
            if self._should_try_alternative():
                alt_cand = self._get_alternative_candidate()
                if alt_cand != self._selected_candidate:
                    self._selected_candidate = alt_cand
            
            # Adaptation phase
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            
            # Update local bests before velocity update
            self._update_local_best_from_personal()
            
            # Velocity and position update
            self._velocity_update_batch()
            self._position_update_batch()
            
            # Evaluate new population
            self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
            
            if actual_n < self.np:
                self.np = actual_n
                self.population = self.population[:actual_n]
                self.velocity = self.velocity[:actual_n]
                self.current_fitness = self.current_fitness[:actual_n]
                break
            
            if stopping_condition():
                break
            
            # Update bests
            self._update_personal_best_batch()
            self._compute_local_best_batch()
            improved = self._compute_global_best()
            
            # Track stagnation for alternative operator switching
            if improved:
                self._stagnation_since_commit = 0
            else:
                self._stagnation_since_commit += 1
            
            # Restart check
            self._restart_if_stagnant()
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
