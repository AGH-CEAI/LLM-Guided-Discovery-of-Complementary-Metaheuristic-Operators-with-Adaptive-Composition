```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: THOMPSON SAMPLING (Discrete Arm Selection, mechanism #1)
    
    The gap table is BLEND-VIABLE but NOT blend-optimal:
      - Maximum gap is 2.8× (Task 5), well below the 10× blend-hostile threshold.
      - However, 5 distinct winners exist with wins spread 1/3/6/7/7.
      - No single operator dominates; different strategies suit different landscapes.
    
    Thompson Sampling is chosen because:
      1. Operators are mutually exclusive (one position update per generation).
      2. Credit assignment is clean: did the generation improve global best?
      3. Rank-based reward (best_fitness in top tercile of last K gens) is
         scale-independent — no magic constants like *1e5.
      4. Probabilistic selection naturally balances exploration/exploitation
         without epsilon-greedy decay schedules.
    
    The dispatcher preserves per-task winner advantages because it can commit
    to a single operator with probability ~1.0 when one clearly dominates
    (Beta(α,β) → peaked at 1.0). No hard-coded weight caps.
    
    Operators (from benchmark winners):
      - 'original': eigenvalue-anisotropic velocity (3 wins)
      - 'spectral': eigenvalue-weighted projection (1 win)  
      - 'graph': graph-Laplacian k-NN modulation (6 wins)
      - 'hybrid': FDC + temporal centroid-drift (7 wins)
      - 'entropy': eigenvalue entropy + fitness-rank (7 wins)
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5 * dim, capped
        self.np = min(max(5 * dim, 20), 300)
        
        # Ring topology neighborhood size (adapts)
        self.neighborhood_size = max(3, dim // 5)
        
        # Acceleration coefficients (adapt)
        self.cognitive_base = 1.496
        self.social_base = 1.496
        
        # Velocity bounds
        self.v_max = 0.2 * (self.upper_bound - self.lower_bound)
        self.v_min = -self.v_max
        
        # Diversity thresholds
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
        # Beta distribution parameters: Beta(alpha, beta)
        # alpha = 1 + successes, beta = 1 + failures
        self._op_alpha = {}
        self._op_beta = {}
        
        # The 5 benchmark-winning position update strategies
        self._operators = [
            'original',   # eigenvalue-anisotropic velocity (3 wins)
            'spectral',   # eigenvalue-weighted projection (1 win)
            'graph',      # graph-Laplacian k-NN (6 wins)
            'hybrid',     # FDC + temporal centroid-drift (7 wins)
            'entropy',    # eigenvalue entropy + fitness-rank (7 wins)
        ]
        for op in self._operators:
            self._op_alpha[op] = 1.0   # Uniform prior: Beta(1,1)
            self._op_beta[op] = 1.0
        
        # Rank-based scoring window
        self._rank_window_size = max(3 * len(self._operators), 15)
        self._rank_fitness_history = []  # (best_fitness, operator) tuples
        
        # Current operator (for Thompson Sampling)
        self._selected_operator = self._operators[0]
        
        # EMA states for hybrid operator
        self._ema_centroid = None
        self._ema_centroid_velocity = None
        self._ema_improvement = 0.0
        
        # Success tracking per particle
        self._success_count = None
        
        # Graph-Laplacian state
        self._centroid_vel_history = None
    
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
        
        self._success_count = np.zeros(self.np)
        self._ema_centroid = None
        self._ema_centroid_velocity = None
        self._ema_improvement = 0.0
        self._centroid_vel_history = []
    
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
    # POSITION UPDATE STRATEGIES (from benchmark winners)
    # -------------------------------------------------------------------------
    
    def _position_update_original(self):
        """Original: eigenvalue-anisotropic velocity modulation (3 wins)."""
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
    
    def _position_update_spectral(self):
        """Spectral: eigenvalue-weighted velocity projection (1 win).
        
        Key insight: project velocity onto principal components, scale each
        component INVERSELY to its eigenvalue. High-eigenvalue directions
        (well-explored) get dampened; low-eigenvalue (collapsed) get amplified.
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
    
    def _position_update_graph(self):
        """Graph-Laplacian: k-NN graph + connected component analysis (6 wins).
        
        Key insight: build k-NN graph, use Laplacian eigenvalues to detect
        clustering. Small spectral gap = clustered → increase exploration.
        """
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
            component_sizes[np.argmax([i in c for c in components])] for i in range(self.np)
        ])
        
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
            d_inv_sqrt_diag = csr_matrix(
                (d_inv_sqrt, (np.arange(self.np), np.arange(self.np))), shape=(self.np, self.np)
            )
            from scipy.sparse import identity
            lap = identity(self.np, format='csr') - d_inv_sqrt_diag @ adj @ d_inv_sqrt_diag
            
            from scipy.sparse.linalg import eigsh
            eigenvalues = eigsh(lap, k=min(5, self.np - 1), return_eigenvectors=False)
            eigenvalues = np.sort(eigenvalues)
            
            spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 1.0
            spectral_gap = np.clip(spectral_gap, 0.0, 1.0)
        except:
            spectral_gap = 1.0
        
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
        
        clustering_explore = 1.0 + 0.5 * (1.0 - spectral_gap)
        size_factor = np.clip(particle_component_size / max(1, self.np), 0.3, 1.0)
        graph_explore = clustering_explore * (2.0 - size_factor)
        graph_explore = np.clip(graph_explore, 0.5, 2.0)
        
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
    
    def _position_update_hybrid(self):
        """Hybrid: FDC exploration/exploitation + temporal centroid-drift (7 wins).
        
        Key insight: FDC-based mechanism during normal search; temporal
        centroid dynamics take over when stagnant (stagnation_counter drives
        drift_weight). Principled switching because stagnation directly
        indicates FDC mechanism has failed.
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
    
    def _position_update_entropy(self):
        """Eigenvalue entropy + fitness-rank velocity modulation (7 wins).
        
        Key insight: eigenvalue spread (entropy of eigenspectrum) captures
        full distribution shape, not just condition number. Fitness rank
        modulates velocity scale per-particle.
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
    # DISPATCHER: maps operator name -> position update method
    # -------------------------------------------------------------------------
    
    def _dispatch_position_update(self, operator):
        """Route to the appropriate position update implementation."""
        if operator == 'original':
            self._position_update_original()
        elif operator == 'spectral':
            self._position_update_spectral()
        elif operator == 'graph':
            self._position_update_graph()
        elif operator == 'hybrid':
            self._position_update_hybrid()
        elif operator == 'entropy':
            self._position_update_entropy()
        else:
            self._position_update_original()
    
    # -------------------------------------------------------------------------
    # THOMPSON SAMPLING: rank-based reward (scale-independent)
    # -------------------------------------------------------------------------
    
    def _compute_rank_score(self, best_fitness):
        """Compute rank-based score: best_fitness in context of recent history.
        
        Score = 1 - normalized_rank, where normalized_rank is the position
        of best_fitness in the sorted recent history.
        
        This is SCALE-INDEPENDENT: no division by raw fitness values,
        no magic constants like *1e5. Score is always in [0,1].
        """
        if len(self._rank_fitness_history) == 0:
            return 0.5
        
        recent_best = min(f for f, _ in self._rank_fitness_history[-self._rank_window_size:])
        recent_worst = max(f for f, _ in self._rank_fitness_history[-self._rank_window_size:])
        
        if recent_worst <= recent_best:
            return 0.5
        
        normalized = (recent_worst - best_fitness) / (recent_worst - recent_best + 1e-10)
        return np.clip(normalized, 0.0, 1.0)
    
    def _update_thompson_parameters(self, operator, gen_best_fitness):
        """Update Beta distribution parameters using rank-based score.
        
        Uses rank-based score (top tercile = success) to avoid scale
        dependence. Beta(α,β) prior starts at Beta(1,1) = uniform.
        """
        rank_score = self._compute_rank_score(gen_best_fitness)
        
        # Top tercile = success (rank_score >= 0.67)
        # This is SCALE-INDEPENDENT: based on relative ranking, not raw values
        is_success = rank_score >= 0.67
        
        if is_success:
            self._op_alpha[operator] += 1.0
        else:
            self._op_beta[operator] += 1.0
        
        # Maintain window
        self._rank_fitness_history.append((gen_best_fitness, operator))
        if len(self._rank_fitness_history) > self._rank_window_size * 2:
            self._rank_fitness_history = self._rank_fitness_history[-self._rank_window_size:]
    
    def _thompson_sample(self):
        """Select operator via Thompson Sampling from Beta distributions.
        
        Samples from Beta(alpha, beta) for each operator and picks
        the highest sample. This naturally balances exploration/exploitation
        without epsilon-greedy schedules.
        
        If an operator has no data (alpha=beta=1), it gets a sample
        drawn from Beta(1,1) = Uniform(0,1), which is high-variance
        and promotes initial exploration.
        """
        samples = {}
        for op in self._operators:
            alpha = max(self._op_alpha[op], 1e-10)
            beta = max(self._op_beta[op], 1e-10)
            samples[op] = np.random.beta(alpha, beta)
        
        return max(samples, key=samples.get)
    
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
        for op in self._operators:
            self._op_alpha[op] = 1.0
            self._op_beta[op] = 1.0
        self._rank_fitness_history = []
        self._selected_operator = np.random.choice(self._operators)
        
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
        
        # Initial rank record
        if self.global_best_fitness < np.inf:
            self._rank_fitness_history.append((self.global_best_fitness, self._selected_operator))
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # Thompson Sampling: select operator probabilistically
            self._selected_operator = self._thompson_sample()
            
            # Adaptation
            self._adapt_inertia_weight()
            self._adapt_neighborhood_size()
            self._update_local_best_from_personal()
            
            # Velocity update (shared base), then position update (SELECTED)
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
            
            # Update Thompson Sampling parameters with rank-based reward
            gen_best = float(np.min(self.current_fitness))
            if self.global_best_fitness < np.inf:
                self._update_thompson_parameters(self._selected_operator, self.global_best_fitness)
            
            self._restart_if_stagnant()
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
```