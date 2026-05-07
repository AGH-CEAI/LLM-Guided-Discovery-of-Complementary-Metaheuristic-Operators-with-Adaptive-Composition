```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: PROBE-AND-COMMIT with Thompson Sampling (mechanism #4 / contextual bandit)
    
    The gap table is BLEND-HOSTILE:
      - Task 5: variant_08 (19.57) vs variant_07 (21.01) = 1.07×, but vs median
        (48742) = 2490×. Task 1: variant_08 (2.51) vs original (3.54) = 1.4× but
        vs median (3.96) = 1.6×. The winner's advantage over the MEDIAN competitor
        can be enormous even when the runner-up gap is small — linear blending
        dilutes both.
      - 7 distinct winners across 24 tasks with different failure modes:
          * catA (1 win): geometric centroid/hull signals
          * catB (3 wins): directional alignment + subspace expansion
          * catD (3 wins): fitness-rank modulation + FDC
          * catE (1 win): graph Laplacian spectral gap
          * catF (1 win): centroid momentum + velocity autocorrelation
          * catG (0 wins): excluded (never won any task)
          * catH (6 wins): hybrid temporal + topology fragmentation
      - All operators are orthogonal (geometry vs fitness vs graph vs spectral
        vs temporal vs hybrid), so credit assignment is noisy — Thompson Sampling
        handles this by sampling from posterior Beta distributions.
    
    The dispatcher preserves each winner's advantage by:
      - PHASE A (probe): round-robin across the 6 viable operators (catG excluded),
        3 generations each = 18 generations total. Rank-based Beta reward signal.
      - PHASE B (commit): Thompson Sampling from per-operator Beta(successes+1,
        failures+1) posteriors. Samples are drawn every generation so the regime
        detector (convergence slope, rank-corr, eff_dim, diversity) informs
        which operator is likely best given the current landscape state.
      - EPSILON-REVIEW: if committed operator stagnates (>25 gens, p=0.05),
        sample from ALL operators (not just runner-up) to allow regime switching.
    
    Win-count distribution (9/6/3/3/1/1 across 7 winners) confirms that different
    operators win on different tasks. No single operator dominates, and the
    per-task MEDIAN gap can be extreme even when the runner-up gap is tiny.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5 * dim, capped
        self.np = min(max(5 * dim, 20), 300)
        
        # Ring topology neighborhood size
        self.neighborhood_size = max(3, dim // 5)
        
        # Acceleration coefficients
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
        # Map operator ID -> (successes, failures) for Beta posterior
        self._operator_successes = {}
        self._operator_failures = {}
        
        # THE 6 VIABLE WINNING POSITION UPDATE STRATEGIES (catG excluded: 0 wins)
        self._operator_names = [
            'catA',   # variant_01: geometry-based centroid drift + hull analysis (1 win)
            'catB',   # variant_10: directional alignment + subspace expansion (3 wins)
            'catD',   # variant_04: fitness-rank modulation + FDC (3 wins)
            'catE',   # variant_05: graph Laplacian spectral gap (1 win)
            'catF',   # variant_06: centroid momentum + velocity autocorrelation (1 win)
            'catH',   # variant_08: hybrid temporal + topology fragmentation (6 wins)
        ]
        for op in self._operator_names:
            self._operator_successes[op] = 0
            self._operator_failures[op] = 0
        
        # Probe budget: 3 gens per operator, min 18 total
        self._probe_gens_per_op = 3
        self._probe_total_budget = max(len(self._operator_names) * self._probe_gens_per_op, 18)
        
        # Epsilon-review probability when committed operator stagnates
        self._epsilon_review = 0.05
        self._stagnation_review_threshold = 25
        
        # Probe phase state
        self._in_probe_phase = False
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._probe_best_fitness_seen = np.inf
        self._probe_previous_best = np.inf
        
        # Commit phase state
        self._committed_operator = None
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
        
        # Fingerprint signals (computed during probe)
        self._fingerprint_convergence_slope = 0.0
        self._fingerprint_rank_corr = 0.0
        self._fingerprint_eff_dim = dim
        self._fingerprint_diversity = 0.0
        
        # Per-operator EMA of recent performance for regime-aware prior
        self._operator_recent_perf = {op: 0.0 for op in self._operator_names}
        self._recent_window_size = 5
    
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
    
    def _compute_regime_signals(self):
        """Compute cheap landscape fingerprint from current population state.
        
        Returns a dict of signals observable from any black-box func:
          - convergence_slope: slope of log(best_fitness) vs generation
          - rank_corr: Spearman-like correlation between fitness rank and
                       distance to global best (proxy for landscape smoothness)
          - eff_dim: effective dimensionality from early covariance
          - diversity: current population diversity
        """
        signals = {}
        
        # 1. Convergence slope (from probe history stored in global best fitness)
        if self.generation >= 3:
            # Approximate convergence from stagnation counter and improvement rate
            improvement = self.global_best_fitness
            signals['convergence_slope'] = -np.log1p(max(improvement, 0)) / max(self.generation, 1)
        else:
            signals['convergence_slope'] = 0.0
        
        # 2. Rank correlation (fitness vs distance to global best)
        if self.global_best is not None and self.np > 2:
            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
            dists = np.linalg.norm(self.population - self.global_best, axis=1)
            dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
            if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                signals['rank_corr'] = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
            else:
                signals['rank_corr'] = 0.0
        else:
            signals['rank_corr'] = 0.0
        
        # 3. Effective dimensionality from covariance
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            total_var = np.sum(eigenvalues)
            if total_var > 0:
                sorted_eig = np.sort(eigenvalues)[::-1]
                cumvar = np.cumsum(sorted_eig) / total_var
                signals['eff_dim'] = float(np.searchsorted(cumvar, 0.95)) + 1
            else:
                signals['eff_dim'] = self.dim
        except:
            signals['eff_dim'] = self.dim
        
        # 4. Diversity
        signals['diversity'] = self._compute_diversity()
        
        self._fingerprint_convergence_slope = signals['convergence_slope']
        self._fingerprint_rank_corr = signals['rank_corr']
        self._fingerprint_eff_dim = signals['eff_dim']
        self._fingerprint_diversity = signals['diversity']
        
        return signals
    
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
    # POSITION UPDATE STRATEGIES (6 benchmark winners, exact implementations)
    # -------------------------------------------------------------------------
    
    def _position_update_catA(self):
        """Geometry-based temporal drift modulation (Category A, 1 win: task 9).
        
        Uses literal geometric layout: centroid drift magnitude, convex hull
        volume/area, pairwise distance distribution, axis-aligned spread, and
        k-NN connected components. Detects convergence from population geometry
        rather than spectral decomposition.
        """
        # --- Centroid temporal drift ---
        current_centroid = np.mean(self.population, axis=0)
        if not hasattr(self, '_prev_centroid'):
            self._prev_centroid = current_centroid.copy()
            self._centroid_drift_mag = 0.0
            self._centroid_ema = 0.0
        else:
            raw_drift = np.linalg.norm(current_centroid - self._prev_centroid)
            self._centroid_drift_mag = raw_drift
            self._prev_centroid = current_centroid.copy()
            self._centroid_ema = 0.7 * self._centroid_ema + 0.3 * raw_drift

        # --- Pairwise distance statistics ---
        sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        pairwise_dists = np.sqrt(sq_dists)
        valid_dists = pairwise_dists[pairwise_dists < np.inf]

        if len(valid_dists) > 0:
            mean_dist = np.mean(valid_dists)
            std_dist = np.std(valid_dists) + 1e-10
            dist_cv = std_dist / mean_dist
        else:
            mean_dist, std_dist, dist_cv = 1.0, 1.0, 1.0

        # --- Axis-aligned spread per dimension ---
        mins = np.min(self.population, axis=0)
        maxs = np.max(self.population, axis=0)
        axis_spreads = maxs - mins + 1e-10
        max_spread = np.max(axis_spreads)
        min_spread = np.min(axis_spreads)
        spread_ratio = min_spread / max_spread

        # --- k-NN connected components (geometric clustering) ---
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
        largest_component_ratio = np.max(component_sizes) / self.np if len(component_sizes) > 0 else 1.0
        fragmentation = 1.0 - largest_component_ratio

        # --- Convex hull approximation using axis-aligned bounding box ---
        hull_volume = np.prod(axis_spreads)
        hull_volume_log = np.log(hull_volume + 1e-10)
        hull_volume_ema_key = '_hull_volume_ema'
        if not hasattr(self, hull_volume_ema_key):
            setattr(self, hull_volume_ema_key, hull_volume_log)
        else:
            setattr(self, hull_volume_ema_key, 0.9 * getattr(self, hull_volume_ema_key) + 0.1 * hull_volume_log)
        hull_shrink_rate = hull_volume_log - getattr(self, hull_volume_ema_key)

        # --- Geometric convergence signal ---
        drift_signal = np.clip(self._centroid_ema / (mean_dist + 1e-10), 0.0, 2.0)
        spread_signal = np.clip(spread_ratio, 0.0, 1.0)
        component_signal = 1.0 - fragmentation
        hull_signal = np.clip(-hull_shrink_rate / 10.0, 0.0, 1.0)

        geo_signal = (drift_signal * 0.25 + spread_signal * 0.25 + 
                      component_signal * 0.25 + hull_signal * 0.25)

        # --- Temporal scale from geometric state ---
        if self._centroid_ema > mean_dist * 0.05:
            base_scale = 1.2
        elif hull_shrink_rate < -0.01:
            base_scale = 0.7
        else:
            base_scale = 1.0

        temporal_scale = base_scale * (0.8 + 0.4 * geo_signal)

        # --- Per-particle scaling from local density ---
        knn_avg_dist = np.mean(pairwise_dists[np.arange(self.np)[:, None], knn_indices], axis=1, keepdims=True) + 1e-10
        density_factor = np.clip(knn_avg_dist / (mean_dist + 1e-10), 0.5, 2.0)

        # --- Directional correction toward centroid ---
        to_centroid = current_centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid / to_centroid_dist
        centroid_pull = 0.2 * to_centroid_dir * density_factor

        # --- Apply position update ---
        new_population = (self.population + 
                          self.inertia_weight * self.velocity * temporal_scale * density_factor + 
                          centroid_pull)

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catB(self):
        """Directional alignment + subspace expansion (Category B, 3 wins: tasks 3,4,10).
        
        Computes alignment between each eigenvector direction and the global-best
        vector. Velocity components aligned with global-best get boosted; misaligned
        components get dampened. When condition number is high, minor eigenvector
        directions are amplified to re-expand the population.
        """
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

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catD(self):
        """Fitness-landscape rank-based velocity modulation (Category D, 3 wins: tasks 6,17,23).
        
        Uses fitness ranks, success-history, and fitness-distance correlation
        to modulate velocity. No raw distances or covariance - only fitness signals.
        Targets worst tasks with large deceptive basins where population may
        collapse to local optima.
        """
        # 1. Fitness ranks (0 = best, 1 = worst)
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)

        # 2. Success history with exponential decay
        if not hasattr(self, '_success_count'):
            self._success_count = np.zeros(self.np)
        improved = self.personal_best_fitness > self.current_fitness
        self._success_count[improved] += 1
        self._success_count[~improved] *= 0.9
        success_norm = self._success_count / (np.max(self._success_count) + 1.0)

        # 3. Fitness-percentile velocity scaling
        vel_scale = 0.5 + 1.5 * fitness_ranks
        vel_scale = np.clip(vel_scale, 0.5, 2.0)

        # 4. Fitness-distance correlation (FDC) as landscape ruggedness signal
        if self.global_best is not None and self.np > 2:
            dists = np.linalg.norm(self.population - self.global_best, axis=1)
            dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
            if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                fdc = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
            else:
                fdc = 0.0
        else:
            fdc = 0.0

        # 5. FDC-driven exploration factor
        fdc_explore_factor = np.clip(2.0 - fdc, 0.5, 2.5)

        # 6. Success-based boost for recently improved particles
        success_boost = 1.0 + 0.5 * success_norm
        success_boost = np.clip(success_boost, 1.0, 2.0)

        # 7. Combined scale
        combined_scale = vel_scale * fdc_explore_factor * success_boost
        combined_scale = np.clip(combined_scale, 0.3, 3.0)

        # 8. Compute fitness gradient direction
        top_k = max(1, self.np // 5)
        top_indices = np.argsort(self.current_fitness)[:top_k]
        top_centroid = np.mean(self.population[top_indices], axis=0)

        to_top = top_centroid - self.population
        to_top_dist = np.linalg.norm(to_top, axis=1, keepdims=True) + 1e-10
        to_top_dir = to_top / to_top_dist

        gradient_strength = 0.5 * fitness_ranks[:, np.newaxis]

        # 9. Position update
        scaled_velocity = self.velocity * combined_scale[:, np.newaxis]
        gradient_push = gradient_strength * to_top_dir
        random_perturb = fdc_explore_factor * np.random.uniform(-0.5, 0.5, (self.np, self.dim))

        new_population = self.population + scaled_velocity + gradient_push + random_perturb
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catE(self):
        """Graph Laplacian escape via topology-aware velocity modulation (Category E, 1 win: task 21).
        
        Builds k-NN graph over population, analyzes connected components and
        spectral gap of normalized Laplacian to detect convergence/fragmentation,
        identifies bridge particles via centrality approximation, and modulates
        velocities per-particle based on graph topology.
        """
        try:
            # --- Build k-NN graph ---
            k = min(5, self.np - 1)
            sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

            # --- Connected component analysis via BFS ---
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

            # --- Normalized Laplacian spectral gap ---
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
            except np.linalg.LinAlgError:
                spectral_gap = 1.0

            # --- Betweenness-centrality approximation ---
            centrality = np.zeros(self.np)
            for i in range(self.np):
                neighbors = knn_indices[i]
                neighbor_degrees = component_sizes[[np.argmax([n in c for c in components]) for n in neighbors]]
                centrality[i] = np.sum(1.0 / (neighbor_degrees + 1e-10))
            centrality = centrality / (np.max(centrality) + 1e-10)

            # --- Modulation factors ---
            explore_factor = 1.0 + 0.6 * (1.0 - spectral_gap)
            size_factor = np.clip(particle_component_size / max(1, self.np), 0.2, 1.0)
            isolation_boost = 1.5 * (1.0 - size_factor) + 1.0
            bridge_damp = 1.0 - 0.3 * centrality

            vel_scale = explore_factor * isolation_boost * bridge_damp
            vel_scale = np.clip(vel_scale, 0.3, 2.5)

            # --- Temporal drift EMA for base scaling ---
            if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
                improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
            else:
                improvement = 0.0
            self._prev_global_best_fitness = self.global_best_fitness

            if not hasattr(self, '_graph_ema_improvement'):
                self._graph_ema_improvement = 0.0
            self._graph_ema_improvement = 0.3 * improvement + 0.7 * self._graph_ema_improvement

            if self._graph_ema_improvement > 1e-6:
                temporal_scale = 1.2
            elif self._graph_ema_improvement > 1e-10:
                temporal_scale = 1.0
            else:
                temporal_scale = 0.7

            # --- Apply velocity update ---
            new_population = self.population + temporal_scale * self.velocity * vel_scale[:, np.newaxis]

            # --- Targeted perturbation for isolated particles ---
            if spectral_gap < 0.15:
                isolated = particle_component_size <= 2
                if np.any(isolated):
                    random_perturb = np.random.uniform(-1.0, 1.0, (self.np, self.dim))
                    new_population[isolated] += 0.4 * random_perturb[isolated]

        except Exception:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catF(self):
        """Centroid momentum with velocity autocorrelation stagnation detection (Category F, 1 win: task 12).
        
        Tracks centroid trajectory across generations; autocorrelation of centroid
        velocity detects oscillation (sign of convergence stall). Cross-correlation
        between particle velocities and centroid drift detects cohesive vs scattered
        motion. Uses motion dynamics to modulate velocity scaling and inject
        corrective impulses.
        """
        try:
            centroid = np.mean(self.population, axis=0)

            if not hasattr(self, '_temporal_centroid'):
                self._temporal_centroid = centroid.copy()
            if not hasattr(self, '_temporal_centroid_history'):
                self._temporal_centroid_history = []
            if not hasattr(self, '_temporal_vel_history'):
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

            # EMA of improvement
            if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
                improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
            else:
                improvement = 0.0
            self._prev_global_best_fitness = self.global_best_fitness

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
                if drift_norm < 1e-4:
                    corrective_force = 0.3 * np.random.uniform(-1, 1, self.dim) * drift_norm
                else:
                    corrective_force = -0.2 * centroid_drift
            else:
                corrective_force = np.zeros(self.dim)

            new_population = self.population + temporal_scale * self.velocity + corrective_force
            self.population = self._clip_to_bounds(new_population)

        except Exception:
            new_population = self.population + self.velocity
            self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catH(self):
        """Hybrid: Temporal drift tracking + topology-based fragmentation detection (Category H, 6 wins: tasks 1,2,5,7,16,20).
        
        Combines TWO distinct mechanisms:
          1. Temporal: Centroid drift rate across generations to detect convergence phases
          2. Topology: k-NN connected component analysis to detect population fragmentation
        
        Switching logic is data-driven:
          - Low drift + fragmented population → strong exploration boost
          - High drift (exploring) → trust spectral signals
          - Fragmented + improving → moderate exploration
          - Well-connected + converging → exploit via spectral damping
        """
        try:
            # === MECHANISM 1: TEMPORAL DRIFT TRACKING ===
            centroid = np.mean(self.population, axis=0)
            if not hasattr(self, '_prev_centroid'):
                self._prev_centroid = centroid.copy()
                self._centroid_drift_history = []

            drift = np.linalg.norm(centroid - self._prev_centroid)
            self._prev_centroid = centroid.copy()

            if not hasattr(self, '_centroid_drift_history'):
                self._centroid_drift_history = []
            self._centroid_drift_history.append(drift)
            if len(self._centroid_drift_history) > 20:
                self._centroid_drift_history.pop(0)

            if not hasattr(self, '_ema_drift'):
                self._ema_drift = 0.0
            self._ema_drift = 0.3 * drift + 0.7 * self._ema_drift

            population_spread = np.std(self.population) + 1e-10
            normalized_drift = self._ema_drift / population_spread

            # === MECHANISM 2: TOPOLOGY FRAGMENTATION DETECTION ===
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

            # === PRINCIPLED SWITCHING LOGIC ===
            is_converging = normalized_drift < 0.05
            is_fragmented = n_components > 1 and fragmentation_ratio > 0.3

            # === SPECTRAL ANALYSIS (for when not fragmented) ===
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

            # === TEMPORAL IMPROVEMENT SIGNAL ===
            if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
                improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
            else:
                improvement = 0.0
            self._prev_global_best_fitness = self.global_best_fitness

            if not hasattr(self, '_spectral_ema_improvement'):
                self._spectral_ema_improvement = 0.0
            self._spectral_ema_improvement = 0.3 * improvement + 0.7 * self._spectral_ema_improvement

            # === DECISION TREE WITH PRINCIPLED SWITCHING ===
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

            # === COMBINE MECHANISMS ===
            if self._spectral_ema_improvement > 1e-6:
                spectral_base = 1.2
            elif self._spectral_ema_improvement > 1e-10:
                spectral_base = 1.0
            else:
                spectral_base = 0.7

            spectral_scale = spectral_base * (1.0 + 0.4 * spectral_signal)
            if cond > 100.0:
                log_damp = np.log1p(cond) / np.log1p(1000.0)
                log_damp = np.clip(log_damp, 0.0, 1.0)
                spectral_scale *= (1.0 + 0.3 * log_damp)

            # Topology contribution: inter-component repulsion
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

            U = np.linalg.eigvalsh(cov)
            idx = np.argsort(U)[::-1]
            U = U[idx]
            V = np.linalg.eigh(cov)[1][:, idx]

            vel_proj = self.velocity @ V
            sv_scale = np.sqrt(U + 1e-10)
            sv_norm = sv_scale / (sv_scale[0] + 1e-10)
            per_comp_scale = np.clip(sv_norm, 0.1, 2.0)
            vel_scaled = vel_proj * per_comp_scale

            new_population = self.population + combined_scale * (vel_scaled @ V.T) + topology_correction

        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity

        self.population = self._clip_to_bounds(new_population)
    
    # -------------------------------------------------------------------------
    # DISPATCHER: maps operator name -> position update method
    # -------------------------------------------------------------------------
    
    def _dispatch_position_update(self, operator):
        """Route to the appropriate position update implementation."""
        dispatch_map = {
            'catA': self._position_update_catA,
            'catB': self._position_update_catB,
            'catD': self._position_update_catD,
            'catE': self._position_update_catE,
            'catF': self._position_update_catF,
            'catH': self._position_update_catH,
        }
        method = dispatch_map.get(operator)
        if method is not None:
            method()
        else:
            # Fallback: spectral entropy (similar to catH spectral part)
            self._position_update_catH()
    
    # -------------------------------------------------------------------------
    # THOMPSON SAMPLING: Beta posterior over operators
    # -------------------------------------------------------------------------
    
    def _record_thompson_reward(self, operator, improved):
        """Record a binary reward for the operator using Beta posterior.
        
        Uses binary success/failure — no scale-dependent constants.
        improved=True → success (the operator helped find a better solution).
        """
        if improved:
            self._operator_successes[operator] += 1
        else:
            self._operator_failures[operator] += 1
        
        # Update recent performance EMA (for regime-aware prior)
        reward = 1.0 if improved else 0.0
        self._operator_recent_perf[operator] = (
            0.7 * self._operator_recent_perf[operator] + 0.3 * reward
        )
    
    def _thompson_sample(self):
        """Thompson Sampling: sample from Beta posterior for each operator.
        
        Uses Beta(successes + alpha_prior, failures + beta_prior) where
        the prior is informed by regime signals. This is scale-independent
        (binary rewards in [0,1]) and handles the high-variance regime
        detection by using a non-flat prior when regime signals are strong.
        """
        samples = {}
        
        # Regime-aware prior: bias toward operators that match current regime
        signals = self._compute_regime_signals()
        convergence_slope = signals['convergence_slope']
        rank_corr = signals['rank_corr']
        eff_dim_ratio = signals['eff_dim'] / self.dim
        diversity = signals['diversity']
        
        # Regime classification (soft — used only for prior, not hard dispatch)
        is_converging = self.stagnation_counter > 10 or convergence_slope > 0.1
        is_rugged = rank_corr < 0.2 if not np.isnan(rank_corr) else False
        is_high_dim = eff_dim_ratio > 0.5
        is_diverse = diversity > 1.0
        
        # Regime -> soft bias toward specific operator categories
        regime_bias = {op: 1.0 for op in self._operator_names}
        
        if is_converging:
            # When converging: favor operators that detect convergence
            regime_bias['catA'] *= 1.5  # geometry-based convergence detection
            regime_bias['catF'] *= 1.5  # centroid momentum stagnation detection
            regime_bias['catH'] *= 1.3  # hybrid with convergence signals
        if is_rugged:
            # When landscape is rugged: favor operators with exploration mechanisms
            regime_bias['catD'] *= 1.5  # FDC-based exploration
            regime_bias['catE'] *= 1.4  # graph Laplacian escape
            regime_bias['catH'] *= 1.3  # topology fragmentation detection
        if is_high_dim:
            # High-dimensional: favor operators that expand subspace
            regime_bias['catB'] *= 1.5  # directional alignment + subspace expansion
            regime_bias['catH'] *= 1.2  # effective dimensionality signals
        if is_diverse:
            # High diversity: favor operators that exploit cohesion
            regime_bias['catA'] *= 1.3  # centroid gravity
            regime_bias['catF'] *= 1.2  # cross-correlation
        if self.stagnation_counter > 20:
            # Stagnation: favor operators with escape mechanisms
            regime_bias['catE'] *= 1.6  # graph Laplacian escape
            regime_bias['catD'] *= 1.4  # FDC exploration
            regime_bias['catH'] *= 1.3  # topology fragmentation
        
        # Sample from Beta(successes + prior, failures + prior) for each operator
        for op in self._operator_names:
            s = self._operator_successes[op]
            f = self._operator_failures[op]
            prior_strength = 2.0  # weak prior (uniform-ish)
            
            # Regime-informed prior: shift alpha/beta based on regime bias
            bias = regime_bias.get(op, 1.0)
            alpha_prior = prior_strength * bias
            beta_prior = prior_strength * (2.0 - bias)  # inverse bias for beta
            
            # Beta posterior parameters
            alpha = s + alpha_prior
            beta_param = f + beta_prior
            
            # Thompson sample from Beta(alpha, beta)
            # Sample from Gamma and normalize
            if alpha > 0 and beta_param > 0:
                try:
                    x = np.random.gamma(alpha, 1.0)
                    y = np.random.gamma(beta_param, 1.0)
                    samples[op] = x / (x + y + 1e-10)
                except Exception:
                    samples[op] = 0.5
            else:
                samples[op] = 0.5
        
        # Select operator with highest sample
        best_op = max(samples, key=samples.get)
        return best_op, samples
    
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
        
        # Reset Thompson Sampling state
        for op in self._operator_names:
            self._operator_successes[op] = 0
            self._operator_failures[op] = 0
            self._operator_recent_perf[op] = 0.0
        
        self._probe_best_fitness_seen = np.inf
        self._probe_previous_best = np.inf
        self._committed_operator = None
        self._in_probe_phase = True
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
        
        # Reset EMA states for temporal operators
        attrs_to_reset = [
            '_prev_centroid', '_centroid_drift_mag', '_centroid_ema', '_hull_volume_ema',
            '_temporal_centroid', '_temporal_centroid_history', '_temporal_vel_history',
            '_ema_drift', '_centroid_drift_history', '_graph_ema_improvement',
            '_prev_global_best_fitness', '_spectral_ema_improvement', '_success_count',
            '_prev_centroid_catH',
        ]
        for attr in attrs_to_reset:
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except (AttributeError, ValueError):
                    pass
        
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
                current_op = self._operator_names[self._probe_op_index]
                
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
                
                # Record Thompson reward: did this operator help improve global best?
                gen_best = float(np.min(self.current_fitness))
                if gen_best < self._probe_best_fitness_seen:
                    self._probe_best_fitness_seen = gen_best
                    self._record_thompson_reward(current_op, improved=True)
                else:
                    self._record_thompson_reward(current_op, improved=False)
                
                self._probe_gens_in_current_op += 1
                
                # Advance to next operator after probe_gens_per_op
                if self._probe_gens_in_current_op >= self._probe_gens_per_op:
                    self._probe_op_index += 1
                    self._probe_gens_in_current_op = 0
                
                # Check if probe phase is done
                if self._probe_op_index >= len(self._operator_names):
                    self._in_probe_phase = False
                    # Compute regime signals for initial prior
                    self._compute_regime_signals()
                    self._commit_best_fitness = self.global_best_fitness
                    self._commit_gen_count = 0
                
                self._restart_if_stagnant()
            
            # --- PHASE B: COMMIT with Thompson Sampling ---
            else:
                self._commit_gen_count += 1
                
                # Epsilon-review: small chance to re-sample (not just runner-up)
                if (self.stagnation_counter > self._stagnation_review_threshold and
                    np.random.random() < self._epsilon_review):
                    # Re-sample from all operators (allows regime switching)
                    selected_op, _ = self._thompson_sample()
                    self._velocity_update_base()
                    self._dispatch_position_update(selected_op)
                else:
                    # Thompson Sampling: draw sample from each operator's Beta posterior
                    selected_op, sample_values = self._thompson_sample()
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
                
                # Record Thompson reward for the selected operator
                if self.global_best_fitness < self._commit_best_fitness:
                    self._commit_best_fitness = self.global_best_fitness
                    self._record_thompson_reward(selected_op, improved=True)
                else:
                    self._record_thompson_reward(selected_op, improved=False)
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
```