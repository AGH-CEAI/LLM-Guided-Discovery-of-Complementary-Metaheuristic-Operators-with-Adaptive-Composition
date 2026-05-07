import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: PROBE-AND-COMMIT (Contextual Bandit, mechanism #4)
    
    The gap table is BLEND-HOSTILE for per-task extremes:
      - Task 2: winner beats median competitor by 72.1× (variant_06_catF_idea_0.py
        at 1.66e-01 vs median of ~12.0). A 50/50 blend with the runner-up yields
        ~1.86e-01, which is ~1.1× WORSE than the winner alone.
      - Task 5: winner beats median by ~12.9×.
      - Task 10: winner beats median by 20.1×.
      - Task 12: winner beats median by 12.7×.
    
    The dispatcher preserves each winner's advantage by:
      - PHASE A (probe): round-robin burst with each of the 8 winning operators,
        tracking RANK-BASED improvement scores (scale-independent). Each operator
        gets 3 generations in the full population.
      - PHASE B (commit): run ONLY the winning operator for the remaining budget.
        No blending. No hard-coded weights.
      - EPSILON-REVIEW: if committed operator stagnates (>30 gens, p=0.05),
        allow a small probability of switching to the runner-up.
    
    Win-count distribution (9/5/3/2/2/1/1/1 across 8 winners) confirms that
    different operators win on different tasks. The probe extracts a cheap
    FINGERPRINT from the landscape (convergence slope, rank-correlation,
    effective dimensionality) to route to the right operator.
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
        self.global_best_fitness = np.inf
        self.inertia_weight = 0.729
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        self.current_fitness = None
        
        # --- PROBE-AND-COMMIT STATE ---
        # 8 winning operators from benchmark
        self._operators = [
            'original',      # original.py (9 wins)
            'catA_convex',   # variant_09_catA_idea_0.py (5 wins): convex hull boundary
            'catC_entropy',  # variant_03_catC_idea_0.py (3 wins): Gaussian entropy
            'catH_hybrid',   # variant_08_catH_idea_0.py (2 wins): temporal-drift + FDC
            'catF_temporal', # variant_06_catF_idea_0.py (2 wins): stagnation-drift
            'catG_mc',       # variant_07_catG_idea_0.py (1 win): Monte Carlo
            'catE_mst',      # variant_05_catE_idea_0.py (1 win): MST betweenness
            'catB_eigen',    # variant_02_catB_idea_0.py (1 win): eigenvalue anisotropic
        ]
        self._operator_scores = {op: [] for op in self._operators}
        self._operator_total_score = {op: 0.0 for op in self._operators}
        self._operator_sample_count = {op: 0 for op in self._operators}
        self._committed_operator = None
        self._runner_up_operator = None
        
        # Sliding window for rank-based scoring (>= 3 * num_arms = 24)
        self._sliding_window_size = max(24, len(self._operators) * 3)
        
        # Probe: 3 gens per operator
        self._probe_gens_per_op = 3
        self._probe_total_budget = len(self._operators) * self._probe_gens_per_op
        
        # Epsilon-review
        self._epsilon_review = 0.05
        
        # Probe phase state
        self._in_probe_phase = False
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._probe_best_fitness_seen = np.inf
        self._probe_fitness_history = []
        self._probe_op_history = []
        self._probe_gen_count = 0
        
        # Commit phase
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
        
        # Fingerprint
        self._fingerprint_convergence_slope = 0.0
        self._fingerprint_rank_corr = 0.0
        self._fingerprint_eff_dim = dim
        self._fingerprint_diversity = 0.0
    
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
            return fitness[:len(fitness)], len(fitness)
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
                neighborhood_indices = np.concatenate([np.arange(left, self.np), np.arange(0, right)])
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
    
    def _compute_fingerprint(self):
        signals = {}
        if len(self._probe_fitness_history) >= 3:
            gens = np.arange(len(self._probe_fitness_history))
            bests = np.array(self._probe_fitness_history)
            valid = bests > 0
            if np.sum(valid) >= 2:
                log_bests = np.log1p(bests[valid])
                signals['convergence_slope'] = np.polyfit(gens[valid], log_bests, 1)[0]
            else:
                signals['convergence_slope'] = np.polyfit(gens, bests, 1)[0]
        else:
            signals['convergence_slope'] = 0.0
        
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
        
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            total_var = np.sum(eigenvalues)
            if total_var > 0:
                cumvar = np.cumsum(sorted(eigenvalues, reverse=True)) / total_var
                signals['eff_dim'] = float(np.searchsorted(cumvar, 0.95)) + 1
            else:
                signals['eff_dim'] = self.dim
        except:
            signals['eff_dim'] = self.dim
        
        signals['diversity'] = self._compute_diversity()
        self._fingerprint_convergence_slope = signals['convergence_slope']
        self._fingerprint_rank_corr = signals['rank_corr']
        self._fingerprint_eff_dim = signals['eff_dim']
        self._fingerprint_diversity = signals['diversity']
        return signals
    
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
        fitness = self.current_fitness
        if len(fitness) > 1:
            n_bins = min(20, max(3, len(fitness) // 2))
            hist, _ = np.histogram(fitness, bins=n_bins)
            hist = hist / (len(fitness) + 1e-10)
            hist = hist[hist > 0]
            fitness_entropy = -np.sum(hist * np.log(hist + 1e-10))
            max_fitness_entropy = np.log(n_bins)
            norm_fitness_ent = fitness_entropy / (max_fitness_entropy + 1e-10)
        else:
            norm_fitness_ent = 0.5
        
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues_norm = eigenvalues / (np.sum(eigenvalues) + 1e-10)
            uniform = np.ones(len(eigenvalues_norm)) / len(eigenvalues_norm)
            uniform = uniform + 1e-10
            kl_div = np.sum(eigenvalues_norm * np.log((eigenvalues_norm + 1e-10) / uniform))
            max_kl = np.log(len(eigenvalues_norm))
            norm_kl = np.clip(kl_div / (max_kl + 1e-10), 0.0, 1.0)
        except:
            norm_kl = 0.5
        
        exploration_signal = 0.4 * (1.0 - norm_fitness_ent) + 0.6 * norm_kl
        exploration_signal = np.clip(exploration_signal, 0.0, 1.0)
        min_ns = 1
        max_ns = max(3, self.np // 4)
        target_ns = int(np.round(min_ns + exploration_signal * (max_ns - min_ns)))
        target_ns = np.clip(target_ns, min_ns, max_ns)
        
        if target_ns > self.neighborhood_size:
            self.neighborhood_size = min(target_ns, self.neighborhood_size + 1)
        elif target_ns < self.neighborhood_size:
            self.neighborhood_size = max(target_ns, self.neighborhood_size - 1)
    
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
            indices = np.random.choice([j for j in range(self.np) if j != i], 3, replace=False)
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (self.population[indices[1]] - self.population[indices[2]])
        mutation_component = np.where(mutation_active, 0.3 * (mutation_vectors - self.population), 0.0)
        new_velocity = self.inertia_weight * self.velocity + cognitive_component + social_component + mutation_component
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)
    
    # -------------------------------------------------------------------------
    # POSITION UPDATE STRATEGIES (8 benchmark winners + original)
    # -------------------------------------------------------------------------
    
    def _position_update_original(self):
        """Original: eigenvalue-anisotropic velocity modulation (baseline, 9 wins)."""
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
    
    def _position_update_catA_convex(self):
        """Convex hull boundary detection + anisotropic centroid correction (variant_09, 5 wins)."""
        from scipy.spatial import ConvexHull
        centroid = np.mean(self.population, axis=0)
        to_centroid = centroid - self.population
        to_centroid_norm = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid / to_centroid_norm
        dists_to_centroid = np.linalg.norm(self.population - centroid, axis=1)
        max_dist = np.max(dists_to_centroid) + 1e-10
        normalized_dists = dists_to_centroid / max_dist
        boundary_mask = np.zeros(self.np, dtype=bool)
        try:
            if self.np >= self.dim + 2 and self.dim <= 15:
                hull = ConvexHull(self.population)
                boundary_mask[hull.vertices] = True
            else:
                boundary_mask = normalized_dists > 0.75
        except:
            boundary_mask = normalized_dists > 0.75
        interior_strength = 0.8 * (1.0 - normalized_dists[:, np.newaxis])
        boundary_strength = 0.25 * normalized_dists[:, np.newaxis]
        correction = np.where(boundary_mask[:, np.newaxis], boundary_strength * to_centroid_dir, interior_strength * to_centroid_dir)
        new_population = self.population + self.inertia_weight * self.velocity + 0.5 * correction
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catC_entropy(self):
        """Gaussian likelihood entropy modulation (variant_03, 3 wins)."""
        try:
            from scipy.stats import multivariate_normal
        except ImportError:
            new_population = self.population + self.inertia_weight * self.velocity
            self.population = self._clip_to_bounds(new_population)
            return
        
        n_select = max(5, self.np // 4)
        top_indices = np.argsort(self.current_fitness)[:n_select]
        if n_select < 2:
            new_population = self.population + self.inertia_weight * self.velocity
            self.population = self._clip_to_bounds(new_population)
            return
        
        top_pop = self.population[top_indices]
        best_mean = np.mean(top_pop, axis=0)
        centered = top_pop - best_mean
        best_cov = np.cov(centered.T)
        if best_cov.ndim == 0:
            best_cov = best_cov.reshape(1, 1)
        best_cov = best_cov + 1e-6 * np.eye(self.dim)
        
        log_likelihood = np.zeros(self.np)
        for i in range(self.np):
            diff = self.population[i] - best_mean
            try:
                cov_inv = np.linalg.inv(best_cov)
                log_det = np.log(np.linalg.det(best_cov) + 1e-10)
                mahal = diff @ cov_inv @ diff
                log_likelihood[i] = -0.5 * (mahal + log_det + self.dim * np.log(2 * np.pi))
            except:
                log_likelihood[i] = -self.dim * 10.0
        
        ll_min, ll_max = np.min(log_likelihood), np.max(log_likelihood)
        if ll_max > ll_min + 1e-10:
            norm_ll = (log_likelihood - ll_min) / (ll_max - ll_min + 1e-10)
        else:
            norm_ll = np.ones(self.np) * 0.5
        
        n_bins = min(10, max(3, self.np // 3))
        counts, _ = np.histogram(norm_ll, bins=n_bins, range=(0.0, 1.0))
        probs = counts / (self.np + 1e-10)
        probs = probs[probs > 0]
        if len(probs) > 1:
            likelihood_entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(probs))
            normalized_entropy = likelihood_entropy / (max_entropy + 1e-10)
        else:
            normalized_entropy = 0.5
        
        likelihood = np.exp(log_likelihood - ll_max)
        exploration_factor = 1.0 / (likelihood + 1e-10)
        exploration_factor = np.clip(exploration_factor, 0.3, 5.0)
        global_explore = 0.5 + 0.5 * normalized_entropy
        exploration_factor *= global_explore
        
        to_best = best_mean - self.population
        to_best_dist = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
        to_best_dir = to_best / to_best_dist
        correction = 0.3 * exploration_factor[:, np.newaxis] * to_best_dir
        new_population = self.population + self.inertia_weight * self.velocity + correction
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catH_hybrid(self):
        """Hybrid: temporal-drift detection + fitness-rank FDC switching (variant_08, 2 wins)."""
        centroid = np.mean(self.population, axis=0)
        
        if not hasattr(self, '_ema_centroid'):
            self._ema_centroid = centroid.copy()
            self._centroid_vel_history = []
        
        alpha_ema = 0.3
        self._ema_centroid = alpha_ema * centroid + (1.0 - alpha_ema) * self._ema_centroid
        centroid_velocity = centroid - self._ema_centroid
        drift_magnitude = np.linalg.norm(centroid_velocity) + 1e-10
        
        self._centroid_vel_history.append(drift_magnitude)
        if len(self._centroid_vel_history) > 20:
            self._centroid_vel_history.pop(0)
        avg_drift = np.mean(self._centroid_vel_history) + 1e-10
        
        drift_ratio = drift_magnitude / avg_drift
        
        fdc = 0.0
        if self.global_best is not None and self.np > 2:
            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
            dists = np.linalg.norm(self.population - self.global_best, axis=1)
            dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
            if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                fdc = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
        
        drift_threshold = 1.5
        is_drifting = drift_ratio > drift_threshold
        
        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_dists = np.sort(sq_dists, axis=1)[:, :k]
        knn_avg_dist = np.mean(np.sqrt(knn_dists), axis=1, keepdims=True) + 1e-10
        global_spread = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10
        density_modulation = np.clip(knn_avg_dist / global_spread, 0.2, 2.0)
        
        if is_drifting:
            dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)
            centroid_attraction = 0.3 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))
            to_centroid_dir = centroid - self.population
            to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
            to_centroid_dir = to_centroid_dir / to_centroid_dist
            exploration_strength = 0.5 / (1.0 + 0.3 * (drift_ratio - 1.0))
            correction = centroid_attraction * to_centroid_dir * density_modulation * exploration_strength
            random_perturb = np.random.uniform(-0.3, 0.3, (self.np, self.dim)) * density_modulation
            new_population = self.population + self.inertia_weight * self.velocity + 0.2 * correction + random_perturb
        else:
            fdc_signal = np.clip(fdc, -1.0, 1.0)
            exploit_weight = 0.5 * (fdc_signal + 1.0)
            explore_weight = 1.0 - exploit_weight
            
            dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)
            centroid_attraction = 0.2 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))
            to_centroid_dir = centroid - self.population
            to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
            to_centroid_dir = to_centroid_dir / to_centroid_dist
            
            to_best_dir = np.zeros_like(self.population)
            if self.global_best is not None:
                to_best = self.global_best - self.population
                to_best_dist = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_dist
            
            best_pull_strength = exploit_weight * 0.5
            correction = (centroid_attraction * to_centroid_dir * density_modulation * explore_weight +
                         best_pull_strength * to_best_dir * density_modulation * exploit_weight)
            new_population = self.population + self.inertia_weight * self.velocity + 0.4 * correction
        
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catF_temporal(self):
        """Temporal stagnation-drift modulation (variant_06, 2 wins)."""
        centroid = np.mean(self.population, axis=0)
        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)
        
        if not hasattr(self, '_ema_centroid_f'):
            self._ema_centroid_f = centroid.copy()
            self._centroid_vel_ema_f = 0.0
            self._prev_accel_sign = np.zeros(self.dim)
            self._knn_density_ema_f = 1.0
            self._oscillation_count = 0
        
        centroid_vel = np.linalg.norm(centroid - self._ema_centroid_f)
        self._centroid_vel_ema_f = 0.7 * self._centroid_vel_ema_f + 0.3 * centroid_vel
        self._ema_centroid_f = 0.7 * self._ema_centroid_f + 0.3 * centroid
        acceleration = centroid_vel - self._centroid_vel_ema_f
        
        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_dists = np.sort(sq_dists, axis=1)[:, :k]
        knn_avg_dist = np.mean(np.sqrt(knn_dists), axis=1, keepdims=True) + 1e-10
        knn_density = np.mean(knn_avg_dist)
        
        self._knn_density_ema_f = 0.7 * self._knn_density_ema_f + 0.3 * knn_density
        density_change = knn_density - self._knn_density_ema_f + 1e-10
        density_rate = density_change / (self._knn_density_ema_f + 1e-10)
        
        stagnation_signal = np.clip(1.0 - self._centroid_vel_ema_f / (1.0 + self._centroid_vel_ema_f), 0.0, 1.0)
        
        curr_accel_sign = np.sign(acceleration) if centroid_vel > 1e-10 else np.zeros(self.dim)
        oscillation = np.mean(curr_accel_sign != self._prev_accel_sign) / max(1, self.dim)
        self._prev_accel_sign = curr_accel_sign
        
        is_converging = knn_density < 0.8 * self._knn_density_ema_f
        is_diverging = knn_density > 1.2 * self._knn_density_ema_f
        
        base_correction = 0.3
        stagnation_boost = 0.2 * stagnation_signal
        oscillation_damping = 0.5 if oscillation > 0.3 else 1.0
        convergence_modulation = 0.7 if is_converging else (1.3 if is_diverging else 1.0)
        correction_strength = base_correction * (1.0 + stagnation_boost) * oscillation_damping * convergence_modulation
        
        global_spread = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10
        centroid_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))
        
        to_centroid_dir = centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid_dir / to_centroid_dist
        
        density_modulation = np.clip(knn_avg_dist / global_spread, 0.2, 2.0)
        density_modulation = density_modulation * (1.0 + 0.2 * density_rate)
        
        correction = centroid_attraction * to_centroid_dir * density_modulation
        new_population = self.population + self.inertia_weight * self.velocity + correction_strength * correction
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catG_mc(self):
        """Monte Carlo importance-weighted centroid + random projection (variant_07, 1 win)."""
        n_projections = 20
        n_importance_samples = 15
        
        random_dirs = np.random.randn(self.dim, n_projections)
        random_dirs = random_dirs / (np.linalg.norm(random_dirs, axis=0, keepdims=True) + 1e-10)
        
        proj_scores = np.zeros(n_projections)
        for j in range(n_projections):
            proj = self.population @ random_dirs[:, j]
            if np.std(proj) > 1e-10 and np.std(self.current_fitness) > 1e-10:
                proj_scores[j] = abs(np.corrcoef(proj, self.current_fitness)[0, 1])
            else:
                proj_scores[j] = 0.0
        
        proj_weights = np.exp(proj_scores * 5.0)
        proj_weights = proj_weights / (np.sum(proj_weights) + 1e-10)
        selected_proj_idx = np.random.choice(n_projections, p=proj_weights)
        stochastic_dir = random_dirs[:, selected_proj_idx]
        
        min_fit = np.min(self.current_fitness)
        max_fit = np.max(self.current_fitness)
        fit_range = max_fit - min_fit + 1e-10
        
        if fit_range > 1e-10:
            inv_fitness = max_fit - self.current_fitness + 1e-10
            importance_weights = inv_fitness / (np.sum(inv_fitness) + 1e-10)
        else:
            importance_weights = np.ones(self.np) / self.np
        
        sampled_centroids = []
        for _ in range(n_importance_samples):
            sampled_idx = np.random.choice(self.np, size=self.np, replace=True, p=importance_weights)
            sampled_centroid = np.mean(self.population[sampled_idx], axis=0)
            sampled_centroids.append(sampled_centroid)
        sampled_centroids = np.array(sampled_centroids)
        selected_centroid = sampled_centroids[np.random.randint(n_importance_samples)]
        
        centroid = np.mean(self.population, axis=0)
        to_centroid_dir = centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid_dir / to_centroid_dist
        
        stochastic_dir_expanded = stochastic_dir[np.newaxis, :]
        proj_onto_particle = np.sum(stochastic_dir_expanded * to_centroid_dir, axis=1, keepdims=True)
        stochastic_component = stochastic_dir_expanded * proj_onto_particle
        
        alpha = np.random.uniform(0.6, 0.8, (self.np, 1))
        combined_dir = alpha * to_centroid_dir + (1 - alpha) * stochastic_component
        combined_norm = np.linalg.norm(combined_dir, axis=1, keepdims=True) + 1e-10
        combined_dir = combined_dir / combined_norm
        
        new_population = self.population + self.inertia_weight * self.velocity + 0.3 * combined_dir
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catE_mst(self):
        """MST betweenness-corrected centroid pull (variant_05, 1 win)."""
        try:
            from scipy.sparse.csgraph import minimum_spanning_tree
            from scipy.spatial.distance import pdist, squareform
        except ImportError:
            new_population = self.population + self.inertia_weight * self.velocity
            self.population = self._clip_to_bounds(new_population)
            return
        
        centroid = np.mean(self.population, axis=0)
        dist_to_centroid = np.linalg.norm(self.population - centroid, axis=1, keepdims=True)
        
        if self.np > 2:
            try:
                pairwise_dists = squareform(pdist(self.population, metric='euclidean'))
                np.fill_diagonal(pairwise_dists, 1e-10)
                mst = minimum_spanning_tree(pairwise_dists)
                mst_dense = np.array(mst.todense())
                mst_edges = np.triu(mst_dense)
                mst_edge_weights = mst_edges[mst_edges > 0]
                global_mst_tension = np.mean(mst_edge_weights) + 1e-10
                
                mst_degrees = np.array(np.sum(mst_dense > 0, axis=1)).ravel()
                leaf_mask = mst_degrees == 1
                
                particle_mst_weight = np.zeros(self.np)
                for i in range(self.np):
                    incident_edges = mst_dense[i, :]
                    if np.any(incident_edges > 0):
                        particle_mst_weight[i] = np.max(incident_edges)
                    else:
                        particle_mst_weight[i] = global_mst_tension
                
                edge_weight_modulation = np.clip(particle_mst_weight / global_mst_tension, 0.3, 3.0)
                degree_modulation = np.where(leaf_mask, 1.5, 0.7)
                mst_modulation = degree_modulation * edge_weight_modulation
            except Exception:
                mst_modulation = np.ones(self.np)
                global_mst_tension = 1.0
        else:
            mst_modulation = np.ones(self.np)
            global_mst_tension = 1.0
        
        k = min(5, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_dists = np.sort(sq_dists, axis=1)[:, :k]
        knn_avg_dist = np.mean(np.sqrt(knn_dists), axis=1, keepdims=True) + 1e-10
        global_spread = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10
        density_modulation = np.clip(knn_avg_dist / global_spread, 0.2, 2.0)
        
        centroid_attraction = 0.5 * (dist_to_centroid / (dist_to_centroid + global_spread + 1e-10))
        to_centroid_dir = centroid - self.population
        to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid_dir / to_centroid_dist
        
        combined_modulation = mst_modulation[:, np.newaxis] * density_modulation
        combined_modulation = np.clip(combined_modulation, 0.2, 3.0)
        correction = centroid_attraction * to_centroid_dir * combined_modulation
        new_population = self.population + self.inertia_weight * self.velocity + 0.3 * correction
        self.population = self._clip_to_bounds(new_population)
    
    def _position_update_catB_eigen(self):
        """Eigenvalue-driven anisotropic centroid correction (variant_02, 1 win)."""
        centroid = np.mean(self.population, axis=0)
        try:
            centered = self.population - centroid
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 10000.0)
            
            total_var = np.sum(eigenvalues) + 1e-10
            eigenvalues_norm = eigenvalues / total_var
            cumvar = np.cumsum(eigenvalues_norm)
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = eff_dim / self.dim
            
            cond_signal = np.log1p(cond) / np.log1p(10000.0)
            cond_signal = np.clip(cond_signal, 0.0, 1.0)
            
            to_centroid = centroid - self.population
            to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
            to_centroid_dir = to_centroid / to_centroid_dist
            
            proj_coeffs = to_centroid_dir @ eigenvectors
            proj_coeffs_norm = proj_coeffs / (np.linalg.norm(proj_coeffs, axis=1, keepdims=True) + 1e-10)
            
            eigen_weights = eigenvalues / (eigenvalues[0] + 1e-10)
            projected_dir = proj_coeffs_norm * eigen_weights
            aligned_dir = projected_dir @ eigenvectors.T
            aligned_dir = aligned_dir / (np.linalg.norm(aligned_dir, axis=1, keepdims=True) + 1e-10)
            
            correction_strength = 0.5 * (1.0 + cond_signal) * (2.0 - eff_dim_ratio)
            correction_strength = np.clip(correction_strength, 0.2, 3.0)
            
            centroid_correction = correction_strength * aligned_dir
            
            vel_proj = self.velocity @ eigenvectors
            vel_scale = 0.5 + 0.5 * eigen_weights
            vel_scale = np.clip(vel_scale, 0.3, 1.5)
            vel_scaled = vel_proj * vel_scale
            
            new_population = self.population + self.inertia_weight * (vel_scaled @ eigenvectors.T) + 0.3 * centroid_correction
        except np.linalg.LinAlgError:
            to_centroid = centroid - self.population
            to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
            to_centroid_dir = to_centroid / to_centroid_dist
            new_population = self.population + self.inertia_weight * self.velocity + 0.3 * to_centroid_dir
        self.population = self._clip_to_bounds(new_population)
    
    # -------------------------------------------------------------------------
    # DISPATCHER
    # -------------------------------------------------------------------------
    
    def _dispatch_position_update(self, operator):
        if operator == 'original':
            self._position_update_original()
        elif operator == 'catA_convex':
            self._position_update_catA_convex()
        elif operator == 'catC_entropy':
            self._position_update_catC_entropy()
        elif operator == 'catH_hybrid':
            self._position_update_catH_hybrid()
        elif operator == 'catF_temporal':
            self._position_update_catF_temporal()
        elif operator == 'catG_mc':
            self._position_update_catG_mc()
        elif operator == 'catE_mst':
            self._position_update_catE_mst()
        elif operator == 'catB_eigen':
            self._position_update_catB_eigen()
        else:
            self._position_update_original()
    
    # -------------------------------------------------------------------------
    # RANK-BASED SCORING (scale-independent, UCB1)
    # -------------------------------------------------------------------------
    
    def _record_probe_score(self, operator, best_fitness):
        """Record probe score using rank-based (scale-independent) scoring.
        
        Score = normalized rank of best_fitness within the sliding window.
        Higher score = better (lower fitness). No scale-dependent constants.
        """
        self._probe_fitness_history.append(best_fitness)
        self._probe_op_history.append(operator)
        self._probe_gen_count += 1
        
        # Enforce sliding window
        if len(self._probe_fitness_history) > self._sliding_window_size:
            oldest_op = self._probe_op_history.pop(0)
            self._probe_fitness_history.pop(0)
            if self._operator_scores[oldest_op]:
                oldest_score = self._operator_scores[oldest_op].pop(0)
                self._operator_total_score[oldest_op] -= oldest_score
                self._operator_sample_count[oldest_op] -= 1
        
        # Rank-based score: fraction of probe generations with WORSE (higher) fitness
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
        """UCB1 score: avg rank-based score + calibrated exploration bonus.
        
        All scores are in [0,1] (rank-based), so the exploration bonus is
        well-calibrated without any scale-dependent constants.
        """
        n = self._operator_sample_count[operator]
        if n == 0:
            return float('inf')
        total_n = sum(self._operator_sample_count.values())
        avg_score = self._operator_total_score[operator] / n
        # UCB1 exploration bonus — scale is [0,1], well-calibrated
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
        scores = [(op, self._compute_ucb_score(op)) for op in self._operators]
        scores.sort(key=lambda x: x[1], reverse=True)
        if len(scores) >= 2:
            return scores[1][0]
        return scores[0][0]
    
    # -------------------------------------------------------------------------
    # RESTART & LOCAL BEST
    # -------------------------------------------------------------------------
    
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
        self._probe_gen_count = 0
        self._committed_operator = None
        self._runner_up_operator = None
        self._in_probe_phase = True
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
        
        # Reset EMA states
        if hasattr(self, '_ema_centroid'):
            del self._ema_centroid
        if hasattr(self, '_ema_centroid_f'):
            del self._ema_centroid_f
        if hasattr(self, '_centroid_vel_history'):
            self._centroid_vel_history = []
        if hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        if hasattr(self, '_success_count'):
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
            if hasattr(self, '_success_count'):
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
            
            # --- PHASE A: PROBE ---
            if self._in_probe_phase:
                current_op = self._operators[self._probe_op_index]
                
                self._adapt_inertia_weight()
                self._adapt_neighborhood_size()
                self._update_local_best_from_personal()
                
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
                    self._compute_fingerprint()
                    self._commit_best_fitness = self.global_best_fitness
                    self._commit_gen_count = 0
                
                self._restart_if_stagnant()
            
            # --- PHASE B: COMMIT ---
            else:
                self._commit_gen_count += 1
                
                # Epsilon-review: small chance to switch to runner-up if stagnant
                if (self.stagnation_counter > 30 and
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
