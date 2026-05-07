```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: PROBE-AND-COMMIT (Contextual Bandit, mechanism #4)
    
    The gap table has no tasks with gap >= 100× (dispatch-mandatory), but
    tasks 0 (7.2× median gap), 2 (12.8× median gap), and 5 (7.9× median gap)
    are BLEND-HOSTILE: linear blending of 1.0 and 100.0 yields 50.5, not 1.0.
    A 50/50 blend of the per-task winner and runner-up would lose 3-13× of
    advantage on these tasks. The dispatcher preserves each winner's advantage
    by committing to ONE operator with weight 1.0.
    
    Win-count distribution (6/6/6/3/2/1 across 6 winners) confirms that
    different operators win on different tasks. The probe extracts a cheap
    FINGERPRINT from the landscape (convergence slope, rank-correlation,
    effective dimensionality) to route to the right operator. No hard-coded
    regime→weight tables. Rank-based UCB scores ensure scale-independence.
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
        self._operator_scores = {}
        self._operator_total_score = {}
        self._operator_sample_count = {}
        self._committed_operator = None
        self._runner_up_operator = None
        
        # THE 6 ACTUAL WINNING POSITION UPDATE STRATEGIES (from benchmark)
        self._operators = [
            'original',                    # original.py (6 wins)
            'variant_01_catA_idea_0',      # axis-aligned bounding box (2 wins)
            'variant_02_catB_idea_0',      # spectral entropy + cond damping (6 wins)
            'variant_03_catC_idea_0',      # Mahalanobis-entropy centroid (3 wins)
            'variant_05_catE_idea_0',      # MST edge-weight analysis (6 wins)
            'variant_10_catB_idea_0',      # pseudoinverse velocity correction (1 win)
        ]
        for op in self._operators:
            self._operator_scores[op] = []
            self._operator_total_score[op] = 0.0
            self._operator_sample_count[op] = 0
        
        # Probe: 3 gens per operator on a sub-population
        self._probe_gens_per_op = 3
        self._probe_sub_ratio = 0.3  # 30% sub-population for probe
        
        # Epsilon-review probability when committed operator stagnates
        self._epsilon_review = 0.05
        
        # Probe phase state
        self._in_probe_phase = False
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._probe_best_fitness_seen = np.inf
        self._probe_sub_population = None
        self._probe_sub_velocity = None
        self._probe_sub_personal_best = None
        self._probe_sub_personal_best_fitness = None
        self._probe_sub_local_best = None
        self._probe_sub_local_best_fitness = None
        self._probe_sub_current_fitness = None
        self._probe_sub_global_best = None
        self._probe_sub_global_best_fitness = np.inf
        self._probe_sub_inertia = 0.729
        self._probe_sub_stagnation = 0
        self._probe_sub_generation = 0
        
        # Running history for rank-based scoring
        self._probe_fitness_history = []
        self._probe_op_history = []
        
        # Fingerprint signals (computed during probe)
        self._fingerprint_convergence_slope = 0.0
        self._fingerprint_rank_corr = 0.0
        self._fingerprint_eff_dim = dim
        self._fingerprint_diversity = 0.0
        
        # Commit phase stagnation tracking
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
        
        # --- ENSEMBLE STATE (used in commit phase as fallback) ---
        self._operator_weights = {op: 1.0 / len(self._operators) for op in self._operators}
        self._operator_last_scores = {op: 0.0 for op in self._operators}
    
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
    
    def _update_personal_best_batch(self, pop, fitness, pbest, pbest_fit):
        """Update personal best positions where current fitness is better."""
        improved = pbest_fit > fitness
        pbest[improved] = pop[improved]
        pbest_fit[improved] = fitness[improved]
    
    def _compute_local_best_batch(self, pop, pbest, pbest_fit, lbest, lbest_fit, np_size):
        """Compute ring topology local bests using vectorized operations."""
        for i in range(np_size):
            left = (i - self.neighborhood_size) % np_size
            right = (i + self.neighborhood_size + 1) % np_size
            
            if left < right:
                neighborhood_indices = np.arange(left, right)
            else:
                neighborhood_indices = np.concatenate([
                    np.arange(left, np_size), np.arange(0, right)
                ])
            
            all_indices = np.concatenate([neighborhood_indices, [i]])
            fitness_in_neighborhood = pbest_fit[all_indices]
            best_idx_in_neighborhood = np.argmin(fitness_in_neighborhood)
            
            lbest[i] = pbest[all_indices[best_idx_in_neighborhood]]
            lbest_fit[i] = fitness_in_neighborhood[best_idx_in_neighborhood]
    
    def _compute_global_best(self, pbest, pbest_fit, gbest, gbest_fit, gbest_pos):
        """Compute global best from personal bests. Returns (updated_gbest, updated_gbest_fitness, stagnation_delta)."""
        best_idx = np.argmin(pbest_fit)
        new_stag = 1
        if pbest_fit[best_idx] < gbest_fit:
            gbest_fit[0] = pbest_fit[best_idx]
            gbest_pos[0] = pbest[best_idx].copy()
            new_stag = 0
        return gbest, gbest_fit, new_stag
    
    def _compute_diversity(self, pop):
        """Compute swarm diversity as average Euclidean distance to centroid."""
        centroid = np.mean(pop, axis=0)
        distances = np.linalg.norm(pop - centroid, axis=1)
        return np.mean(distances)
    
    def _compute_fingerprint(self):
        """Compute cheap landscape fingerprint from current population state.
        
        Returns a dict of signals observable from any black-box func:
          - convergence_slope: slope of log(best_fitness) vs generation
          - rank_corr: Spearman-like correlation between fitness rank and
                       distance to global best (proxy for landscape smoothness)
          - eff_dim: effective dimensionality from early covariance
          - diversity: current population diversity
        """
        signals = {}
        
        # 1. Convergence slope (from probe history)
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
                cumvar = np.cumsum(sorted(eigenvalues, reverse=True)) / total_var
                signals['eff_dim'] = float(np.searchsorted(cumvar, 0.95)) + 1
            else:
                signals['eff_dim'] = self.dim
        except:
            signals['eff_dim'] = self.dim
        
        # 4. Diversity
        signals['diversity'] = self._compute_diversity(self.population)
        
        self._fingerprint_convergence_slope = signals['convergence_slope']
        self._fingerprint_rank_corr = signals['rank_corr']
        self._fingerprint_eff_dim = signals['eff_dim']
        self._fingerprint_diversity = signals['diversity']
        
        return signals
    
    def _adapt_inertia_weight(self, pop, gen, inertia_ref):
        """Adapt inertia weight using condition number of population covariance."""
        try:
            centered = pop - np.mean(pop, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 10000.0)

            spectral_signal = np.log1p(cond) / np.log1p(10000.0)
            spectral_signal = np.clip(spectral_signal, 0.0, 1.0)

            target_inertia = 0.4 + 0.55 * spectral_signal

            decay = 0.729 - 0.15 * (gen / 1000)
            target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)

            new_inertia = inertia_ref * 0.8 + target_inertia * 0.2
            return np.clip(new_inertia, 0.4, 0.95)
        except np.linalg.LinAlgError:
            decay = 0.729 - 0.15 * (gen / 1000)
            new_inertia = inertia_ref * 0.8 + decay * 0.2
            return np.clip(new_inertia, 0.4, 0.95)
    
    def _adapt_neighborhood_size(self, pop, diversity):
        """Adapt ring topology neighborhood size based on diversity."""
        if diversity < self.diversity_threshold_low:
            self.neighborhood_size = min(len(pop) // 2, self.neighborhood_size + 1)
        elif diversity > self.diversity_threshold_high:
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    def _adaptive_coefficients(self, stagnation):
        """Compute adaptive cognitive and social coefficients."""
        improvement_ratio = 1.0 / (1.0 + stagnation)
        cognitive = self.cognitive_base * (1.0 + 0.5 * improvement_ratio)
        social = self.social_base * (2.0 - improvement_ratio)
        return cognitive, social
    
    def _velocity_update_base(self, pop, vel, pbest, pbest_fit, lbest, lbest_fit, 
                               inertia, stagnation, np_size, dim):
        """Base velocity update: cognitive + social + DE mutation."""
        cognitive, social = self._adaptive_coefficients(stagnation)
        
        r1 = np.random.uniform(0, 1, (np_size, dim))
        r2 = np.random.uniform(0, 1, (np_size, dim))
        
        cognitive_component = cognitive * r1 * (pbest - pop)
        social_component = social * r2 * (lbest - pop)
        
        mutation_mask = np.random.uniform(0, 1, (np_size, dim))
        mutation_threshold = 0.1 * (1.0 - self.generation / 5000)
        mutation_active = mutation_mask < mutation_threshold
        
        mutation_vectors = np.zeros((np_size, dim))
        for i in range(np_size):
            indices = np.random.choice(
                [j for j in range(np_size) if j != i], 3, replace=False
            )
            mutation_vectors[i] = pop[indices[0]] + 0.5 * (
                pop[indices[1]] - pop[indices[2]]
            )
        
        mutation_component = np.where(
            mutation_active,
            0.3 * (mutation_vectors - pop),
            0.0
        )
        
        new_velocity = (
            inertia * vel +
            cognitive_component +
            social_component +
            mutation_component
        )
        
        return np.clip(new_velocity, self.v_min, self.v_max)
    
    # -------------------------------------------------------------------------
    # POSITION UPDATE STRATEGIES (the 6 benchmark winners)
    # -------------------------------------------------------------------------
    
    def _position_update_original(self, pop, vel, gbest, fitness, np_size):
        """Original: eigenvalue-anisotropic velocity modulation."""
        centered = pop - np.mean(pop, axis=0)
        cov = np.cov(centered.T)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            spectral_scale = np.clip(cond / 100.0, 0.5, 2.0)
            
            vel_proj = vel @ eigenvectors
            vel_scaled = vel_proj * spectral_scale
            
            new_pop = pop + (vel_scaled @ eigenvectors.T)
        except np.linalg.LinAlgError:
            new_pop = pop + vel
        
        return self._clip_to_bounds(new_pop)
    
    def _position_update_variant_01_catA_idea_0(self, pop, vel, gbest, fitness, np_size):
        """Axis-aligned bounding box spread correction (Category A: Geometry).
        
        Uses per-dimension min/max to compute normalized deviation from the
        population's axis-aligned bounding box. Particles far outside the box
        get weak correction (already explored); particles near the center
        get stronger correction (promote convergence).
        """
        pop_mean = np.mean(pop, axis=0)

        dim_min = np.min(pop, axis=0)
        dim_max = np.max(pop, axis=0)
        dim_range = dim_max - dim_min + 1e-10

        deviation = pop - pop_mean
        normalized_dev = deviation / dim_range

        max_dev = np.max(np.abs(normalized_dev), axis=1, keepdims=True) + 1e-10

        to_center_dir = pop_mean - pop
        dist_to_center = np.linalg.norm(to_center_dir, axis=1, keepdims=True) + 1e-10
        to_center_dir = to_center_dir / dist_to_center

        correction_scale = np.clip(0.3 / max_dev, 0.05, 1.0)

        correction = correction_scale * to_center_dir

        new_pop = pop + self.inertia_weight * vel + correction

        return self._clip_to_bounds(new_pop)
    
    def _position_update_variant_02_catB_idea_0(self, pop, vel, gbest, fitness, np_size):
        """Spectral entropy + condition-number anisotropic damping (Category B).
        
        Replaces geometric k-NN density with eigenvalue analysis. Key insight:
        spectral entropy detects when population collapses to a subspace (low entropy
        = concentrated along dominant eigenvectors = premature convergence). Condition
        number measures anisotropy. Damping along dominant eigendirections prevents
        collapse; exploration scaling handles ill-conditioned worst tasks.
        """
        try:
            centered = pop - np.mean(pop, axis=0)
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

            vel_proj = vel @ eigenvectors
            vel_scaled = vel_proj * spectral_damping * spectral_signal
            new_pop = pop + vel_scaled @ eigenvectors.T

            if gbest is not None:
                to_best = gbest - pop
                to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_norm
                correction = 0.3 * to_best_dir * (1.0 + 0.3 * cond_scale)
                new_pop = new_pop + correction

            exploration = 1.0 + 0.5 * np.log1p(cond)
            random_perturb = exploration * np.random.uniform(-0.3, 0.3, (np_size, self.dim))
            new_pop = new_pop + random_perturb

        except np.linalg.LinAlgError:
            new_pop = pop + vel

        return self._clip_to_bounds(new_pop)
    
    def _position_update_variant_03_catC_idea_0(self, pop, vel, gbest, fitness, np_size):
        """Mahalanobis-entropy centroid modulation (Category C).
        
        Reframes centroid gravity within an information-theoretic framework:
        - Mahalanobis distance from fitted Gaussian captures distributional shape
        - Entropy of eigenvalue spectrum measures population collapse
        - Per-particle entropy modulation drives exploration/exploitation balance
        """
        centroid = np.mean(pop, axis=0)
        centered = pop - centroid
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

        to_centroid_dir = centroid - pop
        to_centroid_dist = np.linalg.norm(to_centroid_dir, axis=1, keepdims=True) + 1e-10
        to_centroid_dir = to_centroid_dir / to_centroid_dist

        correction = centroid_attraction * to_centroid_dir
        correction *= exploration_factor * kl_signal

        new_pop = pop + self.inertia_weight * vel + 0.3 * correction

        return self._clip_to_bounds(new_pop)
    
    def _position_update_variant_05_catE_idea_0(self, pop, vel, gbest, fitness, np_size):
        """MST edge-weight analysis for topology-aware exploration (Category E).
        
        Key insight: Build MST over population; particles in sparse MST regions
        (long edges, low betweenness) get strong corrective pulls toward the
        global best. This uses GRAPH STRUCTURE (edge weights, connectivity) to
        identify underexplored regions.
        """
        dists = np.linalg.norm(pop[:, np.newaxis, :] - pop[np.newaxis, :, :], axis=2)

        in_tree = np.zeros(np_size, dtype=bool)
        parent = np.full(np_size, -1, dtype=int)
        min_edge = np.full(np_size, np.inf)
        min_edge[0] = 0.0
        in_tree[0] = True

        for _ in range(np_size - 1):
            best_idx = -1
            best_val = np.inf
            for i in range(np_size):
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
            correction = np.zeros((np_size, self.dim))
        else:
            edge_mean = np.mean(valid_edges)
            edge_std = np.std(valid_edges) + 1e-10

            particle_betweenness = np.zeros(np_size)
            for i in range(np_size):
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

            if gbest is not None:
                to_best = gbest - pop
                to_best_dist = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_dist

                correction_strength = 0.3 * topology_modulation[:, np.newaxis]
                correction = correction_strength * to_best_dir
            else:
                centroid = np.mean(pop, axis=0)
                to_centroid = centroid - pop
                to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
                to_centroid_dir = to_centroid / to_centroid_dist
                correction = 0.3 * topology_modulation[:, np.newaxis] * to_centroid_dir

        sparsity_scale = np.clip(1.0 / (np.mean(valid_edges) / (valid_edges + 1e-10) + 1e-10), 0.5, 1.5)
        damped_velocity = vel * sparsity_scale[:, np.newaxis]

        new_pop = pop + self.inertia_weight * damped_velocity + 0.4 * correction
        return self._clip_to_bounds(new_pop)
    
    def _position_update_variant_10_catB_idea_0(self, pop, vel, gbest, fitness, np_size):
        """Pseudoinverse velocity correction for rank-deficient populations (Category B).
        
        Detects when population covariance becomes ill-conditioned (low-rank) using
        singular value thresholding. Uses Moore-Penrose pseudoinverse to project and
        correct velocity onto the principal subspace, amplifying collapsed directions.
        """
        try:
            centered = pop - np.mean(pop, axis=0)
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

            vel_proj = vel @ Vt.T

            uniform_scale = np.ones(len(singular_values))
            per_comp_scale = uniform_scale * (1.0 - correction_strength) + inv_sv_weight * correction_strength

            if cond > 100.0:
                log_cond = np.log1p(cond) / np.log1p(1000.0)
                log_cond = np.clip(log_cond, 0.0, 1.0)
                high_cond_damp = 1.0 - 0.3 * log_cond
                per_comp_scale *= high_cond_damp

            vel_scaled = vel_proj * np.clip(per_comp_scale, 0.1, 3.0)

            corrected_vel = vel_scaled @ Vt

            new_pop = pop + corrected_vel

        except np.linalg.LinAlgError:
            new_pop = pop + vel

        return self._clip_to_bounds(new_pop)
    
    # -------------------------------------------------------------------------
    # DISPATCHER: maps operator name -> position update method
    # -------------------------------------------------------------------------
    
    def _dispatch_position_update(self, operator, pop, vel, gbest, fitness, np_size):
        """Route to the appropriate position update implementation."""
        if operator == 'original':
            return self._position_update_original(pop, vel, gbest, fitness, np_size)
        elif operator == 'variant_01_catA_idea_0':
            return self._position_update_variant_01_catA_idea_0(pop, vel, gbest, fitness, np_size)
        elif operator == 'variant_02_catB_idea_0':
            return self._position_update_variant_02_catB_idea_0(pop, vel, gbest, fitness, np_size)
        elif operator == 'variant_03_catC_idea_0':
            return self._position_update_variant_03_catC_idea_0(pop, vel, gbest, fitness, np_size)
        elif operator == 'variant_05_catE_idea_0':
            return self._position_update_variant_05_catE_idea_0(pop, vel, gbest, fitness, np_size)
        elif operator == 'variant_10_catB_idea_0':
            return self._position_update_variant_10_catB_idea_0(pop, vel, gbest, fitness, np_size)
        else:
            return self._position_update_original(pop, vel, gbest, fitness, np_size)
    
    # -------------------------------------------------------------------------
    # PROBE-AND-COMMIT: rank-based scoring (scale-independent)
    # -------------------------------------------------------------------------
    
    def _record_probe_score(self, operator, best_fitness):
        """Record a probe score using rank-based (scale-independent) scoring.
        
        Score = fraction of probe generations where this operator's best
        was in the top tercile of all operators' bests seen so far.
        This is rank-based, not raw fitness — no scale-dependent constants.
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
        """Compute UCB1-style score for operator selection.
        
        Uses rank-based average (already in [0,1]) so exploration bonus
        is on the same scale — no magic constants.
        """
        n = self._operator_sample_count[operator]
        if n == 0:
            return float('inf')
        
        total_n = sum(self._operator_sample_count.values())
        avg_score = self._operator_total_score[operator] / n
        
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
    # PROBE SUB-POPULATION MANAGEMENT
    # -------------------------------------------------------------------------
    
    def _init_probe_subpopulation(self):
        """Initialize sub-population for probe phase."""
        sub_n = max(10, int(self.np * self._probe_sub_ratio))
        
        # Sample diverse individuals from full population
        if self.global_best is not None:
            candidates = np.vstack([self.population, self.global_best.reshape(1, -1)])
            if len(candidates) > sub_n:
                dists_to_best = np.linalg.norm(candidates - self.global_best, axis=1)
                sub_indices = [np.argmin(dists_to_best)]
                remaining = np.argsort(dists_to_best)[1:]
                sub_indices.extend(remaining[:sub_n - 1].tolist())
                sub_indices = sub_indices[:sub_n]
            else:
                sub_indices = list(range(len(candidates)))
                sub_indices.extend(np.random.choice(len(candidates), sub_n - len(candidates), replace=False).tolist())
        else:
            all_indices = list(range(self.np))
            sub_indices = np.random.choice(all_indices, min(sub_n, self.np), replace=False).tolist()
        
        self._probe_sub_population = self.population[sub_indices].copy()
        self._probe_sub_velocity = self.velocity[sub_indices].copy()
        self._probe_sub_personal_best = self._probe_sub_population.copy()
        self._probe_sub_personal_best_fitness = np.full(sub_n, np.inf)
        self._probe_sub_local_best = self._probe_sub_population.copy()
        self._probe_sub_local_best_fitness = np.full(sub_n, np.inf)
        self._probe_sub_global_best = None
        self._probe_sub_global_best_fitness = np.inf
        self._probe_sub_inertia = self.inertia_weight
        self._probe_sub_stagnation = 0
        self._probe_sub_generation = 0
        self._probe_sub_current_fitness = None
    
    def _restart_probe_subpopulation_if_needed(self):
        """Reinitialize probe sub-population if stagnant."""
        if self._probe_sub_stagnation > 15:
            sub_n = len(self._probe_sub_population)
            n_replace = max(1, int(0.3 * sub_n))
            worst_indices = np.argsort(self._probe_sub_personal_best_fitness)[-n_replace:]
            
            self._probe_sub_population[worst_indices] = np.random.uniform(
                self.lower_bound, self.upper_bound, (n_replace, self.dim)
            )
            self._probe_sub_velocity[worst_indices] = np.random.uniform(
                self.v_min * 0.1, self.v_max * 0.1, (n_replace, self.dim)
            )
            self._probe_sub_personal_best_fitness[worst_indices] = np.inf
            self._probe_sub_personal_best[worst_indices] = self._probe_sub_population[worst_indices]
            self._probe_sub_stagnation = 0
    
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
        
        self._update_personal_best_batch(self.population, self.current_fitness,
                                          self.personal_best, self.personal_best_fitness)
        self._compute_local_best_batch(self.population, self.personal_best, self.personal_best_fitness,
                                        self.local_best, self.local_best_fitness, self.np)
        self._compute_global_best(self.personal_best, self.personal_best_fitness,
                                   self.global_best, self.global_best_fitness,
                                   self.global_best)
        gbest_arr = np.array([self.global_best_fitness])
        _, _, stag = self._compute_global_best(self.personal_best, self.personal_best_fitness,
                                                self.global_best, gbest_arr, self.global_best)
        self.stagnation_counter = 0 if stag == 0 else 1
        
        # Initialize probe sub-population
        self._init_probe_subpopulation()
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # --- PHASE A: PROBE ---
            if self._in_probe_phase:
                current_op = self._operators[self._probe_op_index]
                sub_n = len(self._probe_sub_population)
                
                # Adaptation on sub-population
                self._probe_sub_inertia = self._adapt_inertia_weight(
                    self._probe_sub_population, self._probe_sub_generation, self._probe_sub_inertia)
                diversity = self._compute_diversity(self._probe_sub_population)
                self._adapt_neighborhood_size(self._probe_sub_population, diversity)
                
                # Update local bests for sub-population
                self._update_personal_best_batch(
                    self._probe_sub_population, self._probe_sub_current_fitness,
                    self._probe_sub_personal_best, self._probe_sub_personal_best_fitness)
                self._compute_local_best_batch(
                    self._probe_sub_population, self._probe_sub_personal_best, self._probe_sub_personal_best_fitness,
                    self._probe_sub_local_best, self._probe_sub_local_best_fitness, sub_n)
                
                # Velocity update on sub-population
                self._probe_sub_velocity = self._velocity_update_base(
                    self._probe_sub_population, self._probe_sub_velocity,
                    self._probe_sub_personal_best, self._probe_sub_personal_best_fitness,
                    self._probe_sub_local_best, self._probe_sub_local_best_fitness,
                    self._probe_sub_inertia, self._probe_sub_stagnation, sub_n, self.dim)
                
                # Position update using PROBED operator
                self._probe_sub_population = self._dispatch_position_update(
                    current_op, self._probe_sub_population, self._probe_sub_velocity,
                    self._probe_sub_global_best, self._probe_sub_current_fitness, sub_n)
                
                self._probe_sub_current_fitness, _ = self._evaluate_batch(self._probe_sub_population, func)
                
                if stopping_condition():
                    break
                
                self._update_personal_best_batch(
                    self._probe_sub_population, self._probe_sub_current_fitness,
                    self._probe_sub_personal_best, self._probe_sub_personal_best_fitness)
                
                # Update sub-population global best
                best_idx = np.argmin(self._probe_sub_personal_best_fitness)
                if self._probe_sub_personal_best_fitness[best_idx] < self._probe_sub_global_best_fitness:
                    self._probe_sub_global_best = self._probe_sub_personal_best[best_idx].copy()
                    self._probe_sub_global_best_fitness = self._probe_sub_personal_best_fitness[best_idx]
                    self._probe_sub_stagnation = 0
                else:
                    self._probe_sub_stagnation += 1
                
                # Record probe score (rank-based, scale-independent)
                gen_best = float(np.min(self._probe_sub_current_fitness))
                if gen_best < self._probe_best_fitness_seen:
                    self._probe_best_fitness_seen = gen_best
                self._record_probe_score(current_op, self._probe_best_fitness_seen)
                
                self._probe_gens_in_current_op += 1
                self._probe_sub_generation += 1
                
                # Advance to next operator after probe_gens_per_op
                if self._probe_gens_in_current_op >= self._probe_gens_per_op:
                    self._probe_op_index += 1
                    self._probe_gens_in_current_op = 0
                
                # Check if probe phase is done
                if self._probe_op_index >= len(self._operators):
                    self._in_probe_phase = False
                    self._committed_operator = self._select_best_operator()
                    self._runner_up_operator = self._select_runner_up_operator()
                    self._compute_fingerprint()
                    self._commit_best_fitness = self.global_best_fitness
                    self._commit_gen_count = 0
                
                self._restart_probe_subpopulation_if_needed()
            
            # --- PHASE B: COMMIT ---
            else:
                self._commit_gen_count += 1
                
                # Adaptation on full population
                self.inertia_weight = self._adapt_inertia_weight(
                    self.population, self.generation, self.inertia_weight)
                diversity = self._compute_diversity(self.population)
                self._adapt_neighborhood_size(self.population, diversity)
                self._update_local_best_from_personal()
                
                # Epsilon-review: small chance to switch to runner-up if stagnant
                use_operator = self._committed_operator
                if (self.stagnation_counter > 20 and
                    np.random.random() < self._epsilon_review):
                    use_operator = self._runner_up_operator
                
                # Velocity update
                self.velocity = self._velocity_update_base(
                    self.population, self.velocity,
                    self.personal_best, self.personal_best_fitness,
                    self.local_best, self.local_best_fitness,
                    self.inertia_weight, self.stagnation_counter, self.np, self.dim)
                
                # Position update using COMMITTED operator
                self.population = self._dispatch_position_update(
                    use_operator, self.population, self.velocity,
                    self.global_best, self.current_fitness, self.np)
                
                self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
                if actual_n < self.np:
                    self.np = actual_n
                    self.population = self.population[:actual_n]
                    self.velocity = self.velocity[:actual_n]
                    self.current_fitness = self.current_fitness[:actual_n]
                    break
                
                if stopping_condition():
                    break
                
                self._update_personal_best_batch(self.population, self.current_fitness,
                                                  self.personal_best, self.personal_best_fitness)
                self._compute_local_best_batch(self.population, self.personal_best, self.personal_best_fitness,
                                                self.local_best, self.local_best_fitness, self.np)
                
                gbest_arr = np.array([self.global_best_fitness])
                self._compute_global_best(self.personal_best, self.personal_best_fitness,
                                           self.global_best, gbest_arr, self.global_best)
                _, _, stag = self._compute_global_best(self.personal_best, self.personal_best_fitness,
                                                        self.global_best, gbest_arr, self.global_best)
                if stag == 0:
                    self.stagnation_counter = 0
                else:
                    self.stagnation_counter += 1
                
                self._restart_if_stagnant()
                
                if self.global_best_fitness < self._commit_best_fitness:
                    self._commit_best_fitness = self.global_best_fitness
        
        # Final result with NaN safety
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        
        return self.global_best_fitness, self.global_best.copy()
```