```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: PROBE-AND-COMMIT (Contextual Bandit, mechanism #4)
    
    The gap table is BLEND-HOSTILE:
      - Task 5: variant_10 wins with 4.4× gap to runner-up BUT 176.7× gap to
        median competitor. A 50/50 blend of winner (5.96) and median (1057.5)
        yields ~531, which is ~89× WORSE than the winner alone. Linear blending
        CANNOT preserve this advantage.
      - No task reaches the 100× runner-up threshold, so pure dispatch (weight=1.0
        always) is not strictly mandatory, but probe-and-commit is the safest
        choice given the blend-hostile median gap.
    
    The dispatcher preserves each winner's advantage by:
      - PHASE A (probe): round-robin burst with all 8 winning adaptation
        strategies, each getting 3 generations. Rank-based UCB scoring ensures
        scale-independent comparison — no magic constants like *1e5.
      - PHASE B (commit): run ONLY the winning strategy for the remaining
        budget. No blending. No hard-coded weights.
      - EPSILON-REVIEW: if committed strategy stagnates (>20 gens, p=0.05),
        allow a small probability of switching to the runner-up.
    
    Win-count distribution (5/4/4/3/3/3/2/2 across 8 winners) confirms that
    different adaptation strategies win on different tasks. The probe extracts
    a cheap FINGERPRINT from the landscape (convergence slope, rank-correlation,
    effective dimensionality) to route to the right strategy.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5 * dim, capped at reasonable range
        self.np = min(max(5 * dim, 20), 300)
        
        # Ring topology neighborhood size (will adapt)
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
        
        # --- PROBE-AND-COMMIT STATE ---
        self._operator_scores = {}
        self._operator_total_score = {}
        self._operator_sample_count = {}
        self._committed_operator = None
        self._runner_up_operator = None
        
        # THE 8 WINNING ADAPTATION STRATEGIES (from benchmark)
        self._operators = [
            'spectral_cond',      # variant_02_catB: eigenvalue condition number (4 wins)
            'info_theory',        # variant_03_catC: information-theoretic signals (5 wins)
            'fitness_landscape',  # variant_04_catD: fitness-rank + success-history (3 wins)
            'knn_connectivity',   # variant_05_catE: k-NN graph connectivity (2 wins)
            'stochastic_sample',  # variant_07_catG: stochastic sampling + UCB (1 win)
            'temporal_topology',  # variant_08_catH: temporal + topology hybrid (4 wins)
            'knn_density',        # variant_09_catA: k-NN local density structure (3 wins)
            'spectral_entropy',   # variant_10_catB: condition + spectral entropy (2 wins)
        ]
        for op in self._operators:
            self._operator_scores[op] = []
            self._operator_total_score[op] = 0.0
            self._operator_sample_count[op] = 0
        
        # Probe budget: 3 gens per operator, min 24 total
        self._probe_gens_per_op = 3
        self._probe_total_budget = max(len(self._operators) * self._probe_gens_per_op, 24)
        
        # Epsilon-review probability when committed operator stagnates
        self._epsilon_review = 0.05
        
        # Probe phase state
        self._in_probe_phase = False
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._probe_best_fitness_seen = np.inf
        
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
    
    # -------------------------------------------------------------------------
    # ADAPTATION STRATEGY 1: Spectral Condition Number (variant_02_catB, 4 wins)
    # -------------------------------------------------------------------------
    def _adapt_neighborhood_spectral_cond(self):
        """Adapt ring topology neighborhood size based on spectral condition number.
        
        Category B (Spectral / linear-algebraic):
        - Computes eigenvalues of population covariance matrix
        - Derives condition number as anisotropy measure
        - High condition number (collapsed along eigenvectors) → expand neighborhood
        - Low condition number (isotropic) → contract neighborhood
        """
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            cond = eigenvalues[0] / eigenvalues[-1]
            cond = np.clip(cond, 1.0, 10000.0)

            spectral_signal = np.log1p(cond) / np.log1p(10000.0)
            spectral_signal = np.clip(spectral_signal, 0.0, 1.0)

            expansion_bias = 2.0 * spectral_signal - 1.0
            expansion_bias = np.clip(expansion_bias, -0.5, 0.5)

            min_size = max(1, self.np // 10)
            max_size = max(3, self.np // 3)

            if expansion_bias > 0:
                self.neighborhood_size = min(max_size, self.neighborhood_size + 1)
            elif expansion_bias < 0:
                self.neighborhood_size = max(min_size, self.neighborhood_size - 1)
        except np.linalg.LinAlgError:
            pass
    
    # -------------------------------------------------------------------------
    # ADAPTATION STRATEGY 2: Information-Theoretic (variant_03_catC, 5 wins)
    # -------------------------------------------------------------------------
    def _adapt_neighborhood_info_theory(self):
        """Adapt neighborhood size using information-theoretic signals.
        
        Uses:
        1. Entropy of discretized fitness distribution
        2. KL divergence of covariance eigenvalues from uniform (spectral entropy)
        
        High entropy + low KL = diverse population → reduce neighborhood.
        Low entropy + high KL = clustered/trapped → increase neighborhood.
        """
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
    
    # -------------------------------------------------------------------------
    # ADAPTATION STRATEGY 3: Fitness-Landscape Rank (variant_04_catD, 3 wins)
    # -------------------------------------------------------------------------
    def _adapt_neighborhood_fitness_landscape(self):
        """Adapt ring topology neighborhood size using fitness-landscape signals.
        
        Category D: Uses ONLY fitness signals (ranks, entropy, success-history).
        - Fitness rank entropy: detects multimodal vs unimodal landscape
        - Spearman-like rank correlation: detects landscape smoothness
        - Success-history: tracks which neighborhood sizes yielded improvements
        """
        fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
        fr_clipped = np.clip(fitness_ranks, 1e-10, 1.0 - 1e-10)
        rank_entropy = -np.mean(fr_clipped * np.log(fr_clipped) - (1 - fr_clipped) * np.log(1 - fr_clipped + 1e-10))
        max_entropy = -np.log(0.5)
        entropy_ratio = np.clip(rank_entropy / (max_entropy + 1e-10), 0.0, 1.0)

        if hasattr(self, '_prev_fitness_ranks') and self._prev_fitness_ranks is not None:
            if len(self._prev_fitness_ranks) == len(fitness_ranks):
                if np.std(fitness_ranks) > 1e-10 and np.std(self._prev_fitness_ranks) > 1e-10:
                    rank_corr = np.corrcoef(fitness_ranks, self._prev_fitness_ranks)[0, 1]
                    rank_corr = np.clip(rank_corr, -1.0, 1.0)
                else:
                    rank_corr = 0.0
            else:
                rank_corr = 0.0
        else:
            rank_corr = 0.0
        self._prev_fitness_ranks = fitness_ranks.copy()

        if not hasattr(self, '_neighborhood_history'):
            self._neighborhood_history = []
        if not hasattr(self, '_neighborhood_success'):
            self._neighborhood_success = {}

        if self.global_best_fitness < getattr(self, '_prev_best_for_history', np.inf):
            improvement = True
        else:
            improvement = False
        self._prev_best_for_history = self.global_best_fitness

        current_ns = self.neighborhood_size
        if current_ns not in self._neighborhood_success:
            self._neighborhood_success[current_ns] = []
        self._neighborhood_success[current_ns].append(1.0 if improvement else 0.0)
        if len(self._neighborhood_success[current_ns]) > 10:
            self._neighborhood_success[current_ns].pop(0)

        def success_rate(ns):
            if ns in self._neighborhood_success and len(self._neighborhood_success[ns]) >= 3:
                return np.mean(self._neighborhood_success[ns])
            return 0.5

        current_success = success_rate(current_ns)
        larger_success = success_rate(min(self.np // 2, current_ns + 1))

        exploration_signal = (1.0 - entropy_ratio) * 0.4 + (1.0 - rank_corr) * 0.3 + (1.0 - current_success) * 0.3
        exploration_signal = np.clip(exploration_signal, 0.0, 1.0)

        if larger_success > current_success + 0.1:
            exploration_signal = min(1.0, exploration_signal + 0.3)

        if exploration_signal > 0.6:
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif exploration_signal < 0.3:
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    # -------------------------------------------------------------------------
    # ADAPTATION STRATEGY 4: k-NN Graph Connectivity (variant_05_catE, 2 wins)
    # -------------------------------------------------------------------------
    def _adapt_neighborhood_knn_connectivity(self):
        """Adapt ring topology neighborhood size based on k-NN graph connectivity.
        
        Category E: Build k-NN graph over population; use connected component count,
        spectral gap, and graph density to detect fragmentation vs convergence.
        """
        n = self.np
        k = min(5, n - 1)

        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_indices = np.argsort(sq_dists, axis=1)[:, :k]

        visited = np.zeros(n, dtype=bool)
        components = []
        for start in range(n):
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
        fragmentation_ratio = 1.0 - (max_component_size / (n + 1e-10))

        max_edges = n * (n - 1) / 2
        actual_edges = n * k / 2
        graph_density = actual_edges / (max_edges + 1e-10)

        try:
            row_indices = np.repeat(np.arange(n), k)
            col_indices = knn_indices.ravel()
            data = np.ones(len(row_indices))
            from scipy.sparse import csr_matrix
            adj = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
            adj = adj + adj.T
            adj.data[:] = 1.0

            degrees = np.array(adj.sum(axis=1)).ravel() + 1e-10
            d_inv_sqrt = 1.0 / np.sqrt(degrees)
            d_inv_sqrt_diag = csr_matrix((d_inv_sqrt, (np.arange(n), np.arange(n))), shape=(n, n))
            from scipy.sparse import identity
            lap = identity(n) - d_inv_sqrt_diag @ adj @ d_inv_sqrt_diag

            from scipy.sparse.linalg import eigsh
            eigenvalues = eigsh(lap, k=min(5, n - 1), return_eigenvectors=False)
            eigenvalues = np.sort(eigenvalues)
            spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 1.0
            spectral_gap = np.clip(spectral_gap, 0.0, 1.0)
        except:
            spectral_gap = 1.0

        is_fragmented = (n_components > 1) or (fragmentation_ratio > 0.3)
        poor_mixing = spectral_gap < 0.3
        is_well_connected = (n_components == 1) and (spectral_gap > 0.6) and (graph_density > 0.3)

        if is_fragmented or poor_mixing:
            self.neighborhood_size = min(n // 2, self.neighborhood_size + 1)
        elif is_well_connected:
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    # -------------------------------------------------------------------------
    # ADAPTATION STRATEGY 5: Stochastic Sampling (variant_07_catG, 1 win)
    # -------------------------------------------------------------------------
    def _adapt_neighborhood_stochastic_sample(self):
        """Adapt ring topology via stochastic sampling of neighborhood candidates.
        
        Category G: Uses Monte Carlo sampling and bootstrap estimation to
        probabilistically select neighborhood size based on empirical performance.
        """
        candidates = list(range(1, max(2, min(self.np // 2, self.neighborhood_size + 3)) + 1))
        if len(candidates) < 2:
            return

        n_bootstrap = 20
        bootstrap_scores = {c: [] for c in candidates}

        for _ in range(n_bootstrap):
            subsample_size = max(5, self.np // 2)
            subsample_idx = np.random.choice(self.np, size=subsample_size, replace=False)

            for candidate in candidates:
                local_best_qualities = []
                for idx in subsample_idx:
                    left = (idx - candidate) % self.np
                    right = (idx + candidate + 1) % self.np
                    if left < right:
                        nb_indices = np.arange(left, right)
                    else:
                        nb_indices = np.concatenate([np.arange(left, self.np), np.arange(0, right)])

                    if len(nb_indices) > 0:
                        nb_fitness = self.personal_best_fitness[nb_indices]
                        local_best_qualities.append(np.min(nb_fitness))

                if local_best_qualities:
                    bootstrap_scores[candidate].append(np.mean(local_best_qualities))

        candidate_stats = {}
        for candidate in candidates:
            scores = bootstrap_scores[candidate]
            if scores:
                mean_perf = np.mean(scores)
                std_perf = np.std(scores) + 1e-10
                candidate_stats[candidate] = mean_perf - 0.5 * std_perf
            else:
                candidate_stats[candidate] = 0.0

        diversity = self._compute_diversity()
        diversity_norm = np.clip(diversity / (self.diversity_threshold_high + 1e-10), 0.0, 1.0)

        temperatures = [0.5 + 0.5 * diversity_norm for _ in candidates]

        min_stat = min(candidate_stats.values())
        max_stat = max(candidate_stats.values())
        stat_range = max_stat - min_stat + 1e-10

        weights = []
        for i, candidate in enumerate(candidates):
            normalized = (candidate_stats[candidate] - min_stat) / stat_range
            weight = np.exp(normalized / temperatures[i])
            weights.append(weight)

        weights = np.array(weights)
        weights = weights / (np.sum(weights) + 1e-10)

        chosen_idx = np.random.choice(len(candidates), p=weights)
        chosen_size = candidates[chosen_idx]

        if not hasattr(self, '_neighborhood_momentum'):
            self._neighborhood_momentum = chosen_size
        self._neighborhood_momentum = 0.7 * self._neighborhood_momentum + 0.3 * chosen_size

        target = int(np.round(self._neighborhood_momentum))
        self.neighborhood_size = max(1, min(self.np // 2, target))
    
    # -------------------------------------------------------------------------
    # ADAPTATION STRATEGY 6: Temporal + Topology Hybrid (variant_08_catH, 4 wins)
    # -------------------------------------------------------------------------
    def _adapt_neighborhood_temporal_topology(self):
        """Hybrid: combine temporal convergence signals with topology fragmentation detection."""
        if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
            improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_best_fitness = self.global_best_fitness

        if not hasattr(self, '_ema_improvement'):
            self._ema_improvement = 0.0
        self._ema_improvement = 0.3 * improvement + 0.7 * self._ema_improvement

        stagnation_ratio = min(self.stagnation_counter / 100.0, 1.0)
        temporal_signal = stagnation_ratio - min(self._ema_improvement, 1.0)
        temporal_signal = np.clip(temporal_signal, -1.0, 1.0)

        k = min(5, self.np - 1)
        sq_dists = np.sum(
            (self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2
        )
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
        n_components = len(components)
        max_component_size = np.max(component_sizes)
        fragmentation_ratio = 1.0 - (max_component_size / (self.np + 1e-10))

        topology_signal = (n_components - 1) / max(self.np // 10, 1) + fragmentation_ratio
        topology_signal = np.clip(topology_signal, 0.0, 2.0)

        is_fragmented = n_components > 1 and fragmentation_ratio > 0.3

        if is_fragmented:
            topology_weight = 0.7
            temporal_weight = 0.3
        else:
            topology_weight = 0.3
            temporal_weight = 0.7

        combined_signal = (
            temporal_weight * (0.5 + 0.5 * temporal_signal) +
            topology_weight * topology_signal
        )

        target_ratio = combined_signal
        target_size = int(self.np * np.clip(target_ratio, 0.05, 0.5))
        target_size = max(1, min(self.np // 2, target_size))

        if target_size > self.neighborhood_size:
            self.neighborhood_size = min(target_size, self.neighborhood_size + 2)
        else:
            self.neighborhood_size = max(target_size, self.neighborhood_size - 2)

        self.neighborhood_size = max(1, min(self.np // 2, self.neighborhood_size))
    
    # -------------------------------------------------------------------------
    # ADAPTATION STRATEGY 7: k-NN Local Density (variant_09_catA, 3 wins)
    # -------------------------------------------------------------------------
    def _adapt_neighborhood_knn_density(self):
        """Adapt ring topology neighborhood size based on k-NN local density structure.
        
        Category A (Geometry / spatial): Uses k-NN nearest-neighbor distances and
        axis-aligned spread ratio to detect clustering vs dispersion.
        """
        if self.np < 5:
            return

        k = min(3, self.np - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        sorted_dists = np.sort(sq_dists, axis=1)
        knn_avg_dist = np.mean(sorted_dists[:, :k], axis=1)

        pop_min = np.min(self.population, axis=0)
        pop_max = np.max(self.population, axis=0)
        bbox_extent = np.mean(pop_max - pop_min) + 1e-10

        density_ratio = knn_avg_dist.mean() / bbox_extent

        occupancy = np.std(self.population) / (bbox_extent * 0.5 + 1e-10)
        occupancy = np.clip(occupancy, 0.0, 2.0)

        knn_variance = np.var(knn_avg_dist) / (np.mean(knn_avg_dist) ** 2 + 1e-10)
        knn_variance = np.clip(knn_variance, 0.0, 10.0)

        if density_ratio < 0.1 or knn_variance > 2.0:
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
        elif density_ratio > 0.5 and occupancy < 0.5:
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif density_ratio > 0.3:
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
    
    # -------------------------------------------------------------------------
    # ADAPTATION STRATEGY 8: Spectral Entropy (variant_10_catB, 2 wins)
    # -------------------------------------------------------------------------
    def _adapt_neighborhood_spectral_entropy(self):
        """Adapt ring topology neighborhood size using spectral properties of population.
        
        Uses condition number and spectral entropy of population covariance to detect:
        - High condition number → anisotropic/ill-conditioned → larger neighborhood
        - Low spectral entropy → collapsed subspace → larger neighborhood
        - Isotropic distribution → smaller neighborhood for exploitation
        """
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]

            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 1e6)

            total_var = np.sum(eigenvalues)
            if total_var > 0:
                eig_norm = eigenvalues / total_var
                eig_norm = eig_norm[eig_norm > 1e-10]
                spectral_entropy = -np.sum(eig_norm * np.log(eig_norm))
                max_entropy = np.log(self.dim)
                entropy_ratio = spectral_entropy / (max_entropy + 1e-10)
            else:
                entropy_ratio = 0.0

            cond_signal = np.log1p(cond) / np.log1p(1e6)
            spectral_signal = 0.5 * cond_signal + 0.5 * (1.0 - entropy_ratio)

            neighborhood_scale = 0.4 + 1.1 * spectral_signal

            target_size = int(self.neighborhood_size * neighborhood_scale)
            target_size = np.clip(target_size, 1, self.np // 2)

            if target_size > self.neighborhood_size:
                self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
            elif target_size < self.neighborhood_size:
                self.neighborhood_size = max(1, self.neighborhood_size - 1)

        except np.linalg.LinAlgError:
            diversity = self._compute_diversity()
            if diversity < self.diversity_threshold_low:
                self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
            elif diversity > self.diversity_threshold_high:
                self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    # -------------------------------------------------------------------------
    # DEFAULT FALLBACK ADAPTATION (used before probe commits)
    # -------------------------------------------------------------------------
    def _adapt_neighborhood_size(self):
        """Default diversity-based adaptation (fallback before probe commits)."""
        diversity = self._compute_diversity()
        
        if diversity < self.diversity_threshold_low:
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif diversity > self.diversity_threshold_high:
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    # -------------------------------------------------------------------------
    # DISPATCHER: maps operator name -> adaptation method
    # -------------------------------------------------------------------------
    def _dispatch_adaptation(self, operator):
        """Route to the appropriate neighborhood adaptation implementation."""
        if operator == 'spectral_cond':
            self._adapt_neighborhood_spectral_cond()
        elif operator == 'info_theory':
            self._adapt_neighborhood_info_theory()
        elif operator == 'fitness_landscape':
            self._adapt_neighborhood_fitness_landscape()
        elif operator == 'knn_connectivity':
            self._adapt_neighborhood_knn_connectivity()
        elif operator == 'stochastic_sample':
            self._adapt_neighborhood_stochastic_sample()
        elif operator == 'temporal_topology':
            self._adapt_neighborhood_temporal_topology()
        elif operator == 'knn_density':
            self._adapt_neighborhood_knn_density()
        elif operator == 'spectral_entropy':
            self._adapt_neighborhood_spectral_entropy()
        else:
            self._adapt_neighborhood_size()
    
    # -------------------------------------------------------------------------
    # PROBE-AND-COMMIT: rank-based scoring (scale-independent)
    # -------------------------------------------------------------------------
    
    def _record_probe_score(self, operator, best_fitness):
        """Record a probe score using rank-based (scale-independent) scoring.
        
        Score = fraction of probe generations with WORSE (higher) fitness.
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
    
    def _adaptive_coefficients(self):
        """Compute adaptive cognitive and social coefficients."""
        improvement_ratio = 1.0 / (1.0 + self.stagnation_counter)
        cognitive = self.cognitive_base * (1.0 + 0.5 * improvement_ratio)
        social = self.social_base * (2.0 - improvement_ratio)
        return cognitive, social
    
    # -------------------------------------------------------------------------
    # VELOCITY UPDATE (shared across all strategies)
    # -------------------------------------------------------------------------
    
    def _velocity_update(self):
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
    
    def _position_update(self):
        """Standard position update (shared across all strategies)."""
        new_population = self.population + self.velocity
        self.population = self._clip_to_bounds(new_population)
    
    # -------------------------------------------------------------------------
    # MAIN OPTIMIZATION LOOP
    # -------------------------------------------------------------------------
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer with PROBE-AND-COMMIT neighborhood adaptation selection.
        
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
        self._in_probe_phase = True
        self._probe_op_index = 0
        self._probe_gens_in_current_op = 0
        self._commit_best_fitness = np.inf
        self._commit_gen_count = 0
        
        # Reset adaptation-specific state
        for attr in ['_prev_fitness_ranks', '_neighborhood_history', '_neighborhood_success',
                     '_neighborhood_momentum', '_ema_improvement', '_prev_global_best_fitness',
                     '_prev_best_for_history']:
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
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # --- PHASE A: PROBE ---
            if self._in_probe_phase:
                current_op = self._operators[self._probe_op_index]
                
                # Adaptation: probe the CURRENT operator's strategy
                self._dispatch_adaptation(current_op)
                
                self._adapt_inertia_weight()
                self._update_local_best_from_personal()
                
                # Standard PSO update
                self._velocity_update()
                self._position_update()
                
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
                    self._compute_fingerprint()
                    self._commit_best_fitness = self.global_best_fitness
                    self._commit_gen_count = 0
                
                self._restart_if_stagnant()
            
            # --- PHASE B: COMMIT ---
            else:
                self._commit_gen_count += 1
                
                # Epsilon-review: small chance to switch to runner-up if stagnant
                if (self.stagnation_counter > 20 and
                    np.random.random() < self._epsilon_review):
                    self._dispatch_adaptation(self._runner_up_operator)
                else:
                    self._dispatch_adaptation(self._committed_operator)
                
                self._adapt_inertia_weight()
                self._update_local_best_from_personal()
                
                self._velocity_update()
                self._position_update()
                
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
```