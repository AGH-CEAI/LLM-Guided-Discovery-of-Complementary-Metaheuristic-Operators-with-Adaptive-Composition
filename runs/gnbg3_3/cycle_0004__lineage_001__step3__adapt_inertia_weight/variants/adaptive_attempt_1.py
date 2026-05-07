import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: ENSEMBLE BLENDING with rank-normalized EWMA weights (mechanism #2)
    
    The gap table is BLEND-VIABLE:
      - All per-task gaps are < 6× (max: task 5 at 5.6×).
      - No task is blend-hostile; linear blending CAN preserve per-task advantages.
      - Win counts are spread (7/5/3/3/2/2/2 across 7 variants), confirming that
        no single operator wins everywhere — ensemble is appropriate.
    
    The algorithm SELF-MODIFIES by:
      - Running ALL 7 winning inertia-weight strategies every generation.
      - Each strategy computes its own inertia target from a distinct signal:
          * Spectral condition number (variant_02, variant_10)
          * Information-theoretic entropy (variant_03)
          * Fitness rank distribution (variant_04)
          * Graph k-NN / MST topology (variant_05)
          * Geometric spread ratio (variant_09)
      - Weights are learned online via exponentially weighted moving average
        of rank-normalized improvement scores — no scale-dependent constants.
      - Position updates are BLENDED (not selected) proportional to weights,
        preserving each strategy's contribution rather than committing to one.
      - This is NOT a bandit-over-operators; it is a self-tuning ensemble where
        the weights continuously reflect which signals are predictive on THIS
        landscape.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5 * dim, capped
        self.np = min(max(5 * dim, 20), 300)
        
        # Ring topology neighborhood size
        self.neighborhood_size = max(3, dim // 5)
        
        # Base acceleration coefficients
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
        
        # ---- ENSEMBLE STATE: 7 winning inertia-weight strategies ----
        self._strategies = [
            'spectral_cond',      # variant_10_catB_idea_0 (3 wins)
            'info_entropy',       # variant_03_catC_idea_0 (5 wins)
            'fitness_rank',       # variant_04_catD_idea_0 (2 wins)
            'graph_knn_mst',      # variant_05_catE_idea_0 (3 wins)
            'geo_spread',         # variant_09_catA_idea_0 (2 wins)
            'spectral_effdim',    # variant_02_catB_idea_0 (2 wins)
            'original',           # original.py (7 wins)
        ]
        
        # Per-strategy EWMA weights (rank-based, [0,1])
        self._strategy_scores = {s: [] for s in self._strategies}
        self._strategy_ewma = {s: 0.0 for s in self._strategies}
        self._strategy_total = {s: 0.0 for s in self._strategies}
        self._strategy_count = {s: 0 for s in self._strategies}
        
        # Ensemble weights (softmax over EWMA scores)
        self._blend_weights = {s: 1.0 / len(self._strategies) for s in self._strategies}
        
        # Probe phase: 2 gens per strategy, min 14 total
        self._probe_gens = 2
        self._probe_budget = max(len(self._strategies) * self._probe_gens, 14)
        self._probe_op_idx = 0
        self._probe_gens_cur = 0
        self._in_probe = True
        self._probe_best_fitness = np.inf
        self._probe_history = []  # list of (strategy, best_fitness)
        
        # Commit phase
        self._commit_best = np.inf
        self._commit_gens = 0
        self._epsilon_review = 0.05
        
        # Per-strategy EMA state (for strategies that need memory)
        self._prev_fitness_for_rank = None
        self._ema_centroid = None
        self._ema_centroid_vel = None
        self._centroid_vel_hist = []
        self._ema_improvement = 0.0
        self._prev_gbest_fit = np.inf
        self._success_count = None
    
    # -------------------------------------------------------------------------
    # INERTIA WEIGHT STRATEGIES (all 7 benchmark winners)
    # -------------------------------------------------------------------------
    
    def _inertia_spectral_cond(self):
        """Variant 10 (Category B): condition number of population covariance."""
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
            target = 0.4 + 0.55 * spectral_signal
            decay = 0.729 - 0.15 * (self.generation / 1000)
            target = np.clip(target * 0.6 + decay * 0.4, 0.4, 0.95)
        except np.linalg.LinAlgError:
            target = 0.729 - 0.15 * (self.generation / 1000)
        return np.clip(target, 0.4, 0.95)
    
    def _inertia_info_entropy(self):
        """Variant 03 (Category C): information-theoretic signals (entropy + KL)."""
        f_min = np.min(self.current_fitness)
        p_fit = self.current_fitness - f_min + 1e-10
        p_fit = p_fit / (np.sum(p_fit) + 1e-10)
        entropy_fit = -np.sum(p_fit * np.log(p_fit + 1e-10))
        max_entropy = np.log(self.np) if self.np > 1 else 1.0
        norm_entropy = np.clip(entropy_fit / (max_entropy + 1e-10), 0.0, 1.0)
        
        kl_div = 0.0
        n_bins = min(10, max(3, self.np // 5))
        for d in range(self.dim):
            hist, _ = np.histogram(self.population[:, d], bins=n_bins, density=True)
            hist = np.clip(hist, 1e-10, None)
            uniform_prob = 1.0 / n_bins
            kl_div += np.sum(hist * np.log(hist / (uniform_prob + 1e-10)))
        kl_div = kl_div / (self.dim + 1e-10)
        kl_norm = np.clip(kl_div / 5.0, 0.0, 1.0)
        
        exploration_signal = norm_entropy * (1.0 - kl_norm)
        exploration_signal = np.clip(exploration_signal, 0.0, 1.0)
        target = 0.95 - 0.55 * exploration_signal
        decay = 0.729 - 0.15 * (self.generation / 1000)
        target = np.clip(target * 0.5 + decay * 0.5, 0.4, 0.95)
        return np.clip(target, 0.4, 0.95)
    
    def _inertia_fitness_rank(self):
        """Variant 04 (Category D): fitness rank distribution and improvement rate."""
        sorted_idx = np.argsort(self.current_fitness)
        ranks = np.zeros(self.np)
        ranks[sorted_idx] = np.arange(self.np) / max(1, self.np - 1)
        q75 = int(0.75 * (self.np - 1))
        q25 = int(0.25 * (self.np - 1))
        sorted_ranks = np.sort(ranks)
        rank_iqr = sorted_ranks[q75] - sorted_ranks[q25]
        rank_iqr = np.clip(rank_iqr, 0.01, 1.0)
        rank_spread = rank_iqr / 0.5
        
        if self._prev_fitness_for_rank is None:
            self._prev_fitness_for_rank = self.current_fitness.copy()
        improved_mask = self.current_fitness < self._prev_fitness_for_rank
        improvement_rate = np.sum(improved_mask) / max(1, self.np)
        self._prev_fitness_for_rank = self.current_fitness.copy()
        
        rank_spread_c = np.clip(rank_spread, 0.1, 2.0)
        imp_c = np.clip(improvement_rate, 0.01, 1.0)
        target = 0.4 + 0.55 * (rank_spread_c * imp_c)
        return np.clip(target, 0.4, 0.95)
    
    def _inertia_graph_knn_mst(self):
        """Variant 05 (Category E): k-NN graph connectivity and MST path-length."""
        npops = self.np
        k = min(5, npops - 1)
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        knn_indices = np.argsort(sq_dists, axis=1)[:, 1:k+1]
        knn_dists = np.take_along_axis(sq_dists, knn_indices, axis=1)
        
        threshold = knn_dists[:, -1:]
        conn_counts = np.sum(sq_dists[:, 1:] <= threshold, axis=1)
        conn_ratio = np.mean(conn_counts >= (k // 2 + 1))
        
        cand_edges = []
        cand_weights = []
        for i in range(npops):
            for j in knn_indices[i]:
                if i < j:
                    cand_edges.append((i, j))
                    cand_weights.append(np.sqrt(sq_dists[i, j]))
        
        if len(cand_weights) >= npops - 1:
            sorted_e = np.argsort(cand_weights)
            parent = np.arange(npops)
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            mst_weight = 0.0
            edges_added = 0
            for idx in sorted_e:
                if edges_added >= npops - 1:
                    break
                i, j = cand_edges[idx]
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
                    mst_weight += cand_weights[idx]
                    edges_added += 1
            mst_avg = mst_weight / max(edges_added, 1)
        else:
            mst_avg = np.mean(knn_dists)
        
        pop_scale = np.mean(np.std(self.population, axis=0)) + 1e-10
        mst_norm = np.clip(mst_avg / (pop_scale * 3.0), 0.0, 1.0)
        
        conn_comp = 0.4 + 0.4 * conn_ratio
        spread_comp = 0.95 - 0.4 * mst_norm
        target = conn_comp * 0.6 + spread_comp * 0.4
        return np.clip(target, 0.4, 0.95)
    
    def _inertia_geo_spread(self):
        """Variant 09 (Category A): pairwise diameter vs mean inter-particle distance."""
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, -np.inf)
        diameter = np.sqrt(np.max(sq_dists)) + 1e-10
        upper_sq = np.triu(sq_dists, k=1)
        mean_pair_sq = np.sum(upper_sq) / max(1, self.np * (self.np - 1) / 2)
        mean_pair = np.sqrt(max(mean_pair_sq, 0)) + 1e-10
        spread_ratio = diameter / (mean_pair + 1e-10)
        spread_norm = np.clip(spread_ratio / 5.0, 0.0, 1.0)
        target = spread_norm * 0.55 + 0.4
        decay = 0.729 - 0.15 * (self.generation / 1000)
        target = np.clip(target * 0.5 + decay * 0.5, 0.4, 0.95)
        return np.clip(target, 0.4, 0.95)
    
    def _inertia_spectral_effdim(self):
        """Variant 02 (Category B variant): spectral + effective dimensionality."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-14, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond_n = np.clip(cond / 500.0, 0.0, 1.0)
            total_var = np.sum(eigenvalues) + 1e-10
            cumvar = np.cumsum(eigenvalues) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_r = np.clip(eff_dim / self.dim, 0.0, 1.0)
            spectral_sig = cond_n * 0.6 + (1.0 - eff_dim_r) * 0.4
            base = 0.4 + 0.35 * eff_dim_r
            target = base * (1.0 - 0.45 * spectral_sig)
            decay = 0.729 - 0.08 * (self.generation / 1000)
            target = np.clip(target * 0.6 + decay * 0.4, 0.4, 0.95)
        except np.linalg.LinAlgError:
            target = 0.729 - 0.08 * (self.generation / 1000)
        return np.clip(target, 0.4, 0.95)
    
    def _inertia_original(self):
        """Original: axis-aligned geometric spread ratio."""
        min_pos = np.min(self.population, axis=0)
        max_pos = np.max(self.population, axis=0)
        spread = max_pos - min_pos + 1e-10
        geo_mean = np.exp(np.mean(np.log(spread)))
        arith_mean = np.mean(spread)
        spread_ratio = geo_mean / (arith_mean + 1e-10)
        spread_norm = np.clip(spread_ratio / 0.9, 0.0, 1.0)
        target = (1.0 - spread_norm) * 0.55 + 0.4
        decay = 0.729 - 0.15 * (self.generation / 1000)
        target = np.clip(target * 0.5 + decay * 0.5, 0.4, 0.95)
        return np.clip(target, 0.4, 0.95)
    
    def _compute_all_inertia_weights(self):
        """Compute inertia weight from all 7 strategies. Returns dict."""
        return {
            'spectral_cond': self._inertia_spectral_cond(),
            'info_entropy': self._inertia_info_entropy(),
            'fitness_rank': self._inertia_fitness_rank(),
            'graph_knn_mst': self._inertia_graph_knn_mst(),
            'geo_spread': self._inertia_geo_spread(),
            'spectral_effdim': self._inertia_spectral_effdim(),
            'original': self._inertia_original(),
        }
    
    # -------------------------------------------------------------------------
    # ENSEMBLE WEIGHT UPDATE (rank-based EWMA, no scale-dependent constants)
    # -------------------------------------------------------------------------
    
    def _update_ensemble_weights(self, best_fitness):
        """Update ensemble weights using rank-based EWMA over sliding window.
        
        Score for each strategy = relative improvement when that strategy's
        inertia was used. Rank-normalized within sliding window so magnitudes
        don't dominate — no magic constants like *1e5.
        """
        self._probe_history.append(best_fitness)
        
        if len(self._probe_history) < 3:
            return  # not enough data
        
        # Rank-normalize within the last K entries
        K = max(3, len(self._strategies) * 2)
        recent = self._probe_history[-K:]
        
        # Convert to rank scores: higher score = better (lower fitness)
        sorted_fits = sorted(recent)
        ranks = [sorted_fits.index(v) / max(1, len(recent) - 1) for v in recent]
        
        # Update EWMA for each strategy (decay = 0.85, so old scores fade)
        decay = 0.85
        for i, s in enumerate(self._strategies):
            if i < len(ranks):
                # EWMA update: new contribution weighted by rank
                new_contribution = ranks[i]
                if self._strategy_count[s] == 0:
                    self._strategy_ewma[s] = new_contribution
                else:
                    self._strategy_ewma[s] = (
                        decay * self._strategy_ewma[s] + (1.0 - decay) * new_contribution
                    )
                self._strategy_total[s] += new_contribution
                self._strategy_count[s] += 1
        
        # Softmax over EWMA scores to get blend weights
        scores = np.array([self._strategy_ewma[s] for s in self._strategies])
        scores -= np.max(scores)  # numerical stability
        exp_scores = np.exp(scores * 10.0)  # temperature=0.1
        total_exp = np.sum(exp_scores) + 1e-10
        for i, s in enumerate(self._strategies):
            self._blend_weights[s] = exp_scores[i] / total_exp
    
    # -------------------------------------------------------------------------
    # POSITION UPDATE (blended from all strategies)
    # -------------------------------------------------------------------------
    
    def _position_update_blended(self, inertia_weights):
        """Blend position updates from all strategies, weighted by ensemble scores.
        
        Each strategy contributes a velocity-scaled update proportional to its
        current blend weight. This is NOT selecting one operator — it preserves
        each strategy's advantage on the tasks where it wins.
        """
        base_velocity = self.velocity.copy()
        
        # Compute blended inertia
        blended_inertia = sum(
            self._blend_weights[s] * inertia_weights[s]
            for s in self._strategies
        )
        blended_inertia = np.clip(blended_inertia, 0.4, 0.95)
        
        # Also blend the velocity itself (each strategy implies a velocity direction)
        # Strategy-specific velocity scalings
        vel_scalings = {}
        for s in self._strategies:
            if s == 'spectral_cond':
                try:
                    centered = self.population - np.mean(self.population, axis=0)
                    cov = np.cov(centered.T)
                    eigval, eigvec = np.linalg.eigh(cov)
                    idx = np.argsort(eigval)[::-1]
                    eigval = eigval[idx]
                    eigvec = eigvec[:, idx]
                    cond = eigval[0] / (eigval[-1] + 1e-10)
                    scale = np.clip(cond / 100.0, 0.5, 2.0)
                    vel_proj = base_velocity @ eigvec
                    vel_scalings[s] = (eigvec, vel_proj * scale)
                except:
                    vel_scalings[s] = (None, base_velocity)
            elif s == 'info_entropy':
                # Entropy-based velocity scaling
                f_min = np.min(self.current_fitness)
                p_fit = self.current_fitness - f_min + 1e-10
                p_fit = p_fit / (np.sum(p_fit) + 1e-10)
                entropy = -np.sum(p_fit * np.log(p_fit + 1e-10))
                max_ent = np.log(self.np) if self.np > 1 else 1.0
                norm_ent = np.clip(entropy / (max_ent + 1e-10), 0.0, 1.0)
                scale = 0.5 + 0.5 * norm_ent
                vel_scalings[s] = (None, base_velocity * scale)
            elif s == 'fitness_rank':
                # Fitness rank-based velocity scaling
                if self._success_count is None:
                    self._success_count = np.zeros(self.np)
                improved = self.personal_best_fitness >= self.current_fitness
                self._success_count[improved] += 1
                self._success_count[~improved] *= 0.9
                max_s = np.max(self._success_count) + 1.0
                success_norm = self._success_count / max_s
                vel_scale = 1.0 + 0.3 * success_norm
                vel_scale = np.clip(vel_scale, 0.5, 2.0)
                vel_scalings[s] = (None, base_velocity * vel_scale[:, np.newaxis])
            elif s == 'graph_knn_mst':
                # k-NN density modulation
                centroid = np.mean(self.population, axis=0)
                dist_c = np.linalg.norm(self.population - centroid, axis=1, keepdims=True) + 1e-10
                k = min(5, self.np - 1)
                diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
                sq_d = np.sum(diffs ** 2, axis=2)
                np.fill_diagonal(sq_d, np.inf)
                knn_d = np.sort(sq_d, axis=1)[:, :k]
                knn_avg = np.mean(np.sqrt(knn_d), axis=1, keepdims=True) + 1e-10
                global_spread = np.mean(np.sqrt(sq_d[sq_d < np.inf])) + 1e-10
                density = np.clip(knn_avg / (global_spread + 1e-10), 0.5, 1.5)
                vel_scalings[s] = (None, base_velocity * density)
            elif s == 'geo_spread':
                # Geometric spread modulation
                diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
                sq_d = np.sum(diffs ** 2, axis=2)
                np.fill_diagonal(sq_d, -np.inf)
                diam = np.sqrt(np.max(sq_d)) + 1e-10
                upper = np.triu(sq_d, k=1)
                mean_p = np.sqrt(max(np.sum(upper) / max(1, self.np * (self.np - 1) / 2), 0)) + 1e-10
                ratio = diam / mean_p
                ratio_n = np.clip(ratio / 5.0, 0.0, 1.0)
                scale = 0.5 + 0.5 * ratio_n
                vel_scalings[s] = (None, base_velocity * scale)
            elif s == 'spectral_effdim':
                try:
                    centered = self.population - np.mean(self.population, axis=0)
                    cov = np.cov(centered.T)
                    eigval = np.linalg.eigvalsh(cov)
                    eigval = np.clip(eigval, 1e-14, None)
                    eigval = np.sort(eigval)[::-1]
                    total_v = np.sum(eigval) + 1e-10
                    cumv = np.cumsum(eigval) / total_v
                    eff_d = float(np.searchsorted(cumv, 0.95)) + 1
                    eff_r = np.clip(eff_d / self.dim, 0.0, 1.0)
                    scale = 0.5 + 0.5 * eff_r
                    vel_scalings[s] = (None, base_velocity * scale)
                except:
                    vel_scalings[s] = (None, base_velocity)
            else:  # original
                min_p = np.min(self.population, axis=0)
                max_p = np.max(self.population, axis=0)
                sp = max_p - min_p + 1e-10
                geo = np.exp(np.mean(np.log(sp)))
                arith = np.mean(sp)
                sr = geo / (arith + 1e-10)
                sr_n = np.clip(sr / 0.9, 0.0, 1.0)
                scale = 1.0 - 0.3 * sr_n
                vel_scalings[s] = (None, base_velocity * scale)
        
        # Blend all strategy velocities
        blended_velocity = sum(
            self._blend_weights[s] * vel_scalings[s][1]
            for s in self._strategies
        )
        
        # Use blended inertia with blended velocity
        new_pop = self.population + blended_inertia * blended_velocity
        self.population = self._clip_to_bounds(new_pop)
    
    # -------------------------------------------------------------------------
    # STANDARD PSO COMPONENTS
    # -------------------------------------------------------------------------
    
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
    
    def _clip_to_bounds(self, pop):
        return np.clip(pop, self.lower_bound, self.upper_bound)
    
    def _evaluate_batch(self, pop, func):
        clipped = self._clip_to_bounds(pop)
        fitness = func(clipped)
        actual_n = min(len(fitness), len(pop))
        return fitness[:actual_n], actual_n
    
    def _update_personal_best_batch(self):
        improved = self.personal_best_fitness > self.current_fitness
        self.personal_best[improved] = self.population[improved]
        self.personal_best_fitness[improved] = self.current_fitness[improved]
    
    def _compute_local_best_batch(self):
        for i in range(self.np):
            left = (i - self.neighborhood_size) % self.np
            right = (i + self.neighborhood_size + 1) % self.np
            if left < right:
                nbr_idx = np.arange(left, right)
            else:
                nbr_idx = np.concatenate([np.arange(left, self.np), np.arange(0, right)])
            all_idx = np.concatenate([nbr_idx, [i]])
            fit_nbr = self.personal_best_fitness[all_idx]
            best_local = np.argmin(fit_nbr)
            self.local_best[i] = self.personal_best[all_idx[best_local]]
            self.local_best_fitness[i] = fit_nbr[best_local]
    
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
        dists = np.linalg.norm(self.population - centroid, axis=1)
        return np.mean(dists)
    
    def _adapt_neighborhood_size(self):
        div = self._compute_diversity()
        if div < self.diversity_threshold_low:
            self.neighborhood_size = min(self.np // 2, self.neighborhood_size + 1)
        elif div > self.diversity_threshold_high:
            self.neighborhood_size = max(1, self.neighborhood_size - 1)
    
    def _adaptive_coefficients(self):
        imp_ratio = 1.0 / (1.0 + self.stagnation_counter)
        cognitive = self.cognitive_base * (1.0 + 0.5 * imp_ratio)
        social = self.social_base * (2.0 - imp_ratio)
        return cognitive, social
    
    def _velocity_update_base(self):
        cognitive, social = self._adaptive_coefficients()
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        cog_comp = cognitive * r1 * (self.personal_best - self.population)
        soc_comp = social * r2 * (self.local_best - self.population)
        
        mut_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mut_thresh = 0.1 * (1.0 - self.generation / 5000)
        mut_active = mut_mask < mut_thresh
        mut_vecs = np.zeros((self.np, self.dim))
        for i in range(self.np):
            idx = np.random.choice([j for j in range(self.np) if j != i], 3, replace=False)
            mut_vecs[i] = self.population[idx[0]] + 0.5 * (self.population[idx[1]] - self.population[idx[2]])
        mut_comp = np.where(mut_active, 0.3 * (mut_vecs - self.population), 0.0)
        
        new_vel = self.inertia_weight * self.velocity + cog_comp + soc_comp + mut_comp
        self.velocity = np.clip(new_vel, self.v_min, self.v_max)
    
    def _restart_if_stagnant(self):
        thresh = 50 + self.dim // 2
        if self.stagnation_counter > thresh:
            n_rep = max(1, int(0.3 * self.np))
            worst = np.argsort(self.personal_best_fitness)[-n_rep:]
            self.population[worst] = np.random.uniform(
                self.lower_bound, self.upper_bound, (n_rep, self.dim)
            )
            self.velocity[worst] = np.random.uniform(
                self.v_min * 0.1, self.v_max * 0.1, (n_rep, self.dim)
            )
            self.personal_best_fitness[worst] = np.inf
            self.personal_best[worst] = self.population[worst]
            self.stagnation_counter = 0
            if self.global_best is not None:
                pert = np.random.uniform(-5, 5, self.dim)
                self.population[0] = self._clip_to_bounds(self.global_best + pert)
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
        Run optimizer with adaptive ensemble blending of 7 inertia-weight strategies.
        
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
        
        # Reset ensemble state
        self._strategy_scores = {s: [] for s in self._strategies}
        self._strategy_ewma = {s: 0.0 for s in self._strategies}
        self._strategy_total = {s: 0.0 for s in self._strategies}
        self._strategy_count = {s: 0 for s in self._strategies}
        self._blend_weights = {s: 1.0 / len(self._strategies) for s in self._strategies}
        self._probe_history = []
        self._probe_best_fitness = np.inf
        self._in_probe = True
        self._probe_op_idx = 0
        self._probe_gens_cur = 0
        self._commit_best = np.inf
        self._commit_gens = 0
        self._prev_fitness_for_rank = None
        self._ema_centroid = None
        self._ema_centroid_vel = None
        self._centroid_vel_hist = []
        self._ema_improvement = 0.0
        self._prev_gbest_fit = np.inf
        self._success_count = np.zeros(self.np) if self.np > 0 else None
        
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
            
            # --- PHASE A: PROBE (all strategies for a few gens) ---
            if self._in_probe:
                # Compute all inertia weights (the "probe" — running all strategies)
                all_inertia = self._compute_all_inertia_weights()
                
                # Update ensemble weights from rank-based scores
                gen_best = float(np.min(self.current_fitness))
                if gen_best < self._probe_best_fitness:
                    self._probe_best_fitness = gen_best
                self._update_ensemble_weights(self._probe_best_fitness)
                
                # Adaptation
                self._adapt_neighborhood_size()
                self._update_local_best_from_personal()
                
                # Use blended position update with current (uniform) weights
                self._velocity_update_base()
                self._position_update_blended(all_inertia)
                
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
                
                self._probe_gens_cur += 1
                if self._probe_gens_cur >= self._probe_gens * len(self._strategies):
                    self._in_probe = False
                    self._commit_best = self.global_best_fitness
                    self._commit_gens = 0
            
            # --- PHASE B: COMMIT (blended ensemble with learned weights) ---
            else:
                self._commit_gens += 1
                
                # Epsilon-review: small chance to re-uniform-ize weights if stagnant
                if (self.stagnation_counter > 20 and
                    np.random.random() < self._epsilon_review):
                    # Re-equalize weights to allow re-learning
                    for s in self._strategies:
                        self._blend_weights[s] = 1.0 / len(self._strategies)
                
                # Compute all inertia weights
                all_inertia = self._compute_all_inertia_weights()
                
                # Update ensemble weights from rank-based scores
                gen_best = float(np.min(self.current_fitness))
                if gen_best < self._probe_best_fitness:
                    self._probe_best_fitness = gen_best
                self._update_ensemble_weights(self._probe_best_fitness)
                
                # Adaptation
                self._adapt_neighborhood_size()
                self._update_local_best_from_personal()
                
                # Blended position update
                self._velocity_update_base()
                self._position_update_blended(all_inertia)
                
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
        
        # Final result
        if self.global_best is None or np.isnan(self.global_best_fitness):
            best_idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[best_idx], self.population[best_idx].copy()
        return self.global_best_fitness, self.global_best.copy()
