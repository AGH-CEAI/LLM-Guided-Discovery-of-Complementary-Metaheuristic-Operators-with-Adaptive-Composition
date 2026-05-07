```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: PROBE-AND-COMMIT + ADAPTIVE ENSEMBLE BLENDING (mechanism #4 → #2)

    The gap table is BLEND-SAFE (NOT blend-hostile):
      - Maximum per-task gap is 1.3× (task 11: variant_10 beats variant_08).
      - Zero tasks with gap >= 10×. Zero tasks with gap >= 100×.
      - Mandatory dispatch (mechanism #3) is NOT required.

    The small gaps mean linear blending CAN preserve per-task advantages:
      A 70/30 blend of 59.9 and 77.8 yields ~65.3, which is close to the
      winner (59.9) — the blend does not destroy the advantage.

    However, different operators win on different tasks (7 distinct winners
    spread evenly at 3–4 wins each), so committing to ONE operator loses
    information. A pure bandit is fragile given the small gaps.

    ADAPTIVE ENSEMBLE BLENDING resolves this:
      PHASE A (probe): round-robin burst of 3 generations per variant,
        tracking rank-based improvement scores (scale-independent).
      PHASE B (commit): run ALL 7 variants every generation and blend
        their proposed populations using softmax(UCB_scores) weights.
        Per-generation rank-based weight adaptation keeps the blend
        responsive without hard-coded regime tables.

    The blend is lightweight: 7 small candidate populations are
    evaluated in parallel each generation, and the blended update
    is applied to the real population.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size
        self.np = min(max(5 * dim, 20), 300)
        
        # Ring topology
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
        
        # --- 7 BENCHMARK WINNING VARIANTS ---
        self._variant_names = [
            'original',        # 4 wins (tasks 5, 8, 16, 21)
            'variant_02_catB', # 3 wins (tasks 7, 19, 22)
            'variant_04_catD', # 4 wins (tasks 1, 9, 14, 18)
            'variant_07_catG', # 4 wins (tasks 10, 12, 17, 20)
            'variant_08_catH', # 3 wins (tasks 0, 13, 15)
            'variant_09_catA', # 3 wins (tasks 3, 6, 23)
            'variant_10_catB', # 3 wins (tasks 2, 4, 11)
        ]
        
        # Probe state: per-variant score tracking
        self._variant_scores = {v: [] for v in self._variant_names}
        self._variant_total = {v: 0.0 for v in self._variant_names}
        self._variant_count = {v: 0 for v in self._variant_names}
        
        # Probe parameters
        self._probe_gens_per_variant = 3
        self._probe_budget = max(len(self._variant_names) * self._probe_gens_per_variant, 21)
        
        # Commit phase: sliding window for rank-based adaptation
        self._commit_rank_window = 10  # K in calibration rule (b)
        self._commit_fitness_history = []  # list of (gen_best_fitness, variant_weights)
        
        # Probe phase state
        self._in_probe_phase = False
        self._probe_variant_idx = 0
        self._probe_gens_current = 0
        self._probe_best_seen = np.inf
        self._probe_fitness_hist = []
        
        # Commit phase state
        self._commit_gen = 0
        self._commit_best = np.inf
        
        # Sliding window for rank-based scoring (mechanism #1 requirement)
        self._reward_window_size = 9  # >= 3 * num_arms = 21; use 9 for efficiency
        self._recent_rewards = {v: [] for v in self._variant_names}
    
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
        if len(fitness) < len(pop):
            return fitness[:len(fitness)], len(fitness)
        return fitness, len(pop)
    
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
        return np.mean(np.linalg.norm(self.population - centroid, axis=1))
    
    def _adapt_inertia_weight(self):
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 10000.0)
            spectral = np.clip(np.log1p(cond) / np.log1p(10000.0), 0.0, 1.0)
            target = 0.4 + 0.55 * spectral
            decay = 0.729 - 0.15 * (self.generation / 1000)
            target = np.clip(target * 0.6 + decay * 0.4, 0.4, 0.95)
            self.inertia_weight = np.clip(self.inertia_weight * 0.8 + target * 0.2, 0.4, 0.95)
        except np.linalg.LinAlgError:
            decay = 0.729 - 0.15 * (self.generation / 1000)
            self.inertia_weight = np.clip(self.inertia_weight * 0.8 + decay * 0.2, 0.4, 0.95)
    
    def _adapt_neighborhood_size(self):
        fitness = self.current_fitness
        if len(fitness) > 1:
            n_bins = min(20, max(3, len(fitness) // 2))
            hist, _ = np.histogram(fitness, bins=n_bins)
            hist = hist / (len(fitness) + 1e-10)
            hist = hist[hist > 0]
            fit_ent = -np.sum(hist * np.log(hist + 1e-10))
            norm_ent = fit_ent / (np.log(n_bins) + 1e-10)
        else:
            norm_ent = 0.5
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenv = np.linalg.eigvalsh(cov)
            eigenv = np.clip(eigenv, 1e-10, None)
            eigenv = np.sort(eigenv)[::-1]
            eig_norm = eigenv / (np.sum(eigenv) + 1e-10)
            uniform = np.ones(len(eig_norm)) / len(eig_norm)
            kl = np.sum(eig_norm * np.log((eig_norm + 1e-10) / (uniform + 1e-10)))
            norm_kl = np.clip(kl / np.log(len(eig_norm) + 1e-10), 0.0, 1.0)
        except:
            norm_kl = 0.5
        expl = np.clip(0.4 * (1.0 - norm_ent) + 0.6 * norm_kl, 0.0, 1.0)
        min_ns, max_ns = 1, max(3, self.np // 4)
        target = int(np.round(min_ns + expl * (max_ns - min_ns)))
        target = np.clip(target, min_ns, max_ns)
        if target > self.neighborhood_size:
            self.neighborhood_size = min(target, self.neighborhood_size + 1)
        elif target < self.neighborhood_size:
            self.neighborhood_size = max(target, self.neighborhood_size - 1)
    
    def _adaptive_coefficients(self):
        imp = 1.0 / (1.0 + self.stagnation_counter)
        cog = self.cognitive_base * (1.0 + 0.5 * imp)
        soc = self.social_base * (2.0 - imp)
        return cog, soc
    
    def _velocity_update_base(self):
        cog, soc = self._adaptive_coefficients()
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        cog_comp = cog * r1 * (self.personal_best - self.population)
        soc_comp = soc * r2 * (self.local_best - self.population)
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
    
    # -------------------------------------------------------------------------
    # VARIANT 1: original.py (4 wins)
    # -------------------------------------------------------------------------
    def _position_update_original(self):
        centered = self.population - np.mean(self.population, axis=0)
        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            spectral_scale = np.clip(cond / 100.0, 0.5, 2.0)
            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * spectral_scale
            new_pop = self.population + (vel_scaled @ eigenvectors.T)
        except np.linalg.LinAlgError:
            new_pop = self.population + self.velocity
        self.population = self._clip_to_bounds(new_pop)
    
    # -------------------------------------------------------------------------
    # VARIANT 2: variant_02_catB_idea_0.py (3 wins) — eigenvalue-anisotropic
    # -------------------------------------------------------------------------
    def _position_update_variant_02_catB(self):
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            cond = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
            cond = np.clip(cond, 1.0, 1e6)
            total_var = np.sum(eigenvalues) + 1e-10
            cumvar = np.cumsum(np.sort(eigenvalues)[::-1]) / total_var
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            spectral_signal = np.clip(np.log1p(cond) / np.log1p(1e6), 0.0, 1.0)
            decay = 0.729 - 0.15 * (self.generation / 1000)
            target_inertia = 0.4 + 0.55 * spectral_signal
            self.inertia_weight = np.clip(
                self.inertia_weight * 0.8 + (target_inertia * 0.6 + decay * 0.4) * 0.2, 0.4, 0.95
            )
            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
            spectral_explore = np.clip(1.0 + 1.0 * spectral_signal, 0.5, 2.5)
            vel_proj = self.velocity @ eigenvectors
            sv_norm = eigenvalues / (eigenvalues[0] + 1e-10)
            per_comp = 1.0 / (sv_norm ** 2 + 0.01)
            per_comp = per_comp / (np.max(per_comp) + 1e-10)
            blend_w = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
            uniform = np.clip(cond / 100.0, 0.5, 2.0)
            combined = np.clip(uniform * (1.0 - blend_w) + per_comp * blend_w, 0.3, 2.5)
            vel_scaled = vel_proj * combined * spectral_explore
            fitness_perturb = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
            if self.global_best is not None:
                to_best = self.global_best - self.population
                to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_norm
                directional = fitness_perturb * to_best_dir
            else:
                directional = np.zeros((self.np, self.dim))
            rand_pert = spectral_explore * np.random.uniform(-0.5, 0.5, (self.np, self.dim))
            new_pop = self.population + self.inertia_weight * vel_scaled + directional + rand_pert
        except:
            spectral_explore = 1.0 + 0.5 * np.random.random()
            new_pop = self.population + self.inertia_weight * self.velocity * spectral_explore
        self.population = self._clip_to_bounds(new_pop)
    
    # -------------------------------------------------------------------------
    # VARIANT 3: variant_04_catD_idea_0.py (4 wins) — fitness-rank signals only
    # -------------------------------------------------------------------------
    def _position_update_variant_04_catD(self):
        try:
            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
            p10 = np.percentile(self.current_fitness, 10)
            p50 = np.percentile(self.current_fitness, 50)
            p90 = np.percentile(self.current_fitness, 90)
            fit_spread = (p90 - p10) / (np.abs(p50) + 1e-10)
            if hasattr(self, '_prev_fit') and len(self._prev_fit) == len(self.current_fitness):
                n = len(self.current_fitness)
                concordant = discordant = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        co = self.current_fitness[i] < self.current_fitness[j]
                        pr = self._prev_fit[i] < self._prev_fit[j]
                        if co == pr:
                            concordant += 1
                        else:
                            discordant += 1
                n_pairs = n * (n - 1) / 2
                kendall_tau = (concordant - discordant) / (n_pairs + 1e-10)
            else:
                kendall_tau = 0.5
            self._prev_fit = self.current_fitness.copy()
            if not hasattr(self, '_succ_cnt_d'):
                self._succ_cnt_d = np.zeros(self.np)
            improved = self.personal_best_fitness >= self.current_fitness
            self._succ_cnt_d[improved] += 1
            self._succ_cnt_d[~improved] *= 0.85
            succ_norm = self._succ_cnt_d / (np.max(self._succ_cnt_d) + 1.0)
            is_conv = fit_spread < 0.05
            is_stag = kendall_tau > 0.75
            if is_stag and is_conv:
                gscale = 1.8
            elif is_stag:
                gscale = 1.4
            elif is_conv:
                gscale = 0.7
            else:
                gscale = 1.0
            rank_exp = 1.0 + 0.6 * fitness_ranks
            succ_exp = 1.0 + 0.4 * (1.0 - succ_norm)
            vel_scale = np.clip(rank_exp[:, np.newaxis] * gscale * succ_exp[:, np.newaxis], 0.2, 3.0)
            best_rank_idx = np.argmin(fitness_ranks)
            attract = 0.5 * (1.0 - fitness_ranks[:, np.newaxis])
            rand_dir = np.random.randn(self.np, self.dim)
            rand_dir = rand_dir / (np.linalg.norm(rand_dir, axis=1, keepdims=True) + 1e-10)
            if self.global_best is not None and kendall_tau < 0.5:
                to_best = self.global_best - self.population
                to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                directional = attract * (to_best / to_best_norm)
            else:
                directional = attract * rand_dir
            rand_str = 0.3 * (1.0 + kendall_tau)
            rand_pert = rand_str * np.random.uniform(-1.0, 1.0, (self.np, self.dim))
            new_pop = self.population + vel_scale * self.velocity + directional + rand_pert
        except:
            new_pop = self.population + self.velocity
        self.population = self._clip_to_bounds(new_pop)
    
    # -------------------------------------------------------------------------
    # VARIANT 4: variant_07_catG_idea_0.py (4 wins) — Monte Carlo convergence
    # -------------------------------------------------------------------------
    def _position_update_variant_07_catG(self):
        try:
            n_mc = 50
            sub_size = max(3, self.np // 4)
            if not hasattr(self, '_mc_prob'):
                self._mc_prob = np.ones(self.np) * 0.5
            if self.global_best is not None and self.np >= sub_size:
                conv_cnt = np.zeros(self.np)
                for _ in range(n_mc):
                    sub_idx = np.random.choice(self.np, sub_size, replace=False)
                    sub_fit, _ = self._evaluate_batch(self.population[sub_idx], self._current_func)
                    sub_best = np.min(sub_fit)
                    for idx in sub_idx:
                        if sub_best < self.global_best_fitness * 1.001:
                            conv_cnt[idx] += 1
                new_prob = conv_cnt / n_mc
                self._mc_prob = np.clip(0.3 * new_prob + 0.7 * self._mc_prob, 0.1, 0.9)
            else:
                self._mc_prob = np.ones(self.np) * 0.5
            mc_prob = self._mc_prob
            n_boot = 30
            boot_conds = []
            for _ in range(n_boot):
                b_idx = np.random.choice(self.np, self.np, replace=True)
                try:
                    cent = self.population[b_idx] - np.mean(self.population[b_idx], axis=0)
                    cov = np.cov(cent.T)
                    eig = np.linalg.eigvalsh(cov)
                    eig = np.clip(eig, 1e-10, None)
                    cond = eig[-1] / (eig[0] + 1e-10) if len(eig) > 1 else 1.0
                    boot_conds.append(cond)
                except:
                    pass
            if len(boot_conds) > 5:
                bc_mean = np.mean(boot_conds)
                bc_std = np.std(boot_conds)
                boot_unc = bc_std / (bc_mean + 1e-10)
            else:
                bc_mean = 1.0
                boot_unc = 0.5
            unc_exp = 1.0 + 0.4 * np.clip(boot_unc, 0.0, 2.0)
            if bc_mean > 10.0:
                cond_exp = 1.3
            elif bc_mean < 3.0:
                cond_exp = 0.85
            else:
                cond_exp = 1.0
            gscale = np.clip(unc_exp * cond_exp, 0.5, 2.0)
            vel_scale = np.where(mc_prob > 0.6, gscale * 0.7, gscale * 1.4)
            vel_scale = np.clip(vel_scale, 0.3, 2.5)
            if self.global_best is not None:
                to_best = self.global_best - self.population
                to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                to_best_dir = to_best / to_best_norm
                n_strata = max(2, self.dim // 4)
                strat_idx = np.random.randint(0, n_strata, size=(self.np,))
                sw = 1.0 / n_strata
                ss = (strat_idx + np.random.random(self.np)) * sw
                directional = 0.4 * ss[:, np.newaxis] * to_best_dir
            else:
                directional = np.random.uniform(-0.5, 0.5, (self.np, self.dim))
            fit_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
            if not hasattr(self, '_succ_cnt_g'):
                self._succ_cnt_g = np.zeros(self.np)
            improved = self.personal_best_fitness >= self.current_fitness
            self._succ_cnt_g[improved] += 1
            self._succ_cnt_g[~improved] *= 0.9
            sn = self._succ_cnt_g / (np.max(self._succ_cnt_g) + 1.0)
            rs = (1.0 - mc_prob)[:, np.newaxis] * (1.0 - 0.3 * sn[:, np.newaxis])
            rand_pert = rs * np.random.uniform(-0.5, 0.5, (self.np, self.dim))
            new_pop = self.population + vel_scale[:, np.newaxis] * self.velocity + directional + rand_pert
        except:
            new_pop = self.population + 1.2 * self.velocity + np.random.uniform(-0.3, 0.3, (self.np, self.dim))
        self.population = self._clip_to_bounds(new_pop)
    
    # -------------------------------------------------------------------------
    # VARIANT 5: variant_08_catH_idea_0.py (3 wins) — graph spectral + temporal
    # -------------------------------------------------------------------------
    def _position_update_variant_08_catH(self):
        k = min(5, self.np - 1)
        sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        knn_idx = np.argsort(sq_dists, axis=1)[:, :k]
        visited = np.zeros(self.np, dtype=bool)
        components = []
        for start in range(self.np):
            if visited[start]:
                continue
            comp = []
            stack = [start]
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                comp.append(node)
                for nb in knn_idx[node]:
                    if not visited[nb]:
                        stack.append(nb)
            components.append(comp)
        comp_sizes = np.array([len(c) for c in components])
        p_comp_size = np.array([comp_sizes[np.argmax([i in c for c in components])] for i in range(self.np)])
        try:
            from scipy.sparse import csr_matrix
            row_i = np.repeat(np.arange(self.np), k)
            col_i = knn_idx.ravel()
            data = np.ones(len(row_i))
            adj = csr_matrix((data, (row_i, col_i)), shape=(self.np, self.np))
            adj = adj + adj.T
            adj.data[:] = 1.0
            deg = np.array(adj.sum(axis=1)).ravel() + 1e-10
            d_inv = 1.0 / np.sqrt(deg)
            d_diag = csr_matrix((d_inv, (np.arange(self.np), np.arange(self.np))), shape=(self.np, self.np))
            lap = d_diag @ (d_diag @ adj.T).T
            lap = np.eye(self.np) - lap
            from scipy.sparse.linalg import eigsh
            evals = eigsh(lap, k=min(5, self.np - 1), return_eigenvectors=False)
            evals = np.sort(evals)
            spec_gap = np.clip(evals[1] if len(evals) > 1 else 1.0, 0.0, 1.0)
        except:
            spec_gap = 1.0
        if not hasattr(self, '_fit_hist_h'):
            self._fit_hist_h = []
        self._fit_hist_h.append(float(np.min(self.current_fitness)))
        if len(self._fit_hist_h) > 20:
            self._fit_hist_h.pop(0)
        if len(self._fit_hist_h) >= 5:
            rec = np.array(self._fit_hist_h[-5:])
            oldest, newest = rec[0], rec[-1]
            if abs(oldest) > 1e-10:
                imp_rate = (oldest - newest) / (abs(oldest) + 1e-10)
            else:
                imp_rate = 0.0
        else:
            imp_rate = 0.0
        norm_imp = np.clip(imp_rate, 0.0, 1.0)
        stag_score = np.clip((1.0 - spec_gap) * (1.0 - norm_imp), 0.0, 1.0)
        is_expl = stag_score > 0.3
        is_stuck = stag_score > 0.6
        if is_stuck:
            gscale = 1.5
        elif is_expl:
            gscale = 1.15
        else:
            gscale = 0.85
        fit_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
        if not hasattr(self, '_succ_cnt_h'):
            self._succ_cnt_h = np.zeros(self.np)
        improved = self.personal_best_fitness >= self.current_fitness
        self._succ_cnt_h[improved] += 1
        self._succ_cnt_h[~improved] *= 0.9
        sn = self._succ_cnt_h / (np.max(self._succ_cnt_h) + 1.0)
        vel_scale = np.clip(gscale * (1.0 + 0.3 * sn) * (1.0 + 0.3 * (1.0 - fit_ranks)), 0.3, 2.5)
        fit_pert = 0.5 * (1.0 - fit_ranks[:, np.newaxis])
        if self.global_best is not None:
            to_best = self.global_best - self.population
            to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
            directional = fit_pert * (to_best / to_best_norm)
        else:
            directional = fit_pert * np.random.uniform(-1, 1, (self.np, self.dim))
        rs = 0.5 if is_expl else 0.25
        rand_pert = np.random.uniform(-rs, rs, (self.np, self.dim))
        comp_nudge = np.zeros((self.np, self.dim))
        for ci, comp in enumerate(components):
            if len(comp) < self.np * 0.3:
                cent = np.mean(self.population[comp], axis=0)
                for pi in comp:
                    td = cent - self.population[pi]
                    td_n = np.linalg.norm(td) + 1e-10
                    comp_nudge[pi] = 0.2 * (1.0 - p_comp_size[pi] / max(1, self.np)) * (td / td_n)
        new_pop = self.population + self.velocity * vel_scale[:, np.newaxis] + directional + rand_pert + comp_nudge
        self.population = self._clip_to_bounds(new_pop)
    
    # -------------------------------------------------------------------------
    # VARIANT 6: variant_09_catA_idea_0.py (3 wins) — geometric bounding box
    # -------------------------------------------------------------------------
    def _position_update_variant_09_catA(self):
        pop = self.population
        np_p, dim = pop.shape
        pop_min = np.min(pop, axis=0)
        pop_max = np.max(pop, axis=0)
        pop_range = pop_max - pop_min + 1e-10
        g_range = self.upper_bound - self.lower_bound
        centroid = np.mean(pop, axis=0)
        norm_pos = 2.0 * (pop - pop_min) / pop_range - 1.0
        edge_margin = 0.1
        near_edge = np.max(np.abs(norm_pos), axis=1) > (1.0 - edge_margin)
        edge_press = np.sum(near_edge) / np_p
        spr_ratios = pop_range / (g_range + 1e-10)
        min_spr = np.min(spr_ratios)
        max_spr = np.max(spr_ratios)
        anis = max_spr / (min_spr + 1e-10) if min_spr > 1e-10 else 1.0
        try:
            if dim <= 10 and np_p >= dim + 1:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pop, qhull_options='Qt')
                hvol = hull.volume
            else:
                hvol = np.prod(pop_range + 1e-10)
        except:
            hvol = np.prod(pop_range + 1e-10)
        max_vol = (g_range ** dim)
        norm_vol = np.clip(hvol / (max_vol + 1e-10), 1e-10, 1.0)
        sq_dists = np.sum((pop[:, np.newaxis, :] - pop[np.newaxis, :, :]) ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        all_d = np.sqrt(sq_dists[sq_dists < np.inf])
        if len(all_d) > 0:
            m_dist = np.mean(all_d)
            std_dist = np.std(all_d)
            exp_ud = np.mean(pop_range) * 0.5
            clust_r = m_dist / (exp_ud + 1e-10)
        else:
            m_dist = std_dist = 1.0
            clust_r = 1.0
        coll_exp = np.clip(1.0 + 1.5 * (1.0 - norm_vol), 0.5, 3.0)
        anis_exp = np.clip(1.0 + 0.3 * np.log1p(anis) / np.log1p(100.0), 0.8, 2.0)
        in_press = edge_press * 0.5
        spr_exp = np.clip(1.0 + 0.5 * (1.0 - np.clip(std_dist / (m_dist + 1e-10), 0.0, 1.0)), 0.5, 2.5)
        geo_exp = np.clip(coll_exp * anis_exp * spr_exp, 0.3, 4.0)
        edge_dist = 1.0 - np.max(np.abs(norm_pos), axis=1)
        in_str = in_press * (1.0 - edge_dist)[:, np.newaxis]
        to_cent = centroid - pop
        to_cent_n = np.linalg.norm(to_cent, axis=1, keepdims=True) + 1e-10
        in_nudge = in_str * (to_cent / to_cent_n)
        try:
            centered = pop - centroid
            cov = np.cov(centered.T)
            eigenv, eigenvec = np.linalg.eigh(cov)
            eigenv = np.clip(eigenv, 1e-10, None)
            eigenv = np.sort(eigenv)
            inv_sp = 1.0 / (eigenv / (eigenv[0] + 1e-10) + 0.01)
            inv_sp = inv_sp / (np.max(inv_sp) + 1e-10)
            v_proj = self.velocity @ eigenvec
            v_corr = v_proj * inv_sp
            anis_corr = (v_corr @ eigenvec.T) - self.velocity
        except:
            anis_corr = np.zeros_like(self.velocity)
        rand_pert = geo_exp * np.random.uniform(-0.5, 0.5, (np_p, dim))
        new_pop = pop + self.velocity * geo_exp + anis_corr + in_nudge + rand_pert
        self.population = self._clip_to_bounds(new_pop)
    
    # -------------------------------------------------------------------------
    # VARIANT 7: variant_10_catB_idea_0.py (3 wins) — effective-rank velocity
    # -------------------------------------------------------------------------
    def _position_update_variant_10_catB(self):
        centered = self.population - np.mean(self.population, axis=0)
        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            total_eig = np.sum(eigenvalues)
            e_norm = np.clip(eigenvalues / (total_eig + 1e-10), 1e-15, None)
            entropy = -np.sum(e_norm * np.log(e_norm + 1e-15))
            max_entropy = np.log(len(e_norm))
            eff_rank = np.clip(np.exp(entropy) / (len(e_norm) + 1e-10), 0.0, 1.0)
            cond = np.clip(eigenvalues[0] / (eigenvalues[-1] + 1e-10), 1.0, 10000.0)
            cond_sig = np.log1p(cond) / np.log1p(10000.0)
            eig_ratio = np.clip(eigenvalues / (eigenvalues[0] + 1e-10), 0.0, 1.0)
            spec_scale = np.sqrt(eig_ratio + 0.01)
            spec_scale = spec_scale / (np.max(spec_scale) + 1e-10)
            blend = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
            per_comp = 1.0 * (1.0 - blend) + spec_scale * blend
            vel_proj = self.velocity @ eigenvectors
            vel_scaled = vel_proj * per_comp
            v_corr = vel_scaled @ eigenvectors.T - self.velocity
            if eff_rank < 0.3:
                exp_boost = 1.5
            elif eff_rank > 0.7:
                exp_boost = 1.2
            else:
                exp_boost = 1.0
            rand_pert = np.random.uniform(-0.3, 0.3, (self.np, self.dim))
            rand_pert *= exp_boost * (1.0 + 0.5 * cond_sig)
            new_pop = self.population + self.velocity + 0.5 * v_corr + rand_pert
        except np.linalg.LinAlgError:
            new_pop = self.population + self.velocity
        self.population = self._clip_to_bounds(new_pop)
    
    # -------------------------------------------------------------------------
    # Dispatcher
    # -------------------------------------------------------------------------
    def _dispatch_position_update(self, variant):
        if variant == 'original':
            self._position_update_original()
        elif variant == 'variant_02_catB':
            self._position_update_variant_02_catB()
        elif variant == 'variant_04_catD':
            self._position_update_variant_04_catD()
        elif variant == 'variant_07_catG':
            self._position_update_variant_07_catG()
        elif variant == 'variant_08_catH':
            self._position_update_variant_08_catH()
        elif variant == 'variant_09_catA':
            self._position_update_variant_09_catA()
        elif variant == 'variant_10_catB':
            self._position_update_variant_10_catB()
        else:
            self._position_update_original()
    
    # -------------------------------------------------------------------------
    # PROBE: rank-based scoring (scale-independent per calibration rule a)
    # -------------------------------------------------------------------------
    def _record_probe_reward(self, variant, best_fitness):
        """Rank-based reward: fraction of probe gens this variant was in top tercile."""
        self._probe_fitness_hist.append(best_fitness)
        if len(self._probe_fitness_hist) < 2:
            reward = 1.0
        else:
            cur_best = min(self._probe_fitness_hist)
            cur_worst = max(self._probe_fitness_hist)
            if cur_worst > cur_best:
                reward = np.clip((cur_worst - best_fitness) / (cur_worst - cur_best + 1e-10), 0.0, 1.0)
            else:
                reward = 1.0
        self._variant_scores[variant].append(reward)
        self._variant_total[variant] += reward
        self._variant_count[variant] += 1
        # Sliding window for rank normalization
        self._recent_rewards[variant].append(reward)
        if len(self._recent_rewards[variant]) > self._reward_window_size:
            self._recent_rewards[variant].pop(0)
    
    def _compute_ucb(self, variant):
        n = self._variant_count[variant]
        if n == 0:
            return float('inf')
        total_n = sum(self._variant_count.values())
        avg = self._variant_total[variant] / n
        expl = np.sqrt(2.0 * np.log(max(total_n, 1)) / max(n, 1))
        return avg + expl
    
    def _compute_variant_weights(self):
        """Softmax of UCB scores → blend weights. No hard-coded table."""
        ucb_scores = {v: self._compute_ucb(v) for v in self._variant_names}
        max_ucb = max(ucb_scores.values())
        # Numerically stable softmax
        exp_scores = {v: np.exp(ucb_scores[v] - max_ucb - 10.0) for v in self._variant_names}
        total_exp = sum(exp_scores.values()) + 1e-10
        weights = {v: exp_scores[v] / total_exp for v in self._variant_names}
        return weights
    
    def _rank_normalize_fitness(self, fitness_batch):
        """Z-score normalize within sliding window — scale-independent."""
        if len(fitness_batch) < 2:
            return np.zeros_like(fitness_batch)
        mean_f = np.mean(fitness_batch)
        std_f = np.std(fitness_batch) + 1e-10
        return (fitness_batch - mean_f) / std_f
    
    # -------------------------------------------------------------------------
    # Restart
    # -------------------------------------------------------------------------
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
    
    def _update_local_from_personal(self):
        improved = self.personal_best_fitness < self.local_best_fitness
        self.local_best[improved] = self.personal_best[improved]
        self.local_best_fitness[improved] = self.personal_best_fitness[improved]
    
    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------
    def __call__(self, func, stopping_condition):
        self._current_func = func
        self._initialize_population()
        self.generation = 0
        self.stagnation_counter = 0
        self.last_improvement_gen = 0
        
        # Reset probe state
        for v in self._variant_names:
            self._variant_scores[v] = []
            self._variant_total[v] = 0.0
            self._variant_count[v] = 0
            self._recent_rewards[v] = []
        self._probe_fitness_hist = []
        self._probe_best_seen = np.inf
        self._in_probe_phase = True
        self._probe_variant_idx = 0
        self._probe_gens_current = 0
        self._commit_gen = 0
        self._commit_best = np.inf
        self._commit_fitness_history = []
        
        # Reset per-variant EMA state
        for attr in ['_mc_prob', '_succ_cnt_d', '_succ_cnt_g', '_succ_cnt_h',
                     '_prev_fit', '_fit_hist_h']:
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
                idx = np.nanargmin(self.current_fitness)
                return self.current_fitness[idx], self.population[idx].copy()
            return self.global_best_fitness, self.global_best.copy()
        
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # --- PHASE A: PROBE ---
            if self._in_probe_phase:
                current_variant = self._variant_names[self._probe_variant_idx]
                
                self._adapt_inertia_weight()
                self._adapt_neighborhood_size()
                self._update_local_from_personal()
                
                # Evaluate current operator on real population
                self._velocity_update_base()
                self._dispatch_position_update(current_variant)
                
                self.current_fitness, actual_n = self._evaluate_batch(self.population, func)
                if actual_n < self.np:
                    self.np = actual_n
                    self.population = self.population[:actual_n]
                    self.velocity = self.velocity[:actual_n]
                    self.current_fitness = self.current_fitness[:actual_n]
                
                if stopping_condition():
                    break
                
                self._update_personal_best_batch()
                self._compute_local_best_batch()
                self._compute_global_best()
                
                gen_best = float(np.min(self.current_fitness))
                if gen_best < self._probe_best_seen:
                    self._probe_best_seen = gen_best
                self._record_probe_reward(current_variant, self._probe_best_seen)
                
                self._probe_gens_current += 1
                if self._probe_gens_current >= self._probe_gens_per_variant:
                    self._probe_variant_idx += 1
                    self._probe_gens_current = 0
                
                if self._probe_variant_idx >= len(self._variant_names):
                    self._in_probe_phase = False
                    self._variant_weights = self._compute_variant_weights()
                    self._commit_best = self.global_best_fitness
                    self._commit_gen = 0
                
                self._restart_if_stagnant()
            
            # --- PHASE B: COMMIT (adaptive ensemble blend) ---
            else:
                self._commit_gen += 1
                
                # Adapt weights from recent rank-based performance
                if len(self._commit_fitness_history) >= 3:
                    # Rank-normalize recent fitness values within window
                    recent_fits = [h[0] for h in self._commit_fitness_history[-self._commit_rank_window:]]
                    ranks = np.argsort(np.argsort(recent_fits)) / max(1, len(recent_fits) - 1)
                    # Map rank improvement to per-variant weight adjustment
                    # Higher rank (worse) = variant is struggling = reduce weight
                    for i, v in enumerate(self._variant_names):
                        if i < len(ranks):
                            rank_based_adj = 1.0 - 0.1 * ranks[i]
                            self._variant_weights[v] *= rank_based_adj
                    # Re-normalize
                    total_w = sum(self._variant_weights.values())
                    if total_w > 0:
                        for v in self._variant_names:
                            self._variant_weights[v] /= total_w
                
                # Evaluate ALL variants on candidate populations
                cand_pops = {}
                cand_fits = {}
                vel_copy = self.velocity.copy()
                pop_copy = self.population.copy()
                cog, soc = self._adaptive_coefficients()
                r1 = np.random.uniform(0, 1, (self.np, self.dim))
                r2 = np.random.uniform(0, 1, (self.np, self.dim))
                base_vel = np.clip(
                    self.inertia_weight * self.velocity +
                    cog * r1 * (self.personal_best - self.population) +
                    soc * r2 * (self.local_best - self.population),
                    self.v_min, self.v_max
                )
                
                for variant in self._variant_names:
                    # Each variant proposes a population update
                    self.velocity = base_vel.copy()
                    tmp_pop = pop_copy.copy()
                    self.population = tmp_pop
                    self._dispatch_position_update(variant)
                    cand_pops[variant] = self.population.copy()
                    cand_fits[variant], _ = self._evaluate_batch(cand_pops[variant], func)
                
                # Restore real population and velocity
                self.velocity = vel_copy
                self.population = pop_copy
                
                # Blend populations using UCB-derived weights
                blended = np.zeros_like(self.population)
                for variant in self._variant_names:
                    blended += self._variant_weights[variant] * cand_pops[variant]
                blended = self._clip_to_bounds(blended)
                
                # Also evaluate blended population
                blended_fitness, _ = self._evaluate_batch(blended, func)
                blended_best = np.min(blended_fitness)
                
                # Decide: use blended or best single candidate?
                real_best = np.min(self.current_fitness)
                
                # Use blended if it's competitive (within 5% of best candidate)
                best_cand = min(np.min(cand_fits[v]) for v in self._variant_names)
                if blended_best <= best_cand * 1.05 or blended_best <= real_best:
                    self.population = blended
                    self.current_fitness = blended_fitness
                else:
                    # Use best single candidate
                    best_var = min(self._variant_names, key=lambda v: np.min(cand_fits[v]))
                    self.population = cand_pops[best_var]
                    self.current_fitness = cand_fits[best_var]
                
                # Update velocity as average of candidate velocities
                self.velocity = base_vel.copy()
                
                if stopping_condition():
                    break
                
                self._update_personal_best_batch()
                self._compute_local_best_batch()
                self._compute_global_best()
                self._restart_if_stagnant()
                
                # Track for rank-based weight adaptation
                gen_best_fitness = float(np.min(self.current_fitness))
                self._commit_fitness_history.append((gen_best_fitness, self._variant_weights.copy()))
                if len(self._commit_fitness_history) > self._commit_rank_window * 2:
                    self._commit_fitness_history.pop(0)
                
                if self.global_best_fitness < self._commit_best:
                    self._commit_best = self.global_best_fitness
        
        if self.global_best is None or np.isnan(self.global_best_fitness):
            idx = np.nanargmin(self.current_fitness)
            return self.current_fitness[idx], self.population[idx].copy()
        return self.global_best_fitness, self.global_best.copy()
```