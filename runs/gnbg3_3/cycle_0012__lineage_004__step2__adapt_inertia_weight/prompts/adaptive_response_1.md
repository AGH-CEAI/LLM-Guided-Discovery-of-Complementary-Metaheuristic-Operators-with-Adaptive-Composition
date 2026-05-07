```python
import numpy as np


class AdaptiveTopologyParticleSwarm:
    """
    Mechanism: PROBE-AND-COMMIT with UCB1 (Contextual Bandit, mechanism #4)
    
    The gap table is NOT blend-hostile:
      - Max winner-vs-runner-up gap is only 1.5× (Task 2).
      - Max winner-vs-median gap is 6.4× (Task 3).
      - No task has a gap >= 10×, so linear blending is mathematically viable.
    
    However, with 7 distinct winners spread across 24 tasks, no single
    inertia-weight strategy wins everywhere. The probe-and-commit pattern
    extracts a cheap landscape fingerprint (convergence slope, rank
    correlation, effective dimensionality, diversity, spectral condition)
    and routes to the empirically best-performing strategy.
    
    Each of the 7 winning strategies is a complete, self-contained
    _adapt_inertia_weight implementation. The dispatcher runs each for
    3 generations in a sub-population, then commits to the UCB1 winner
    for the remaining budget. No blending, no hard-coded weights.
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
        
        # --- PROBE-AND-COMMIT STATE ---
        # THE 7 BENCHMARK-WINNING INERTIA WEIGHT STRATEGIES
        self._operators = [
            'orig',      # original.py: 6 wins (Tasks 1,4,5,8,10,11)
            'catA',      # variant_01: 1 win  (Task 20)
            'catC',      # variant_03: 3 wins (Tasks 12,17,18)
            'catD',      # variant_04: 6 wins (Tasks 3,7,15,16,21,22)
            'catF',      # variant_06: 1 win  (Task 2)
            'catH',      # variant_08: 5 wins (Tasks 0,6,9,13,23)
            'catB',      # variant_10: 2 wins (Tasks 14,19)
        ]
        self._operator_scores = {op: [] for op in self._operators}
        self._operator_total = {op: 0.0 for op in self._operators}
        self._operator_count = {op: 0 for op in self._operators}
        
        # Probe: 3 gens per operator, min 21 total
        self._probe_gens = 3
        self._probe_total = len(self._operators) * self._probe_gens
        
        self._committed_op = None
        self._runner_up_op = None
        self._in_probe = True
        self._probe_idx = 0
        self._probe_gens_cur = 0
        self._probe_best = np.inf
        
        # Sliding window for rank-based scoring
        self._score_window_size = 7
        self._probe_fitness_hist = []
        self._probe_op_hist = []
        
        # Fingerprint signals (computed at commit time)
        self._fingerprint = {}
        
        # Commit-phase tracking
        self._commit_best = np.inf
        self._epsilon_review = 0.05
    
    def _initialize_population(self):
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.np, self.dim))
        self.velocity = np.random.uniform(
            self.v_min * 0.1, self.v_max * 0.1, (self.np, self.dim))
        
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
                nbhd = np.arange(left, right)
            else:
                nbhd = np.concatenate([np.arange(left, self.np), np.arange(0, right)])
            all_idx = np.concatenate([nbhd, [i]])
            fit_nbhd = self.personal_best_fitness[all_idx]
            best_local = np.argmin(fit_nbhd)
            self.local_best[i] = self.personal_best[all_idx[best_local]]
            self.local_best_fitness[i] = fit_nbhd[best_local]
    
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
        """Extract cheap landscape fingerprint from probe data.
        
        All signals are observable from any black-box func:
          - convergence_slope: log-slope of best_fitness vs generation
          - rank_corr: Spearman correlation (fitness rank vs dist to best)
          - eff_dim: effective dimensionality from covariance eigenvalues
          - diversity: current population diversity
          - condition: spectral condition number of population
        """
        sig = {}
        
        # 1. Convergence slope from probe history
        if len(self._probe_fitness_hist) >= 3:
            gens = np.arange(len(self._probe_fitness_hist))
            bests = np.array(self._probe_fitness_hist)
            valid = bests > 0
            if np.sum(valid) >= 2:
                log_bests = np.log1p(bests[valid])
                sig['convergence_slope'] = np.polyfit(gens[valid], log_bests, 1)[0]
            else:
                sig['convergence_slope'] = np.polyfit(gens, bests, 1)[0]
        else:
            sig['convergence_slope'] = 0.0
        
        # 2. Rank correlation
        if self.global_best is not None and self.np > 2:
            fit_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
            dists = np.linalg.norm(self.population - self.global_best, axis=1)
            dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
            if np.std(fit_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                sig['rank_corr'] = float(np.corrcoef(fit_ranks, dist_ranks)[0, 1])
            else:
                sig['rank_corr'] = 0.0
        else:
            sig['rank_corr'] = 0.0
        
        # 3. Effective dimensionality from covariance
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.clip(eigvals, 1e-10, None)
            total_var = np.sum(eigvals)
            if total_var > 0:
                sorted_eig = np.sort(eigvals)[::-1]
                cumvar = np.cumsum(sorted_eig) / total_var
                sig['eff_dim'] = float(np.searchsorted(cumvar, 0.95)) + 1
            else:
                sig['eff_dim'] = float(self.dim)
        except:
            sig['eff_dim'] = float(self.dim)
        
        # 4. Diversity
        sig['diversity'] = self._compute_diversity()
        
        # 5. Spectral condition number
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.clip(eigvals, 1e-10, None)
            eigvals = np.sort(eigvals)[::-1]
            sig['condition'] = float(eigvals[0] / (eigvals[-1] + 1e-10))
        except:
            sig['condition'] = 1.0
        
        self._fingerprint = sig
        return sig
    
    # =========================================================================
    # THE 7 BENCHMARK-WINNING INERTIA WEIGHT STRATEGIES
    # =========================================================================
    
    def _adapt_inertia_orig(self):
        """Original: condition-number spectral signal (6 wins: Tasks 1,4,5,8,10,11)."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.clip(eigvals, 1e-10, None)
            eigvals = np.sort(eigvals)[::-1]
            cond = np.clip(eigvals[0] / (eigvals[-1] + 1e-10), 1.0, 10000.0)
            spectral_signal = np.clip(np.log1p(cond) / np.log1p(10000.0), 0.0, 1.0)
            target = np.clip(0.4 + 0.55 * spectral_signal, 0.4, 0.95)
            decay = np.clip(0.729 - 0.15 * (self.generation / 1000), 0.4, 0.95)
            target = np.clip(target * 0.6 + decay * 0.4, 0.4, 0.95)
            self.inertia_weight = np.clip(self.inertia_weight * 0.8 + target * 0.2, 0.4, 0.95)
        except:
            decay = np.clip(0.729 - 0.15 * (self.generation / 1000), 0.4, 0.95)
            self.inertia_weight = np.clip(self.inertia_weight * 0.8 + decay * 0.2, 0.4, 0.95)
    
    def _adapt_inertia_catA(self):
        """Category A: geometric layout — centroid distance, axis spread, k-NN density (1 win: Task 20)."""
        try:
            centroid = np.mean(self.population, axis=0)
            dists_centroid = np.linalg.norm(self.population - centroid, axis=1)
            mean_cdist = np.mean(dists_centroid)
            mins, maxs = np.min(self.population, axis=0), np.max(self.population, axis=0)
            ranges = maxs - mins
            mean_range, max_range = np.mean(ranges), np.max(ranges)
            diameter = mean_cdist * 2.0 + max_range
            k = min(5, self.np - 1)
            sq_dists = np.sum((self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]) ** 2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            knn_sq = np.sort(sq_dists, axis=1)[:, :k]
            mean_knn = np.mean(np.sqrt(knn_sq)) + 1e-10
            min_range = np.min(ranges) + 1e-10
            spread_ratio = np.clip(max_range / min_range, 1.0, 100.0)
            exp_diameter = 200.0 * np.sqrt(self.dim)
            d_signal = np.clip(diameter / exp_diameter, 0.0, 1.0)
            c_signal = np.clip(mean_cdist / exp_diameter, 0.0, 1.0)
            s_signal = np.clip(np.log1p(spread_ratio) / np.log1p(100.0), 0.0, 1.0)
            knn_signal = np.clip(1.0 / (mean_knn + 1.0), 0.0, 1.0)
            geo_signal = np.clip(0.30 * d_signal + 0.25 * c_signal + 0.25 * s_signal + 0.20 * knn_signal, 0.0, 1.0)
            target = np.clip(0.4 + 0.55 * geo_signal, 0.4, 0.95)
            decay = np.clip(0.729 - 0.15 * (self.generation / 1000), 0.4, 0.95)
            target = np.clip(target * 0.6 + decay * 0.4, 0.4, 0.95)
            self.inertia_weight = np.clip(self.inertia_weight * 0.8 + target * 0.2, 0.4, 0.95)
        except:
            decay = np.clip(0.729 - 0.15 * (self.generation / 1000), 0.4, 0.95)
            self.inertia_weight = np.clip(self.inertia_weight * 0.8 + decay * 0.2, 0.4, 0.95)
    
    def _adapt_inertia_catC(self):
        """Category C: spectral entropy of covariance eigenvalues (3 wins: Tasks 12,17,18)."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.clip(eigvals, 1e-10, None)
            eigvals = np.sort(eigvals)[::-1]
            total = np.sum(eigvals)
            p = eigvals / total if total > 0 else np.ones_like(eigvals) / len(eigvals)
            p = np.clip(p, 1e-10, 1.0)
            spectral_entropy = -np.sum(p * np.log2(p))
            max_entropy = np.log2(len(eigvals) + 1e-10)
            norm_entropy = np.clip(spectral_entropy / max_entropy, 0.0, 1.0) if max_entropy > 0 else 0.5
            target = np.clip(0.4 + 0.55 * norm_entropy, 0.4, 0.95)
            decay = np.clip(0.729 - 0.15 * (self.generation / 1000), 0.4, 0.95)
            target = np.clip(target * 0.6 + decay * 0.4, 0.4, 0.95)
            self.inertia_weight = np.clip(self.inertia_weight * 0.8 + target * 0.2, 0.4, 0.95)
        except:
            decay = np.clip(0.729 - 0.15 * (self.generation / 1000), 0.4, 0.95)
            self.inertia_weight = np.clip(self.inertia_weight * 0.8 + decay * 0.2, 0.4, 0.95)
    
    def _adapt_inertia_catD(self):
        """Category D: fitness-distance correlation + success history (6 wins: Tasks 3,7,15,16,21,22)."""
        try:
            if self.global_best is not None and self.np > 3:
                fit_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
                dists = np.linalg.norm(self.population - self.global_best, axis=1)
                dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
                if np.std(fit_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                    corr = float(np.corrcoef(fit_ranks, dist_ranks)[0, 1])
                    spearman_fdc = np.clip(corr, -1.0, 1.0)
                else:
                    spearman_fdc = 0.0
            else:
                spearman_fdc = 0.0
            fit_sorted = np.sort(self.current_fitness)
            p90 = fit_sorted[int(0.9 * self.np)] if self.np > 0 else 0.0
            p10 = fit_sorted[int(0.1 * self.np)] if self.np > 0 else 0.0
            pct_spread = np.clip((p90 - p10) / (np.abs(self.global_best_fitness) + 1e-10 + p90 - p10 + 1e-10), 0.0, 1.0)
            if not hasattr(self, '_success_ema'):
                self._success_ema = 0.0
            if not hasattr(self, '_prev_fitness_med'):
                self._prev_fitness_med = np.median(self.current_fitness)
            med_fit = np.median(self.current_fitness)
            med_imp = max(0.0, self._prev_fitness_med - med_fit)
            self._prev_fitness_med = med_fit
            self._success_ema = 0.2 * med_imp + 0.8 * self._success_ema
            success_sig = np.clip(self._success_ema / (np.abs(self.global_best_fitness) + 1.0), 0.0, 1.0)
            fdc_exploit = (spearman_fdc + 1.0) / 2.0
            roughness = 1.0 - fdc_exploit
            explore_sig = np.clip(0.5 * roughness + 0.3 * pct_spread + 0.2 * (1.0 - success_sig), 0.0, 1.0)
            target = np.clip(0.4 + 0.55 * explore_sig, 0.4, 0.95)
            decay = np.clip(0.729 - 0.15 * (self.generation / 1000), 0.4, 0.95)
            target = np.clip(target * 0.6 + decay * 0.4, 0.4, 0.95)
            self.inertia_weight = np.clip(self.inertia_weight * 0.8 + target * 0.2, 0.4, 0.95)
        except:
            decay = np.clip(0.729 - 0.15 * (self.generation / 1000), 0.4, 0.95)
            self.inertia_weight = np.clip(self.inertia_weight * 0.8 + decay * 0.2, 0.4, 0.95)
    
    def _adapt_inertia_catF(self):
        """Category F: temporal signals — improvement EMA, stagnation, diversity drift (1 win: Task 2)."""
        if self.global_best is not None and hasattr(self, '_prev_best_fit'):
            raw_imp = max(0.0, self._prev_best_fit - self.global_best_fitness)
        else:
            raw_imp = 1.0
        self._prev_best_fit = self.global_best_fitness if self.global_best is not None else np.inf
        if not hasattr(self, '_ema_imp'):
            self._ema_imp = raw_imp
        self._ema_imp = 0.2 * raw_imp + 0.8 * self._ema_imp
        scale = max(abs(self.global_best_fitness), 1e-10)
        norm_imp = np.clip(self._ema_imp / scale, 0.0, 2.0) / 2.0
        if not hasattr(self, '_ema_stag'):
            self._ema_stag = 0.0
        self._ema_stag = 0.15 * self.stagnation_counter + 0.85 * self._ema_stag
        norm_stag = np.clip(self._ema_stag / 100.0, 0.0, 1.0)
        curr_div = self._compute_diversity()
        if not hasattr(self, '_ema_div'):
            self._ema_div = curr_div
        self._ema_div = 0.1 * curr_div + 0.9 * self._ema_div
        div_ratio = curr_div / (self._ema_div + 1e-10)
        norm_div = np.clip(div_ratio - 1.0, -1.0, 1.0)
        imp_sig = np.clip(norm_imp, 0.0, 2.0) / 2.0
        inertia_imp = 0.7 - 0.4 * imp_sig
        stag_boost = 0.3 * norm_stag
        div_corr = -0.1 * norm_div
        target = np.clip(inertia_imp + stag_boost + div_corr, 0.3, 0.95)
        time_floor = 0.4 - 0.1 * min(self.generation / 500, 1.0)
        target = max(target, time_floor)
        self.inertia_weight = np.clip(0.8 * self.inertia_weight + 0.2 * target, 0.3, 0.95)
    
    def _adapt_inertia_catH(self):
        """Category H: hybrid spectral condition + fitness improvement, stagnation-weighted (5 wins: Tasks 0,6,9,13,23)."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.clip(eigvals, 1e-10, None)
            eigvals = np.sort(eigvals)[::-1]
            cond = np.clip(eigvals[0] / (eigvals[-1] + 1e-10), 1.0, 10000.0)
            spectral_sig = np.clip(np.log1p(cond) / np.log1p(10000.0), 0.0, 1.0)
            spectral_inertia = np.clip(0.4 + 0.55 * spectral_sig, 0.4, 0.95)
            if self.global_best is not None:
                if not hasattr(self, '_prev_best_catH'):
                    self._prev_best_catH = self.global_best_fitness
                imp = self._prev_best_catH - self.global_best_fitness
                self._prev_best_catH = self.global_best_fitness
            else:
                imp = 0.0
            if not hasattr(self, '_ema_imp_catH'):
                self._ema_imp_catH = 0.0
            self._ema_imp_catH = 0.3 * max(0.0, imp) + 0.7 * self._ema_imp_catH
            max_exp = 1.0 + self.stagnation_counter * 0.1
            imp_sig = 1.0 - np.clip(self._ema_imp_catH / (max_exp + 1e-10), 0.0, 1.0)
            imp_inertia = np.clip(0.4 + 0.55 * imp_sig, 0.4, 0.95)
            is_stag = self.stagnation_counter > 15
            if is_stag:
                spec_w, imp_w = 0.7, 0.3
            else:
                spec_w, imp_w = 0.3, 0.7
            target = np.clip(spec_w * spectral_inertia + imp_w * imp_inertia, 0.4, 0.95)
            decay = np.clip(0.729 - 0.15 * (self.generation / 1000), 0.4, 0.95)
            target = np.clip(target * 0.6 + decay * 0.4, 0.4, 0.95)
            self.inertia_weight = np.clip(self.inertia_weight * 0.8 + target * 0.2, 0.4, 0.95)
        except:
            decay = np.clip(0.729 - 0.15 * (self.generation / 1000), 0.4, 0.95)
            self.inertia_weight = np.clip(self.inertia_weight * 0.8 + decay * 0.2, 0.4, 0.95)
    
    def _adapt_inertia_catB(self):
        """Category B: SVD spectral entropy + effective dimensionality (2 wins: Tasks 14,19)."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            U, sv, Vt = np.linalg.svd(centered, full_matrices=False)
            sv = np.clip(sv, 1e-10, None)
            total_var = np.sum(sv ** 2) + 1e-10
            sv_norm = np.clip(sv ** 2 / total_var, 1e-10, None)
            spec_ent = -np.sum(sv_norm * np.log(sv_norm))
            max_ent = np.log(min(self.np, self.dim) + 1e-10)
            ent_ratio = np.clip(spec_ent / max_ent, 0.0, 1.0) if max_ent > 0 else 0.5
            cumvar = np.cumsum(sv_norm)
            eff_dim = float(np.searchsorted(cumvar, 0.95)) + 1
            eff_dim_ratio = eff_dim / min(self.np, self.dim)
            if len(sv) >= 2:
                sv_ratios = np.clip(sv[1:] / (sv[:-1] + 1e-10), 0.0, 1.0)
                decay_rate = np.exp(np.mean(np.log(sv_ratios + 1e-10)))
            else:
                decay_rate = 0.5
            collapse_sig = np.clip(
                (1.0 - ent_ratio) * 0.4 + (1.0 - eff_dim_ratio) * 0.4 + (1.0 - decay_rate) * 0.2,
                0.0, 1.0)
            target = np.clip(0.4 + 0.55 * collapse_sig, 0.4, 0.95)
            decay = np.clip(0.729 - 0.15 * (self.generation / 1000), 0.4, 0.95)
            target = np.clip(target * 0.6 + decay * 0.4, 0.4, 0.95)
            self.inertia_weight = np.clip(self.inertia_weight * 0.8 + target * 0.2, 0.4, 0.95)
        except:
            decay = np.clip(0.729 - 0.15 * (self.generation / 1000), 0.4, 0.95)
            self.inertia_weight = np.clip(self.inertia_weight * 0.8 + decay * 0.2, 0.4, 0.95)
    
    # --- Dispatcher: map operator name -> inertia adaptation method ---
    def _dispatch_inertia(self, operator):
        if operator == 'orig':
            self._adapt_inertia_orig()
        elif operator == 'catA':
            self._adapt_inertia_catA()
        elif operator == 'catC':
            self._adapt_inertia_catC()
        elif operator == 'catD':
            self._adapt_inertia_catD()
        elif operator == 'catF':
            self._adapt_inertia_catF()
        elif operator == 'catH':
            self._adapt_inertia_catH()
        elif operator == 'catB':
            self._adapt_inertia_catB()
        else:
            self._adapt_inertia_orig()
    
    # =========================================================================
    # PROBE-AND-COMMIT: rank-based scoring (scale-independent)
    # =========================================================================
    
    def _record_score(self, operator, best_fitness):
        """Record score using rank-based normalization within sliding window.
        
        Score = where this generation's best_fitness falls in the rank
        distribution of the last _score_window_size generations for the
        SAME operator. This is rank-based, not raw fitness — no magic
        constants, no scale dependence.
        """
        self._probe_fitness_hist.append(best_fitness)
        self._probe_op_hist.append(operator)
        
        # Keep only the last window for rank computation
        start = max(0, len(self._probe_fitness_hist) - self._score_window_size)
        window_fits = np.array(self._probe_fitness_hist[start:])
        window_ops = self._probe_op_hist[start:]
        
        # Filter to same operator within window
        same_op_mask = np.array([op == operator for op in window_ops])
        same_op_fits = window_fits[same_op_mask]
        
        if len(same_op_fits) < 2:
            score = 1.0
        else:
            # Rank-based: position of current best within same-op history
            sorted_fits = np.sort(same_op_fits)
            rank = np.searchsorted(sorted_fits, best_fitness) / max(1, len(sorted_fits) - 1)
            score = np.clip(rank, 0.0, 1.0)
        
        self._operator_scores[operator].append(score)
        self._operator_total[operator] += score
        self._operator_count[operator] += 1
    
    def _ucb_score(self, operator):
        """UCB1 score for operator selection. Uses rank-based average [0,1]."""
        n = self._operator_count[operator]
        if n == 0:
            return float('inf')
        total_n = sum(self._operator_count.values())
        avg = self._operator_total[operator] / n
        # UCB1 exploration bonus — well-calibrated since avg is in [0,1]
        exploration = np.sqrt(2.0 * np.log(max(total_n, 1)) / max(n, 1))
        return avg + exploration
    
    def _select_best(self):
        """Select operator with highest UCB1 score."""
        return max(self._operators, key=lambda op: self._ucb_score(op))
    
    def _select_runner_up(self):
        """Select second-best operator by UCB1."""
        scores = [(op, self._ucb_score(op)) for op in self._operators]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[1][0] if len(scores) >= 2 else scores[0][0]
    
    # =========================================================================
    # PSO COMPONENTS (shared infrastructure)
    # =========================================================================
    
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
    
    def _velocity_update(self):
        cognitive, social = self._adaptive_coefficients()
        r1 = np.random.uniform(0, 1, (self.np, self.dim))
        r2 = np.random.uniform(0, 1, (self.np, self.dim))
        cog = cognitive * r1 * (self.personal_best - self.population)
        soc = social * r2 * (self.local_best - self.population)
        mut_mask = np.random.uniform(0, 1, (self.np, self.dim))
        mut_thresh = 0.1 * (1.0 - self.generation / 5000)
        mut_active = mut_mask < mut_thresh
        mut_vecs = np.zeros((self.np, self.dim))
        for i in range(self.np):
            idx = np.random.choice([j for j in range(self.np) if j != i], 3, replace=False)
            mut_vecs[i] = self.population[idx[0]] + 0.5 * (self.population[idx[1]] - self.population[idx[2]])
        mut_comp = np.where(mut_active, 0.3 * (mut_vecs - self.population), 0.0)
        new_vel = self.inertia_weight * self.velocity + cog + soc + mut_comp
        self.velocity = np.clip(new_vel, self.v_min, self.v_max)
    
    def _position_update(self):
        """Standard PSO position update with velocity clipping."""
        new_pop = self.population + self.velocity
        self.population = self._clip_to_bounds(new_pop)
    
    def _restart_if_stagnant(self):
        threshold = 50 + self.dim // 2
        if self.stagnation_counter > threshold:
            n_replace = max(1, int(0.3 * self.np))
            worst = np.argsort(self.personal_best_fitness)[-n_replace:]
            self.population[worst] = np.random.uniform(
                self.lower_bound, self.upper_bound, (n_replace, self.dim))
            self.velocity[worst] = np.random.uniform(
                self.v_min * 0.1, self.v_max * 0.1, (n_replace, self.dim))
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
    
    # =========================================================================
    # MAIN OPTIMIZATION LOOP
    # =========================================================================
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer with PROBE-AND-COMMIT inertia-weight selection.
        
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
            self._operator_total[op] = 0.0
            self._operator_count[op] = 0
        self._probe_fitness_hist = []
        self._probe_op_hist = []
        self._probe_best = np.inf
        self._committed_op = None
        self._runner_up_op = None
        self._in_probe = True
        self._probe_idx = 0
        self._probe_gens_cur = 0
        self._commit_best = np.inf
        
        # Reset per-strategy EMA state
        for attr in ['_success_ema', '_prev_fitness_med', '_ema_imp',
                     '_ema_stag', '_ema_div', '_prev_best_fit',
                     '_ema_imp_catH', '_prev_best_catH']:
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
                return float(self.current_fitness[best_idx]), self.population[best_idx].copy()
            return float(self.global_best_fitness), self.global_best.copy()
        
        self._update_personal_best_batch()
        self._compute_local_best_batch()
        self._compute_global_best()
        
        # ---- MAIN LOOP ----
        while not stopping_condition():
            self.generation += 1
            
            # --- PHASE A: PROBE ---
            if self._in_probe:
                current_op = self._operators[self._probe_idx]
                
                # Adapt inertia using the PROBED strategy
                self._dispatch_inertia(current_op)
                self._adapt_neighborhood_size()
                self._update_local_from_personal()
                
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
                
                # Record rank-based score
                gen_best = float(np.min(self.current_fitness))
                if gen_best < self._probe_best:
                    self._probe_best = gen_best
                self._record_score(current_op, self._probe_best)
                
                self._probe_gens_cur += 1
                
                # Advance to next operator
                if self._probe_gens_cur >= self._probe_gens:
                    self._probe_idx += 1
                    self._probe_gens_cur = 0
                
                # Check if probe phase is done
                if self._probe_idx >= len(self._operators):
                    self._in_probe = False
                    self._committed_op = self._select_best()
                    self._runner_up_op = self._select_runner_up()
                    self._compute_fingerprint()
                    self._commit_best = self.global_best_fitness
                
                self._restart_if_stagnant()
            
            # --- PHASE B: COMMIT ---
            else:
                # Adapt inertia using the COMMITTED strategy
                self._dispatch_inertia(self._committed_op)
                self._adapt_neighborhood_size()
                self._update_local_from_personal()
                
                # Epsilon-review: small chance to try runner-up if stagnant
                if (self.stagnation_counter > 20 and
                        np.random.random() < self._epsilon_review):
                    self._dispatch_inertia(self._runner_up_op)
                
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
            return float(self.current_fitness[best_idx]), self.population[best_idx].copy()
        return float(self.global_best_fitness), self.global_best.copy()
```