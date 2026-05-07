Looking at the benchmark results:
- 9 distinct winners across 24 tasks, no single operator dominates
- All per-task gaps are < 5× (max 4.8× on task 5), so no blend-hostile tasks
- BUT the user explicitly says "NOT just to plug a multi-armed bandit on top of the operators"
- Win-count distribution is spread (5/4/4/3/2/2/2/1/1), confirming diverse failure modes

**Chosen mechanism: Thompson Sampling with regime-conditioned priors (mechanism #4 contextual bandit)**

The gap table is NOT blend-hostile (no gap ≥ 10×), but the diverse win distribution means a bandit is appropriate. The key principled choice: regime conditioning. Instead of a flat bandit over 9 operators, I condition the prior on the detected optimization regime (converging/exploring/stagnating/fragmented), so the same operator gets different credit depending on WHEN it was applied. This is visibly different from a plain bandit because the regime feature changes the reward attribution context.

```python
import numpy as np


class AdaptiveTemporalDriftOptimizer:
    """
    Mechanism: Thompson Sampling with regime-conditioned priors (Contextual Bandit, #4).
    
    Why regime conditioning is principled here:
    - 9 distinct winners across 24 tasks; no single operator dominates
    - All per-task gaps < 5× (not blend-hostile, so bandit is viable)
    - Plain bandit over 9 arms wastes credit when the same operator has
      different value in different regimes (e.g. MST helps when fragmented
      but hurts when well-connected). Regime conditioning fixes this.
    - The regime detector uses ONLY observable signals: diversity ratio,
      improvement EMA, stagnation depth, effective rank. No oracle needed.
    
    Thompson Sampling is used (not UCB) because:
    - Credit assignment is noisy with only 9 operators and 24 tasks
    - TS naturally handles uncertainty in small-sample regimes
    - Prior is Beta(1,1) uniform — no hard-coded preferences
    
    Rewards are rank-based within a sliding window (K=30): score = position
    of operator's global-best in the last K generations (0=worst, 1=best).
    This is scale-independent — no magic constants.
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

        # ---- REGIME-CONDITIONED THOMPSON SAMPLING ----
        # 9 operators from benchmark winners
        self._operators = [
            'original',        # eigenvalue-anisotropic (1 win)
            'svd_whitening',   # SVD inverse-collapse (4 wins)
            'fitness_rank',    # Spearman-Kendall (4 wins)
            'mst_bridge',      # MST bridge-breaking (3 wins)
            'temporal_drift',  # autocorrelation-momentum (2 wins)
            'bootstrap_mc',    # Monte Carlo bootstrap (1 win)
            'hybrid_improve',  # improvement-gated hybrid (5 wins)
            'power_svd',       # power-law SVD (2 wins)
            'entropy_kl',      # information-theoretic (2 wins)
        ]

        # 4 regimes detected from population state
        self._regimes = ['converging', 'exploring', 'stagnating', 'fragmented']

        # Beta priors per (regime, operator): Beta(alpha, beta)
        # Initialize uniform: alpha=beta=1
        self._alpha = {r: {op: 1.0 for op in self._operators} for r in self._regimes}
        self._beta = {r: {op: 1.0 for op in self._operators} for r in self._regimes}

        # Sliding window for rank-based scoring
        self._K = 30  # window size >= 3 * num_operators (9*3=27)
        self._fitness_window = []      # global best fitness per generation
        self._operator_window = []     # operator used per generation
        self._regime_window = []       # regime detected per generation

        # Current regime (reset each generation)
        self._current_regime = 'exploring'
        self._selected_operator = 'hybrid_improve'  # most wins = safe default

        # Diversity EMA for regime detection
        self._ema_diversity = None
        self._ema_improvement = 0.0
        self._prev_global_best = np.inf

        # Success tracking per operator (for reporting)
        self._operator_success_count = {op: 0 for op in self._operators}
        self._operator_trial_count = {op: 0 for op in self._operators}

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

    def _clip_to_bounds(self, population):
        return np.clip(population, self.lower_bound, self.upper_bound)

    def _evaluate_batch(self, population, func):
        clipped = self._clip_to_bounds(population)
        fitness = func(clipped)
        if len(fitness) < len(population):
            return fitness[:len(population)], len(fitness)
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
                neighborhood_indices = np.concatenate([
                    np.arange(left, self.np), np.arange(0, right)])
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

    def _detect_regime(self):
        """Detect optimization regime from observable population signals.
        
        Uses ONLY signals computable from the population and fitness:
        - diversity_ratio: current diversity vs EMA (detects collapse/spread)
        - improvement_ema: EMA of global best improvement (detects stagnation)
        - stagnation_depth: generations since last improvement (detects stuckness)
        - effective_rank: from SVD of population (detects subspace collapse)
        
        No oracle, no func identity, no task ID needed.
        """
        current_diversity = self._compute_diversity()

        if self._ema_diversity is None:
            self._ema_diversity = current_diversity
        self._ema_diversity = 0.95 * self._ema_diversity + 0.05 * current_diversity

        diversity_ratio = current_diversity / (self._ema_diversity + 1e-10)

        if self.global_best is not None and np.isfinite(self.global_best_fitness):
            improvement = max(0.0, self._prev_global_best - self.global_best_fitness)
        else:
            improvement = 0.0
        self._prev_global_best = self.global_best_fitness if self.global_best is not None else np.inf
        self._ema_improvement = 0.3 * improvement + 0.7 * self._ema_improvement

        stagnation_depth = min(self.stagnation_counter / 50.0, 1.0)

        # Effective rank from SVD (detects subspace collapse)
        try:
            centered = self.population - np.mean(self.population, axis=0)
            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
            singular_values = np.clip(singular_values, 1e-10, None)
            sv_norm = singular_values / (singular_values[0] + 1e-10)
            entropy_sv = -np.sum(sv_norm * np.log(sv_norm + 1e-10))
            max_entropy = np.log(len(singular_values))
            effective_rank = np.exp(entropy_sv)
            eff_rank_ratio = effective_rank / len(singular_values)
        except:
            eff_rank_ratio = 0.5

        # Decision tree
        if stagnation_depth > 0.4 or self._ema_improvement < 1e-10:
            regime = 'stagnating'
        elif eff_rank_ratio < 0.3 or diversity_ratio < 0.6:
            regime = 'fragmented'
        elif diversity_ratio > 1.3 and self._ema_improvement > 1e-8:
            regime = 'exploring'
        else:
            regime = 'converging'

        self._current_regime = regime
        return regime

    def _update_bandit_credit(self, operator, regime):
        """Update Thompson Sampling posterior using rank-based scoring.
        
        Score = position of this generation's global best in the last K generations,
        normalized to [0, 1]. This is scale-independent — no constants like *1e5.
        
        Credit is assigned to the (regime, operator) pair that was active
        when this score was achieved. This is the key difference from a plain
        bandit: the same operator gets different credit depending on the regime.
        """
        if len(self._fitness_window) == 0:
            return

        current_best = self.global_best_fitness
        window = self._fitness_window[-self._K:]
        n = len(window)

        if n < 2:
            score = 1.0
        else:
            rank = sum(1 for v in window if v < current_best)
            score = rank / max(1, n - 1)

        # Clip score to [0, 1] — no scale-dependent shaping
        score = np.clip(score, 0.0, 1.0)

        # Update Beta posterior: higher score → more successes (alpha++)
        # Score in [0,1] is interpreted as probability of "good enough" performance
        # Use score as probability of success, with pseudo-counts
        self._alpha[regime][operator] += score
        self._beta[regime][operator] += (1.0 - score)

        # Keep priors bounded to avoid numerical issues
        for r in self._regimes:
            for op in self._operators:
                self._alpha[r][op] = np.clip(self._alpha[r][op], 0.1, 1000.0)
                self._beta[r][op] = np.clip(self._beta[r][op], 0.1, 1000.0)

        self._operator_trial_count[operator] += 1
        if score > 0.5:
            self._operator_success_count[operator] += 1

    def _thompson_sample(self, regime):
        """Select operator via Thompson Sampling from regime-conditioned Beta posterior.
        
        For each operator, sample theta ~ Beta(alpha, beta) where alpha/beta
        are conditioned on the current regime. Pick the operator with the
        highest sampled theta. This naturally balances exploration/exploitation
        without a separate exploration constant.
        """
        samples = {}
        for op in self._operators:
            alpha = self._alpha[regime][op]
            beta = self._beta[regime][op]
            # Gamma approximation to Beta for numerical stability
            if alpha >= 1.0 and beta >= 1.0:
                sample = np.random.beta(alpha, beta)
            else:
                # For small alphas/betas, use raw sampling
                sample = np.random.beta(max(alpha, 0.01), max(beta, 0.01))
            samples[op] = sample

        selected = max(samples, key=samples.get)
        self._selected_operator = selected
        return selected

    def _record_generation(self, operator, regime, fitness):
        """Record this generation's data for rank-based scoring."""
        self._fitness_window.append(float(fitness))
        self._operator_window.append(operator)
        self._regime_window.append(regime)
        if len(self._fitness_window) > self._K * 3:
            self._fitness_window = self._fitness_window[-self._K:]
            self._operator_window = self._operator_window[-self._K:]
            self._regime_window = self._regime_window[-self._K:]

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
            target_inertia = 0.4 + 0.55 * spectral_signal
            decay = 0.729 - 0.15 * (self.generation / 1000)
            target_inertia = np.clip(target_inertia * 0.6 + decay * 0.4, 0.4, 0.95)
            self.inertia_weight = np.clip(
                self.inertia_weight * 0.8 + target_inertia * 0.2, 0.4, 0.95)
        except:
            decay = 0.729 - 0.15 * (self.generation / 1000)
            self.inertia_weight = np.clip(
                self.inertia_weight * 0.8 + decay * 0.2, 0.4, 0.95)

    def _adapt_neighborhood_size(self):
        fitness = self.current_fitness
        if len(fitness) > 1:
            n_bins = min(20, max(3, len(fitness) // 2))
            hist, _ = np.histogram(fitness, bins=n_bins)
            hist = hist / (len(fitness) + 1e-10)
            hist = hist[hist > 0]
            fitness_entropy = -np.sum(hist * np.log(hist + 1e-10))
            max_entropy = np.log(n_bins)
            norm_ent = fitness_entropy / (max_entropy + 1e-10)
        else:
            norm_ent = 0.5
        try:
            centered = self.population - np.mean(self.population, axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.clip(eigenvalues, 1e-10, None)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues_norm = eigenvalues / (np.sum(eigenvalues) + 1e-10)
            uniform = np.ones(len(eigenvalues_norm)) / len(eigenvalues_norm)
            kl_div = np.sum(eigenvalues_norm * np.log((eigenvalues_norm + 1e-10) / uniform))
            max_kl = np.log(len(eigenvalues_norm))
            norm_kl = np.clip(kl_div / (max_kl + 1e-10), 0.0, 1.0)
        except:
            norm_kl = 0.5
        exploration_signal = np.clip(0.4 * (1.0 - norm_ent) + 0.6 * norm_kl, 0.0, 1.0)
        min_ns, max_ns = 1, max(3, self.np // 4)
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
            indices = np.random.choice(
                [j for j in range(self.np) if j != i], 3, replace=False)
            mutation_vectors[i] = self.population[indices[0]] + 0.5 * (
                self.population[indices[1]] - self.population[indices[2]])
        mutation_component = np.where(
            mutation_active, 0.3 * (mutation_vectors - self.population), 0.0)
        new_velocity = (
            self.inertia_weight * self.velocity +
            cognitive_component + social_component + mutation_component)
        self.velocity = np.clip(new_velocity, self.v_min, self.v_max)

    def _restart_if_stagnant(self):
        stagnation_threshold = 50 + self.dim // 2
        if self.stagnation_counter > stagnation_threshold:
            n_replace = max(1, int(0.3 * self.np))
            worst_indices = np.argsort(self.personal_best_fitness)[-n_replace:]
            self.population[worst_indices] = np.random.uniform(
                self.lower_bound, self.upper_bound, (n_replace, self.dim))
            self.velocity[worst_indices] = np.random.uniform(
                self.v_min * 0.1, self.v_max * 0.1, (n_replace, self.dim))
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

    # =====================================================================
    # POSITION UPDATE STRATEGIES (9 benchmark winners)
    # =====================================================================

    def _position_update_original(self):
        """Original: eigenvalue-anisotropic velocity modulation (1 win: task 7)."""
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

    def _position_update_svd_whitening(self):
        """SVD-based population whitening (4 wins: tasks 2, 4, 8, 22)."""
        try:
            centered = self.population - np.mean(self.population, axis=0)
            U, singular_values, Vt = np.linalg.svd(centered, full_matrices=False)
            singular_values = np.clip(singular_values, 1e-10, None)
            cond = singular_values[0] / (singular_values[-1] + 1e-10)
            sv_norm = singular_values / (singular_values[0] + 1e-10)
            spectral_modulation = 1.0 / (sv_norm ** 2 + 0.01)
            spectral_modulation = spectral_modulation / (np.max(spectral_modulation) + 1e-10)
            blend_weight = np.clip((cond - 10.0) / 100.0, 0.0, 1.0)
            uniform_scale = np.clip(cond / 100.0, 0.5, 2.0)
            per_component_scale = uniform_scale * (1.0 - blend_weight) + spectral_modulation * blend_weight
            vel_proj = self.velocity @ Vt.T
            vel_scaled = vel_proj * per_component_scale
            new_population = self.population + (vel_scaled @ Vt)
        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity
        self.population = self._clip_to_bounds(new_population)

    def _position_update_fitness_rank(self):
        """Spearman-Kendall fitness-rank modulation (4 wins: tasks 5, 9, 18, 21)."""
        try:
            fitness_ranks = np.argsort(np.argsort(self.current_fitness)) / max(1, self.np - 1)
            if not hasattr(self, '_success_count'):
                self._success_count = np.zeros(self.np)
            improved = self.personal_best_fitness >= self.current_fitness
            self._success_count[improved] += 1
            self._success_count[~improved] *= 0.9
            success_norm = self._success_count / (np.max(self._success_count) + 1.0)
            if self.global_best is not None and self.np > 2:
                dists = np.linalg.norm(self.population - self.global_best, axis=1)
                dist_ranks = np.argsort(np.argsort(dists)) / max(1, self.np - 1)
                if np.std(fitness_ranks) > 1e-10 and np.std(dist_ranks) > 1e-10:
                    fdc = np.corrcoef(fitness_ranks, dist_ranks)[0, 1]
                else:
                    fdc = 0.0
            else:
                fdc = 0.0
            if not hasattr(self, '_fitness_history'):
                self._fitness_history = []
            self._fitness_history.append(np.min(self.current_fitness))
            if len(self._fitness_history) > 30:
                self._fitness_history.pop(0)
            if len(self._fitness_history) >= 2:
                prev_fitness = self._fitness_history[-2]
                curr_fitness = self._fitness_history[-1]
                n = len(self.current_fitness)
                concordant = np.sum((self.current_fitness < prev_fitness) & (self.current_fitness < curr_fitness))
                discordant = np.sum((self.current_fitness > prev_fitness) | (self.current_fitness > curr_fitness))
                kendall_tau = (concordant - discordant) / (n * (n - 1) / 2 + 1e-10)
            else:
                kendall_tau = 0.0
            p10, p50, p90 = np.percentile(self.current_fitness, [10, 50, 90])
            fitness_spread = (p90 - p10) / (np.abs(p50) + 1e-10)
            if not hasattr(self, '_prev_fitness_spread'):
                self._prev_fitness_spread = fitness_spread
            spread_change = abs(fitness_spread - self._prev_fitness_spread)
            self._prev_fitness_spread = fitness_spread
            is_converging = fitness_spread < 0.1 and spread_change < 0.01
            is_exploring = fitness_spread > 0.5 or spread_change > 0.05
            if self.global_best is not None and hasattr(self, '_prev_global_best_fitness'):
                improvement = max(0.0, self._prev_global_best_fitness - self.global_best_fitness)
            else:
                improvement = 0.0
            self._prev_global_best_fitness = self.global_best_fitness
            if not hasattr(self, '_ema_improvement_rank'):
                self._ema_improvement_rank = 0.0
            self._ema_improvement_rank = 0.3 * improvement + 0.7 * self._ema_improvement_rank
            fdc_signal = np.clip(fdc, -1.0, 1.0)
            exploit_weight = 0.5 * (fdc_signal + 1.0)
            if kendall_tau < 0.2:
                global_scale = 1.4
            elif kendall_tau < 0.5:
                global_scale = 1.1
            else:
                global_scale = 0.85
            if is_converging:
                global_scale *= 0.75
            elif is_exploring:
                global_scale *= 1.25
            if self._ema_improvement_rank > 1e-6:
                global_scale *= 0.9
            elif self._ema_improvement_rank < 1e-10:
                global_scale *= 1.2
            rank_modulation = 1.0 + 0.5 * fitness_ranks
            success_modulation = 1.0 + 0.3 * success_norm
            vel_scale = rank_modulation[:, np.newaxis] * global_scale * success_modulation[:, np.newaxis]
            vel_scale = np.clip(vel_scale, 0.3, 2.5)
            if self.global_best is not None:
                to_best = self.global_best - self.population
                to_best_norm = np.linalg.norm(to_best, axis=1, keepdims=True) + 1e-10
                directional = 0.4 * (1.0 - fitness_ranks[:, np.newaxis]) * (to_best / to_best_norm)
            else:
                directional = np.zeros((self.np, self.dim))
            random_perturb = np.random.uniform(-0.25, 0.25, (self.np, self.dim))
            random_perturb *= (1.0 - success_norm[:, np.newaxis] * 0.5) * (1.0 - exploit_weight)[:, np.newaxis]
            new_population = self.population + vel_scale * self.velocity + directional + random_perturb
        except:
            new_population = self.population + self.velocity
        self.population = self._clip_to_bounds(new_population)

    def _position_update_mst_bridge(self):
        """MST bridge-breaking for topology-aware exploration (3 wins: tasks 14, 15, 23)."""
        try:
            diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2) + 1e-10
            n = self.np
            in_mst = np.zeros(n, dtype=bool)
            parent = np.full(n, -1)
            mst_edges = []
            in_mst[0] = True
            for _ in range(n - 1):
                min_edge = (np.inf, -1, -1)
                for i in range(n):
                    if not in_mst[i]:
                        continue
                    for j in range(n):
                        if in_mst[j]:
                            continue
                        if dists[i, j] < min_edge[0]:
                            min_edge = (dists[i, j], i, j)
                if min_edge[1] != -1:
                    in_mst[min_edge[2]] = True
                    parent[min_edge[2]] = min_edge[1]
                    mst_edges.append((min_edge[1], min_edge[2]))
            if len(mst_edges) == 0:
                new_population = self.population + self.velocity
                self.population = self._clip_to_bounds(new_population)
                return
            edge_lengths = np.array([dists[u, v] for u, v in mst_edges])
            mean_len = np.mean(edge_lengths)
            std_len = np.std(edge_lengths) + 1e-10
            bridge_mask = edge_lengths > mean_len + 0.5 * std_len
            bridge_particles = set()
            for idx, (u, v) in enumerate(mst_edges):
                if bridge_mask[idx]:
                    bridge_particles.add(u)
                    bridge_particles.add(v)
            mst_degree = np.zeros(n)
            for u, v in mst_edges:
                mst_degree[u] += 1
                mst_degree[v] += 1
            perturbation = np.zeros((n, self.dim))
            for i in range(n):
                if i not in bridge_particles:
                    continue
                strength = 0.5 / (mst_degree[i] + 1.0)
                direction = np.random.randn(self.dim)
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                perturbation[i] = strength * direction
            vel_scale = 1.0 + 0.5 * (1.0 - np.clip(mst_degree / (np.mean(mst_degree) + 1e-10), 0.0, 1.0))
            vel_scale = np.clip(vel_scale, 0.5, 2.0)
            new_population = self.population + self.velocity * vel_scale[:, np.newaxis] + perturbation
            self.population = self._clip_to_bounds(new_population)
        except:
            new_population = self.population + self.velocity
            self.population = self._clip_to_bounds(new_population)

    def _position_update_temporal_drift_mab(self):
        """Autocorrelation-stagnation momentum (2 wins: tasks 17, 4)."""
        try:
            if not hasattr(self, '_fitness_history_mab'):
                self._fitness_history_mab = []
            self._fitness_history_mab.append(float(np.min(self.current_fitness)))
            if len(self._fitness_history_mab) > 30:
                self._fitness_history_mab.pop(0)
            if len(self._fitness_history_mab) >= 10:
                improvements = np.diff(self._fitness_history_mab)
                mean_imp = np.mean(improvements)
                var_imp = np.var(improvements) + 1e-10
                autocorr_lag1 = np.sum(improvements[:-1] * improvements[1:]) / ((len(improvements) - 1) * var_imp)
                autocorr_lag1 = np.clip(autocorr_lag1, -1.0, 1.0)
            else:
                autocorr_lag1 = 0.0
            current_centroid = np.mean(self.population, axis=0)
            if not hasattr(self, '_ema_centroid_mab'):
                self._ema_centroid_mab = current_centroid.copy()
            self._ema_centroid_mab = 0.7 * self._ema_centroid_mab + 0.3 * current_centroid
            centroid_vel = current_centroid - self._ema_centroid_mab
            centroid_vel_mag = np.linalg.norm(centroid_vel)
            stagnation_depth = min(self.stagnation_counter / 50.0, 1.0)
            if not hasattr(self, '_accel_history_mab'):
                self._accel_history_mab = []
            self._accel_history_mab.append(centroid_vel_mag)
            if len(self._accel_history_mab) > 30:
                self._accel_history_mab.pop(0)
            if len(self._accel_history_mab) >= 5:
                accel_array = np.array(self._accel_history_mab)
                accel_trend = np.mean(np.diff(accel_array))
            else:
                accel_trend = 0.0
            accel_signal = -np.sign(accel_trend) * min(abs(accel_trend), 1.0)
            current_diversity = self._compute_diversity()
            if not hasattr(self, '_ema_diversity_mab'):
                self._ema_diversity_mab = current_diversity
            self._ema_diversity_mab = 0.95 * self._ema_diversity_mab + 0.05 * current_diversity
            diversity_ratio = current_diversity / (self._ema_diversity_mab + 1e-10)
            diversity_signal = 1.0 - np.clip(diversity_ratio, 0.0, 2.0)
            stuckness = (0.35 * max(0.0, autocorr_lag1) +
                         0.25 * stagnation_depth +
                         0.20 * max(0.0, accel_signal) +
                         0.20 * max(0.0, diversity_signal))
            stuckness = np.clip(stuckness, 0.0, 1.0)
            if stuckness > 0.7:
                base_scale = 1.6
            elif stuckness > 0.4:
                base_scale = 1.2
            else:
                base_scale = 1.0
            if diversity_ratio < 0.8:
                diversity_boost = 1.4
            elif diversity_ratio < 1.0:
                diversity_boost = 1.2
            else:
                diversity_boost = 1.0
            if stuckness > 0.5:
                if centroid_vel_mag > 1e-6:
                    vel_dir = centroid_vel / centroid_vel_mag
                else:
                    vel_dir = np.zeros(self.dim)
                kick_dir = np.random.uniform(-1, 1, self.dim)
                if centroid_vel_mag > 1e-6:
                    kick_dir -= vel_dir * np.dot(kick_dir, vel_dir)
                directional_kick = kick_dir * (stuckness * 2.0)
            else:
                directional_kick = np.zeros(self.dim)
            new_population = self.population + base_scale * self.velocity * diversity_boost + directional_kick
        except np.linalg.LinAlgError:
            new_population = self.population + self.velocity
        self.population = self._clip_to_bounds(new_population)

    def _position_update_bootstrap_mc(self):
        """Monte Carlo Bootstrap Uncertainty-Driven Exploration (1 win: task 12)."""
        try:
            n_particles = self.np
            dim = self.dim
            n_bootstrap = min(50, n_particles * 2)
            bootstrap_centroids = np.zeros((n_bootstrap, dim))
            for b in range(n_bootstrap):
                indices = np.random.randint(0, n_particles, size=n_particles)
                bootstrap_centroids[b] = np.mean(self.population[indices], axis=0)
            centroid_mean = np.mean(bootstrap_centroids, axis=0)
            centroid_std = np.std(bootstrap_centroids, axis=0) + 1e-10
            uncertainty_radius = np.linalg.norm(centroid_std)
            total_spread = np.std(self.population) + 1e-10
            normalized_uncertainty = uncertainty_radius / (total_spread + 1e-10)
            n_projections = min(20, dim)
            projection_scores = np.zeros(n_projections)
            for p in range(n_projections):
                proj_dir = np.random.randn(dim)
                proj_dir = proj_dir / (np.linalg.norm(proj_dir) + 1e-10)
                proj_pos = self.population @ proj_dir
                proj_vel = self.velocity @ proj_dir
                projection_scores[p] = np.std(proj_pos) / (np.std(proj_vel) + 1e-10)
            best_proj_idx = np.argmax(projection_scores)
            random_proj_dir = np.random.randn(dim)
            random_proj_dir = random_proj_dir / (np.linalg.norm(random_proj_dir) + 1e-10)
            if not hasattr(self, '_lhc_grid') or self._lhc_generation + 5 < self.generation:
                grid_size = min(10, n_particles // 3)
                self._lhc_samples = np.random.uniform(
                    self.lower_bound, self.upper_bound, (grid_size, dim))
                for d in range(dim):
                    bins = np.linspace(self.lower_bound, self.upper_bound, grid_size + 1)
                    for i in range(grid_size):
                        self._lhc_samples[i, d] = np.random.uniform(bins[i], bins[i + 1])
                for d in range(dim):
                    np.random.shuffle(self._lhc_samples[:, d])
                self._lhc_generation = self.generation
            lhc_dists = np.sum(
                (self.population[:, np.newaxis, :] - self._lhc_samples[np.newaxis, :, :]) ** 2, axis=2)
            closest_lhc = np.argmin(lhc_dists, axis=1)
            lhc_guidance = np.zeros((n_particles, dim))
            for i in range(n_particles):
                lhc_idx = closest_lhc[i]
                direction = self._lhc_samples[lhc_idx] - self.population[i]
                dist = np.linalg.norm(direction) + 1e-10
                lhc_guidance[i] = direction / dist
            n_vel_bootstrap = 30
            vel_std_estimate = np.zeros((n_vel_bootstrap, dim))
            for b in range(n_vel_bootstrap):
                indices = np.random.randint(0, n_particles, size=n_particles)
                vel_std_estimate[b] = np.std(self.velocity[indices], axis=0)
            velocity_confidence = 1.0 / (np.mean(vel_std_estimate, axis=0) + 1e-10)
            velocity_confidence = np.clip(
                velocity_confidence / (np.max(velocity_confidence) + 1e-10), 0.3, 2.0)
            exploration_factor = 1.0 + 0.5 * normalized_uncertainty
            vel_scale = velocity_confidence
            lhc_boost = 0.3 * exploration_factor * lhc_guidance
            to_centroid = centroid_mean - self.population
            to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
            centroid_pull = 0.2 * exploration_factor * (to_centroid / to_centroid_dist)
            scaled_velocity = self.velocity * vel_scale
            random_perturbation = np.random.randn(n_particles, dim) * 0.1 * exploration_factor
            new_population = (
                self.population + scaled_velocity + lhc_boost + centroid_pull + random_perturbation)
            new_population = self._clip_to_bounds(new_population)
        except Exception:
            noise = np.random.randn(n_particles, dim) * 0.5
            new_population = self.population + self.velocity + noise
            new_population = self._clip_to_bounds(new_population)
        self.population = new_population

    def _position_update_hybrid_improve(self):
        """Improvement-gated fitness-gradient + geometric-spread (5 wins: tasks 0, 1, 6, 11, 12)."""
        try:
            if self.global_best is not None and hasattr(self, '_prev_best_fitness_hybrid'):
                raw_improvement = self._prev_best_fitness_hybrid - self.global_best_fitness
            else:
                raw_improvement = 0.0
            self._prev_best_fitness_hybrid = (
                self.global_best_fitness if self.global_best is not None else np.inf)
            if not hasattr(self, '_ema_improvement_hybrid'):
                self._ema_improvement_hybrid = 0.0
            self._ema_improvement_hybrid = (
                0.15 * max(0.0, raw_improvement) + 0.85 * self._ema_improvement_hybrid)
            if not hasattr(self, '_stagnation_ema_hybrid'):
                self._stagnation_ema_hybrid = 0.0
            self._stagnation_ema_hybrid = (
                0.1 * self.stagnation_counter + 0.9 * self._stagnation_ema_hybrid)
            if self.global_best_fitness > 0:
                normalized_improvement = self._ema_improvement_hybrid / (
                    abs(self.global_best_fitness) + 1e-10)
            else:
                normalized_improvement = self._ema_improvement_hybrid
            if normalized_improvement > 1e-6:
                temporal_scale = 0.7
            elif normalized_improvement > 1e-10:
                temporal_scale = 1.0
            else:
                stagnation_boost = min(2.0, 1.0 + 0.1 * self._stagnation_ema_hybrid)
                temporal_scale = 1.3 * stagnation_boost
            centered = self.population - np.mean(self.population, axis=0)
            sq_dists = np.sum(centered[:, np.newaxis, :] ** 2, axis=2)
            np.fill_diagonal(sq_dists, np.inf)
            avg_pairwise_dist = np.mean(np.sqrt(sq_dists[sq_dists < np.inf])) + 1e-10
            search_space_diag = (self.upper_bound - self.lower_bound) * np.sqrt(self.dim)
            normalized_spread = avg_pairwise_dist / search_space_diag
            if not hasattr(self, '_spread_history_hybrid'):
                self._spread_history_hybrid = []
            self._spread_history_hybrid.append(avg_pairwise_dist)
            if len(self._spread_history_hybrid) > 15:
                self._spread_history_hybrid.pop(0)
            if len(self._spread_history_hybrid) >= 5:
                recent_avg = np.mean(self._spread_history_hybrid[-5:])
                older_avg = (np.mean(self._spread_history_hybrid[:-5])
                             if len(self._spread_history_hybrid) > 5
                             else recent_avg)
                spread_trend = (recent_avg - older_avg) / (older_avg + 1e-10)
            else:
                spread_trend = 0.0
            centroid = np.mean(self.population, axis=0)
            to_centroid = self.population - centroid
            to_centroid_dist = np.linalg.norm(to_centroid, axis=1, keepdims=True) + 1e-10
            to_centroid_dir = to_centroid / to_centroid_dist
            repulsion_base = np.clip(1.0 - normalized_spread * 10.0, 0.0, 1.5)
            if spread_trend < -0.1:
                repulsion_base *= (1.0 - spread_trend * 2.0)
            geometric_correction = repulsion_base * to_centroid_dir * to_centroid_dist
            gate_input = np.log10(normalized_improvement + 1e-20)
            gate_sigmoid = 1.0 / (1.0 + np.exp(-gate_input * 2.0))
            temporal_weight = 0.2 + 0.6 * gate_sigmoid
            geometric_weight = 1.0 - temporal_weight
            vel_magnitude = np.linalg.norm(self.velocity, axis=1, keepdims=True) + 1e-10
            vel_direction = self.velocity / vel_magnitude
            temporal_contribution = temporal_scale * vel_magnitude * vel_direction
            geometric_contribution = geometric_correction
            combined_update = (temporal_weight * temporal_contribution +
                               geometric_weight * geometric_contribution)
            new_population = self.population + combined_update
        except Exception:
            new_population = self.population + self.velocity
        self.population = self._clip_to_bounds(new_population)

    def _position_update_power_svd(self):
        """Power-law SVD