import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTATION MECHANISM: #4 — PROBE-AND-COMMIT (Contextual Bandit specialization).
    
    The per-task gap table is BLEND-HOSTILE: 7 tasks show winner-to-runner-up gaps
    >=100× (up to 5.5M× on Task 5). Linear blending mathematically cannot preserve
    these per-task advantages — a 50/50 blend of 1 and 100 is 50.5, not 1. The
    dispatcher preserves the winner's advantage by committing exclusively to ONE
    operator (weight=1.0) after a short probe phase, with no blending post-commit.
    
    Three candidate operators are probed:
      A: original.py compass mutation (wins 21/24 tasks)
      B: variant_06 temporal-momentum compass (wins 2/24 tasks)  
      C: variant_10 spectral compass (wins 1/24 tasks)
    
    Phase A (PROBE, ~5% of budget): Round-robin across operators on a sub-population.
    Phase B (COMMIT): Run only the empirically-best operator for remaining budget.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(max(kwargs.get('NP', 6 * dim), 4 * dim), min(10 * dim, 500))
        self.F = 0.6
        self.CR = 0.85
        self.max_iter = kwargs.get('max_iter', 20000)
        self.penalty_factor = kwargs.get('penalty', 1e9)
        
        # --- Probe-and-commit state ---
        self._committed_operator = None  # 0=original, 1=temporal, 2=spectral
        self._probe_completed = False
        self._probe_gen = 0
        
        # Operator names for bookkeeping
        self._operator_names = ['original', 'temporal_momentum', 'spectral']
        
        # Probe tracking: per-operator rank-based improvement scores
        self._probe_scores = np.zeros(3)
        self._probe_counts = np.zeros(3)
        self._probe_history = []  # List of dicts per probe gen
        
        # --- Legacy attributes (used by operators and adaptation) ---
        self.archive = []
        self.archive_max = self.NP * 3
        self.stagnation_counter = 0
        self.prev_best_fitness = np.inf
        self.diversity_threshold = 1e-6
        
        self.F_history = [self.F]
        self.CR_history = [self.CR]
        
        # Strategy pool (kept for compatibility)
        self.strategy_names = ['rand', 'best', 'current_to_rand_best', 'local_ring', 'two_rail']
        self.strategy_scores = np.ones(len(self.strategy_names))
        self._strategy_funcs = [
            self._mutate_rand,
            self._mutate_best,
            self._mutate_current_to_rand_best,
            self._mutate_local_ring,
            self._mutate_two_rail,
        ]
        
        # EMA state for temporal momentum operator
        self._dir_ema = None
        self._ema_alpha = 0.3
        self._prev_population = None
        self._prev_fitness = None
        
        # Post-commit fallback state
        self._fallback_tried = False
        self._fallback_operator = None
        self._stagnation_since_commit = 0
        
        # --- Probe budget calibration ---
        # Use 5% of max_iter, minimum 5 generations per operator
        self._probe_n_gen = max(5, int(0.05 * self.max_iter))
        self._probe_n_gen = min(self._probe_n_gen, 15)  # Cap at 15
        
        # Sub-population size for probe (at least 20, at most NP//2)
        self._probe_NP = min(max(20, self.NP // 2), self.NP)
        
        # Per-operator probe budget
        self._probe_ops = [0, 1, 2]  # Round-robin order
        
    def _initialize_population(self, NP=None):
        """Initialize population with optional custom size for probe."""
        n = NP if NP is not None else self.NP
        low, high = -100.0, 100.0
        pop = np.random.uniform(low, high, (n, self.dim))
        pop += np.random.randn(n, self.dim) * 5.0
        return np.clip(pop, low, high)
    
    def _compute_fitness_ranking(self, fitness):
        """Rank fitness values in [0, 1]. Handles NaN."""
        valid = ~np.isnan(fitness)
        if valid.sum() == 0:
            return np.zeros_like(fitness)
        order = np.argsort(fitness[valid])
        ranks = np.empty(valid.sum())
        ranks[order] = np.arange(valid.sum())
        full_ranks = np.full_like(fitness, 0.5)
        full_ranks[valid] = ranks / max(valid.sum() - 1, 1)
        return full_ranks
    
    # -------------------------------------------------------------------------
    # OPERATOR A: Original compass mutation (baseline, wins 21/24 tasks)
    # -------------------------------------------------------------------------
    def _mutate_operator_A(self, population, fitness):
        """Original compass mutation: score 3 directions per individual, pick best."""
        NP = population.shape[0]
        fitness_ranks = self._compute_fitness_ranking(fitness)
        mutants = np.empty_like(population)
        
        for i in range(NP):
            others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
            np.random.shuffle(others)
            r1, r2, r3, r4 = others[:4]
            
            i_rank = fitness_ranks[i]
            
            # Direction A: DE/rand/1
            dA = population[r1] + self.F * (population[r2] - population[r3])
            score_A = i_rank - min(fitness_ranks[[r1, r2, r3]].min(), i_rank + 0.1)
            
            # Direction B: DE/best/1
            best_idx = np.argmin(fitness_ranks)
            dB = population[best_idx] + self.F * (population[r1] - population[r2])
            score_B = i_rank - fitness_ranks[best_idx]
            
            # Direction C: current-to-pbest with archive
            p = 0.1
            top_p = int(np.ceil(p * NP))
            pbest_candidates = np.argsort(fitness_ranks)[:top_p]
            pbest = population[np.random.choice(pbest_candidates)]
            dC = population[i] + self.F * (pbest - population[r1]) + self.F * (population[r3] - population[r4])
            score_C = i_rank - fitness_ranks[pbest_candidates].min()
            
            candidates = [(dA, score_A), (dB, score_B), (dC, score_C)]
            best_dir, _ = max(candidates, key=lambda x: x[1])
            mutants[i] = best_dir
        
        return mutants
    
    # -------------------------------------------------------------------------
    # OPERATOR B: Temporal momentum compass (variant_06, wins 2/24 tasks)
    # -------------------------------------------------------------------------
    def _mutate_operator_B(self, population, fitness):
        """Temporal momentum compass: blend current direction with EMA of past successes."""
        NP = population.shape[0]
        
        # Initialize temporal momentum tracking
        if self._dir_ema is None or self._dir_ema.shape != (NP, self.dim):
            self._dir_ema = np.zeros((NP, self.dim))
        
        fitness_ranks = self._compute_fitness_ranking(fitness)
        mutants = np.empty_like(population)
        
        for i in range(NP):
            others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
            np.random.shuffle(others)
            r1, r2, r3, r4 = others[:4]
            
            i_rank = fitness_ranks[i]
            
            # Direction A: DE/rand/1 style
            dA = population[r1] + self.F * (population[r2] - population[r3])
            score_A = i_rank - min(fitness_ranks[[r1, r2, r3]].min(), i_rank + 0.1)
            
            # Direction B: DE/best/1 style
            best_idx = np.argmin(fitness_ranks)
            dB = population[best_idx] + self.F * (population[r1] - population[r2])
            score_B = i_rank - fitness_ranks[best_idx]
            
            # Direction C: pbest with archive
            p = 0.1
            top_p = int(np.ceil(p * NP))
            pbest_candidates = np.argsort(fitness_ranks)[:top_p]
            pbest = population[np.random.choice(pbest_candidates)]
            dC = population[i] + self.F * (pbest - population[r1]) + self.F * (population[r3] - population[r4])
            score_C = i_rank - fitness_ranks[pbest_candidates].min()
            
            # Select best direction for this individual
            candidates = [(dA, score_A), (dB, score_B), (dC, score_C)]
            best_dir, _ = max(candidates, key=lambda x: x[1])
            
            # Blend with temporal momentum (EMA of past directions)
            ema_strength = 1.0 / (1.0 + np.linalg.norm(self._dir_ema[i]) + 1e-10)
            momentum_weight = 0.3 * ema_strength
            mutants[i] = (1 - momentum_weight) * best_dir + momentum_weight * self._dir_ema[i]
            
            # Update EMA: track direction toward better fitness
            if self._prev_population is not None and self._prev_fitness is not None:
                if fitness[i] < self._prev_fitness[i]:
                    delta_dir = best_dir - self._dir_ema[i]
                    self._dir_ema[i] = (1 - self._ema_alpha) * self._dir_ema[i] + self._ema_alpha * delta_dir
        
        # Store for next generation comparison
        self._prev_population = population.copy()
        self._prev_fitness = fitness.copy()
        
        return mutants
    
    # -------------------------------------------------------------------------
    # OPERATOR C: Spectral compass (variant_10, wins 1/24 tasks)
    # -------------------------------------------------------------------------
    def _mutate_operator_C(self, population, fitness):
        """Spectral compass: eigendecomposition of covariance, scale mutation by inverse eigenvalue."""
        NP, dim = population.shape
        
        ranks = self._compute_fitness_ranking(fitness)
        
        # Compute population covariance matrix
        centroid = population.mean(axis=0)
        centered = population - centroid
        cov = (centered.T @ centered) / max(NP - 1, 1)
        
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            std_pop = np.std(population, axis=0) + 1e-10
            return population + self.F * np.random.randn(NP, dim) * std_pop
        
        # Sort by descending eigenvalue
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        
        total_var = np.sum(eigenvalues)
        if total_var > 0:
            explained_ratio = eigenvalues / total_var
        else:
            explained_ratio = np.ones(dim) / dim
        
        # Effective dimensionality: number of PCs for 95% variance
        cumsum = np.cumsum(explained_ratio)
        n_effective = int(np.searchsorted(cumsum, 0.95)) + 1
        n_effective = min(n_effective, dim)
        
        # Mutation scaling inversely to eigenvalue (low variance = high exploration)
        base_scale = 1.0 / np.sqrt(eigenvalues + 1e-10)
        axis_scaling = base_scale / (np.max(base_scale) + 1e-10)
        
        mutants = np.empty_like(population)
        
        for i in range(NP):
            # Direction toward p-best (top 10%)
            p = 0.1
            top_k = max(1, int(np.ceil(p * NP)))
            pbest_indices = np.argsort(ranks)[:top_k]
            pbest = population[np.random.choice(pbest_indices)]
            
            direction = pbest - population[i]
            
            # Transform to eigenspace and scale
            z = eigenvectors.T @ direction
            z_scaled = z * axis_scaling
            direction_scaled = eigenvectors @ z_scaled
            
            # Rank-based exploration factor
            rank_factor = 0.5 + 0.5 * (1.0 - ranks[i])
            
            # Random perturbation along principal axes
            random_perturb = np.zeros(dim)
            for j in range(n_effective):
                random_perturb += eigenvectors[:, j] * np.random.randn() * axis_scaling[j]
            
            mutants[i] = population[i] + self.F * rank_factor * direction_scaled + \
                         0.3 * self.F * random_perturb
        
        return np.clip(mutants, -100.0, 100.0)
    
    # -------------------------------------------------------------------------
    # DISPATCH: Select operator based on committed choice
    # -------------------------------------------------------------------------
    def _dispatch_operator(self, population, fitness):
        """Route to the committed operator (or probe operator)."""
        op = self._committed_operator
        if op == 0:
            return self._mutate_operator_A(population, fitness)
        elif op == 1:
            return self._mutate_operator_B(population, fitness)
        elif op == 2:
            return self._mutate_operator_C(population, fitness)
        else:
            return self._mutate_operator_A(population, fitness)  # Safe fallback
    
    # -------------------------------------------------------------------------
    # PROBE PHASE: Extract landscape fingerprint and rank operators
    # -------------------------------------------------------------------------
    def _extract_fingerprint(self, pop_history, fit_history):
        """
        Extract cheap landscape fingerprint from probe history.
        Returns dict with observable signals that don't require oracle knowledge.
        """
        gens = len(pop_history)
        if gens < 2:
            return {}
        
        # Signal 1: Convergence rate (log-scale improvement per generation)
        best_fits = [np.nanmin(f) for f in fit_history]
        valid = [(i, f) for i, f in enumerate(best_fits) if not np.isnan(f) and f > 0]
        if len(valid) >= 3:
            log_best = [np.log(max(f, 1e-15)) for _, f in valid]
            # Slope of log(best) vs generation (more negative = faster convergence)
            xs = np.array([x for x, _ in valid], dtype=float)
            ys = np.array(log_best)
            # Simple linear regression slope
            n = len(xs)
            if n > 1:
                mean_x, mean_y = xs.mean(), ys.mean()
                slope = np.sum((xs - mean_x) * (ys - mean_y)) / (np.sum((xs - mean_x)**2) + 1e-10)
            else:
                slope = 0.0
        else:
            slope = 0.0
        
        # Signal 2: Effective dimensionality from early covariance
        if len(pop_history) >= 2:
            early_pop = pop_history[min(1, len(pop_history)-1)]
            centroid = early_pop.mean(axis=0)
            centered = early_pop - centroid
            cov = (centered.T @ centered) / max(len(early_pop) - 1, 1)
            try:
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = np.sort(eigvals)[::-1]
                total_var = np.sum(eigvals)
                if total_var > 0:
                    explained = eigvals / total_var
                    cumsum = np.cumsum(explained)
                    eff_dim = int(np.searchsorted(cumsum, 0.95)) + 1
                else:
                    eff_dim = self.dim
            except:
                eff_dim = self.dim
        else:
            eff_dim = self.dim
        
        # Signal 3: Fitness variance across population (proxy for landscape ruggedness)
        fit_variances = []
        for f in fit_history:
            valid_f = f[~np.isnan(f)]
            if len(valid_f) > 1:
                fit_variances.append(np.var(valid_f))
        fit_var_trend = 0.0
        if len(fit_variances) >= 3:
            fit_var_trend = fit_variances[-1] - fit_variances[0]  # Increasing = more exploration needed
        
        # Signal 4: Stagnation index (fraction of gens with no improvement)
        improvements = []
        for i in range(1, len(best_fits)):
            if not np.isnan(best_fits[i]) and not np.isnan(best_fits[i-1]):
                imp = best_fits[i-1] - best_fits[i]
                improvements.append(1.0 if imp > 1e-10 else 0.0)
        stagnation_idx = 0.0
        if len(improvements) > 0:
            stagnation_idx = 1.0 - np.mean(improvements)
        
        # Signal 5: Rank stability (correlation of fitness ranks across gens)
        rank_stability = 1.0
        if len(fit_history) >= 2:
            valid = ~(np.isnan(fit_history[-1]) | np.isnan(fit_history[-2]))
            if valid.sum() > 3:
                r_prev = self._compute_fitness_ranking(fit_history[-2])
                r_curr = self._compute_fitness_ranking(fit_history[-1])
                # Spearman-like: correlation of ranks
                valid_mask = valid
                if valid_mask.sum() > 2:
                    rank_stability = np.corrcoef(r_prev[valid_mask], r_curr[valid_mask])[0, 1]
                    rank_stability = 0.5 + 0.5 * rank_stability  # Map to [0,1]
        
        return {
            'convergence_slope': slope,
            'effective_dim': eff_dim / max(self.dim, 1),
            'fit_var_trend': fit_var_trend,
            'stagnation_idx': stagnation_idx,
            'rank_stability': rank_stability,
        }
    
    def _score_probe_results(self):
        """
        Score each operator based on rank-based improvement during probe.
        Uses rank-normalized scores within the sliding window — no scale-dependent constants.
        """
        if len(self._probe_history) < 2:
            return
        
        # Per-operator: list of (best_rank_improvement, success_rate) per generation
        op_improvements = {0: [], 1: [], 2: []}
        op_successes = {0: [], 1: [], 2: []}
        
        for rec in self._probe_history:
            op = rec['operator']
            best_rank = rec['best_rank']  # Lower is better
            improvement = rec.get('rank_improvement', 0.0)
            success = rec.get('improved', False)
            
            op_improvements[op].append(improvement)
            op_successes[op].append(1.0 if success else 0.0)
        
        # Compute per-operator score: rank-normalized improvement (z-score within window)
        window_size = len(self._probe_history)
        
        for op in [0, 1, 2]:
            if len(op_improvements[op]) == 0:
                continue
            
            improvements = np.array(op_improvements[op])
            successes = np.array(op_successes[op])
            
            # Z-score normalization within the operator's own history window
            if len(improvements) >= 2:
                mean_imp = np.mean(improvements)
                std_imp = np.std(improvements) + 1e-10
                z_imp = (improvements[-1] - mean_imp) / std_imp
            else:
                z_imp = improvements[-1] if len(improvements) > 0 else 0.0
            
            # Success rate (last 3 measurements)
            recent_success = np.mean(successes[-3:]) if len(successes) > 0 else 0.0
            
            # Combined score: z-score of improvement + success rate
            # Both on comparable scales (z-score and [0,1])
            score = 0.6 * np.clip(z_imp, -3, 3) + 0.4 * recent_success
            
            self._probe_scores[op] = score
            self._probe_counts[op] = len(op_improvements[op])
    
    def _commit_to_operator(self):
        """
        Select the best operator based on probe scores.
        Tie-breaking prefers operator 0 (original) as it has the strongest prior (21 wins).
        """
        self._score_probe_results()
        
        # Check if any operator was sampled enough
        min_samples = 2
        valid_ops = [op for op in [0, 1, 2] if self._probe_counts[op] >= min_samples]
        
        if len(valid_ops) == 0:
            # Not enough samples — default to original (strongest prior)
            self._committed_operator = 0
        else:
            # Pick operator with highest score
            valid_scores = {op: self._probe_scores[op] for op in valid_ops}
            best_op = max(valid_scores, key=valid_scores.get)
            
            # If best score is not clearly better than original (within 0.5 z-units), prefer original
            orig_score = self._probe_scores.get(0, -999)
            best_score = valid_scores[best_op]
            
            if best_op != 0 and (best_score - orig_score) < 0.5:
                self._committed_operator = 0  # Prefer original on tie
            else:
                self._committed_operator = best_op
        
        self._probe_completed = True
        
        # Reset EMA for committed operator if needed
        if self._committed_operator == 1:
            self._dir_ema = None
            self._prev_population = None
            self._prev_fitness = None
    
    # -------------------------------------------------------------------------
    # Post-commit fallback mechanism
    # -------------------------------------------------------------------------
    def _should_try_fallback(self, f_opt):
        """Check if we should try a runner-up operator due to stagnation."""
        if self._fallback_tried:
            return False
        
        improved = self.prev_best_fitness - f_opt > 1e-8
        if improved:
            self._stagnation_since_commit = 0
        else:
            self._stagnation_since_commit += 1
        
        # After 30 generations of stagnation post-commit, try fallback
        if self._stagnation_since_commit > 30:
            # Find runner-up from probe scores
            scores = list(self._probe_scores)
            if len(scores) >= 2:
                sorted_ops = np.argsort(scores)[::-1]
                runner_up = sorted_ops[0]  # Best non-committed
                if runner_up != self._committed_operator:
                    self._fallback_operator = runner_up
                    return True
        return False
    
    # -------------------------------------------------------------------------
    # Core DE operations (unchanged from original)
    # -------------------------------------------------------------------------
    def _mutate_rand(self, population, idx, F):
        others = np.concatenate([np.arange(idx), np.arange(idx + 1, self.NP)])
        r = others[np.random.choice(len(others), 3, replace=False)]
        return population[r[0]] + F * (population[r[1]] - population[r[2]])
    
    def _mutate_best(self, population, idx, F):
        best_idx = np.argmin(population.sum(axis=1))
        others = np.concatenate([np.arange(idx), np.arange(idx + 1, self.NP)])
        r = others[np.random.choice(len(others), 2, replace=False)]
        return population[best_idx] + F * (population[r[0]] - population[r[1]])
    
    def _mutate_current_to_rand_best(self, population, idx, F):
        i = idx
        others = np.concatenate([np.arange(i), np.arange(i + 1, self.NP)])
        r = others[np.random.choice(len(others), 4, replace=False)]
        return population[i] + F * (population[r[0]] - population[i]) + F * (population[r[1]] - population[r[2]])
    
    def _mutate_local_ring(self, population, idx, F):
        left = (idx - 2) % self.NP
        right = (idx + 2) % self.NP
        others = [left, (idx - 1) % self.NP, (idx + 1) % self.NP, right]
        r = np.array(others)[np.random.choice(4, 3, replace=False)]
        return population[r[0]] + F * (population[r[1]] - population[r[2]])
    
    def _mutate_two_rail(self, population, idx, F):
        others = np.concatenate([np.arange(idx), np.arange(idx + 1, self.NP)])
        r = others[np.random.choice(len(others), 4, replace=False)]
        return population[r[0]] + population[r[1]] + F * (population[r[2]] - population[r[3]]) - population[idx]
    
    def _crossover_batch(self, population, mutants):
        """Binomial crossover with dimension-wise CR."""
        NP = population.shape[0]
        CR = np.clip(self.CR + np.random.randn(NP) * 0.1, 0.5, 0.98)
        mask = np.random.rand(NP, self.dim) < CR[:, np.newaxis]
        mask[np.arange(NP), np.random.randint(0, self.dim, NP)] = True
        trials = np.where(mask, mutants, population)
        return np.clip(trials, -100.0, 100.0)
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """One-to-one elitist selection."""
        better = trial_fitness < fitness
        new_pop = np.where(better[:, np.newaxis], trials, population)
        new_fit = np.where(better, trial_fitness, fitness)
        return new_pop, new_fit, better
    
    def _compute_diversity(self, population):
        """Population spread: average pairwise Euclidean distance."""
        centroid = population.mean(axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return distances.mean()
    
    def _adapt_parameters(self, improvement_rate):
        """Adapt F and CR based on k-NN graph topology."""
        if not hasattr(self, '_current_population') or self._current_population is None:
            return
        
        pop = self._current_population
        if pop is None or len(pop) < 5:
            return
        
        NP, dim = pop.shape
        k = max(2, min(5, NP // 10))
        
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
        nearest_sq_dists = sq_dists[np.arange(NP)[:, np.newaxis], nearest_indices]
        
        avg_edge_length = np.mean(np.sqrt(nearest_sq_dists))
        
        clustering_coeffs = np.zeros(NP)
        for i in range(NP):
            neighbors = set(nearest_indices[i])
            if len(neighbors) < 2:
                clustering_coeffs[i] = 0.0
                continue
            edges_among_neighbors = 0
            possible_edges = 0
            for ni in neighbors:
                for nj in neighbors:
                    if ni < nj:
                        possible_edges += 1
                        if nj in set(nearest_indices[ni]):
                            edges_among_neighbors += 1
            clustering_coeffs[i] = edges_among_neighbors / max(possible_edges, 1)
        
        avg_clustering = np.mean(clustering_coeffs)
        
        clustering_factor = 1.0 + 0.15 * (avg_clustering - 0.3)
        self.F = np.clip(self.F * clustering_factor, 0.3, 2.0)
        
        median_edge = np.median(np.sqrt(sq_dists[sq_dists != np.inf]))
        if avg_edge_length > median_edge * 0.5:
            self.CR = np.clip(self.CR * 1.08, 0.1, 0.95)
        else:
            self.CR = np.clip(self.CR * 0.93, 0.1, 0.95)
        
        if improvement_rate > 0.15:
            self.F = np.clip(self.F * 1.05, 0.3, 2.0)
        elif improvement_rate < 0.03:
            self.F = np.clip(self.F * 1.12, 0.3, 2.0)
        
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
    
    def _check_stagnation(self, best_fitness):
        improved = self.prev_best_fitness - best_fitness > 1e-8
        if improved:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        self.prev_best_fitness = best_fitness
        return self.stagnation_counter > 50
    
    # -------------------------------------------------------------------------
    # MAIN OPTIMIZATION LOOP
    # -------------------------------------------------------------------------
    def __call__(self, func, stopping_condition):
        """
        PROBE-AND-COMMIT optimization loop.
        
        Phase A (PROBE): Round-robin across operators on a sub-population.
        Phase B (COMMIT): Run only the empirically-best operator on full population.
        """
        # --- Initialize sub-population for probe ---
        probe_pop = self._initialize_population(NP=self._probe_NP)
        probe_fitness = func(probe_pop)
        
        if len(probe_fitness) < self._probe_NP:
            probe_fitness = np.full(self._probe_NP, np.nan)
        
        probe_best_idx = np.nanargmin(probe_fitness)
        probe_f_opt = probe_fitness[probe_best_idx]
        probe_prev_best = probe_f_opt
        
        # Tracking for fingerprint extraction
        probe_pop_history = [probe_pop.copy()]
        probe_fit_history = [probe_fitness.copy()]
        prev_probe_best = probe_f_opt
        
        # --- PROBE PHASE: Round-robin across operators ---
        probe_op_idx = 0
        probe_gens = 0
        
        while probe_gens < self._probe_n_gen and not stopping_condition():
            # Select operator for this probe generation (round-robin)
            op = self._probe_ops[probe_op_idx % len(self._probe_ops)]
            
            # Generate mutants using this operator
            if op == 0:
                probe_mutants = self._mutate_operator_A(probe_pop, probe_fitness)
            elif op == 1:
                probe_mutants = self._mutate_operator_B(probe_pop, probe_fitness)
            else:
                probe_mutants = self._mutate_operator_C(probe_pop, probe_fitness)
            
            # Crossover
            probe_trials = self._crossover_batch(probe_pop, probe_mutants)
            
            # Evaluate
            probe_trial_fit = func(probe_trials)
            if len(probe_trial_fit) < len(probe_trials):
                if len(probe_trial_fit) == 0:
                    break
                probe_trial_fit = np.full(len(probe_trials), np.nan)
            
            # Selection
            probe_pop, probe_fitness, probe_improved = self._select_survivors_batch(
                probe_pop, probe_fitness, probe_trials, probe_trial_fit
            )
            
            # Track probe results
            curr_best = np.nanmin(probe_fitness)
            prev_best = prev_probe_best
            rank_improvement = (prev_best - curr_best) / (prev_best + 1e-10)
            rank_improvement = np.clip(rank_improvement, -10, 10)  # Bound extreme values
            
            rec = {
                'operator': op,
                'generation': probe_gens,
                'best_rank': np.nanmin(self._compute_fitness_ranking(probe_fitness)),
                'best_fitness': curr_best,
                'rank_improvement': rank_improvement,
                'improved': curr_best < prev_best,
            }
            self._probe_history.append(rec)
            
            # Update best
            if curr_best < probe_f_opt:
                probe_f_opt = curr_best
            
            prev_probe_best = curr_best
            probe_pop_history.append(probe_pop.copy())
            probe_fit_history.append(probe_fitness.copy())
            
            probe_op_idx += 1
            probe_gens += 1
            
            if stopping_condition():
                return probe_f_opt, probe_pop[np.nanargmin(probe_fitness)]
        
        # --- Commit to the best operator ---
        self._commit_to_operator()
        
        # --- COMMIT PHASE: Full population with committed operator ---
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        prev_best_fitness = f_opt
        iteration = 0
        
        while not stopping_condition():
            # Use committed operator exclusively (with rare fallback on stagnation)
            use_fallback = self._should_try_fallback(f_opt)
            
            if use_fallback and self._fallback_operator is not None:
                op = self._fallback_operator
            else:
                op = self._committed_operator
            
            # Generate mutants
            if op == 0:
                mutants = self._mutate_operator_A(population, fitness)
            elif op == 1:
                mutants = self._mutate_operator_B(population, fitness)
            else:
                mutants = self._mutate_operator_C(population, fitness)
            
            # Crossover
            trials = self._crossover_batch(population, mutants)
            
            # Evaluate
            trial_fitness = func(trials)
            
            if len(trial_fitness) < len(trials):
                if len(trial_fitness) == 0:
                    break
                trial_fitness = np.full(len(trials), np.nan)
            
            # Selection
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Update best
            current_best_idx = np.nanargmin(fitness)
            if fitness[current_best_idx] < f_opt:
                f_opt = fitness[current_best_idx]
                x_opt = population[current_best_idx].copy()
                self._stagnation_since_commit = 0  # Reset stagnation on improvement
            
            # Parameter adaptation
            self._current_population = population
            improvement_rate = np.mean(improved_mask)
            self._adapt_parameters(improvement_rate)
            
            # Diversity injection
            diversity = self._compute_diversity(population)
            if diversity < self.diversity_threshold:
                population = self._inject_diversity(population, fitness)
                fitness = func(population)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            
            # Stagnation check
            if self._check_stagnation(f_opt):
                population, f_opt, x_opt = self._restart_if_needed(population, fitness)
                fitness = func(population)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            
            iteration += 1
            
            if stopping_condition():
                break
        
        return f_opt, x_opt
    
    # -------------------------------------------------------------------------
    # Helper methods (used during commit phase)
    # -------------------------------------------------------------------------
    def _inject_diversity(self, population, fitness):
        """Replace worst individuals with archive + random samples."""
        if len(self.archive) < 2:
            return population
        
        archive_concat = np.vstack(self.archive)
        if len(archive_concat) < self.NP // 4:
            return population
        
        n_replace = max(self.NP // 5, 2)
        worst_indices = np.argsort(fitness)[-n_replace:]
        
        for idx in worst_indices:
            if np.random.rand() < 0.5 and len(archive_concat) > 0:
                donor_idx = np.random.randint(len(archive_concat))
                donor = archive_concat[donor_idx]
                noise = np.random.randn(self.dim) * 10.0
                population[idx] = np.clip(donor + noise, -100.0, 100.0)
            else:
                population[idx] = np.random.uniform(-100.0, 100.0, self.dim)
        
        return population
    
    def _restart_if_needed(self, population, fitness):
        """Restart based on k-NN graph connectivity analysis."""
        NP, dim = population.shape
        
        k = max(2, min(5, NP // 8))
        
        diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
        
        connectivity = np.zeros(NP)
        for i in range(NP):
            neighbors = nearest_indices[i]
            for n in neighbors:
                connectivity[n] += 1
        
        clustering = np.zeros(NP)
        for i in range(NP):
            neighbors = set(nearest_indices[i])
            if len(neighbors) < 2:
                clustering[i] = 0.0
                continue
            edges = 0
            possible = 0
            for ni in neighbors:
                for nj in neighbors:
                    if ni < nj:
                        possible += 1
                        if nj in set(nearest_indices[ni]):
                            edges += 1
            clustering[i] = edges / max(possible, 1)
        
        fitness_ranks = self._compute_fitness_ranking(fitness)
        isolation_score = (1.0 - connectivity / (connectivity.max() + 1e-10)) * 0.5 + \
                          (1.0 - clustering) * 0.3 + \
                          fitness_ranks * 0.2
        
        n_replace = max(NP // 4, 3)
        replace_indices = np.argsort(isolation_score)[-n_replace:]
        
        n_best = min(5, NP)
        best_indices = np.argsort(fitness)[:n_best]
        
        new_individuals = []
        for idx in replace_indices:
            best_source = None
            best_graph_dist = -1
            for b_idx in best_indices:
                n1 = set(nearest_indices[idx])
                n2 = set(nearest_indices[b_idx])
                common = len(n1 & n2)
                graph_dist_approx = 1.0 / (common + 0.1)
                if graph_dist_approx > best_graph_dist:
                    best_graph_dist = graph_dist_approx
                    best_source = b_idx
            
            if best_source is None:
                best_source = best_indices[0]
            
            source = population[best_source]
            direction = population[best_source] - population[idx]
            mutation_scale = np.random.uniform(0.8, 1.5)
            
            local_edges = sq_dists[idx, nearest_indices[idx]]
            avg_local_dist = np.mean(np.sqrt(local_edges))
            
            new_point = source + mutation_scale * self.F * direction + \
                        np.random.randn(dim) * avg_local_dist * 0.5
            new_individuals.append(np.clip(new_point, -100.0, 100.0))
        
        for i, idx in enumerate(replace_indices):
            population[idx] = new_individuals[i]
        
        best_idx = np.argmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        self.stagnation_counter = 0
        
        return population, f_opt, x_opt
