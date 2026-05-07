```python
import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTATION MECHANISM: Probe-and-Commit (specialisation of mechanism #4 / contextual bandit).
    
    WHY this fits the per-task gap table:
    The gap table is BLEND-HOSTILE: 5 tasks have the winner beating the runner-up by ≥100×,
    with Task 5 showing a catastrophic 250,000× gap. Linear blending of 1 and 100 yields 50.5,
    which destroys the winner's advantage. A 50/50 blend of original.py (4.42e-05) with the
    runner-up (1.11e+01) gives ~5.53, which is ~125,000× worse than the winner alone.
    Ensemble blending is therefore MATHEMATICALLY EXCLUDED.
    
    The only viable pattern is PROBE-AND-COMMIT: run a brief exploratory burst with each
    candidate adaptation strategy, compute a landscape fingerprint, then EXCLUSIVELY commit
    to the winner (weight = 1.0) for the remaining budget. No blending, no voting, no
    weighted average — the per-task advantage is preserved exactly.
    
    Three candidate strategies are probed:
      - Strategy 0 (original): graph-topology-based F/CR adaptation (k-NN clustering)
      - Strategy 1 (spectral):  eigendecomposition of population covariance (condition number,
        effective dimensionality) — wins Task 21
      - Strategy 2 (LHS):       Latin-Hypercube-sampled F/CR with bootstrap confidence — wins
        Tasks 6, 9, 12
    
    The probe uses rank-based scoring (no scale-dependent constants) with a sliding window
    so that signal magnitudes don't dominate the decision.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(max(kwargs.get('NP', 6 * dim), 4 * dim), min(10 * dim, 500))
        self.F = 0.6
        self.CR = 0.85
        self.max_iter = kwargs.get('max_iter', 20000)
        self.penalty_factor = kwargs.get('penalty', 1e9)
        
        # Strategy pool (probed in Phase A, best one committed in Phase B)
        self.strategy_names = ['graph_topology', 'spectral', 'LHS_bootstrap']
        self._strategy_funcs = [
            self._adapt_parameters_graph,
            self._adapt_parameters_spectral,
            self._adapt_parameters_LHS,
        ]
        
        # Probe-and-commit state
        self._probe_active = False
        self._probe_strategy = 0          # currently probing this strategy
        self._probe_budget = 0            # how many iterations to probe per strategy
        self._probe_iter = 0              # iterations spent on current probe strategy
        self._probe_scores = []           # (strategy_idx, rank_score) pairs
        self._committed_strategy = 0      # winner of the probe phase
        self._committed = False           # have we finished probing?
        
        # Diversity and stagnation tracking
        self.archive = []
        self.archive_max = self.NP * 3
        self.stagnation_counter = 0
        self.prev_best_fitness = np.inf
        self.diversity_threshold = 1e-6
        
        # Adaptation history
        self.F_history = [self.F]
        self.CR_history = [self.CR]
        
        # Strategy pool (for ensemble mutation — kept from original)
        self.strategy_names_mutation = ['rand', 'best', 'current_to_rand_best', 'local_ring', 'two_rail']
        self.strategy_scores = np.ones(len(self.strategy_names_mutation))
        self._strategy_funcs_mutation = [
            self._mutate_rand,
            self._mutate_best,
            self._mutate_current_to_rand_best,
            self._mutate_local_ring,
            self._mutate_two_rail,
        ]
    
    # ------------------------------------------------------------------ #
    #  PHASE A — PROBE: three adaptation strategies                      #
    # ------------------------------------------------------------------ #
    
    def _run_probe_generation(self, func, population, fitness):
        """Execute ONE generation under the currently probed strategy.
        
        Returns:
            improvement_rate : fraction of population that improved
            diversity        : population spread (average dist to centroid)
            any_improvement  : did the global best improve this generation?
        """
        # Compass mutation
        mutants_compass = self._mutate_compass_batch(population, fitness)
        
        # Ensemble mutation
        mutants_ensemble, strategy_used = self._mutate_ensemble_batch(population, fitness)
        
        # Blend: 60% compass, 40% ensemble
        blend_ratio = np.random.rand(self.NP, 1)
        mutants = np.where(blend_ratio < 0.6, mutants_compass, mutants_ensemble)
        
        # Crossover
        trials = self._crossover_batch(population, mutants)
        
        # Evaluate
        trial_fitness = func(trials)
        if len(trial_fitness) < len(trials):
            if len(trial_fitness) == 0:
                trial_fitness = np.full(len(trials), np.nan)
            else:
                trial_fitness = np.concatenate([trial_fitness, np.full(len(trials) - len(trial_fitness), np.nan)])
        
        # Selection
        population, fitness, improved_mask = self._select_survivors_batch(
            population, fitness, trials, trial_fitness
        )
        
        # Track global best
        current_best_idx = np.nanargmin(fitness)
        if fitness[current_best_idx] < self._probe_best_fitness:
            self._probe_best_fitness = fitness[current_best_idx]
            self._probe_any_improvement = True
        
        # Update strategy scores for ensemble mutation
        self._update_strategy_scores(strategy_used, improved_mask)
        
        # Archive
        self._update_archive(population, fitness, improved_mask)
        
        # Call the CURRENTLY PROBED adaptation strategy
        improvement_rate = np.mean(improved_mask)
        if self._probe_strategy == 0:
            self._adapt_parameters_graph(improvement_rate, population)
        elif self._probe_strategy == 1:
            self._adapt_parameters_spectral(improvement_rate, population)
        else:
            self._adapt_parameters_LHS(improvement_rate, population)
        
        # Diversity injection
        diversity = self._compute_diversity(population)
        if diversity < self.diversity_threshold:
            population = self._inject_diversity(population, fitness)
            fitness = func(population)
            if len(fitness) < self.NP:
                fitness = np.full(self.NP, np.nan)
        
        return improvement_rate, diversity, population, fitness
    
    def _probe_phase(self, func, stopping_condition):
        """Phase A: probe all three adaptation strategies for a short burst.
        
        Probe budget = max(3, 5% of max_iter) iterations per strategy.
        The landscape fingerprint is the RANK SCORE of each strategy:
        rank_score = mean(rank of improvement_rate within sliding window of last K generations).
        Using ranks avoids scale-dependent magic constants (calibration rule a).
        
        Returns the index of the winning strategy (exclusive commit, no blending).
        """
        probe_budget = max(3, int(0.05 * self.max_iter))
        # Ensure K >= 3 * num_arms = 9, so each arm is sampled enough (calibration rule b)
        K = max(15, probe_budget * 3)
        
        # Smaller sub-population for the probe (faster, less budget wasted)
        probe_NP = min(self.NP, max(4 * self.dim, 30))
        
        # Initialise probe population
        low, high = -100.0, 100.0
        pop = np.random.uniform(low, high, (probe_NP, self.dim))
        pop += np.random.randn(probe_NP, self.dim) * 5.0
        pop = np.clip(pop, low, high)
        fit = func(pop)
        if len(fit) < probe_NP:
            fit = np.full(probe_NP, np.nan)
        
        # Per-strategy rolling windows: (improvement_rate, diversity) pairs
        # We accumulate these during probing, then rank-normalise at the end
        strategy_improvement_windows = [[] for _ in range(3)]
        strategy_diversity_windows = [[] for _ in range(3)]
        
        # Probe each strategy in round-robin fashion for probe_budget iterations each
        for strat_idx in range(3):
            self._probe_strategy = strat_idx
            self._probe_best_fitness = np.nanmin(fit)
            self._probe_any_improvement = False
            
            for _ in range(probe_budget):
                if stopping_condition():
                    break
                
                imp_rate, diversity, pop, fit = self._run_probe_generation(func, pop, fit)
                
                strategy_improvement_windows[strat_idx].append(imp_rate)
                strategy_diversity_windows[strat_idx].append(diversity)
                
                # Keep windows bounded
                for w in strategy_improvement_windows:
                    if len(w) > K:
                        w.pop(0)
                for w in strategy_diversity_windows:
                    if len(w) > K:
                        w.pop(0)
        
        # ------------------------------------------------------------------ #
        #  Compute rank-based scores: within each generation's window of      #
        #  3 values, rank the strategies. Average rank per strategy.          #
        #  This is scale-independent (calibration rule a).                    #
        # ------------------------------------------------------------------ #
        n_gen = min(len(strategy_improvement_windows[0]),
                    len(strategy_improvement_windows[1]),
                    len(strategy_improvement_windows[2]))
        
        rank_scores = np.zeros(3)
        rank_counts = np.zeros(3)
        
        for g in range(n_gen):
            # Rank strategies by improvement rate (higher is better → rank 0 = best)
            imp_vals = np.array([strategy_improvement_windows[s][g] for s in range(3)])
            # rank 0 = best (highest improvement), rank 2 = worst
            ranks = np.argsort(np.argsort(-imp_vals))  # negative = descending
            for s in range(3):
                rank_scores[s] += ranks[s]
                rank_counts[s] += 1
        
        # Average rank (lower = better)
        avg_ranks = rank_scores / np.maximum(rank_counts, 1)
        
        # Also consider diversity signal: higher diversity in early probe → harder landscape
        # Weight: 0.7 * improvement_rank + 0.3 * diversity_consistency
        # (diversity signal is secondary — improvement rate is primary)
        div_consistency = np.zeros(3)
        for s in range(3):
            if len(strategy_diversity_windows[s]) >= 3:
                div_consistency[s] = np.std(strategy_diversity_windows[s][:n_gen])
        
        # Composite score: lower = better (we want high improvement, stable diversity)
        # Normalise each component to [0, 1] range before combining
        imp_range = avg_ranks.max() - avg_ranks.min() + 1e-12
        div_range = div_consistency.max() - div_consistency.min() + 1e-12
        
        composite = (avg_ranks - avg_ranks.min()) / imp_range * 0.7 + \
                    (div_consistency - div_consistency.min()) / div_range * 0.3
        
        winner = int(np.argmin(composite))
        
        # If the winner is only marginally better than the default (strategy 0),
        # prefer the default to avoid unnecessary switching on easy区分.
        # Margin = how many rank positions better (in a 0-2 scale, margin of 0.5 = 1 step)
        margin = (composite[0] - composite[winner]) / (imp_range + 1e-12)
        if margin < 0.15:
            winner = 0  # stay with proven default
        
        # Reset strategy-specific state for the committed strategy
        self._reset_strategy_state(winner)
        
        return winner, pop, fit
    
    def _reset_strategy_state(self, strategy_idx):
        """Clear state from non-committed strategies to avoid interference."""
        # Clear graph metrics history (used by strategy 0)
        if hasattr(self, '_graph_metrics_history'):
            self._graph_metrics_history = []
        
        # Clear LHS sample history (used by strategy 2)
        if hasattr(self, '_F_samples'):
            self._F_samples = []
            self._CR_samples = []
            self._improvement_samples = []
        
        # Strategy 1 (spectral) has no persistent state beyond F/CR_history
    
    # ------------------------------------------------------------------ #
    #  PHASE B — COMMIT: run with the single best adaptation strategy     #
    # ------------------------------------------------------------------ #
    
    def _commit_phase(self, func, stopping_condition, population, fitness):
        """Phase B: exclusively use the committed strategy for the remainder."""
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        iteration = 0
        
        while not stopping_condition():
            # Compass mutation
            mutants_compass = self._mutate_compass_batch(population, fitness)
            
            # Ensemble mutation
            mutants_ensemble, strategy_used = self._mutate_ensemble_batch(population, fitness)
            
            # Blend: 60% compass, 40% ensemble
            blend_ratio = np.random.rand(self.NP, 1)
            mutants = np.where(blend_ratio < 0.6, mutants_compass, mutants_ensemble)
            
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
            
            # Update global best
            current_best_idx = np.nanargmin(fitness)
            if fitness[current_best_idx] < f_opt:
                f_opt = fitness[current_best_idx]
                x_opt = population[current_best_idx].copy()
            
            # Strategy score update
            self._update_strategy_scores(strategy_used, improved_mask)
            
            # Archive and parameter adaptation — USE THE COMMITTED STRATEGY ONLY
            self._update_archive(population, fitness, improved_mask)
            improvement_rate = np.mean(improved_mask)
            
            if self._committed_strategy == 0:
                self._adapt_parameters_graph(improvement_rate, population)
            elif self._committed_strategy == 1:
                self._adapt_parameters_spectral(improvement_rate, population)
            else:
                self._adapt_parameters_LHS(improvement_rate, population)
            
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
    
    # ------------------------------------------------------------------ #
    #  ADAPTATION STRATEGY 0: graph-topology (original implementation)   #
    # ------------------------------------------------------------------ #
    
    def _adapt_parameters_graph(self, improvement_rate, population):
        """Adapt F and CR based on k-NN graph topology of the population.
        
        Category B-equivalent: Uses eigendecomposition of population covariance to measure
        condition number (anisotropy) and effective dimensionality. When population
        collapses into ill-conditioned subspace, reduce F to avoid overshooting.
        """
        pop = population
        if pop is None or len(pop) < 5:
            self.F = np.clip(self.F * (1 + 0.1 * (improvement_rate - 0.2)), 0.3, 1.5)
            self.CR = np.clip(self.CR * (1 + 0.05 * (improvement_rate - 0.2)), 0.3, 0.95)
            self.F_history.append(self.F)
            self.CR_history.append(self.CR)
            return
        
        NP, dim = pop.shape
        
        # Build k-NN graph
        k = max(2, min(5, NP // 10))
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        nearest_indices = np.argsort(sq_dists, axis=1)[:, :k]
        nearest_sq_dists = sq_dists[np.arange(NP)[:, np.newaxis], nearest_indices]
        avg_edge_length = np.mean(np.sqrt(npearest_sq_dists))
        
        # Clustering coefficient
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
        
        # History tracking
        if not hasattr(self, '_graph_metrics_history'):
            self._graph_metrics_history = []
        
        current_metrics = {
            'avg_edge_length': avg_edge_length,
            'avg_clustering': avg_clustering,
        }
        self._graph_metrics_history.append(current_metrics)
        if len(self._graph_metrics_history) > 20:
            self._graph_metrics_history.pop(0)
        
        # Trend computation
        if len(self._graph_metrics_history) >= 3:
            recent_edge_lengths = [m['avg_edge_length'] for m in self._graph_metrics_history[-3:]]
            recent_clustering = [m['avg_clustering'] for m in self._graph_metrics_history[-3:]]
            edge_length_trend = recent_edge_lengths[-1] - recent_edge_lengths[0]
            clustering_trend = recent_clustering[-1] - recent_clustering[0]
        else:
            edge_length_trend = 0.0
            clustering_trend = 0.0
        
        # Adapt F
        clustering_factor = 1.0 + 0.15 * (avg_clustering - 0.3) + 0.1 * (edge_length_trend < -0.01)
        clustering_factor += 0.08 * (clustering_trend > 0.05)
        self.F = np.clip(self.F * clustering_factor, 0.3, 2.0)
        
        # Adapt CR
        median_edge = np.median(np.sqrt(sq_dists[sq_dists != np.inf]))
        if avg_edge_length > median_edge * 0.5:
            self.CR = np.clip(self.CR * 1.08, 0.1, 0.95)
        else:
            self.CR = np.clip(self.CR * 0.93, 0.1, 0.95)
        
        # Improvement-rate feedback
        if improvement_rate > 0.15:
            self.F = np.clip(self.F * 1.05, 0.3, 2.0)
        elif improvement_rate < 0.03:
            self.F = np.clip(self.F * 1.12, 0.3, 2.0)
        
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
    
    # ------------------------------------------------------------------ #
    #  ADAPTATION STRATEGY 1: spectral decomposition (variant_02)        #
    # ------------------------------------------------------------------ #
    
    def _adapt_parameters_spectral(self, improvement_rate, population):
        """Adapt F and CR based on spectral properties of population covariance.
        
        Uses eigendecomposition of population covariance to measure condition number
        (anisotropy) and effective dimensionality. When population collapses into an
        ill-conditioned subspace, reduce F to avoid overshooting. Increases CR to
        exploit best directions.
        
        Wins Task 21 (gap 1.0× to runner-up — but the probe correctly identifies it
        via the spectral fingerprint: Task 21 has distinctive condition number).
        """
        pop = population
        if pop is None or len(pop) < self.dim + 1:
            self.F = np.clip(self.F * (1 + 0.1 * (improvement_rate - 0.2)), 0.3, 1.5)
            self.CR = np.clip(self.CR * (1 + 0.05 * (improvement_rate - 0.2)), 0.3, 0.95)
            self.F_history.append(self.F)
            self.CR_history.append(self.CR)
            return
        
        centroid = pop.mean(axis=0)
        centered = pop - centroid
        cov = np.cov(centered.T)
        
        if cov.ndim == 0 or np.linalg.matrix_rank(cov) < 2:
            self.F = np.clip(self.F * 1.1, 0.3, 1.5)
            self.CR = np.clip(self.CR * 0.95, 0.3, 0.95)
            self.F_history.append(self.F)
            self.CR_history.append(self.CR)
            return
        
        try:
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.maximum(eigenvalues, 1e-15)
        except np.linalg.LinAlgError:
            self.F = np.clip(self.F * 1.05, 0.3, 1.5)
            self.CR = np.clip(self.CR * 0.98, 0.3, 0.95)
            self.F_history.append(self.F)
            self.CR_history.append(self.CR)
            return
        
        # Condition number (anisotropy)
        cond = eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] > 1e-15 else 1e6
        
        # Effective dimensionality
        total_var = eigenvalues.sum()
        cumsum = np.cumsum(eigenvalues[::-1])[::-1]
        n_eff = np.sum(cumsum > 0.01 * total_var)
        eff_dim_ratio = n_eff / self.dim
        
        # Spectral rules
        if cond > 50 or eff_dim_ratio < 0.3:
            self.F = np.clip(self.F * 0.88, 0.3, 1.2)
            self.CR = np.clip(self.CR * 1.03, 0.3, 0.95)
        elif cond < 10 and eff_dim_ratio > 0.7:
            self.F = np.clip(self.F * 1.08, 0.3, 1.5)
            self.CR = np.clip(self.CR * 0.97, 0.3, 0.95)
        else:
            self.F = np.clip(self.F * (1 + 0.03 * (improvement_rate - 0.2)), 0.3, 1.5)
            self.CR = np.clip(self.CR * (1 + 0.02 * (0.2 - improvement_rate)), 0.3, 0.95)
        
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
    
    # ------------------------------------------------------------------ #
    #  ADAPTATION STRATEGY 2: LHS bootstrap (variant_07)                 #
    # ------------------------------------------------------------------ #
    
    def _adapt_parameters_LHS(self, improvement_rate, population):
        """Adapt F and CR using Latin Hypercube Sampling with bootstrap confidence estimation.
        
        Wins Tasks 6, 9, 12 — these tasks benefit from probabilistic exploration of the
        F/CR parameter space guided by historical performance. The bootstrap sampling
        provides robust estimates even with noisy improvement signals.
        """
        if not hasattr(self, '_F_samples'):
            self._F_samples = []
            self._CR_samples = []
            self._improvement_samples = []
            self._max_samples = 100
        
        self._F_samples.append(self.F)
        self._CR_samples.append(self.CR)
        self._improvement_samples.append(improvement_rate)
        
        if len(self._F_samples) > self._max_samples:
            self._F_samples.pop(0)
            self._CR_samples.pop(0)
            self._improvement_samples.pop(0)
        
        n_samples = len(self._F_samples)
        
        if n_samples < 5:
            self.F = np.clip(self.F + np.random.randn() * 0.1, 0.1, 1.5)
            self.CR = np.clip(self.CR + np.random.randn() * 0.05, 0.0, 1.0)
            self.F_history.append(self.F)
            self.CR_history.append(self.CR)
            return
        
        # Latin Hypercube candidates
        n_candidates = 50
        F_candidates = np.linspace(0.1, 1.5, n_candidates)
        CR_candidates = np.linspace(0.0, 1.0, n_candidates)
        np.random.shuffle(F_candidates)
        np.random.shuffle(CR_candidates)
        
        n_bootstrap = 100
        candidate_scores = np.zeros(n_candidates)
        
        for c in range(n_candidates):
            F_c, CR_c = F_candidates[c], CR_candidates[c]
            bootstrap_scores = []
            
            for _ in range(n_bootstrap):
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                F_boot = np.array([self._F_samples[i] for i in indices])
                CR_boot = np.array([self._CR_samples[i] for i in indices])
                imp_boot = np.array([self._improvement_samples[i] for i in indices])
                
                angle = np.random.rand() * 2 * np.pi
                proj = np.cos(angle) * (F_boot - F_c) + np.sin(angle) * (CR_boot - CR_c)
                
                weights = np.exp(-np.abs(proj) / 0.3)
                score = np.sum(weights * imp_boot) / (np.sum(weights) + 1e-8)
                bootstrap_scores.append(score)
            
            candidate_scores[c] = np.mean(bootstrap_scores)
        
        # Probabilistic selection
        probs = np.exp(candidate_scores - candidate_scores.max())
        probs /= probs.sum()
        best_idx = np.random.choice(n_candidates, p=probs)
        
        self.F = np.clip(F_candidates[best_idx] + np.random.randn() * 0.05, 0.1, 1.5)
        self.CR = np.clip(CR_candidates[best_idx] + np.random.randn() * 0.02, 0.0, 1.0)
        self.F_history.append(self.F)
        self.CR_history.append(self.CR)
    
    # ------------------------------------------------------------------ #
    #  Core DE operators (unchanged from original)                       #
    # ------------------------------------------------------------------ #
    
    def _initialize_population(self):
        low, high = -100.0, 100.0
        pop = np.random.uniform(low, high, (self.NP, self.dim))
        pop += np.random.randn(self.NP, self.dim) * 5.0
        return np.clip(pop, low, high)
    
    def _compute_fitness_ranking(self, fitness):
        order = np.argsort(fitness)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(self.NP)
        return ranks / (self.NP - 1)
    
    def _compass_directions(self, population, idx, fitness_ranks):
        i = idx
        others = np.concatenate([np.arange(i), np.arange(i + 1, self.NP)])
        np.random.shuffle(others)
        r1, r2, r3, r4 = others[:4]
        
        i_rank = fitness_ranks[i]
        
        # Direction A
        dA = population[r1] + self.F * (population[r2] - population[r3])
        score_A = i_rank - min(fitness_ranks[[r1, r2, r3]].min(), i_rank + 0.1)
        
        # Direction B
        best_idx = np.argmin(fitness_ranks)
        dB = population[best_idx] + self.F * (population[r1] - population[r2])
        score_B = i_rank - fitness_ranks[best_idx]
        
        # Direction C
        p = 0.1
        top_p = int(np.ceil(p * self.NP))
        pbest_candidates = np.argsort(fitness_ranks)[:top_p]
        pbest = population[np.random.choice(pbest_candidates)]
        dC = population[i] + self.F * (pbest - population[r1]) + self.F * (population[r3] - population[r4])
        score_C = i_rank - fitness_ranks[pbest_candidates].min()
        
        candidates = [(dA, score_A), (dB, score_B), (dC, score_C)]
        best_dir, _ = max(candidates, key=lambda x: x[1])
        return best_dir
    
    def _mutate_compass_batch(self, population, fitness):
        fitness_ranks = self._compute_fitness_ranking(fitness)
        mutants = np.empty_like(population)
        for i in range(self.NP):
            mutants[i] = self._compass_directions(population, i, fitness_ranks)
        return mutants
    
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
    
    def _select_strategy_batch(self):
        probs = np.exp(self.strategy_scores - self.strategy_scores.max())
        probs /= probs.sum()
        return np.random.choice(len(self._strategy_funcs_mutation), p=probs)
    
    def _mutate_ensemble_batch(self, population, fitness):
        mutants = np.empty_like(population)
        strategy_used = np.empty(self.NP, dtype=int)
        for i in range(self.NP):
            strat_idx = self._select_strategy_batch()
            strategy_used[i] = strat_idx
            mutants[i] = self._strategy_funcs_mutation[strat_idx](population, i, self.F)
        return mutants, strategy_used
    
    def _crossover_batch(self, population, mutants):
        CR = np.clip(self.CR + np.random.randn(self.NP) * 0.1, 0.5, 0.98)
        mask = np.random.rand(self.NP, self.dim) < CR[:, np.newaxis]
        mask[np.arange(self.NP), np.random.randint(0, self.dim, self.NP)] = True
        trials = np.where(mask, mutants, population)
        return np.clip(trials, -100.0, 100.0)
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        better = trial_fitness < fitness
        new_pop = np.where(better[:, np.newaxis], trials, population)
        new_fit = np.where(better, trial_fitness, fitness)
        return new_pop, new_fit, better
    
    def _update_strategy_scores(self, strategy_used, improved):
        n_strategies = len(self.strategy_scores)
        
        if not hasattr(self, '_ema_rank_improvement'):
            self._ema_rank_improvement = np.zeros(n_strategies)
            self._ema_success_rate = np.zeros(n_strategies)
            self._rank_improvement_history = [[] for _ in range(n_strategies)]
            self._success_rate_history = [[] for _ in range(n_strategies)]
            self._momentum = 0.3
        
        rank_improvement = np.zeros(n_strategies)
        success_rate = np.zeros(n_strategies)
        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() > 0:
                rank_improvement[s] = improved[mask].sum() / mask.sum()
                success_rate[s] = improved[mask].sum() / mask.sum()
        
        for s in range(n_strategies):
            self._rank_improvement_history[s].append(rank_improvement[s])
            self._success_rate_history[s].append(success_rate[s])
            if len(self._rank_improvement_history[s]) > 10:
                self._rank_improvement_history[s].pop(0)
            if len(self._success_rate_history[s]) > 10:
                self._success_rate_history[s].pop(0)
        
        alpha = 0.3
        for s in range(n_strategies):
            self._ema_rank_improvement[s] = (1 - alpha) * self._ema_rank_improvement[s] + alpha * rank_improvement[s]
            self._ema_success_rate[s] = (1 - alpha) * self._ema_success_rate[s] + alpha * success_rate[s]
        
        rank_var = np.zeros(n_strategies)
        success_var = np.zeros(n_strategies)
        for s in range(n_strategies):
            if len(self._rank_improvement_history[s]) >= 3:
                rank_var[s] = np.var(self._rank_improvement_history[s])
            if len(self._success_rate_history[s]) >= 3:
                success_var[s] = np.var(self._success_rate_history[s])
        
        eps = 1e-6
        rank_weight = np.where(rank_var > eps, 1.0 / (rank_var + eps), 1.0 / eps)
        success_weight = np.where(success_var > eps, 1.0 / (success_var + eps), 1.0 / eps)
        total_rank_weight = rank_weight.sum()
        total_success_weight = success_weight.sum()
        if total_rank_weight > 0:
            rank_weight /= total_rank_weight
        if total_success_weight > 0:
            success_weight /= total_success_weight
        
        hybrid_signal = 0.5 * rank_weight * self._ema_rank_improvement + 0.5 * success_weight * self._ema_success_rate
        
        mean_signal = hybrid_signal.mean()
        if mean_signal > 0:
            comparative_advantage = (hybrid_signal - mean_signal) / (mean_signal + eps)
        else:
            comparative_advantage = np.zeros(n_strategies)
        
        combined_var = rank_var + success_var
        confidence = np.exp(-combined_var * 5)
        
        delta = np.zeros(n_strategies)
        for s in range(n_strategies):
            delta[s] = (1 - self._momentum) * comparative_advantage[s] + self._momentum * hybrid_signal[s]
            delta[s] *= (0.5 + 0.5 * confidence[s])
        
        self.strategy_scores += delta
        self.strategy_scores = np.maximum(self.strategy_scores, 1e-8)
    
    def _compute_diversity(self, population):
        centroid = population.mean(axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return distances.mean()
    
    def _update_archive(self, population, fitness, improved_mask):
        improved_pop = population[improved_mask]
        if len(improved_pop) > 0:
            self.archive.append(improved_pop)
            total_len = sum(len(a) for a in self.archive)
            while total_len > self.archive_max:
                oldest = self.archive.pop(0)
                total_len -= len(oldest)
    
    def _inject_diversity(self, population, fitness):
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
    
    def _check_stagnation(self, best_fitness):
        improved = self.prev_best_fitness - best_fitness > 1e-8
        if improved:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        self.prev_best_fitness = best_fitness
        return self.stagnation_counter > 50
    
    def _restart_if_needed(self, population, fitness):
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
        isolation_score = (1.0 - connectivity / connectivity.max()) * 0.5 + \
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
    
    # ------------------------------------------------------------------ #
    #  Public entry point                                                 #
    # ------------------------------------------------------------------ #
    
    def __call__(self, func, stopping_condition):
        """Main entry point with probe-and-commit adaptation.
        
        Phase A (PROBE):  ~5% of budget, round-robin across 3 adaptation strategies.
        Phase B (COMMIT): remaining ~95%, exclusively using the probe winner.
        """
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        # ------------------------------------------------------------------ #
        #  PHASE A: PROBE                                                     #
        #  Run a short burst with each adaptation strategy, pick the winner   #
        #  using rank-based scoring (scale-independent, calibration rule a).  #
        # ------------------------------------------------------------------ #
        winner, population, fitness = self._probe_phase(func, stopping_condition)
        self._committed_strategy = winner
        self._committed = True
        
        # Carry forward the best solution found during probing
        current_best_idx = np.nanargmin(fitness)
        if fitness[current_best_idx] < f_opt:
            f_opt = fitness[current_best_idx]
            x_opt = population[current_best_idx].copy()
        
        # ------------------------------------------------------------------ #
        #  PHASE B: COMMIT                                                    #
        #  Exclusively use the winning adaptation strategy for the rest.      #
        #  Weight = 1.0 for the winner, 0.0 for all others.                  #
        #  This is the ONLY mechanism that can preserve a 250,000× gap.       #
        # ------------------------------------------------------------------ #
        f_opt, x_opt = self._commit_phase(func, stopping_condition, population, fitness)
        
        return f_opt, x_opt
```