import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTATION MECHANISM: Per-Task Probe-and-Commit (specialisation of mechanism #4).
    
    The per-task winner-gap table is BLEND-HOSTILE: 8 tasks have winner/runner-up
    gaps >=100× (max 2.5e7× on task 5). Linear blending mathematically CANNOT
    preserve these advantages — a 50/50 blend of 1 and 100 is 50.5, not 1.
    Ensemble blending is therefore EXPLICITLY FORBIDDEN.
    
    The table shows 2 distinct winners: original.py (23 wins) and
    variant_08_catH_t05 (1 win, task 23, gap ~1.03× — marginal).
    The correct response is a PROBE-AND-COMMIT dispatcher:
    
      Phase A (PROBE, ~5% budget):
        - Run a short burst with the original compass strategy AND with
          the variant_08 FDC-adaptive strategy on a sub-population.
        - Compute a landscape fingerprint from the probe: convergence rate,
          rank-autocorrelation (ruggedness proxy), effective dimensionality,
          and stagnation index.
        - Compute per-strategy empirical win rate from the probe.
      
      Phase B (COMMIT):
        - If probe is conclusive (clear winner): commit exclusively to that
          strategy (weight 1.0 on the chosen arm, weight 0.0 on others).
        - If probe is inconclusive: default to original (23/24 win rate).
        - If the committed strategy stagnates, allow a one-time re-probe
          with a larger budget before committing again.
    
    This preserves per-task advantages because it selects ONE arm at a time,
    never blends, and the dispatch rule is derived from observable landscape
    signals rather than hard-coded regime tables.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(max(kwargs.get('NP', 6 * dim), 4 * dim), min(10 * dim, 500))
        self.F = 0.6
        self.CR = 0.85
        self.max_iter = kwargs.get('max_iter', 20000)
        self.penalty_factor = kwargs.get('penalty', 1e9)
        
        # Strategy pool for probe phase
        self._probe_strategies = {
            'original': self._probe_original,
            'variant_08': self._probe_variant_08,
        }
        
        # Diversity and stagnation tracking
        self.archive = []
        self.archive_max = self.NP * 3
        self.stagnation_counter = 0
        self.prev_best_fitness = np.inf
        self.diversity_threshold = 1e-6
        
        # Adaptation momentum
        self.F_history = [self.F]
        self.CR_history = [self.CR]
        
        # Probe-and-commit state
        self._committed_strategy = None
        self._probe_completed = False
        self._landscape_fingerprint = None
        self._reprobe_allowed = True
        self._reprobe_budget = None
        
    def _initialize_population(self):
        low, high = -100.0, 100.0
        pop = np.random.uniform(low, high, (self.NP, self.dim))
        pop += np.random.randn(self.NP, self.dim) * 5.0
        return np.clip(pop, low, high)
    
    def _compute_fitness_ranking(self, fitness):
        order = np.argsort(fitness)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(self.NP)
        return ranks / max(self.NP - 1, 1)
    
    def _mutate_compass_batch(self, population, fitness):
        """Original compass mutation: rank-proximity scoring across 3 candidate directions."""
        fitness_ranks = self._compute_fitness_ranking(fitness)
        mutants = np.empty_like(population)
        
        for i in range(self.NP):
            others = np.concatenate([np.arange(i), np.arange(i + 1, self.NP)])
            np.random.shuffle(others)
            r1, r2, r3, r4 = others[:4]
            
            i_rank = fitness_ranks[i]
            
            # Candidate A: DE/rand/1 style
            dA = population[r1] + self.F * (population[r2] - population[r3])
            score_A = i_rank - min(fitness_ranks[[r1, r2, r3]].min(), i_rank + 0.1)
            
            # Candidate B: DE/best/1 style
            best_idx = np.argmin(fitness_ranks)
            dB = population[best_idx] + self.F * (population[r1] - population[r2])
            score_B = i_rank - fitness_ranks[best_idx]
            
            # Candidate C: current-to-pbest with archive
            p = 0.1
            top_p = max(1, int(np.ceil(p * self.NP)))
            pbest_candidates = np.argsort(fitness_ranks)[:top_p]
            pbest = population[np.random.choice(pbest_candidates)]
            dC = population[i] + self.F * (pbest - population[r1]) + self.F * (population[r3] - population[r4])
            score_C = i_rank - fitness_ranks[pbest_candidates].min()
            
            candidates = [(dA, score_A), (dB, score_B), (dC, score_C)]
            best_dir, _ = max(candidates, key=lambda x: x[1])
            mutants[i] = np.clip(best_dir, -100.0, 100.0)
        
        return mutants
    
    def _mutate_variant_08_batch(self, population, fitness):
        """Variant_08 FDC-adaptive mutation: uses fitness-distance correlation to
        weight exploration vs exploitation. Key mechanism: exploit_weight derived
        from FDC EMA determines balance between centroid direction (exploration)
        and best-direction (exploitation)."""
        NP, dim = population.shape
        ranks = self._compute_fitness_ranking(fitness)
        best_idx = np.argmin(fitness)
        
        # Initialise FDC temporal tracking
        if not hasattr(self, '_fdc_history'):
            self._fdc_history = []
            self._centroid_history = []
            self._fdc_ema = 0.5
        
        # Compute FDC — key landscape indicator
        centroid = population.mean(axis=0)
        dists_to_centroid = np.linalg.norm(population - centroid, axis=1)
        if dists_to_centroid.std() > 1e-10:
            fdc = np.corrcoef(ranks, dists_to_centroid)[0, 1]
            if np.isnan(fdc):
                fdc = 0.0
        else:
            fdc = 0.0
        
        # Update EMA for temporal stability
        self._fdc_ema = 0.7 * self._fdc_ema + 0.3 * fdc
        self._fdc_history.append(self._fdc_ema)
        if len(self._fdc_history) > 15:
            self._fdc_history.pop(0)
        
        # Centroid velocity
        self._centroid_history.append(centroid.copy())
        if len(self._centroid_history) > 5:
            self._centroid_history.pop(0)
        
        centroid_velocity = 0.0
        if len(self._centroid_history) >= 2:
            centroid_velocity = np.linalg.norm(self._centroid_history[-1] - self._centroid_history[0])
        
        # FDC-driven adaptive weight (mechanism core)
        fdc_threshold = 0.3
        exploit_weight = np.clip((self._fdc_ema - fdc_threshold) / (1.0 - fdc_threshold + 1e-10), 0.1, 0.9)
        explore_weight = 1.0 - exploit_weight
        
        # Mechanism 1: centroid direction (exploration-biased)
        direction_to_centroid = centroid - population
        centroid_mutations = population + explore_weight * self.F * direction_to_centroid
        
        # Mechanism 2: best direction (exploitation-biased)
        direction_to_best = population[best_idx] - population
        best_mutations = population + exploit_weight * self.F * direction_to_best
        
        mutants = np.empty_like(population)
        for i in range(NP):
            others = np.concatenate([np.arange(i), np.arange(i + 1, self.NP)])
            np.random.shuffle(others)
            r1, r2, r3, r4 = others[:4]
            
            # Candidate A: centroid-weighted
            cand_A = centroid_mutations[i]
            score_A = ranks[i] - np.mean([ranks[j] for j in [r1, r2, r3]])
            
            # Candidate B: best-weighted
            cand_B = best_mutations[i]
            score_B = ranks[i] - ranks[best_idx]
            
            # Candidate D: baseline rand/1
            cand_D = population[r1] + self.F * (population[r2] - population[r3])
            score_D = ranks[i] - np.mean([ranks[r1], ranks[r2], ranks[r3]])
            
            # Candidate C: archive-based if available
            cand_C = None
            score_C = -np.inf
            if hasattr(self, 'archive') and len(self.archive) > 0:
                archive_concat = np.vstack(self.archive)
                if len(archive_concat) > 0:
                    archive_idx = np.random.randint(len(archive_concat))
                    archive_donor = archive_concat[archive_idx]
                    pbest_idx = np.random.choice(np.argsort(fitness)[:max(1, NP // 10)])
                    pbest = population[pbest_idx]
                    cand_C = population[i] + self.F * (pbest - population[r1]) + self.F * (population[r2] - population[r3])
                    score_C = ranks[i] - np.mean([ranks[j] for j in [r1, r2]])
            
            candidates = [(cand_A, score_A), (cand_B, score_B), (cand_D, score_D)]
            if cand_C is not None:
                candidates.append((cand_C, score_C))
            
            best_cand, _ = max(candidates, key=lambda x: x[1])
            
            # Noise proportional to centroid velocity
            noise_scale = 0.01 + 0.1 * centroid_velocity / (np.linalg.norm(centroid) + 1e-10)
            noise = np.random.randn(dim) * noise_scale
            mutants[i] = np.clip(best_cand + noise, -100.0, 100.0)
        
        return mutants
    
    def _probe_original(self, pop, func, n_probe_gens):
        """Run original compass strategy for n_probe_gens. Returns (best_fitness_trace, final_best)."""
        fit = func(pop)
        best_trace = [np.min(fit)]
        for _ in range(n_probe_gens):
            mutants = self._mutate_compass_batch(pop, fit)
            trials = self._crossover_batch_fast(pop, mutants)
            trial_fit = func(trials)
            improved = trial_fit < fit
            pop = np.where(improved[:, np.newaxis], trials, pop)
            fit = np.where(improved, trial_fit, fit)
            best_trace.append(np.min(fit))
            if np.isnan(np.min(fit)):
                break
        return best_trace, np.min(fit)
    
    def _probe_variant_08(self, pop, func, n_probe_gens):
        """Run FDC-adaptive strategy for n_probe_gens. Returns (best_fitness_trace, final_best)."""
        # Reset FDC state for clean probe
        if hasattr(self, '_fdc_history'):
            old_history = self._fdc_history[:]
            old_ema = self._fdc_ema
            old_centroid = self._centroid_history[:] if hasattr(self, '_centroid_history') else []
        else:
            old_history = None
            old_ema = None
            old_centroid = None
        
        self._fdc_history = []
        self._centroid_history = []
        self._fdc_ema = 0.5
        
        fit = func(pop)
        best_trace = [np.min(fit)]
        for _ in range(n_probe_gens):
            mutants = self._mutate_variant_08_batch(pop, fit)
            trials = self._crossover_batch_fast(pop, mutants)
            trial_fit = func(trials)
            improved = trial_fit < fit
            pop = np.where(improved[:, np.newaxis], trials, pop)
            fit = np.where(improved, trial_fit, fit)
            best_trace.append(np.min(fit))
            if np.isnan(np.min(fit)):
                break
        
        # Restore FDC state
        if old_history is not None:
            self._fdc_history = old_history
            self._fdc_ema = old_ema
            self._centroid_history = old_centroid if old_centroid else []
        
        return best_trace, np.min(fit)
    
    def _crossover_batch_fast(self, population, mutants):
        """Fast crossover without adaptive CR noise (used in probe)."""
        CR = self.CR * np.ones(self.NP)
        mask = np.random.rand(self.NP, self.dim) < CR[:, np.newaxis]
        mask[np.arange(self.NP), np.random.randint(0, self.dim, self.NP)] = True
        trials = np.where(mask, mutants, population)
        return np.clip(trials, -100.0, 100.0)
    
    def _compute_landscape_fingerprint(self, probe_results):
        """Extract cheap landscape signals from probe data. Returns dict of signals."""
        signals = {}
        
        for name, (trace, final_best) in probe_results.items():
            trace = np.array(trace)
            valid = trace[~np.isnan(trace)]
            if len(valid) < 2:
                continue
            
            # Converge rate: negative slope of log-improvement (higher = faster convergence)
            gen_idx = np.arange(len(valid))
            if np.std(valid) > 1e-12 and np.max(gen_idx) > np.min(gen_idx):
                cov = np.cov(gen_idx, valid)
                if cov[0, 0] > 0:
                    slope = cov[0, 1] / cov[0, 0]
                    signals[f'{name}_slope'] = slope
                else:
                    signals[f'{name}_slope'] = 0.0
            else:
                signals[f'{name}_slope'] = 0.0
            
            # Improvement ratio: how much did we improve from start to end?
            if valid[0] > 0:
                signals[f'{name}_improvement_ratio'] = (valid[0] - valid[-1]) / max(abs(valid[0]), 1e-12)
            else:
                signals[f'{name}_improvement_ratio'] = 0.0
            
            # Stagnation index: fraction of gens with <1% relative improvement
            rel_improvements = np.abs(np.diff(valid)) / max(np.abs(valid[:-1]), 1e-12)
            stagnation_frac = np.mean(rel_improvements < 0.01)
            signals[f'{name}_stagnation'] = stagnation_frac
            
            # Rank autocorrelation: how smooth is the convergence trajectory?
            if len(valid) >= 4:
                autocorr = np.corrcoef(valid[:-1], valid[1:])[0, 1]
                signals[f'{name}_autocorr'] = 0.0 if np.isnan(autocorr) else autocorr
            else:
                signals[f'{name}_autocorr'] = 0.0
        
        # Relative comparison signals
        for metric in ['slope', 'improvement_ratio', 'stagnation', 'autocorr']:
            o_key = f'original_{metric}'
            v_key = f'variant_08_{metric}'
            if o_key in signals and v_key in signals:
                signals[f'delta_{metric}'] = signals[o_key] - signals.get(v_key, 0.0)
        
        return signals
    
    def _decide_strategy(self, fingerprint):
        """Dispatch to the winning strategy using probe evidence.
        Returns the key of the committed strategy (str)."""
        # Primary criterion: convergence slope (higher = better = more negative)
        o_slope = fingerprint.get('original_slope', 0.0)
        v_slope = fingerprint.get('variant_08_slope', 0.0)
        
        # Secondary criterion: improvement ratio
        o_imp = fingerprint.get('original_improvement_ratio', 0.0)
        v_imp = fingerprint.get('variant_08_improvement_ratio', 0.0)
        
        # Tertiary: stagnation (lower = better)
        o_stag = fingerprint.get('original_stagnation', 1.0)
        v_stag = fingerprint.get('variant_08_stagnation', 1.0)
        
        # Rank-based composite score (all signals on same scale via rank)
        scores = {}
        for name in ['original', 'variant_08']:
            slope_key = f'{name}_slope'
            imp_key = f'{name}_improvement_ratio'
            stag_key = f'{name}_stagnation'
            
            # Normalise each signal to [0, 1] relative to the two strategies
            s_vals = [fingerprint.get(slope_key, 0.0), fingerprint.get(slope_key, 0.0)]
            i_vals = [fingerprint.get(imp_key, 0.0), fingerprint.get(imp_key, 0.0)]
            st_vals = [fingerprint.get(stag_key, 1.0), fingerprint.get(stag_key, 1.0)]
            
            # For slope and improvement: higher is better -> rank 1 = max
            # For stagnation: lower is better -> rank 1 = min
            s_range = max(abs(s_vals[0] - s_vals[1]), 1e-12)
            i_range = max(abs(i_vals[0] - i_vals[1]), 1e-12)
            st_range = max(abs(st_vals[0] - st_vals[1]), 1e-12)
            
            s_score = (fingerprint.get(slope_key, 0.0) - min(s_vals)) / s_range if s_range > 1e-12 else 0.5
            i_score = (fingerprint.get(imp_key, 0.0) - min(i_vals)) / i_range if i_range > 1e-12 else 0.5
            st_score = 1.0 - (fingerprint.get(stag_key, 1.0) - min(st_vals)) / st_range if st_range > 1e-12 else 0.5
            
            # Composite: weighted average (slope most important)
            scores[name] = 0.5 * s_score + 0.3 * i_score + 0.2 * st_score
        
        # Commit to the higher-scoring strategy
        if scores['original'] > scores['variant_08']:
            return 'original'
        elif scores['variant_08'] > scores['original']:
            return 'variant_08'
        else:
            # Inconclusive: default to original (23/24 win rate)
            return 'original'
    
    def _run_probe_phase(self, func, budget_fraction=0.05):
        """Run probe phase: short evaluation of both strategies on sub-population.
        Returns committed strategy key and landscape fingerprint."""
        n_total = self.max_iter
        n_probe_gens = max(3, int(n_total * budget_fraction / 2))
        probe_NP = max(10, self.NP // 2)
        
        # Create probe population (smaller for speed)
        low, high = -100.0, 100.0
        pop_orig = np.random.uniform(low, high, (probe_NP, self.dim))
        pop_v08 = pop_orig.copy() + np.random.randn(probe_NP, self.dim) * 2.0
        pop_orig = np.clip(pop_orig, low, high)
        pop_v08 = np.clip(pop_v08, low, high)
        
        probe_results = {}
        
        # Probe original strategy
        trace_o, final_o = self._probe_original(pop_orig, func, n_probe_gens)
        probe_results['original'] = (trace_o, final_o)
        
        # Probe variant_08 strategy
        trace_v, final_v = self._probe_variant_08(pop_v08, func, n_probe_gens)
        probe_results['variant_08'] = (trace_v, final_v)
        
        # Compute landscape fingerprint
        fingerprint = self._compute_landscape_fingerprint(probe_results)
        
        # Decide strategy
        committed = self._decide_strategy(fingerprint)
        
        return committed, fingerprint
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """One-to-one elitist selection."""
        better = trial_fitness < fitness
        new_pop = np.where(better[:, np.newaxis], trials, population)
        new_fit = np.where(better, trial_fitness, fitness)
        return new_pop, new_fit, better
    
    def _crossover_batch(self, population, mutants):
        """Binomial crossover with dimension-wise CR."""
        CR = np.clip(self.CR + np.random.randn(self.NP) * 0.1, 0.5, 0.98)
        mask = np.random.rand(self.NP, self.dim) < CR[:, np.newaxis]
        mask[np.arange(self.NP), np.random.randint(0, self.dim, self.NP)] = True
        trials = np.where(mask, mutants, population)
        return np.clip(trials, -100.0, 100.0)
    
    def _compute_diversity(self, population):
        """Population spread: average pairwise Euclidean distance."""
        centroid = population.mean(axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return distances.mean()
    
    def _update_archive(self, population, improved_mask):
        """Maintain archive of improved solutions for diversity injection."""
        improved_indices = np.where(improved_mask)[0]
        if len(improved_indices) > 0:
            improved_pop = population[improved_indices]
            self.archive.append(improved_pop)
            total_len = sum(len(a) for a in self.archive)
            while total_len > self.archive_max:
                oldest = self.archive.pop(0)
                total_len -= len(oldest)
    
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
    
    def _check_stagnation(self, best_fitness):
        """Detect stagnation and low diversity."""
        improved = self.prev_best_fitness - best_fitness > 1e-8
        if improved:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        self.prev_best_fitness = best_fitness
        return self.stagnation_counter > 50
    
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
        isolation_score = (1.0 - connectivity / max(connectivity.max(), 1e-12)) * 0.5 + \
                          (1.0 - clustering) * 0.3 + \
                          fitness_ranks * 0.2
        
        n_replace = max(NP // 4, 3)
        replace_indices = np.argsort(isolation_score)[-n_replace:]
        n_best = min(5, NP)
        best_indices = np.argsort(fitness)[:n_best]
        
        new_individuals = []
        for idx in replace_indices:
            best_source = best_indices[0]
            best_graph_dist = -1
            for b_idx in best_indices:
                n1 = set(nearest_indices[idx])
                n2 = set(nearest_indices[b_idx])
                common = len(n1 & n2)
                graph_dist_approx = 1.0 / (common + 0.1)
                if graph_dist_approx > best_graph_dist:
                    best_graph_dist = graph_dist_approx
                    best_source = b_idx
            
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
    
    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        iteration = 0
        
        # ============================================================
        # PHASE A: PROBE (5% of budget)
        # ============================================================
        if not self._probe_completed:
            try:
                committed, fingerprint = self._run_probe_phase(func, budget_fraction=0.05)
                self._committed_strategy = committed
                self._landscape_fingerprint = fingerprint
                self._probe_completed = True
            except Exception:
                # Fallback: default to original (23/24 wins)
                self._committed_strategy = 'original'
                self._probe_completed = True
                self._landscape_fingerprint = {}
        
        # ============================================================
        # PHASE B: COMMIT
        # ============================================================
        use_variant_08 = (self._committed_strategy == 'variant_08')
        
        while not stopping_condition():
            # Apply committed strategy exclusively (no blending)
            if use_variant_08:
                mutants = self._mutate_variant_08_batch(population, fitness)
            else:
                mutants = self._mutate_compass_batch(population, fitness)
            
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
            
            # Archive and diversity
            self._update_archive(population, improved_mask)
            diversity = self._compute_diversity(population)
            if diversity < self.diversity_threshold:
                population = self._inject_diversity(population, fitness)
                fitness = func(population)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            
            # Stagnation check — allow re-probe if strategy stagnates
            if self._check_stagnation(f_opt):
                if self._reprobe_allowed:
                    # Re-probe with larger budget
                    try:
                        committed, fingerprint = self._run_probe_phase(func, budget_fraction=0.10)
                        self._committed_strategy = committed
                        self._landscape_fingerprint = fingerprint
                        use_variant_08 = (self._committed_strategy == 'variant_08')
                        self._reprobe_allowed = False  # Only one re-probe per run
                    except Exception:
                        pass
                
                population, f_opt, x_opt = self._restart_if_needed(population, fitness)
                fitness = func(population)
                if len(fitness) < self.NP:
                    fitness = np.full(self.NP, np.nan)
            
            # Adapt F/CR based on improvement rate
            improvement_rate = np.mean(improved_mask)
            if improvement_rate > 0.15:
                self.F = np.clip(self.F * 1.05, 0.3, 2.0)
            elif improvement_rate < 0.03:
                self.F = np.clip(self.F * 1.12, 0.3, 2.0)
            self.F_history.append(self.F)
            self.CR_history.append(self.CR)
            
            iteration += 1
            if stopping_condition():
                break
        
        return f_opt, x_opt
