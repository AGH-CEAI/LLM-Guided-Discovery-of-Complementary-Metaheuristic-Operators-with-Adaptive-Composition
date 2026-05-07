```python
import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTATION MECHANISM: PROBE-AND-COMMIT (Contextual Bandit, Mechanism #4, specialized).
    
    The per-task gap table is BLEND-HOSTILE: 6 tasks have winner/runner-up gaps >= 100×,
    and 8 tasks have gaps >= 10×. A linear blend of values 1e-8 and 1e-2 yields 5e-3,
    destroying 500,000× of the winner's advantage. Ensemble blending is mathematically
    excluded. The dispatcher preserves per-task advantages by committing to ONE variant
    at a time (weight = 1.0 for the chosen arm, 0.0 for others).
    
    Strategy: Probe ~5-8% of budget with round-robin crossover variants, extract
    landscape fingerprints (convergence rate, diversity, separability proxy), then
    commit to the empirically best variant for the remaining budget.
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(max(kwargs.get('NP', 6 * dim), 4 * dim), min(10 * dim, 500))
        self.F = 0.6
        self.CR = 0.85
        self.max_iter = kwargs.get('max_iter', 20000)
        self.penalty_factor = kwargs.get('penalty', 1e9)
        
        # Strategy pool with softmax-converted success history
        self.strategy_names = ['rand', 'best', 'current_to_rand_best', 'local_ring', 'two_rail']
        self.strategy_scores = np.ones(len(self.strategy_names))
        self._strategy_funcs = [
            self._mutate_rand,
            self._mutate_best,
            self._mutate_current_to_rand_best,
            self._mutate_local_ring,
            self._mutate_two_rail,
        ]
        
        # Diversity and stagnation tracking
        self.archive = []
        self.archive_max = self.NP * 3
        self.stagnation_counter = 0
        self.prev_best_fitness = np.inf
        self.diversity_threshold = 1e-6
        
        # Adaptation momentum
        self.F_history = [self.F]
        self.CR_history = [self.CR]
        
        # ===== CROSSOVER VARIANT ADAPTATION =====
        # Four crossover strategies to probe
        self._crossover_variants = [
            self._crossover_original,
            self._crossover_variant_01_centroid,   # 4 wins: centroid-proximity weighting
            self._crossover_variant_07_lhs_beta,   # 2 wins: LHS-guided adaptive CR
            self._crossover_variant_02_svd,        # 1 win: SVD-guided spectral CR
        ]
        self._variant_names = ['original', 'centroid_weighted', 'lhs_beta_adaptive', 'svd_spectral']
        
        # Probe state
        self._probe_active = False
        self._probe_variant_idx = 0
        self._probe_budget = 0
        self._probe_total_budget = 0
        self._probe_generation = 0
        self._probe_variant_fitness = []   # List of (variant_idx, best_fitness) per probe gen
        self._probe_best_fitness_history = []  # Track best-so-far per probe generation
        
        # Commit state
        self._committed_variant = None
        self._committed_variant_name = None
        self._post_commit_stagnation = 0
        self._epsilon_revisit = 0.05  # Small probability to try runner-up on stagnation
        
        # Variant win tracking (empirical)
        self._variant_ranksums = np.zeros(len(self._crossover_variants))
        self._variant_counts = np.zeros(len(self._crossover_variants))
        
        # Landscape fingerprint (computed during probe)
        self._fingerprint = {}
        
        # Per-variant state for adaptive crossover variants
        # variant_07 state
        self._cr_success_history = []
        self._cr_alpha = 2.0
        self._cr_beta = 5.0
        
    def _crossover_original(self, population, mutants):
        """Standard binomial crossover (original implementation)."""
        NP, dim = population.shape
        CR = np.clip(self.CR + np.random.randn(NP) * 0.1, 0.5, 0.98)
        mask = np.random.rand(NP, dim) < CR[:, np.newaxis]
        mask[np.arange(NP), np.random.randint(0, dim, NP)] = True
        trials = np.where(mask, mutants, population)
        return np.clip(trials, -100.0, 100.0)
    
    def _crossover_variant_01_centroid(self, population, mutants):
        """Geometry/spatial crossover: weight by distance to population centroid."""
        NP, dim = population.shape
        
        # Compute population centroid
        centroid = population.mean(axis=0)
        
        # Compute per-parent centroid distances
        target_dists = np.linalg.norm(population - centroid, axis=1)
        mutant_dists = np.linalg.norm(mutants - centroid, axis=1)
        
        # Convert distances to proximity weights (closer = higher weight)
        eps = 1e-10
        target_weights = 1.0 / (target_dists + eps)
        mutant_weights = 1.0 / (mutant_dists + eps)
        
        # Normalize per pair
        total_weights = target_weights + mutant_weights
        target_weights /= total_weights
        mutant_weights /= total_weights
        
        # Per-dimension crossover mask (standard binomial)
        CR = getattr(self, 'CR', 0.85)
        j_rand = np.random.randint(dim)
        rand_mask = np.random.rand(NP, dim) < CR
        rand_mask[:, j_rand] = True
        
        # Base trial from binomial crossover
        trials = np.where(rand_mask, mutants, population)
        
        # Apply centroid-proximity weighting
        for i in range(NP):
            if target_weights[i] > mutant_weights[i]:
                blend = 0.25 * mutant_weights[i]
                trials[i] = trials[i] * (1 - blend) + population[i] * blend
            else:
                blend = 0.25 * target_weights[i]
                trials[i] = trials[i] * (1 - blend) + mutants[i] * blend
        
        return trials
    
    def _crossover_variant_07_lhs_beta(self, population, mutants):
        """LHS-guided binomial crossover with adaptive Monte Carlo CR sampling."""
        NP, dim = population.shape
        
        # Monte Carlo: sample CR from Beta distribution per individual
        # Adapt Beta parameters based on recent success
        if len(self._cr_success_history) >= 10:
            recent_success = np.mean(self._cr_success_history[-10:])
            target_mean = 0.5 + 0.3 * (recent_success - 0.1) / 0.9
            target_mean = np.clip(target_mean, 0.1, 0.9)
            self._cr_alpha = target_mean * 10
            self._cr_beta = (1 - target_mean) * 10
        
        # Sample CR per individual from Beta distribution
        CR_samples = np.random.beta(max(self._cr_alpha, 0.1), max(self._cr_beta, 0.1), size=NP)
        CR_samples = np.clip(CR_samples, 0.1, 0.95)
        
        # Generate LHS points for dimension selection
        lhs_base = (np.arange(NP) + np.random.rand(NP)) / NP
        np.random.shuffle(lhs_base)
        
        trials = np.empty_like(population)
        
        for i in range(NP):
            j_rand = np.random.randint(dim)
            cr = CR_samples[i]
            mask = np.random.rand(dim) < cr
            if not mask.any():
                mask[j_rand] = True
            trials[i] = np.where(mask, mutants[i], population[i])
        
        # Track CR success for adaptation (use diversity as proxy signal)
        trial_diversity = np.std(trials, axis=0).mean()
        pop_diversity = np.std(population, axis=0).mean()
        cr_success = trial_diversity / (pop_diversity + 1e-10) if pop_diversity > 1e-10 else 0.5
        cr_success = np.clip(cr_success, 0.0, 1.0)
        self._cr_success_history.append(cr_success)
        if len(self._cr_success_history) > 50:
            self._cr_success_history = self._cr_success_history[-50:]
        
        return trials
    
    def _crossover_variant_02_svd(self, population, mutants):
        """Binomial crossover with spectral-weighted per-dimension CR using SVD."""
        NP, dim = population.shape
        trials = np.empty_like(population)
        
        # Compute SVD of population matrix
        try:
            U, s, Vt = np.linalg.svd(population, full_matrices=False)
        except np.linalg.LinAlgError:
            mask = np.random.rand(NP, dim) < self.CR
            trials = np.where(mask, mutants, population)
            return trials
        
        # Compute effective dimensionality
        threshold = s[0] * 1e-6 if s[0] > 0 else 1e-12
        effective_dim = np.sum(s > threshold)
        effective_dim = max(1, min(effective_dim, dim))
        
        base_cr = getattr(self, 'CR', 0.85)
        
        # Compute spectral weights
        s_normalized = s / (s.sum() + 1e-12)
        
        if dim <= len(s_normalized):
            spectral_weights = s_normalized[:dim]
        else:
            spectral_weights = np.zeros(dim)
            spectral_weights[:len(s_normalized)] = s_normalized
            spectral_weights[len(s_normalized):] = s_normalized[-1] if len(s_normalized) > 0 else 0.0
        
        if spectral_weights.sum() > 0:
            spectral_weights = spectral_weights * dim / spectral_weights.sum()
        
        cr_per_dim = np.clip(base_cr * (0.5 + 0.5 * spectral_weights), 0.0, 0.99)
        
        cr_noise = np.random.uniform(-0.05, 0.05, dim)
        cr_per_dim_noisy = np.clip(cr_per_dim + cr_noise, 0.0, 0.99)
        
        mask = np.random.rand(NP, dim) < cr_per_dim_noisy
        
        force_cross = np.random.randint(0, dim, size=NP)
        force_mask = np.zeros((NP, dim), dtype=bool)
        force_mask[np.arange(NP), force_cross] = True
        mask = mask | force_mask
        
        trials = np.where(mask, mutants, population)
        return trials
    
    def _compute_fitness_ranking(self, fitness):
        order = np.argsort(fitness)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(self.NP)
        return ranks / (self.NP - 1) if self.NP > 1 else np.zeros_like(fitness)
    
    def _compass_directions(self, population, idx, fitness_ranks):
        """Sample 3 candidate mutation directions, score by rank improvement potential."""
        i = idx
        others = np.concatenate([np.arange(i), np.arange(i + 1, self.NP)])
        np.random.shuffle(others)
        r1, r2, r3, r4 = others[:4]
        
        i_rank = fitness_ranks[i]
        
        candidates = []
        dA = population[r1] + self.F * (population[r2] - population[r3])
        score_A = i_rank - min(fitness_ranks[[r1, r2, r3]].min(), i_rank + 0.1)
        
        best_idx = np.argmin(fitness_ranks)
        dB = population[best_idx] + self.F * (population[r1] - population[r2])
        score_B = i_rank - fitness_ranks[best_idx]
        
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
        """Vectorized compass mutation."""
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
        """Softmax-weighted strategy selection."""
        probs = np.exp(self.strategy_scores - self.strategy_scores.max())
        probs /= probs.sum()
        return np.random.choice(len(self._strategy_funcs), p=probs)
    
    def _mutate_ensemble_batch(self, population, fitness):
        """Use adaptive strategy pool to generate mutants."""
        mutants = np.empty_like(population)
        strategy_used = np.empty(self.NP, dtype=int)
        
        for i in range(self.NP):
            strat_idx = self._select_strategy_batch()
            strategy_used[i] = strat_idx
            mutants[i] = self._strategy_funcs[strat_idx](population, i, self.F)
        
        return mutants, strategy_used
    
    def _crossover_batch(self, population, mutants):
        """Dispatch to the committed (or currently probed) crossover variant."""
        if self._probe_active:
            # During probe, use round-robin variant
            variant_idx = self._probe_variant_idx
            variant_name = self._variant_names[variant_idx]
        elif self._committed_variant is not None:
            # Commit to the best variant from probe
            variant_idx = self._committed_variant
            variant_name = self._committed_variant_name
        else:
            # Fallback to original
            variant_idx = 0
            variant_name = 'original'
        
        return self._crossover_variants[variant_idx](population, mutants)
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """One-to-one elitist selection."""
        better = trial_fitness < fitness
        new_pop = np.where(better[:, np.newaxis], trials, population)
        new_fit = np.where(better, trial_fitness, fitness)
        return new_pop, new_fit, better
    
    def _update_strategy_scores(self, strategy_used, improved):
        """Hybrid credit: rank improvement + success rate."""
        n_strategies = len(self.strategy_scores)
        n = len(strategy_used)

        if not hasattr(self, '_ema_rank_improvement'):
            self._ema_rank_improvement = np.zeros(n_strategies)
            self._ema_success_rate = np.zeros(n_strategies)
            self._rank_improvement_history = [[] for _ in range(n_strategies)]
            self._success_rate_history = [[] for _ in range(n_strategies)]
            self._momentum = 0.3

        rank_improvement = np.zeros(n_strategies)
        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() > 0:
                rank_improvement[s] = improved[mask].sum() / mask.sum()

        success_rate = np.zeros(n_strategies)
        for s in range(n_strategies):
            mask = strategy_used == s
            if mask.sum() > 0:
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
    
    def _adapt_parameters(self, improvement_rate):
        """Adapt F and CR based on k-NN graph topology of the population."""
        if not hasattr(self, '_graph_population'):
            self._graph_population = None
            self._graph_metrics_history = []

        if not hasattr(self, '_current_population'):
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
        graph_diameter = np.max(np.sqrt(nearest_sq_dists))

        current_metrics = {
            'avg_edge_length': avg_edge_length,
            'avg_clustering': avg_clustering,
            'graph_diameter': graph_diameter
        }
        self._graph_metrics_history.append(current_metrics)

        if len(self._graph_metrics_history) > 20:
            self._graph_metrics_history.pop(0)

        if len(self._graph_metrics_history) >= 3:
            recent_edge_lengths = [m['avg_edge_length'] for m in self._graph_metrics_history[-3:]]
            recent_clustering = [m['avg_clustering'] for m in self._graph_metrics_history[-3:]]
            edge_length_trend = recent_edge_lengths[-1] - recent_edge_lengths[0]
            clustering_trend = recent_clustering[-1] - recent_clustering[0]
        else:
            edge_length_trend = 0.0
            clustering_trend = 0.0

        clustering_factor = 1.0 + 0.15 * (avg_clustering - 0.3) + 0.1 * (edge_length_trend < -0.01)
        clustering_factor += 0.08 * (clustering_trend > 0.05)
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
    
    def _compute_diversity(self, population):
        """Population spread: average pairwise Euclidean distance."""
        centroid = population.mean(axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return distances.mean()
    
    def _update_archive(self, population, fitness, improved_mask):
        """Maintain archive of improved solutions for diversity injection."""
        improved_pop = population[improved_mask]
        if len(improved_pop) > 0:
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
    
    def _compute_landscape_fingerprint(self, pop, fitness, gen):
        """Extract cheap landscape features from current population state."""
        fp = {}
        NP, dim = pop.shape
        
        # 1. Convergence rate (slope of log-best over recent generations)
        self._probe_best_fitness_history.append(np.min(fitness))
        if len(self._probe_best_fitness_history) >= 3:
            recent = np.array(self._probe_best_fitness_history[-5:])
            valid = recent > 0
            if np.sum(valid) >= 2:
                log_recent = np.log(recent[valid] + 1e-15)
                gen_indices = np.arange(len(recent))[valid]
                if len(gen_indices) >= 2:
                    fp['convergence_slope'] = np.polyfit(gen_indices, log_recent, 1)[0] if len(gen_indices) >= 2 else 0.0
                else:
                    fp['convergence_slope'] = 0.0
            else:
                fp['convergence_slope'] = 0.0
        else:
            fp['convergence_slope'] = 0.0
        
        # 2. Fitness rank entropy (higher = more diverse fitness landscape)
        ranks = self._compute_fitness_ranking(fitness)
        hist, _ = np.histogram(ranks, bins=min(10, NP), range=(0, 1))
        hist = hist / (NP + 1e-10)
        hist = hist[hist > 0]
        fp['rank_entropy'] = -np.sum(hist * np.log(hist + 1e-10))
        
        # 3. Effective dimensionality from covariance (fraction of eigenvalues > 95% variance)
        if NP > dim:
            cov = np.cov(pop.T)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]
            eigvals = np.maximum(eigvals, 0)
            total_var = np.sum(eigvals)
            if total_var > 1e-10:
                cumvar = np.cumsum(eigvals) / total_var
                fp['effective_dim_frac'] = np.searchsorted(cumvar, 0.95) / len(eigvals)
            else:
                fp['effective_dim_frac'] = 1.0
        else:
            fp['effective_dim_frac'] = 1.0
        
        # 4. Population diversity (average distance to centroid)
        centroid = pop.mean(axis=0)
        dists = np.linalg.norm(pop - centroid, axis=1)
        fp['pop_diversity'] = np.mean(dists)
        fp['pop_diversity_std'] = np.std(dists)
        
        # 5. Separability proxy: axis-aligned improvement vs diagonal improvement
        # Small perturbation in each coordinate separately
        if gen < 3 and dim <= 20:
            best_idx = np.argmin(fitness)
            x_best = pop[best_idx].copy()
            axis_imp = 0
            diag_imp = 0
            for d in range(min(dim, 10)):
                step = np.zeros(dim)
                step[d] = 0.1 * (np.random.choice([1, -1]))
                f_plus = func_single(x_best + step) if 'func_single' in dir() else None
                if f_plus is not None and f_plus < fitness[best_idx]:
                    axis_imp += 1
            fp['separability_proxy'] = axis_imp / min(dim, 10)
        else:
            fp['separability_proxy'] = 0.5
        
        # 6. Stagnation index (fraction of recent gens with no improvement)
        if len(self._probe_best_fitness_history) >= 5:
            recent_best = np.array(self._probe_best_fitness_history[-5:])
            improvements = np.diff(recent_best)
            fp['stagnation_frac'] = np.mean(improvements >= 0)
        else:
            fp['stagnation_frac'] = 0.0
        
        return fp
    
    def _decide_committed_variant(self):
        """Decide which crossover variant to commit to based on probe results.
        
        Uses rank-based scoring to avoid scale-dependent magic constants.
        """
        n_variants = len(self._crossover_variants)
        
        if len(self._probe_variant_fitness) == 0:
            # No probe data: default to original
            self._committed_variant = 0
            self._committed_variant_name = 'original'
            return
        
        # Collect all probe fitness values for rank normalization
        all_fitness_values = [entry[1] for entry in self._probe_variant_fitness]
        
        # For each variant, compute average rank (lower rank = better fitness)
        variant_best_fitness = {i: [] for i in range(n_variants)}
        for variant_idx, best_fit in self._probe_variant_fitness:
            variant_best_fitness[variant_idx].append(best_fit)
        
        # Compute mean best fitness per variant
        variant_mean_best = np.zeros(n_variants)
        variant_count = np.zeros(n_variants)
        for v in range(n_variants):
            if len(variant_best_fitness[v]) > 0:
                variant_mean_best[v] = np.mean(variant_best_fitness[v])
                variant_count[v] = len(variant_best_fitness[v])
        
        # Rank-based scoring: rank variants by mean best fitness
        valid_mask = variant_count > 0
        if np.sum(valid_mask) == 0:
            self._committed_variant = 0
            self._committed_variant_name = 'original'
            return
        
        # Rank the variants (1 = best, higher = worse)
        rank_scores = np.zeros(n_variants)
        sorted_indices = np.argsort(variant_mean_best[valid_mask])
        for rank, v_idx in enumerate(sorted_indices):
            actual_v = np.where(valid_mask)[0][rank]
            rank_scores[actual_v] = rank + 1
        
        # Also consider: convergence slope (more negative = faster convergence)
        # and final diversity maintained
        # Normalize rank scores to [0, 1] where 0 = best
        max_rank = len(self._crossover_variants)
        rank_scores_norm = rank_scores / max_rank
        
        # Weighted composite: primarily rank-based, with small bonus for more samples
        sample_bonus = np.clip(variant_count / (np.max(variant_count) + 1e-10), 0, 1) * 0.1
        composite_scores = rank_scores_norm - sample_bonus
        
        # Pick the variant with lowest composite score
        winner = np.argmin(composite_scores)
        
        # Verify the winner has enough samples
        if variant_count[winner] < 2:
            # Not enough data: default to original
            self._committed_variant = 0
            self._committed_variant_name = 'original'
        else:
            self._committed_variant = int(winner)
            self._committed_variant_name = self._variant_names[winner]
        
        # Store fingerprint for potential rule-based override
        if len(self._probe_best_fitness_history) >= 3:
            recent_best = np.array(self._probe_best_fitness_history[-3:])
            if np.all(recent_best < recent_best[0] * 0.9):
                # Rapid convergence: centroid-weighted or original tends to do well
                pass  # Keep the rank-based decision
            elif np.std(recent_best) / (np.mean(recent_best) + 1e-10) < 0.01:
                # Flat fitness landscape: SVD-guided might help
                pass  # Keep the rank-based decision
    
    def _initialize_population(self):
        low, high = -100.0, 100.0
        pop = np.random.uniform(low, high, (self.NP, self.dim))
        pop += np.random.randn(self.NP, self.dim) * 5.0
        return np.clip(pop, low, high)
    
    def __call__(self, func, stopping_condition):
        # Initialize probe budget: ~6% of max_iter, minimum 5 generations per variant
        probe_min_gens = 5
        n_variants = len(self._crossover_variants)
        self._probe_total_budget = max(n_variants * probe_min_gens, int(self.max_iter * 0.06))
        self._probe_budget = 0
        self._probe_active = True
        self._probe_variant_idx = 0
        self._probe_generation = 0
        self._probe_variant_fitness = []
        self._probe_best_fitness_history = []
        
        # Initialize population
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        iteration = 0
        
        while not stopping_condition():
            # Store current population for adaptation methods
            self._current_population = population
            
            # --- PROBE PHASE ---
            if self._probe_active:
                # Use round-robin: cycle through variants each generation
                current_variant = self._probe_variant_idx
                
                # Compass mutation
                mutants_compass = self._mutate_compass_batch(population, fitness)
                mutants_ensemble, strategy_used = self._mutate_ensemble_batch(population, fitness)
                blend_ratio = np.random.rand(self.NP, 1)
                mutants = np.where(blend_ratio < 0.6, mutants_compass, mutants_ensemble)
                
                # Crossover using current probe variant
                trials = self._crossover_variants[current_variant](population, mutants)
                
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
                
                # Track probe results
                current_best_idx = np.nanargmin(fitness)
                current_best_fitness = fitness[current_best_idx]
                self._probe_variant_fitness.append((current_variant, current_best_fitness))
                
                if current_best_fitness < f_opt:
                    f_opt = current_best_fitness
                    x_opt = population[current_best_idx].copy()
                
                # Update best fitness history
                self._probe_best_fitness_history.append(current_best_fitness)
                
                # Advance probe state
                self._probe_generation += 1
                self._probe_budget += 1
                self._probe_variant_idx = (self._probe_variant_idx + 1) % n_variants
                
                # Check if probe phase is complete
                if self._probe_budget >= self._probe_total_budget:
                    # Compute landscape fingerprint
                    self._fingerprint = self._compute_landscape_fingerprint(
                        population, fitness, self._probe_generation
                    )
                    
                    # Decide which variant to commit to
                    self._decide_committed_variant()
                    
                    # Exit probe phase
                    self._probe_active = False
                    self._post_commit_stagnation = 0
                    
                    # Strategy score update (only during probe)
                    self._update_strategy_scores(strategy_used, improved_mask)
                    improvement_rate = np.mean(improved_mask)
                    self._adapt_parameters(improvement_rate)
                
                iteration += 1
                if stopping_condition():
                    break
                continue
            
            # --- COMMIT PHASE ---
            # Compass mutation
            mutants_compass = self._mutate_compass_batch(population, fitness)
            mutants_ensemble, strategy_used = self._mutate_ensemble_batch(population, fitness)
            blend_ratio = np.random.rand(self.NP, 1)
            mutants = np.where(blend_ratio < 0.6, mutants_compass, mutants_ensemble)
            
            # Crossover using committed variant (with small epsilon-revisit probability)
            if self._committed_variant is not None and np.random.rand() > self._epsilon_revisit:
                trials = self._crossover_variants[self._committed_variant](population, mutants)
            else:
                # Small probability to try a different variant during stagnation
                alt_variant = np.random.randint(0, len(self._crossover_variants))
                trials = self._crossover_variants[alt_variant](population, mutants)
            
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
                self._post_commit_stagnation = 0
            else:
                self._post_commit_stagnation += 1
            
            # Strategy score update
            self._update_strategy_scores(strategy_used, improved_mask)
            
            # Archive and parameter adaptation
            self._update_archive(population, fitness, improved_mask)
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
```