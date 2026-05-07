```python
import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTIVE MECHANISM: Probe-and-Commit (mechanism #4, contextual bandit with probe).
    
    Per-task gap analysis reveals BLEND-HOSTILE landscape:
    - 8 tasks have winner/runner-up gap >= 100× (dispatch-mandatory)
    - Worst gap: task 5 = 25,276,021× (original beats variant_05 by that margin)
    - A 50/50 blend of 1e-8 and 1e4 = 5e3, which is 500,000× worse than the winner
    - Two distinct winners: original.py (23 tasks) and variant_08_catH (1 task)
    
    The probe-and-commit pattern preserves per-task advantages:
    1. PHASE A (PROBE ~5%): Run both operators on a small sub-population.
       Extract rank-based reward (scale-free). Update Beta priors via Thompson sampling.
    2. PHASE B (COMMIT ~95%): Sample from Beta posteriors, commit to ONE arm
       with weight 1.0. Linear blending is mathematically FORBIDDEN because
       it cannot preserve 100× advantages (rule f from calibration).
    
    Why this fits: Different tasks have dramatically different winners (23 vs 1).
    A bandit that commits to ONE operator per-task is the only mechanism that
    can preserve 100× advantages. Rank-based Thompson sampling avoids scale-
    dependent magic constants (rules a, b from calibration).
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
        
        # Probe-and-commit state (mechanism #4)
        self._probe_alpha = np.ones(2)   # Beta prior: arm 0 = original compass, arm 1 = variant_08
        self._probe_beta = np.ones(2)
        self._committed_arm = None       # Set after probe
        self._probe_complete = False
        self._arm_wins = np.zeros(2)     # Track wins per arm during probe
        self._arm_counts = np.zeros(2)   # Track samples per arm
    
    def _initialize_population(self):
        low, high = -100.0, 100.0
        pop = np.random.uniform(low, high, (self.NP, self.dim))
        pop += np.random.randn(self.NP, self.dim) * 5.0
        return np.clip(pop, low, high)
    
    def _compute_fitness_ranking(self, fitness):
        order = np.argsort(fitness)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(fitness))
        return ranks / max(len(fitness) - 1, 1)
    
    def _mutate_compass_batch(self, population, fitness):
        """Original compass mutation (arm 0): samples 3 candidate directions per individual."""
        fitness_ranks = self._compute_fitness_ranking(fitness)
        mutants = np.empty_like(population)
        n = len(population)
        
        for i in range(n):
            i_rank = fitness_ranks[i]
            others = np.concatenate([np.arange(i), np.arange(i + 1, n)])
            np.random.shuffle(others)
            r1, r2, r3, r4 = others[:4]
            
            # DE/rand/1
            dA = population[r1] + self.F * (population[r2] - population[r3])
            score_A = i_rank - min(fitness_ranks[r1], fitness_ranks[r2], fitness_ranks[r3])
            
            # DE/best/1
            best_idx = np.argmin(fitness_ranks)
            dB = population[best_idx] + self.F * (population[r1] - population[r2])
            score_B = i_rank - fitness_ranks[best_idx]
            
            # Current-to-pbest
            p = 0.1
            top_p = max(1, int(np.ceil(p * n)))
            pbest_candidates = np.argsort(fitness_ranks)[:top_p]
            pbest = population[np.random.choice(pbest_candidates)]
            dC = population[i] + self.F * (pbest - population[r1]) + self.F * (population[r3] - population[r4])
            score_C = i_rank - fitness_ranks[pbest_candidates].min()
            
            best_dir, _ = max([(dA, score_A), (dB, score_B), (dC, score_C)], key=lambda x: x[1])
            mutants[i] = best_dir
        
        return mutants
    
    def _mutate_variant_08(self, population, fitness):
        """Variant_08 (arm 1): FDC-based hybrid mutation with fitness-rank landscape analysis."""
        NP, dim = population.shape
        ranks = self._compute_fitness_ranking(fitness)
        best_idx = np.argmin(fitness)
        
        # Initialize temporal tracking (F: temporal/dynamical signal)
        if not hasattr(self, '_fdc_history'):
            self._fdc_history = []
            self._centroid_history = []
            self._fdc_ema = 0.5
        
        # Compute fitness-distance correlation (D: fitness-landscape signal)
        centroid = population.mean(axis=0)
        dists_to_centroid = np.linalg.norm(population - centroid, axis=1)
        fdc = 0.0
        if dists_to_centroid.std() > 1e-10:
            corr = np.corrcoef(ranks, dists_to_centroid)[0, 1]
            if not np.isnan(corr):
                fdc = corr
        
        # Update EMA of FDC for temporal stability
        self._fdc_ema = 0.7 * self._fdc_ema + 0.3 * fdc
        self._fdc_history.append(self._fdc_ema)
        if len(self._fdc_history) > 15:
            self._fdc_history.pop(0)
        
        # Compute centroid velocity
        self._centroid_history.append(centroid.copy())
        if len(self._centroid_history) > 5:
            self._centroid_history.pop(0)
        
        centroid_velocity = 0.0
        if len(self._centroid_history) >= 2:
            centroid_velocity = np.linalg.norm(self._centroid_history[-1] - self._centroid_history[0])
        
        # Data-driven switching: FDC > threshold means good landscape -> exploit
        fdc_threshold = 0.3
        exploit_weight = np.clip((self._fdc_ema - fdc_threshold) / (1.0 - fdc_threshold + 1e-10), 0.1, 0.9)
        explore_weight = 1.0 - exploit_weight
        
        # Mechanism A: Spatial centroid direction (A: geometric/spatial)
        direction_to_centroid = centroid - population
        centroid_mutations = population + explore_weight * self.F * direction_to_centroid
        
        # Mechanism B: Fitness-rank based direction (D: fitness-landscape)
        direction_to_best = population[best_idx] - population
        best_mutations = population + exploit_weight * self.F * direction_to_best
        
        mutants = np.empty_like(population)
        
        for i in range(NP):
            others = np.concatenate([np.arange(i), np.arange(i + 1, self.NP)])
            np.random.shuffle(others)
            r1, r2, r3, r4 = others[:4]
            
            cand_A = centroid_mutations[i]
            score_A = ranks[i] - np.mean([ranks[j] for j in [r1, r2, r3]])
            
            cand_B = best_mutations[i]
            score_B = ranks[i] - ranks[best_idx]
            
            # Mechanism C: Archive-based (diversity injection)
            cand_C = None
            score_C = -np.inf
            if hasattr(self, 'archive') and len(self.archive) > 0:
                archive_concat = np.vstack(self.archive)
                if len(archive_concat) > 0:
                    archive_idx = np.random.randint(len(archive_concat))
                    archive_donor = archive_concat[archive_idx]
                    pbest_archive = population[np.random.choice(np.argsort(fitness)[:max(1, NP // 10)])]
                    cand_C = population[i] + self.F * (pbest_archive - population[r1]) + self.F * (population[r2] - population[r3])
                    score_C = ranks[i] - np.mean([ranks[j] for j in [r1, r2]])
            
            # Mechanism D: Random baseline
            cand_D = population[r1] + self.F * (population[r2] - population[r3])
            score_D = ranks[i] - np.mean([ranks[r1], ranks[r2], ranks[r3]])
            
            candidates = [(cand_A, score_A), (cand_B, score_B), (cand_D, score_D)]
            if cand_C is not None:
                candidates.append((cand_C, score_C))
            
            best_cand, _ = max(candidates, key=lambda x: x[1])
            
            # Noise proportional to centroid velocity
            noise_scale = 0.01 + 0.1 * centroid_velocity / (np.linalg.norm(centroid) + 1e-10)
            noise = np.random.randn(dim) * noise_scale
            mutants[i] = np.clip(best_cand + noise, -100.0, 100.0)
        
        return mutants
    
    def _crossover_batch(self, population, mutants):
        """Binomial crossover."""
        CR = np.clip(self.CR + np.random.randn(self.NP) * 0.1, 0.5, 0.98)
        mask = np.random.rand(self.NP, self.dim) < CR[:, np.newaxis]
        mask[np.arange(self.NP), np.random.randint(0, self.dim, self.NP)] = True
        trials = np.where(mask, mutants, population)
        return np.clip(trials, -100.0, 100.0)
    
    def _run_probe(self, func, stopping_condition):
        """PHASE A: Probe both arms with small sub-population. Rank-based Thompson sampling."""
        probe_NP = max(10, self.NP // 4)
        probe_iters = min(50, max(5, int(0.05 * self.max_iter)))
        
        probe_pop = np.random.uniform(-100.0, 100.0, (probe_NP, self.dim))
        probe_fit = func(probe_pop)
        if len(probe_fit) < probe_NP:
            probe_fit = np.full(probe_NP, np.nan)
        
        for p_iter in range(probe_iters):
            if stopping_condition():
                return
            
            # Alternate arms each probe iteration (clean comparison)
            arm = p_iter % 2
            
            # Apply committed arm's mutation
            if arm == 0:
                mutants = self._mutate_compass_batch(probe_pop, probe_fit)
            else:
                mutants = self._mutate_variant_08(probe_pop, probe_fit)
            
            # Crossover
            CR_p = np.clip(self.CR + np.random.randn(probe_NP) * 0.1, 0.5, 0.98)
            mask = np.random.rand(probe_NP, self.dim) < CR_p[:, np.newaxis]
            mask[np.arange(probe_NP), np.random.randint(0, self.dim, probe_NP)] = True
            trials = np.clip(np.where(mask, mutants, probe_pop), -100.0, 100.0)
            
            trial_fit = func(trials)
            if len(trial_fit) < probe_NP:
                trial_fit = np.full(probe_NP, np.nan)
            
            # Rank-based reward (scale-free, rule a)
            # Reward = normalized rank improvement of best trial vs population median
            pop_ranks = self._compute_fitness_ranking(probe_fit)
            trial_ranks = self._compute_fitness_ranking(trial_fit)
            
            pop_best_rank = np.nanmin(pop_ranks)
            trial_best_rank = np.nanmin(trial_ranks)
            
            # Rank improvement: positive if trials improved the best rank
            rank_improvement = pop_best_rank - trial_best_rank
            
            # Normalize to [0,1] using rank range (scale-free)
            reward = np.clip(rank_improvement * probe_NP + 0.5, 0.0, 1.0)
            
            # Update Beta posterior via Thompson sampling
            self._probe_alpha[arm] += reward
            self._probe_beta[arm] += (1.0 - reward)
            
            # Track wins for diagnostics
            if trial_best_rank < pop_best_rank - 1e-9:
                self._arm_wins[arm] += 1
            self._arm_counts[arm] += 1
            
            # Selection
            better = trial_fit < probe_fit
            probe_pop = np.where(better[:, np.newaxis], trials, probe_pop)
            probe_fit = np.where(better, trial_fit, probe_fit)
    
    def _commit_arm(self):
        """PHASE B: Thompson sample from Beta posteriors and commit to one arm."""
        # Thompson sampling: draw from Beta posteriors
        try:
            sample0 = np.random.beta(max(self._probe_alpha[0], 1e-6), max(self._probe_beta[0], 1e-6))
            sample1 = np.random.beta(max(self._probe_alpha[1], 1e-6), max(self._probe_beta[1], 1e-6))
        except Exception:
            sample0, sample1 = 1.0, 0.0
        
        self._committed_arm = 0 if sample0 >= sample1 else 1
        self._probe_complete = True
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """One-to-one elitist selection."""
        better = trial_fitness < fitness
        new_pop = np.where(better[:, np.newaxis], trials, population)
        new_fit = np.where(better, trial_fitness, fitness)
        return new_pop, new_fit, better
    
    def _update_strategy_scores(self, strategy_used, improved):
        """Hybrid credit: rank improvement + success rate, weighted by signal confidence."""
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

        alpha_param = 0.3
        for s in range(n_strategies):
            self._ema_rank_improvement[s] = (1 - alpha_param) * self._ema_rank_improvement[s] + alpha_param * rank_improvement[s]
            self._ema_success_rate[s] = (1 - alpha_param) * self._ema_success_rate[s] + alpha_param * success_rate[s]

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

        if not hasattr(self, '_graph_metrics_history'):
            self._graph_metrics_history = []

        current_metrics = {
            'avg_edge_length': avg_edge_length,
            'avg_clustering': avg_clustering,
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
        isolation_score = (1.0 - connectivity / max(connectivity.max(), 1e-10)) * 0.5 + \
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

        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        self.stagnation_counter = 0

        return population, f_opt, x_opt
    
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
        r = others[np.random.choice(len(others), 4, replace