```python
import numpy as np


class AdaptiveCompassDE:
    """
    ADAPTATION MECHANISM: Thompson Sampling Bandit with Rank-Based Rewards (mechanism #1)
    
    Rationale: The per-task gap table shows NO blend-hostile tasks (all gaps < 5× for
    winner vs runner-up). Four distinct variants win across 21 tasks, with variant_05
    winning 10 tasks and variant_01 winning 6. A bandit can learn which restart
    strategy works best for the current problem's landscape. Rank-based rewards avoid
    scale-dependent magic constants. A sliding window of size 3*num_arms ensures each
    arm is sampled enough times for reliable estimation.
    
    Key design decisions:
    - 4 arms: original, variant_01 (geometric clustering), variant_05 (k-NN connectivity),
      variant_07 (bootstrap covariance)
    - Thompson Sampling with Beta priors for binary success/failure
    - Rank-normalized reward within sliding window to avoid scale dependence
    - Warm start with uniform prior (no strong prior assumption)
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
        
        # ============================================================
        # RESTART BANDIT: Thompson Sampling with rank-based rewards
        # ============================================================
        # 4 arms: original, variant_01, variant_05, variant_07
        self._restart_arm_names = ['original', 'variant_01', 'variant_05', 'variant_07']
        self._num_restart_arms = len(self._restart_arm_names)
        
        # Beta prior: alpha = successes + 1, beta = failures + 1
        # Start with uniform prior (alpha=1, beta=1)
        self._restart_alpha = np.ones(self._num_restart_arms)
        self._restart_beta = np.ones(self._num_restart_arms)
        
        # Sliding window for rank-normalized rewards
        self._window_size = max(12, 3 * self._num_restart_arms)  # 12 for 4 arms
        self._restart_reward_history = [[] for _ in range(self._num_restart_arms)]
        self._restart_generation = 0
        self._restart_uses = np.zeros(self._num_restart_arms, dtype=int)
        
        # Track restart quality: improvement ratio after each restart
        self._restart_best_before = []
        self._restart_best_after = []
        
        # Current selected arm (for credit assignment)
        self._current_restart_arm = None
        
        # Small epsilon to avoid division by zero in rank normalization
        self._eps = 1e-10
        
    def _select_restart_arm_thompson(self):
        """Thompson Sampling: sample from Beta posterior, pick arm with highest sample."""
        samples = np.array([
            np.random.beta(self._restart_alpha[i], self._restart_beta[i])
            for i in range(self._num_restart_arms)
        ])
        selected = int(np.argmax(samples))
        self._current_restart_arm = selected
        self._restart_uses[selected] += 1
        return selected
    
    def _compute_rank_reward(self, arm_idx):
        """
        Compute rank-normalized reward for an arm within its sliding window.
        Returns percentile rank (0.0 to 1.0) of the most recent reward
        relative to all rewards in the window.
        
        If window has < 3 entries, use a conservative estimate based on
        the arm's empirical success rate vs population mean.
        """
        history = self._restart_reward_history[arm_idx]
        
        if len(history) < 3:
            # Conservative: use empirical success rate
            total_success = self._restart_alpha[arm_idx] - 1
            total_trials = self._restart_alpha[arm_idx] + self._restart_beta[arm_idx] - 2
            if total_trials > 0:
                return total_success / total_trials
            return 0.5  # Uninformative prior
        
        # Rank-normalized reward: percentile of latest reward in window
        latest = history[-1]
        rank = sum(1 for r in history if r <= latest) / len(history)
        return rank
    
    def _update_restart_bandit(self, arm_idx, fitness_before, fitness_after, generation):
        """
        Update bandit with rank-normalized reward based on restart improvement.
        
        Reward signal: relative improvement ratio, rank-normalized within window.
        - fitness_before: best fitness before restart
        - fitness_after: best fitness after restart (could be same if no improvement yet)
        - generation: current generation number
        
        Uses rank-based credit to avoid scale-dependent constants.
        """
        # Compute improvement ratio (avoid division by near-zero)
        denom = max(abs(fitness_before), 1e-10, abs(fitness_after))
        if fitness_before > fitness_after:
            # Improvement: positive reward
            improvement_ratio = (fitness_before - fitness_after) / denom
        else:
            # No improvement or worse: negative/small reward
            improvement_ratio = -0.1 * (fitness_after - fitness_before) / denom
        
        # Store in sliding window
        self._restart_reward_history[arm_idx].append(improvement_ratio)
        
        # Maintain window size
        if len(self._restart_reward_history[arm_idx]) > self._window_size:
            self._restart_reward_history[arm_idx].pop(0)
        
        # Compute rank-normalized reward
        rank_reward = self._compute_rank_reward(arm_idx)
        
        # Update Beta posterior based on rank reward
        # Binary success/failure based on whether rank_reward > 0.5
        is_success = rank_reward > 0.5
        
        if is_success:
            self._restart_alpha[arm_idx] += 1.0
        else:
            self._restart_beta[arm_idx] += 1.0
        
        # Decay old evidence slightly to allow adaptation to regime changes
        decay_factor = 0.99
        self._restart_alpha[arm_idx] *= decay_factor
        self._restart_beta[arm_idx] *= decay_factor
        
        # Ensure minimum for numerical stability
        self._restart_alpha[arm_idx] = max(self._restart_alpha[arm_idx], 0.5)
        self._restart_beta[arm_idx] = max(self._restart_beta[arm_idx], 0.5)
    
    def _restart_if_needed(self, population, fitness):
        """
        Adaptive restart: selected by Thompson Sampling bandit.
        
        Selects among 4 restart strategies based on learned posterior:
        0: original (full restart with best preserved)
        1: variant_01 (convex hull + geometric clustering)
        2: variant_05 (k-NN graph connectivity analysis)
        3: variant_07 (stochastic bootstrap covariance)
        """
        # Record best before restart
        best_idx = np.argmin(fitness)
        best_fitness_before = fitness[best_idx]
        best_solution_before = population[best_idx].copy()
        
        # Select restart strategy via Thompson Sampling
        arm_idx = self._select_restart_arm_thompson()
        arm_name = self._restart_arm_names[arm_idx]
        
        # Execute selected restart strategy
        if arm_idx == 0:
            # Original restart
            new_pop, best_fitness_after, best_solution_after = self._restart_original(population, fitness)
        elif arm_idx == 1:
            # variant_01: convex hull + geometric clustering
            new_pop, best_fitness_after, best_solution_after = self._restart_variant_01(population, fitness)
        elif arm_idx == 2:
            # variant_05: k-NN graph connectivity
            new_pop, best_fitness_after, best_solution_after = self._restart_variant_05(population, fitness)
        else:
            # variant_07: stochastic bootstrap
            new_pop, best_fitness_after, best_solution_after = self._restart_variant_07(population, fitness)
        
        # Update bandit with rank-normalized reward
        self._update_restart_bandit(arm_idx, best_fitness_before, best_fitness_after, self._restart_generation)
        self._restart_generation += 1
        
        # Reset stagnation counter
        self.stagnation_counter = 0
        
        return new_pop, best_fitness_after, best_solution_after
    
    def _restart_original(self, population, fitness):
        """Full restart with diversity injection (original behavior)."""
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        new_pop = self._initialize_population()
        new_pop[0] = best_solution
        self.archive = []
        
        return new_pop, best_fitness, best_solution
    
    def _restart_variant_01(self, population, fitness):
        """Restart using convex hull and geometric clustering."""
        NP, dim = population.shape
        
        centroid = population.mean(axis=0)
        centroid_dists = np.linalg.norm(population - centroid, axis=1)
        
        diffs = population[:, np.newaxis, :] - population[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        
        n_extreme = max(2, NP // 4)
        extreme_mask = np.zeros(NP, dtype=bool)
        sorted_by_dist = np.argsort(centroid_dists)[::-1]
        extreme_mask[sorted_by_dist[:n_extreme]] = True
        extreme_indices = np.where(extreme_mask)[0]
        
        k = max(3, min(8, NP // 5))
        nearest = np.argsort(sq_dists, axis=1)[:, :k]
        
        median_dist = np.median(np.sqrt(sq_dists[sq_dists != np.inf]))
        local_density = np.sum(np.sqrt(sq_dists) < median_dist, axis=1)
        
        n_clusters = max(2, NP // 6)
        cluster_rep_indices = []
        remaining = np.ones(NP, dtype=bool)
        remaining[extreme_indices] = False
        
        for _ in range(n_clusters):
            if not remaining.any():
                break
            densities = np.where(remaining, local_density, np.inf)
            rep_idx = np.argmin(densities)
            cluster_rep_indices.append(rep_idx)
            neighbor_mask = np.isin(nearest[rep_idx], np.where(remaining)[0])
            nearby = nearest[rep_idx][neighbor_mask]
            remaining[nearby] = False
            remaining[rep_idx] = False
        
        new_pop = []
        selected = set(extreme_indices.tolist() + cluster_rep_indices)
        
        for idx in extreme_indices:
            new_pop.append(population[idx].copy())
        
        for idx in cluster_rep_indices:
            if len(new_pop) < NP:
                new_pop.append(population[idx].copy())
        
        axis_ranges = population.max(axis=0) - population.min(axis=0)
        axis_ranges = np.maximum(axis_ranges, 1e-6)
        
        while len(new_pop) < NP:
            t = np.random.rand(dim)
            corner = population.min(axis=0) + t * axis_ranges
            point = centroid + 0.5 * (corner - centroid) + np.random.randn(dim) * np.sqrt(axis_ranges) * 0.1
            new_pop.append(np.clip(point, -100.0, 100.0))
        
        best_idx = np.argmin(fitness)
        return np.array(new_pop[:NP]), fitness[best_idx], population[best_idx].copy()
    
    def _restart_variant_05(self, population, fitness):
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
        isolation_score = (1.0 - connectivity / (connectivity.max() + self._eps)) * 0.5 + \
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
        return population, fitness[best_idx], population[best_idx].copy()
    
    def _restart_variant_07(self, population, fitness):
        """Stochastic restart via Monte Carlo bootstrap from elite solutions."""
        n, dim = population.shape
        bounds = (-100.0, 100.0)
        
        n_elite = max(3, n // 4)
        elite_idx = np.argpartition(fitness, n_elite)[:n_elite]
        elite_pop = population[elite_idx]
        
        n_bootstrap = 100
        bootstrap_indices = np.random.randint(0, len(elite_pop), size=(n_bootstrap, len(elite_pop)))
        bootstrap_samples = elite_pop[bootstrap_indices]
        
        weights = 1.0 / (fitness[elite_idx] + 1e-10)
        weights /= weights.sum()
        weighted_mean = np.average(elite_pop, axis=0, weights=weights)
        
        centered = elite_pop - weighted_mean
        cov = np.zeros((dim, dim))
        for i in range(len(elite_pop)):
            cov += weights[i] * np.outer(centered[i], centered[i])
        
        cov += np.eye(dim) * 1e-6
        
        best_f = np.min(fitness)
        noise_scale = max(50.0, 200.0 / (best_f + 1.0))
        
        new_pop = np.zeros((n, dim))
        
        for i in range(n):
            for _ in range(10):
                sample = np.random.multivariate_normal(weighted_mean, cov * noise_scale)
                if np.all((sample >= bounds[0]) & (sample <= bounds[1])):
                    new_pop[i] = sample
                    break
            else:
                new_pop[i] = np.random.uniform(bounds[0], bounds[1], dim)
        
        best_idx = np.argmin(fitness)
        new_pop[0] = population[best_idx]
        second_best = np.argmin(np.where(np.arange(n) != best_idx, fitness, np.inf))
        new_pop[1] = population[second_best]
        
        self.archive.append(population[elite_idx].copy())
        
        return new_pop, fitness[best_idx], population[best_idx].copy()
    
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
        """Vectorized compass mutation: compute direction scores, select best per individual."""
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
        """Binomial crossover with dimension-wise CR."""
        CR = np.clip(self.CR + np.random.randn(self.NP) * 0.1, 0.5, 0.98)
        mask = np.random.rand(self.NP, self.dim) < CR[:, np.newaxis]
        mask[np.arange(self.NP), np.random.randint(0, self.dim, self.NP)] = True
        trials = np.where(mask, mutants, population)
        return np.clip(trials, -100.0, 100.0)
    
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
    
    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        fitness = func(population)
        
        if len(fitness) < self.NP:
            fitness = np.full(self.NP, np.nan)
        
        best_idx = np.nanargmin(fitness)
        x_opt = population[best_idx].copy()
        f_opt = fitness[best_idx]
        
        iteration = 0
        
        while not stopping_condition():
            # Store population reference for graph analysis
            self._current_population = population
            
            # Compass mutation (primary strategy)
            mutants_compass = self._mutate_compass_batch(population, fitness)
            
            # Ensemble mutation (secondary strategy pool)
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
            
            # Update best
            current_best_idx = np.nanargmin(fitness)
            if fitness[current_best_idx] < f_opt:
                f_opt = fitness[current_best_idx]
                x_opt = population[current_best_idx].copy()
            
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