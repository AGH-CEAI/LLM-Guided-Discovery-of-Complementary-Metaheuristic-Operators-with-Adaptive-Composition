Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically picks (or blends) the best `_select_survivors_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_catA_idea_0  variant_02_catB_idea_0  variant_03_catC_idea_0  variant_04_catD_idea_0  variant_05_catE_idea_0  variant_06_catF_idea_0  variant_07_catG_idea_0  variant_08_catH_idea_0  variant_09_catA_idea_0  variant_10_catB_idea_0  
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      1.118196e-07            1.022542e-07            -inf                    7.376071e-07            2.225942e-05            1.079764e-07            1.063755e-07            nan                     nan                     2.911387e-03            1.232743e-07            
1      1.923990e+00            2.537726e+00            -inf                    3.127861e+00            1.737239e+00            1.872531e+00            3.359958e+00            nan                     nan                     3.915138e+00            3.409768e+00            
2      5.657046e-05            5.432755e-05            -inf                    5.139710e-02            2.233481e-04            6.403032e-05            8.956917e-05            nan                     nan                     2.697491e+00            4.689573e-05            
3      2.592829e+00            2.269933e+00            -inf                    8.858183e+00            3.623920e+00            2.639575e+00            2.607959e+00            nan                     nan                     9.605845e+00            2.345656e+00            
4      8.468928e-01            1.044809e+00            -inf                    1.368448e+00            1.214671e+00            9.467436e-01            1.159655e+00            nan                     nan                     1.215530e+00            9.322804e-01            
5      1.346859e+02            2.689989e+01            -inf                    1.302465e+03            2.325247e+02            9.047262e+00            4.872885e+01            nan                     nan                     2.201354e+03            9.902434e-01            
6      2.654227e+02            3.017061e+02            -inf                    6.224826e+02            5.525992e+02            1.839980e+02            2.775999e+02            nan                     nan                     1.973574e+02            1.264960e+02            
7      6.396815e+00            5.011403e+00            -inf                    1.001025e+01            7.016564e+00            5.813932e+00            6.074388e+00            nan                     nan                     1.042167e+01            6.080346e+00            
8      1.599741e+00            1.734266e+00            -inf                    2.881390e+00            1.827022e+00            2.044001e+00            1.614214e+00            nan                     nan                     3.428619e+00            1.457589e+00            
9      4.692224e+01            3.847302e+01            -inf                    5.475689e+01            4.431869e+01            4.240022e+01            4.214260e+01            nan                     nan                     4.254989e+01            4.150430e+01            
10     1.081160e+00            1.012930e+00            -inf                    1.197769e+00            1.632243e+00            1.010144e+00            1.517179e+00            nan                     nan                     1.536518e+01            1.037143e+00            
11     4.068769e+02            3.870390e+02            -inf                    3.592595e+02            4.367658e+02            3.648479e+02            3.200979e+02            nan                     nan                     3.981723e+02            3.901319e+02            
12     2.837961e+00            2.584712e+00            -inf                    3.083504e+00            3.600397e+00            2.894557e+00            2.662883e+00            nan                     nan                     1.725384e+01            2.616604e+00            
13     3.128272e+01            2.906171e+01            -inf                    2.874516e+01            3.407228e+01            3.111516e+01            3.091327e+01            nan                     nan                     2.948621e+01            3.115262e+01            
14     7.427308e+00            7.301928e+00            -inf                    8.126290e+00            4.026584e+00            7.223933e+00            4.469149e+00            nan                     nan                     1.058810e+01            7.026694e+00            
15     3.591257e+00            3.486133e+00            -inf                    3.603729e+00            3.662852e+00            3.622133e+00            3.641987e+00            nan                     nan                     3.568557e+00            3.505156e+00            
16     5.372293e+03            4.772035e+03            -inf                    4.217142e+03            6.255382e+03            4.599085e+03            5.143857e+03            nan                     nan                     4.908582e+03            5.960240e+03            
17     7.239167e+04            8.726470e+04            -inf                    8.594502e+04            1.245427e+05            7.860444e+04            1.078931e+05            nan                     nan                     7.670089e+04            7.183941e+04            
18     5.912787e+01            5.894850e+01            -inf                    5.938454e+01            6.597316e+01            5.733073e+01            6.200104e+01            nan                     nan                     5.685338e+01            5.491741e+01            
19     1.185155e+02            1.326171e+02            -inf                    1.338481e+02            1.076694e+02            1.559861e+02            1.281933e+02            nan                     nan                     1.822597e+02            1.374934e+02            
20     2.008751e+01            1.997467e+01            -inf                    2.011780e+01            2.237376e+01            1.996335e+01            1.987732e+01            nan                     nan                     2.005084e+01            2.019495e+01            
21     5.688817e+00            5.688240e+00            -inf                    5.629098e+00            5.632190e+00            5.651786e+00            5.858693e+00            nan                     nan                     5.703141e+00            5.671615e+00            
22     1.318928e+01            1.336401e+01            -inf                    1.370453e+01            1.388500e+01            1.329557e+01            1.272456e+01            nan                     nan                     1.453907e+01            1.362826e+01            
23     5.451930e+01            5.093115e+01            -inf                    6.300661e+01            4.704710e+01            5.215551e+01            4.169311e+01            nan                     nan                     6.189125e+01            5.293774e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: variant_01_catA_idea_0.py  (error=1.022542e-07)
Task  1: variant_04_catD_idea_0.py  (error=1.737239e+00)
Task  2: variant_10_catB_idea_0.py  (error=4.689573e-05)
Task  3: variant_01_catA_idea_0.py  (error=2.269933e+00)
Task  4: original.py  (error=8.468928e-01)
Task  5: variant_10_catB_idea_0.py  (error=9.902434e-01)
Task  6: variant_10_catB_idea_0.py  (error=1.264960e+02)
Task  7: variant_01_catA_idea_0.py  (error=5.011403e+00)
Task  8: variant_10_catB_idea_0.py  (error=1.457589e+00)
Task  9: variant_01_catA_idea_0.py  (error=3.847302e+01)
Task 10: variant_05_catE_idea_0.py  (error=1.010144e+00)
Task 11: variant_06_catF_idea_0.py  (error=3.200979e+02)
Task 12: variant_01_catA_idea_0.py  (error=2.584712e+00)
Task 13: variant_03_catC_idea_0.py  (error=2.874516e+01)
Task 14: variant_04_catD_idea_0.py  (error=4.026584e+00)
Task 15: variant_01_catA_idea_0.py  (error=3.486133e+00)
Task 16: variant_03_catC_idea_0.py  (error=4.217142e+03)
Task 17: variant_10_catB_idea_0.py  (error=7.183941e+04)
Task 18: variant_10_catB_idea_0.py  (error=5.491741e+01)
Task 19: variant_04_catD_idea_0.py  (error=1.076694e+02)
Task 20: variant_06_catF_idea_0.py  (error=1.987732e+01)
Task 21: variant_03_catC_idea_0.py  (error=5.629098e+00)
Task 22: variant_06_catF_idea_0.py  (error=1.272456e+01)
Task 23: variant_06_catF_idea_0.py  (error=4.169311e+01)

WIN COUNTS:
  variant_01_catA_idea_0.py: 6 wins
  variant_10_catB_idea_0.py: 6 wins
  variant_06_catF_idea_0.py: 4 wins
  variant_04_catD_idea_0.py: 3 wins
  variant_03_catC_idea_0.py: 3 wins
  original.py: 1 wins
  variant_05_catE_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_01_catA_idea_0.py (6 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Greedy selection with centroid-distance geometric diversity maintenance."""
        improved_mask = trial_fitness < fitness

        new_population = population.copy()
        new_fitness = fitness.copy()

        # Compute current population centroid (fitness-weighted for robustness)
        weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
        weights = weights / np.sum(weights)
        current_centroid = np.sum(population * weights[:, np.newaxis], axis=0)

        # Compute current average distance to centroid (geometric spread metric)
        current_distances = np.linalg.norm(population - current_centroid, axis=1)
        current_avg_spread = np.mean(current_distances)

        # Only apply geometric bonus when population has meaningful spread
        min_spread_threshold = 1e-6
        use_geometric_bonus = current_avg_spread > min_spread_threshold

        if use_geometric_bonus:
            # Apply selection with geometric diversity bonus
            for i in range(len(population)):
                if trial_fitness[i] < fitness[i]:
                    # Compute new centroid if this trial replaces current individual
                    temp_weights = weights.copy()
                    temp_weights[i] = 1.0 / (trial_fitness[i] - np.min(fitness) + 1e-10)
                    temp_weights = temp_weights / np.sum(temp_weights)
                    temp_centroid = np.sum(
                        np.where(np.arange(len(population))[:, np.newaxis] == i, 
                                 trials[i], population) * temp_weights[:, np.newaxis], 
                        axis=0
                    )

                    # Approximate new distances to centroid (only for changed individual)
                    old_dist_i = current_distances[i]
                    new_dist_i = np.linalg.norm(trials[i] - temp_centroid)

                    # Compute diversity gain: positive if trial increases geometric spread
                    diversity_gain = new_dist_i - old_dist_i
                    normalized_gain = diversity_gain / (current_avg_spread + 1e-10)

                    # Small bonus for geometric improvement (scaled to not override fitness)
                    diversity_bonus = 0.02 * normalized_gain

                    # Composite score: fitness + diversity bonus (lower is better)
                    composite_score = trial_fitness[i] + diversity_bonus

                    if composite_score < fitness[i]:
                        new_population[i] = trials[i]
                        new_fitness[i] = trial_fitness[i]
                        improved_mask[i] = True

        # Fallback: standard greedy selection for non-improvers and when geometry negligible
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]

        return new_population, new_fitness, improved_mask
```

# --- From variant_03_catC_idea_0.py (3 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """
        Information-theoretic survivor selection using entropy and KL divergence.
        Treats the population as a probability distribution; selects candidates
        that improve fitness AND contribute to population entropy/diversity.
        """
        improved_mask = trial_fitness < fitness

        # Compute entropy from fitness distribution (histogram-based)
        # This measures the diversity of fitness values in the population
        fitness_flat = fitness.ravel()
        valid_fitness = fitness_flat[~np.isinf(fitness_flat)]

        if len(valid_fitness) >= 10:
            hist, bin_edges = np.histogram(valid_fitness, bins='auto', density=True)
            hist = np.maximum(hist, 1e-10)
            bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1.0
            population_entropy = -np.sum(hist * np.log(hist)) * bin_width
        else:
            population_entropy = 1.0  # Default for small populations

        new_population = population.copy()
        new_fitness = fitness.copy()

        # For each improved candidate, compute KL divergence-based information gain
        for i in range(len(population)):
            if improved_mask[i]:
                trial_val = trial_fitness[i]
                parent_val = fitness[i]

                # Estimate probability of parent vs trial in current distribution
                # Using Gaussian kernel approximation around the candidate
                sigma = np.std(valid_fitness) + 1e-10

                # Compute p(parent) and q(trial) under current distribution model
                p_parent = np.exp(-0.5 * ((parent_val - valid_fitness) / sigma) ** 2)
                p_trial = np.exp(-0.5 * ((trial_val - valid_fitness) / sigma) ** 2)

                p_parent = np.maximum(p_parent.mean(), 1e-10)
                p_trial = np.maximum(p_trial.mean(), 1e-10)

                # KL divergence: information gain from replacing parent with trial
                kl_gain = np.log(p_trial / p_parent + 1e-10)

                # Normalize fitness improvement to [0,1] range for combination
                fitness_range = np.max(valid_fitness) - np.min(valid_fitness) + 1e-10
                norm_improvement = (parent_val - trial_val) / fitness_range

                # Weighted combination: balance fitness improvement with information gain
                # Higher alpha emphasizes information-theoretic diversity
                alpha = 0.7
                score = alpha * norm_improvement + (1 - alpha) * kl_gain

                # Accept if score is positive (net benefit to population)
                if score > 0:
                    new_population[i] = trials[i]
                    new_fitness[i] = trial_fitness[i]

        return new_population, new_fitness, improved_mask
```

# --- From variant_04_catD_idea_0.py (3 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """
        Adaptive selection: greedy when improving, rank-based when stagnant.
        Uses Spearman correlation and temporal EMA for regime detection.
        Category D: Fitness-landscape / rank-based reasoning only.
        """
        # Compute raw improvement rate
        improved_mask_greedy = trial_fitness < fitness
        improvement_rate = np.mean(improved_mask_greedy)

        # Initialize EMA state for temporal regime detection
        if not hasattr(self, 'sel_ewma_short'):
            self.sel_ewma_short = improvement_rate
            self.sel_ewma_long = improvement_rate

        # Update short/long EMAs with different smoothing (temporal dynamics)
        self.sel_ewma_short = 0.3 * improvement_rate + 0.7 * self.sel_ewma_short
        self.sel_ewma_long = 0.1 * improvement_rate + 0.9 * self.sel_ewma_long

        # Compute Spearman rank correlation between parent fitness and trial fitness
        # Low correlation indicates rugged/deceptive landscape where greedy fails
        n = len(fitness)
        if n > 3:
            ranks_f = np.argsort(np.argsort(fitness)) + 1
            ranks_tf = np.argsort(np.argsort(trial_fitness)) + 1
            # Handle edge case where all values identical
            std_f = np.std(fitness)
            std_tf = np.std(trial_fitness)
            if std_f > 1e-12 and std_tf > 1e-12:
                cov = np.mean((fitness - np.mean(fitness)) * (trial_fitness - np.mean(trial_fitness)))
                spearman_proxy = cov / (std_f * std_tf + 1e-12)
            else:
                spearman_proxy = 1.0
        else:
            spearman_proxy = 1.0

        # Regime detection using fitness-landscape signals
        short_trending_up = self.sel_ewma_short > self.sel_ewma_long
        both_stagnant = self.sel_ewma_short < 0.08 and self.sel_ewma_long < 0.12
        low_correlation = spearman_proxy < 0.25  # Rugged landscape signal

        # Switch to rank-based selection when: declining success OR stagnant OR rugged
        use_rank_selection = (not short_trending_up) or both_stagnant or low_correlation

        if use_rank_selection:
            # Rank-based survival: combine population + trials, select best NP by rank
            # This maintains diversity by potentially keeping inferior-but-different parents
            combined_pop = np.vstack([population, trials])
            combined_fitness = np.concatenate([fitness, trial_fitness])

            n_pop = len(population)

            # Compute ranks: 1 = best (lowest fitness)
            rank_order = np.argsort(combined_fitness)
            ranks = np.empty_like(rank_order)
            ranks[rank_order] = np.arange(len(rank_order)) + 1

            # Select top NP individuals
            selected_indices = np.argsort(ranks)[:n_pop]

            new_population = combined_pop[selected_indices]
            new_fitness = combined_fitness[selected_indices]
            improved_mask = selected_indices >= n_pop  # True if selected individual was from trials

            return new_population, new_fitness, improved_mask
        else:
            # Standard greedy selection when population is making consistent progress
            new_population = population.copy()
            new_fitness = fitness.copy()

            improved_mask = trial_fitness < fitness
            new_population[improved_mask] = trials[improved_mask]
            new_fitness[improved_mask] = trial_fitness[improved_mask]

            return new_population, new_fitness, improved_mask
```

# --- From variant_05_catE_idea_0.py (1 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Greedy selection biased by graph betweenness centrality to maintain population connectivity."""
        from scipy.sparse.csgraph import shortest_path
        from scipy.sparse import csr_matrix

        pop_size, dim = population.shape

        # Build k-NN graph and compute betweenness centrality for each individual
        k = max(2, min(5, pop_size // 4))
        dist_matrix = np.zeros((pop_size, pop_size))
        for i in range(pop_size):
            diffs = population - population[i]
            dist_matrix[i] = np.sqrt(np.sum(diffs * diffs, axis=1))
        np.fill_diagonal(dist_matrix, np.inf)

        # k-NN adjacency
        knn_indices = np.argsort(dist_matrix, axis=1)[:, :k]
        row_idx = np.repeat(np.arange(pop_size), k)
        col_idx = knn_indices.flatten()
        data = np.ones(len(row_idx))
        adj = csr_matrix((data, (row_idx, col_idx)), shape=(pop_size, pop_size))
        adj = adj + adj.T
        adj = (adj > 0).astype(float)

        # Compute betweenness centrality via Floyd-Warshall path counts
        n_paths = np.zeros((pop_size, pop_size))
        dists, preds = shortest_path(adj, directed=False, unweighted=True, return_predecessors=True)
        dists = np.nan_to_num(dists, nan=np.inf)

        for start in range(pop_size):
            for end in range(pop_size):
                if start != end and dists[start, end] < np.inf:
                    current = end
                    while current != start:
                        prev = preds[start, current]
                        if prev < 0:
                            break
                        n_paths[start, prev] += 1
                        n_paths[prev, end] += 1
                        current = prev

        centrality = np.zeros(pop_size)
        for i in range(pop_size):
            for j in range(i + 1, pop_size):
                if dists[i, j] < np.inf:
                    c_ij = n_paths[i, j] + n_paths[j, i]
                    centrality[i] += c_ij
                    centrality[j] += c_ij
        centrality = centrality / (centrality.max() + 1e-10)

        # Selection: prefer individuals with higher centrality when fitness is similar
        new_population = population.copy()
        new_fitness = fitness.copy()
        improved_mask = np.zeros(pop_size, dtype=bool)

        for i in range(pop_size):
            trial_f = trial_fitness[i]
            parent_f = fitness[i]

            if trial_f < parent_f:
                new_population[i] = trials[i]
                new_fitness[i] = trial_f
                improved_mask[i] = True
            elif np.abs(trial_f - parent_f) < 1e-14:
                # For equal fitness, probabilistically prefer higher centrality
                trial_centrality = 0.5  # New individual has neutral centrality
                if np.random.random() < (centrality[i] - trial_centrality + 0.5):
                    new_population[i] = trials[i]
                    new_fitness[i] = trial_f
                    improved_mask[i] = True

        return new_population, new_fitness, improved_mask
```

# --- From variant_06_catF_idea_0.py (4 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Temporal regime-adaptive selection with EMA-based stagnation detection."""
        improved_mask = trial_fitness < fitness

        # Initialize temporal state on first call
        if not hasattr(self, 'gen_improvement_ema'):
            self.gen_improvement_ema = np.zeros(self.np)  # Per-individual EMA of improvement
            self.pop_improvement_ema_short = 0.0
            self.pop_improvement_long = 0.0
            self.temporal_regime = 'neutral'  # 'improving', 'stagnant', 'declining'
            self.regime_patience = 0
            self.stagnation_window = []

        # Compute per-individual improvement magnitude (negative = improvement)
        improvement_mag = trial_fitness - fitness
        improvement_mag = np.where(improved_mask, improvement_mag, 0.0)

        # Update per-individual EMA of improvement magnitude
        alpha_indiv = 0.3
        self.gen_improvement_ema = alpha_indiv * improvement_mag + (1 - alpha_indiv) * self.gen_improvement_ema

        # Compute population-level improvement signal
        pop_improvement = np.sum(np.where(improved_mask, fitness - trial_fitness, 0.0))

        # Update population EMAs with different time constants
        alpha_short = 0.4
        alpha_long = 0.1
        self.pop_improvement_ema_short = alpha_short * pop_improvement + (1 - alpha_short) * self.pop_improvement_ema_short
        self.pop_improvement_long = alpha_long * pop_improvement + (1 - alpha_long) * self.pop_improvement_long

        # Maintain rolling window of improvement signals
        self.stagnation_window.append(np.mean(improved_mask))
        if len(self.stagnation_window) > 12:
            self.stagnation_window.pop(0)

        # Temporal regime detection via EMA divergence and window statistics
        window_mean = np.mean(self.stagnation_window)
        ema_trend = self.pop_improvement_ema_short - self.pop_improvement_long

        if ema_trend > 0.01 and window_mean > 0.25:
            self.temporal_regime = 'improving'
            self.regime_patience = 0
        elif ema_trend < -0.005 or window_mean < 0.15:
            self.regime_patience += 1
            if self.regime_patience > 5:
                self.temporal_regime = 'stagnant'
        else:
            self.temporal_regime = 'neutral'
            self.regime_patience = max(0, self.regime_patience - 1)

        # Compute temporal fitness: blend current fitness with EMA-based history
        # Individuals that consistently improve get a "temporal bonus" (lower effective fitness)
        temporal_bonus = 0.1 * self.gen_improvement_ema  # Positive EMA = consistent improver
        effective_fitness = fitness - temporal_bonus
        effective_trial_fitness = trial_fitness.copy()

        # Determine selection pressure based on temporal regime
        if self.temporal_regime == 'improving':
            # High selection pressure: trust current generation's greedy signal
            selection_mask = trial_fitness < fitness
        elif self.temporal_regime == 'stagnant':
            # Low selection pressure: use temporal fitness, favor consistent improvers
            # Allow some non-improving trials if they have good temporal history
            temporal_improvement_mask = effective_trial_fitness < effective_fitness
            # Also accept trials that improve individuals with poor temporal history
            poor_temporal_mask = self.gen_improvement_ema < np.percentile(self.gen_improvement_ema, 30)
            selection_mask = temporal_improvement_mask | (improved_mask & poor_temporal_mask)
        else:
            # Neutral: standard greedy with slight temporal weighting
            selection_mask = trial_fitness < fitness

        # Apply selection
        new_population = population.copy()
        new_fitness = fitness.copy()

        new_population[selection_mask] = trials[selection_mask]
        new_fitness[selection_mask] = trial_fitness[selection_mask]

        return new_population, new_fitness, improved_mask
```

# --- From variant_10_catB_idea_0.py (6 wins) ---
```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
            """Greedy selection with spectral diversity modulation."""
            improved_mask = trial_fitness < fitness
            improvement = fitness - trial_fitness

            # Compute population covariance eigenvalue spread
            centered = population - np.mean(population, axis=0)
            cov = np.cov(centered.T)

            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]

            # Eigenvalue spread: ratio of largest to smallest non-zero eigenvalue
            # High spread → anisotropic population → reduce selection pressure
            if eigvals[0] > 1e-12 and eigvals[-1] > 1e-12:
                condition = eigvals[0] / eigvals[-1]
            else:
                condition = 1.0

            # Modulation: in ill-conditioned populations, accept smaller improvements
            # log(condition) maps: 1→0, 10→2.3, 100→4.6, 1000→6.9
            diversity_bonus = np.log1p(condition)

            # Scaled improvement relative to population fitness range
            fit_range = np.ptp(fitness)
            fit_range = max(fit_range, 1e-10)
            scaled_improvement = improvement / fit_range

            # Accept if improved OR if scaled improvement exceeds diversity threshold
            accept_threshold = 0.01 / (1.0 + 0.5 * diversity_bonus)
            accept_mask = improved_mask | (scaled_improvement > accept_threshold)

            new_population = population.copy()
            new_fitness = fitness.copy()

            new_population[accept_mask] = trials[accept_mask]
            new_fitness[accept_mask] = trial_fitness[accept_mask]

            return new_population, new_fitness, accept_mask
```

────────────────────────────────────────────────────────────────────────
ADAPTATION MECHANISM MENU — pick ONE primary mechanism and motivate it.
Different mechanisms are appropriate to different operator types. A bandit
over six near-equivalent metrics is wasted; an ensemble that VOTES gives
a useful new signal even when the candidates are similar.

  1) DISCRETE ARM SELECTION (multi-armed bandit family)
     - Thompson Sampling, UCB1, EXP3, ε-greedy
     - Best when: operators are mutually exclusive, easy to credit-assign,
       reward is on a stable scale.
     - REQUIRED if you use this: rank-normalize rewards within a sliding
       window so magnitudes don't dominate. NEVER hard-code multipliers
       like *1e5 or *1e10 — those are scale-dependent magic constants.

  2) ENSEMBLE / VOTING / BLENDING
     - Run all candidates every generation and combine their outputs:
       weighted average, median, soft-vote, rank-aggregation.
     - Weights can be learned online (e.g. exponentially weighted majority).
     - Best when: candidates are correlated but each captures a different
       failure mode. Robust on its own — does not need credit assignment.

  3) CONDITIONAL / RULE-BASED DISPATCH
     - Hand-coded rules that pick the operator from a feature of the
       current population state: stagnation count, condition number of C,
       fitness-rank entropy, generation index, restart count, etc.
     - Best when: the failure modes are diagnosable. Fast convergence
       because there is no exploration cost.

  4) CONTEXTUAL BANDIT / FEATURE-CONDITIONED SELECTION
     - Like (1) but the arm-distribution depends on a feature vector
       describing the current state of the search (population shape,
       improvement rate, diversity, generation, ...).
     - Best when: different operators are best in different regimes and
       you can measure the regime cheaply.

  5) HYPER-HEURISTIC / SCHEDULE LEARNING
     - Maintain a *sequence* of operators rather than a single per-step
       choice (e.g. operator A for k_1 generations, then B for k_2).
     - Schedule itself is adapted (e.g. via genetic encoding of the
       sequence, or adapting k_i with success-history).
     - Best when: operators benefit from being applied in bursts.

  6) SELF-ADAPTIVE PARAMETERS (continuous, not discrete)
     - Instead of picking among discrete operators, evolve continuous
       parameters that morph the operator (e.g. CR, F, σ, mutation
       radius). Parameters can be encoded per-individual and inherited.
     - Best when: the operator family is a continuous family.

  7) CO-EVOLUTION
     - Run TWO populations: one of solutions, one of operators / operator
       configurations. Operators evolve based on their measured fitness.
     - Best when: the operator design space is large and you want
       genuine novelty.

  8) RESTART-AWARE / EPOCHED ADAPTATION
     - Different operators on different restart epochs. Memory of which
       operators worked in past epochs informs the current epoch's
       distribution.
     - Best when: the algorithm restarts often (multimodal landscapes).

  9) META-CONTROLLED ADAPTATION
     - A small online controller (e.g. a learned linear policy or a
       hand-tuned PID-like loop) maps measured signals to operator
       weights. The controller's gains are themselves adapted slowly.
     - Best when: the dynamics are reasonably smooth.

────────────────────────────────────────────────────────────────────────
CALIBRATION RULES (mandatory — these are the most common failure modes):

a) NEVER use scale-dependent magic constants in reward shaping.
   FORBIDDEN: `np.log1p(diversity_improvement * 1e5)`,
              `np.log1p(func_improvement * 1e10)`,
              any hard-coded multiplier > 100.
   ALLOWED:   rank-based credit (was this generation in the top tercile of
              the last K generations?), z-score normalization, relative
              improvement (delta / |incumbent|), success/failure binary.

b) If you use a sliding window of size K, declare K as a parameter and
   make K >= 3 * num_arms so each arm is sampled enough times to be
   estimable. If num_arms is large, prefer ensemble (mechanism #2) over
   discrete selection.

c) Mixed reward signals (e.g. 0.6*diversity + 0.4*fitness) MUST be on
   the same scale. Either both rank-based, both z-scored, or both
   relative-improvement. Mixing log1p(x*1e5) with log1p(y*1e10) is wrong.

d) The mechanism must be VISIBLY DIFFERENT from a plain bandit-over-
   operators if the variants are highly correlated. If you can't credit-
   assign reliably, switch mechanism — don't try to fix a broken
   reward signal with more clipping/decay.

e) Initial state matters. Don't start with `alpha = beta = 1` (uniform
   prior) if you have no reason to think the prior is uninformative —
   the original method may already be a strong choice.

────────────────────────────────────────────────────────────────────────

DECISION GUIDE — choose your mechanism by looking at the data above:

- If ONE variant clearly dominates (>50% of wins) → use mechanism #3
  (rule-based dispatch with that variant as default + fallback).
- If wins are spread roughly evenly across many variants → mechanism #1
  (bandit) or #4 (contextual bandit) is appropriate.
- If variants compute correlated quantities and the differences look
  like rescalings → mechanism #2 (ensemble/voting), NOT a bandit. A
  bandit cannot learn between near-identical arms.
- If there is a clear regime split (variant X wins early, variant Y
  wins late) → mechanism #5 (schedule) or #4 (contextual).
- If the operator is a continuous family → mechanism #6 (self-adaptive
  parameters) is more powerful than picking among discrete snapshots.

REQUIREMENTS:
- Respond with the COMPLETE class code inside a single ```python``` block.
- Include `import numpy as np` at the top.
- At the START of the class docstring, write a SHORT comment naming the
  adaptation mechanism you chose (from the menu above) and 1–2 lines
  explaining why it fits the benchmark results.
- You may add new attributes in __init__ and new private helper methods.
- Do NOT change signatures of existing public methods (__call__, etc.).
- The class MUST be callable as: `obj(func, stopping_condition)` → (f_opt, x_opt).
- Handle ALL edge cases: no -inf, no NaN, no crashes, clip to bounds.
- If you use any of mechanisms #1, #4, #5, #8: rewards MUST be rank-based
  or z-scored within a sliding window. NO scale-dependent constants.
- If the variants are highly correlated (the typical case for diversity-
  metric variants), mechanism #2 (ensemble) is strongly preferred.

Original algorithm:
```python
import numpy as np


class AdaptiveDirectionalDE:
    """
    Adaptive Directional Differential Evolution (ADDE)
    
    A DE variant with:
    - Ring topology for mutation parent selection
    - Directional memory that biases mutation toward successful directions
    - Adaptive F and Cr based on historical success rates
    - Composite mutation: directional + rand/1 blended adaptively
    - Diversity-triggered restart mechanism
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 60), 300)  # Population size
        self.lower = -100.0
        self.upper = 100.0
        
        # Control parameters
        self.F_base = 0.5
        self.Cr_base = 0.3
        self.F = self.F_base
        self.Cr = self.Cr_base
        
        # Directional memory
        self.direction_memory_size = 5
        self.successful_directions = []
        self.direction_decay = 0.95
        
        # Adaptation tracking
        self.adaptation_window = 10
        self.F_success_history = []
        self.Cr_success_history = []
        
        # Restart control
        self.stagnation_threshold = 50
        self.min_diversity_threshold = 1e-6
        self.generation = 0
        
    def __call__(self, func, stopping_condition):
        return self._optimize(func, stopping_condition)
    
    def _initialize_population(self, func):
        """Initialize using Latin Hypercube Sampling for better spread."""
        sampler = np.linspace(self.lower, self.upper, self.np)
        population = np.zeros((self.np, self.dim))
        
        for d in range(self.dim):
            indices = np.random.permutation(self.np)
            offsets = np.random.uniform(0, 1, self.np)
            population[:, d] = sampler[indices] + offsets * (sampler[1] - sampler[0])
        
        population = self._clip_to_bounds(population)
        fitness = func(population)
        
        if len(fitness) < self.np:
            fitness = np.resize(fitness, self.np)
            fitness[np.isnan(fitness)] = np.inf
        
        best_idx = np.argmin(fitness)
        return population, fitness, fitness[best_idx], population[best_idx].copy()
    
    def _clip_to_bounds(self, population):
        """Hard clip all candidates to search bounds."""
        return np.clip(population, self.lower, self.upper)
    
    def _build_ring_topology(self):
        """Create ring neighbor indices for each population member."""
        indices = np.arange(self.np)
        ring_prev = np.roll(indices, 1)
        ring_next = np.roll(indices, -1)
        return ring_prev, ring_next
    
    def _compute_population_centroid(self, population, fitness):
        """Compute fitness-weighted centroid for directional bias."""
        weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
        weights = weights / np.sum(weights)
        centroid = np.sum(population * weights[:, np.newaxis], axis=0)
        return centroid
    
    def _update_direction_memory(self, successful_mutations, successful_mask):
        """Track successful mutation directions for adaptive biasing."""
        if not np.any(successful_mask):
            self.successful_directions = [
                d * self.direction_decay for d in self.successful_directions
            ]
            return
        
        for mutation_vec in successful_mutations[successful_mask]:
            self.successful_directions.append(mutation_vec.copy())
        
        if len(self.successful_directions) > self.direction_memory_size:
            self.successful_directions = self.successful_directions[-self.direction_memory_size:]
    
    def _compute_directional_bias(self, target_indices):
        """Compute directional bias from historical successful mutations."""
        if len(self.successful_directions) == 0:
            return np.zeros((len(target_indices), self.dim))
        
        direction_stack = np.array(self.successful_directions)
        weights = np.exp(-0.5 * np.arange(len(direction_stack))[::-1])
        weights = weights / np.sum(weights)
        
        weighted_direction = np.sum(direction_stack * weights[:, np.newaxis], axis=0)
        directional_bias = np.tile(weighted_direction, (len(target_indices), 1))
        
        return 0.3 * directional_bias
    
    def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Generate mutation vectors using composite strategy:
        - Ring-based rand/1 component
        - Directional component from historical memory
        """
        np_pop = len(population)
        trials = np.zeros((np_pop, self.dim))
        
        centroid = self._compute_population_centroid(population, fitness)
        directional_bias = self._compute_directional_bias(np.arange(np_pop))
        
        # Select ring-based parents
        r1 = ring_prev
        r2 = ring_next
        
        # Standard DE rand/1 mutation
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        # Adaptive F with perturbation based on diversity
        current_F = self.F * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)
        
        # Base mutation: ring-based differential
        base_mutation = rand1_vectors + current_F * (rand2_vectors - population)
        
        # Directional component: bias toward successful regions
        directional_component = directional_bias
        
        # Blend based on adaptation success
        blend_weight = 0.2 + 0.3 * (1.0 - self.Cr)
        trials = (1 - blend_weight) * base_mutation + blend_weight * (centroid + directional_component)
        
        return trials, current_F
    
    def _crossover_batch(self, population, mutant_batch):
        """Binomial crossover with adaptive Cr."""
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        # Per-individual Cr with bounded distribution
        Cr_individual = np.clip(
            self.Cr + 0.1 * np.random.randn(np_pop),
            0.1, 0.9
        )
        
        # Ensure at least one dimension is crossed
        j_rand = np.random.randint(0, dim, size=np_pop)
        
        for i in range(np_pop):
            mask = np.random.rand(dim) < Cr_individual[i]
            mask[j_rand[i]] = True
            trials[i] = np.where(mask, mutant_batch[i], population[i])
        
        return trials, Cr_individual
    
    def _evaluate_batch(self, batch, func):
        """Evaluate batch, handle budget exhaustion gracefully."""
        batch = self._clip_to_bounds(batch)
        fitness = func(batch)
        
        if len(fitness) < len(batch):
            valid_len = len(fitness)
            fitness = np.resize(fitness, len(batch))
            fitness[valid_len:] = np.inf
        
        fitness = np.where(np.isnan(fitness), np.inf, fitness)
        return fitness
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Greedy selection with diversity maintenance."""
        improved_mask = trial_fitness < fitness
        
        new_population = population.copy()
        new_fitness = fitness.copy()
        
        new_population[improved_mask] = trials[improved_mask]
        new_fitness[improved_mask] = trial_fitness[improved_mask]
        
        return new_population, new_fitness, improved_mask
    
    def _compute_diversity(self, population):
        """Compute population diversity via average pairwise distance."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Adapt F and Cr using temporal EMA dynamics and parameter drift detection."""
        success_rate = np.mean(improved_mask)

        # Initialize EMA state on first call
        if not hasattr(self, 'success_ewma_short'):
            self.success_ewma_short = success_rate
            self.success_ewma_long = success_rate
            self.stagnation_counter = 0
            self.F_history = [self.F]
            self.Cr_history = [self.Cr]

        # Update EMAs with different smoothing constants
        alpha_short = 0.3   # Fast EMA for short-term dynamics
        alpha_long = 0.1    # Slow EMA for trend detection
        self.success_ewma_short = alpha_short * success_rate + (1 - alpha_short) * self.success_ewma_short
        self.success_ewma_long = alpha_long * success_rate + (1 - alpha_long) * self.success_ewma_long

        # Track parameter history for drift detection
        self.F_history.append(F_used)
        self.Cr_history.append(F_used)
        if len(self.F_history) > 15:
            self.F_history.pop(0)
            self.Cr_history.pop(0)

        # Compute parameter drift from historical mean
        F_mean = np.mean(self.F_history)
        Cr_mean = np.mean(self.Cr_history)
        F_drift = self.F / (F_mean + 1e-10)
        Cr_drift = self.Cr / (Cr_mean + 1e-10)

        # Regime detection via EMA divergence
        short_trending_up = self.success_ewma_short > self.success_ewma_long
        short_trending_down = self.success_ewma_short < self.success_ewma_long
        both_stagnant = self.success_ewma_short < 0.1 and self.success_ewma_long < 0.15

        # Compute momentum-adjusted base adjustments
        momentum_factor = np.clip(F_drift, 0.7, 1.4)

        if short_trending_up and self.success_ewma_short > 0.25:
            # Exploitation phase: success accelerating, increase convergence pressure
            F_adjustment = 1.05 * momentum_factor
            Cr_adjustment = 1.08
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
        elif short_trending_down and self.success_ewma_short < 0.2:
            # Exploration phase: success declining, inject diversity
            F_adjustment = 0.88 / momentum_factor
            Cr_adjustment = 0.92
            self.stagnation_counter += 1
        elif both_stagnant:
            # Deep stagnation: large perturbation to escape local traps
            F_adjustment = 0.75
            Cr_adjustment = 1.15
            self.stagnation_counter += 2
        else:
            # Neutral regime: gentle correction toward baseline
            delta = self.success_ewma_short - self.success_ewma_long
            F_adjustment = 1.0 + 0.08 * delta
            Cr_adjustment = 1.0 - 0.05 * delta

        # Apply adjustments with drift correction
        self.F = np.clip(self.F * F_adjustment, 0.1, 1.5)
        self.Cr = np.clip(self.Cr * Cr_adjustment, 0.1, 0.9)
    
    def _restart_if_stagnant(self, population, fitness, best_fitness, best_solution):
        """Reinitialize portion of population if stagnation detected."""
        current_best = np.min(fitness)
        
        if current_best >= best_fitness:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            best_fitness = current_best
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx].copy()
        
        diversity = self._compute_diversity(population)
        
        if (self.stagnation_counter > self.stagnation_threshold or 
            diversity < self.min_diversity_threshold):
            
            n_replace = max(self.np // 5, 2)
            replace_indices = np.random.choice(self.np, n_replace, replace=False)
            
            for idx in replace_indices:
                population[idx] = np.random.uniform(
                    self.lower, self.upper, self.dim
                )
            
            fitness[replace_indices] = np.inf
            self.stagnation_counter = 0
            self.successful_directions = []
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop orchestrating all batched operations."""
        population, fitness, best_fitness, best_solution = self._initialize_population(func)
        
        self.stagnation_counter = 0
        self.generation = 0
        
        ring_prev, ring_next = self._build_ring_topology()
        
        while not stopping_condition():
            self.generation += 1
            
            # Build mutation batch
            mutant_batch, F_used = self._mutate_batch(
                population, fitness, ring_prev, ring_next
            )
            
            # Build crossover batch
            trial_batch, Cr_used = self._crossover_batch(population, mutant_batch)
            
            # Check budget before evaluation
            if stopping_condition():
                break
            
            # Evaluate trial batch
            trial_fitness = self._evaluate_batch(trial_batch, func)
            
            if stopping_condition():
                break
            
            # Select survivors
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trial_batch, trial_fitness
            )
            
            # Update best solution
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
            
            # Compute successful mutations for direction memory
            successful_mutations = trial_batch - population
            self._update_direction_memory(successful_mutations, improved_mask)
            
            # Adapt control parameters
            self._adapt_parameters(improved_mask, F_used, Cr_used)
            
            # Check for stagnation and restart if needed
            population, fitness, best_fitness, best_solution = self._restart_if_stagnant(
                population, fitness, best_fitness, best_solution
            )
            
            # Rebuild topology after potential restart
            ring_prev, ring_next = self._build_ring_topology()
        
        return best_fitness, best_solution

```