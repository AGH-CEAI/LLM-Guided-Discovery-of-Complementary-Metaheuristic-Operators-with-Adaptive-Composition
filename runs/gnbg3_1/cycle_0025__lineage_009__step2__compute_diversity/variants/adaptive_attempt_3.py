import numpy as np


class AdaptiveDiversityCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best diversity computation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 6 diversity computation strategies (original + 5 variants)
    - Thompson Sampling for operator selection
    - Sliding window credit assignment with diversity improvement reward
    - Automatic restart on stagnation
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.NP = min(240, max(80, 8 * dim))
        self.bounds = (-100.0, 100.0)
        self.lb = np.full(dim, -100.0)
        self.ub = np.full(dim, 100.0)
        
        # CMA-ES inspired parameters
        self.mu = self.NP // 4
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        
        # Learning rates
        self.cs = (self.mueff + 2.0) / (self.dim + self.mueff + 5.0)
        self.cc = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        self.ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        self.damping = 1.0 + np.maximum(0.0, np.sqrt(self.mueff) - 1.0)
        
        # Restart parameters
        self.max_stagnation = 50 + self.dim * 3
        self.min_diversity = 1e-6 * (self.ub[0] - self.lb[0])
        
        # Adaptive diversity operator selection parameters
        self.num_diversity_ops = 6
        self.diversity_op_names = ['original', 'variant_02', 'variant_04', 'variant_06', 'variant_08', 'variant_09']
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_diversity_ops, dtype=float)
        self.beta = np.ones(self.num_diversity_ops, dtype=float)
        
        # Sliding window for credit assignment
        self.reward_window_size = 10
        self.operator_rewards = {i: [] for i in range(self.num_diversity_ops)}
        self.operator_counts = np.zeros(self.num_diversity_ops, dtype=float)
        self.selection_counts = np.zeros(self.num_diversity_ops, dtype=float)
        
        # Runtime state
        self.generation = 0
        self.current_diversity_op = 0
        self.L = None
    
    def _clip_to_bounds(self, x):
        """Clip solution to bounds."""
        return np.clip(x, self.lb, self.ub)
    
    def _ensure_positive_definite(self, C):
        """Ensure matrix is symmetric and positive definite."""
        C = 0.5 * (C + C.T)
        min_eig = np.min(np.linalg.eigvalsh(C))
        if min_eig < 1e-10:
            C += (1e-7 - min_eig) * np.eye(self.dim)
        return C
    
    def __call__(self, func, stopping_condition):
        self.func = func
        self._initialize_population()
        
        while not stopping_condition():
            self._sample_trials_batch()
            
            if stopping_condition():
                break
                
            self._evaluate_batch()
            
            if len(self.trial_fitness) < len(self.trials):
                self.trials = self.trials[:len(self.trial_fitness)]
                
            if len(self.trial_fitness) < self.NP:
                break
                
            self._update_best()
            
            if stopping_condition():
                break
                
            self._select_survivors_batch()
            self._adapt_step_size()
            
            # Select diversity computation operator using Thompson Sampling
            self._select_diversity_operator_thompson()
            
            # Update operator rewards based on diversity improvement
            self._update_diversity_operator_rewards()
            
            self._check_stagnation()
            self._restart_if_needed()
        
        return self.f_opt, self.x_opt
    
    def _initialize_population(self):
        """Initialize population using Sobol quasi-random sequences."""
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(self.dim, scramble=True)
            samples = sampler.random(self.NP)
            samples = qmc.scale(samples, self.lb, self.ub)
        except Exception:
            samples = np.zeros((self.NP, self.dim))
            for d in range(self.dim):
                bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
                bin_width = bins[1] - bins[0]
                samples[:, d] = bins[:-1] + np.random.random(self.NP) * bin_width
            for d in range(self.dim):
                samples[:, d] = samples[np.random.permutation(self.NP), d]

        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = self.func(self.population)

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        self.diversity_opt_prev = self._compute_diversity()

        self.stagnation_counter = 0
        self.generation = 0
        self.current_diversity_op = 0

        self._select_diversity_operator_thompson()

        self._reset_operator_state()
    
    def _reset_operator_state(self):
        """Reset all operator-specific state variables."""
        self.L = None
        if hasattr(self, 'pc_weighted'):
            self.pc_weighted = np.zeros(self.dim)
        if hasattr(self, 'p_cross'):
            self.p_cross = np.zeros(self.dim)
        if hasattr(self, 'prev_y_mean'):
            delattr(self, 'prev_y_mean')
        if hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.improvement_ema = 1.0
    
    def _sample_trials_batch(self):
        """Sample new trial population using Cholesky decomposition."""
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        try:
            L = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            diag_C = np.diag(self.C)
            diag_C = np.maximum(diag_C, 1e-10)
            L = np.diag(np.sqrt(diag_C))

        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)
        self.trials = self._clip_to_bounds(self.trials)
    
    def _evaluate_batch(self):
        """Evaluate all trial candidates in single batch call."""
        self.trial_fitness = self.func(self.trials)
    
    def _update_best(self):
        """Update best solution if trial population contains improvement."""
        trial_best_idx = np.argmin(self.trial_fitness)
        trial_best_fit = float(np.asarray(self.trial_fitness[trial_best_idx]).flatten()[0])
        if trial_best_fit < self.f_opt:
            self.f_opt = trial_best_fit
            self.x_opt = self.trials[trial_best_idx].copy()
    
    def _select_survivors_batch(self):
        """Select survivors via elitist (mu, lambda)-selection."""
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])
        
        sorted_indices = np.argsort(combined_fit)
        self.population = combined_pop[sorted_indices[:self.NP]]
        self.fitness = combined_fit[sorted_indices[:self.NP]]
        
        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
    
    def _adapt_step_size(self):
        """Adapt step-size using momentum-enhanced improvement tracking with diversity-based damping."""
        recent_improved = 0
        recent_total = 0
        total_improvement = 0.0

        for i in range(self.NP):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                if self.trial_fitness[i] < self.fitness[i]:
                    recent_improved += 1
                    diff = self.fitness[i] - self.trial_fitness[i]
                    total_improvement += max(diff, 0.0)
                recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = recent_improved / recent_total
        success_rate = np.clip(success_rate, 0.0, 1.0)

        avg_improvement = total_improvement / max(recent_improved, 1)
        improvement_magnitude = np.log1p(max(avg_improvement, 1e-15))

        if not hasattr(self, 'improvement_ema'):
            self.improvement_ema = improvement_magnitude
        self.improvement_ema = 0.7 * self.improvement_ema + 0.3 * improvement_magnitude

        target_rate = 0.25
        adaptation = (success_rate - target_rate) / target_rate
        adaptation += 0.3 * np.tanh(self.improvement_ema - 1.0)
        adaptation = np.clip(adaptation, -0.8, 0.8)

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        diversity_boost = 1.0 + 2.0 * (1.0 - diversity)
        diversity_boost = np.clip(diversity_boost, 0.5, 3.0)

        damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim)) / diversity_boost
        damping_adaptive = np.clip(damping_adaptive, 0.05, 200.0)

        self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _select_diversity_operator_thompson(self):
        """Select diversity operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_diversity_op = int(np.argmax(samples))
        self.operator_counts[self.current_diversity_op] += 1
        self.selection_counts[self.current_diversity_op] += 1
        return self.current_diversity_op
    
    def _update_diversity_operator_rewards(self):
        """Credit assignment via diversity improvement with stagnation awareness."""
        current_diversity = self._compute_diversity()
        diversity_improvement = current_diversity - self.diversity_opt_prev
        self.diversity_opt_prev = current_diversity

        func_improvement = max(0.0, self.f_opt_prev - self.f_opt)

        if not hasattr(self, 'operator_cumulative_reward'):
            self.operator_cumulative_reward = np.zeros(self.num_diversity_ops, dtype=float)
        if not hasattr(self, 'operator_decay_sum'):
            self.operator_decay_sum = np.zeros(self.num_diversity_ops, dtype=float)

        decay = 0.95
        self.operator_cumulative_reward *= decay
        self.operator_decay_sum *= decay

        if diversity_improvement > 0:
            div_reward = float(np.log1p(diversity_improvement * 1e5) / 10.0)
        else:
            div_reward = 0.0

        if func_improvement > 0:
            func_reward = float(np.log1p(func_improvement * 1e10) / 10.0)
        else:
            func_reward = 0.0

        reward = 0.6 * div_reward + 0.4 * func_reward
        reward = float(np.clip(reward, -10.0, 10.0))

        is_stagnant = self.stagnation_counter > self.max_stagnation // 2

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        if is_stagnant and diversity < 0.1:
            reward *= 2.0

        self.operator_cumulative_reward[self.current_diversity_op] += reward
        self.operator_decay_sum[self.current_diversity_op] += 1.0

        norm = max(self.operator_decay_sum[self.current_diversity_op], 1.0)
        normalized_reward = self.operator_cumulative_reward[self.current_diversity_op] / norm
        normalized_reward = float(np.clip(normalized_reward, 0.0, 10.0))

        op = self.current_diversity_op
        self.operator_rewards[op].append(normalized_reward)

        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)

        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            mean_reward = np.clip(mean_reward, 0.0, 1.0)
            var_reward = float(np.var(self.operator_rewards[op]))
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.beta[op] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            sum_reward = np.clip(sum_reward, 0.0, 10.0)
            if sum_reward > 0:
                self.alpha[op] = 1.0 + sum_reward
            else:
                self.beta[op] = 1.0 - sum_reward
    
    def _compute_diversity_original(self):
        """Compute diversity using effective sample size of the covariance matrix.

        Combines trace-based spread with eigenvalue-based shape penalty.
        ESS = geometric_mean(eigenvalues) / arithmetic_mean(eigenvalues):
          ~1.0 for spherical (good) → full diversity
          ~0.0 for elongated (bad) → wasted dimensions

        Diversity = normalized_trace * sqrt(ESS) penalizes both:
          - Low overall spread (collapsed population)
          - High elongation (narrow ridge, CMA-ES losing rank)
        """
        diag = np.diag(self.C)
        trace = np.sum(diag)
        expected_trace = self.dim * ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        trace_norm = trace / max(expected_trace, 1e-10)

        eigvals = np.linalg.eigvalsh(self.C)
        eigvals = np.clip(eigvals, 1e-15, None)
        geo_mean = np.exp(np.mean(np.log(eigvals)))
        arith_mean = np.mean(eigvals)
        ess = geo_mean / max(arith_mean, 1e-15)

        diversity = trace_norm * np.sqrt(ess)
        return float(np.clip(diversity, 1e-15, None))
    
    def _compute_diversity_variant_02(self):
        """Compute diversity using average pairwise Euclidean distance in decision space.

        Directly measures spatial spread of population rather than relying on
        covariance matrix statistics. This catches population collapse that
        eigenvalue-based metrics miss on ill-conditioned problems.

        Diversity = log(1 + mean_pairwise_dist / expected_dist)
          where expected_dist is the mean distance in a uniformly distributed population

        Log scaling provides sensitivity across orders of magnitude while avoiding
        numerical instability when distances are very small.
        """
        if len(self.population) < 2:
            return 1e-15

        pop = self.population.reshape(len(self.population), -1)
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        pairwise_dists = np.sqrt(np.sum(diffs ** 2, axis=2))

        n = len(self.population)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        mean_pairwise_dist = np.mean(pairwise_dists[mask])

        search_space_diameter = np.sqrt(self.dim) * (self.ub[0] - self.lb[0])
        expected_dist = search_space_diameter / np.sqrt(2.0)

        dist_ratio = mean_pairwise_dist / max(expected_dist, 1e-10)
        diversity = np.log1p(dist_ratio)

        return float(np.clip(diversity, 1e-15, None))
    
    def _compute_diversity_variant_04(self):
        """Compute diversity using hybrid fitness variance, condition number, and spread.

        Unlike the eigenvalue-based ESS approach, this combines:
        1. Fitness variance (relative) — detects population collapse into local optima
        2. Condition number penalty — detects elongated covariance (narrow ridges)
        3. Actual population spread — measures true geometric extent

        This directly addresses the failure mode where the population is trapped
        in a local optimum with low fitness diversity and/or highly elongated covariance.
        """
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)
        rel_var = float(np.clip(rel_var, 1e-10, 1.0))
        fitness_component = np.sqrt(rel_var)

        eigvals = np.linalg.eigvalsh(self.C)
        eigvals = np.clip(eigvals, 1e-15, None)
        eig_min = float(np.min(eigvals))
        eig_max = float(np.max(eigvals))
        cond = eig_max / max(eig_min, 1e-15)
        cond_penalty = 1.0 / np.log1p(cond)
        cond_penalty = float(np.clip(cond_penalty, 1e-10, 1.0))

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        spread = pop_variance / max(expected_var, 1e-10)
        spread = float(np.clip(spread, 1e-10, 1.0))

        diversity = (fitness_component * cond_penalty * spread) ** (1.0 / 3.0)
        return float(np.clip(diversity, 1e-15, None))
    
    def _compute_diversity_variant_06(self):
        """Compute diversity using direct population metrics.

        Combines three complementary measures:
        1. Normalized mean pairwise distance (actual spatial spread)
        2. Relative fitness variance (selection pressure indicator)
        3. Solution-space coverage ratio (bound utilization)

        This avoids reliance on covariance matrix eigenvalues which become
        unreliable when the algorithm struggles on difficult tasks.
        """
        pop = self.population[:len(self.fitness)]
        n = len(pop)
        if n < 2:
            return 1e-15

        sample_size = min(50, n)
        indices = np.random.choice(n, sample_size, replace=False)
        pop_sample = pop[indices]

        pairwise_dists = []
        for i in range(sample_size):
            diffs = pop_sample - pop_sample[i]
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            pairwise_dists.extend(dists[i + 1:])

        mean_pairwise_dist = np.mean(pairwise_dists) if pairwise_dists else 0.0
        max_possible_dist = np.sqrt(self.dim) * (self.ub[0] - self.lb[0])
        spatial_spread = mean_pairwise_dist / max(max_possible_dist, 1e-10)
        spatial_spread = np.clip(spatial_spread, 0.0, 1.0)

        fit_arr = np.asarray(self.fitness).flatten()
        fit_var = np.var(fit_arr)
        fit_range = np.ptp(fit_arr) + 1e-10
        fit_scale = max(abs(self.f_opt), 1.0, fit_range)
        fitness_diversity = np.clip(fit_var / (fit_scale ** 2 + 1e-10), 0.0, 1.0)

        pop_min = np.min(pop, axis=0)
        pop_max = np.max(pop, axis=0)
        span = pop_max - pop_min
        search_span = self.ub - self.lb
        coverage = np.mean(span / np.maximum(search_span, 1e-10))
        coverage = np.clip(coverage, 0.0, 1.0)

        diversity = (0.5 * spatial_spread + 0.3 * fitness_diversity + 0.2 * coverage)
        return float(np.clip(diversity, 1e-15, None))
    
    def _compute_diversity_variant_08(self):
        """Compute diversity using pairwise distances in population space.

        Directly measures spatial distribution of population via mean/median
        Euclidean distances between individuals. Normalized by search space
        diameter gives scale-invariant metric that detects:
          - Population collapse (all individuals clustered)
          - Deceptive local optima trapping (clustering away from global optimum)
          - Loss of exploration (low pairwise distances)
        """
        pop = self.population[:len(self.fitness)]
        n = len(pop)

        if n < 2:
            return float(1e-15)

        sq_dists = np.sum((pop[:, np.newaxis, :] - pop[np.newaxis, :, :]) ** 2, axis=2)

        upper_tri_indices = np.triu_indices(n, k=1)
        pairwise_sq_dists = sq_dists[upper_tri_indices]

        if len(pairwise_sq_dists) == 0:
            return float(1e-15)

        mean_sq_dist = np.mean(pairwise_sq_dists)
        median_sq_dist = np.median(pairwise_sq_dists)

        mean_dist = np.sqrt(mean_sq_dist)
        median_dist = np.sqrt(median_sq_dist)

        search_diameter = np.sqrt(self.dim) * (self.ub[0] - self.lb[0])
        expected_dist = search_diameter / np.sqrt(2.0)

        mean_dist_norm = mean_dist / max(expected_dist, 1e-10)
        median_dist_norm = median_dist / max(expected_dist, 1e-10)

        dists_to_best = np.sqrt(np.sum((pop - self.x_opt) ** 2, axis=1))
        mean_dist_to_best = np.mean(dists_to_best)

        clustering_factor = mean_dist_to_best / max(expected_dist, 1e-10)
        clustering_penalty = np.clip(clustering_factor, 0.0, 1.0)

        spread_metric = (mean_dist_norm + median_dist_norm) / 2.0
        spread_metric = np.clip(spread_metric, 1e-15, 1.0)

        diversity = spread_metric * np.sqrt(1.0 - 0.5 * clustering_penalty)

        return float(np.clip(diversity, 1e-15, None))
    
    def _compute_diversity_variant_09(self):
        """Compute diversity using percentile-robust and condition-aware metrics.

        Key differences from ESS-based approach:
        - Uses MAX-normalized spread (catches catastrophic collapse better)
        - Uses fitness percentile range (detects local optima trapping)
        - Uses condition number directly (detects rank deficiency)
        - Uses population spread ratio (detects severe elongation)

        This makes the metric more sensitive to extreme states that cause
        catastrophic failure on multimodal/ill-conditioned tasks.
        """
        pop_diffs = self.population - self.mean
        max_sq_dist = np.max(np.sum(pop_diffs ** 2, axis=1))
        expected_max_sq = self.dim * ((self.ub[0] - self.lb[0]) / 3.0) ** 2
        spread_norm = np.sqrt(max_sq_dist) / np.sqrt(max(expected_max_sq, 1e-10))
        spread_norm = np.clip(spread_norm, 0.0, 2.0)

        sorted_fit = np.sort(self.fitness)
        fit_range = sorted_fit[-1] - sorted_fit[0]
        fit_median = sorted_fit[len(sorted_fit) // 2]
        fit_p10 = sorted_fit[max(0, len(sorted_fit) // 10 - 1)]
        fit_percentile_range = (fit_median - fit_p10) / (abs(fit_median) + abs(fit_p10) + 1e-10)
        fit_percentile_range = np.clip(fit_percentile_range, 0.0, 1.0)

        eigvals = np.linalg.eigvalsh(self.C)
        eigvals = np.clip(eigvals, 1e-15, None)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        cond = eig_max / max(eig_min, 1e-15)
        cond_norm = 1.0 / np.log1p(cond)
        cond_norm = np.clip(cond_norm, 0.0, 1.0)

        diag = np.diag(self.C)
        diag = np.maximum(diag, 1e-15)
        max_var = np.max(diag)
        min_var = np.min(diag)
        spread_ratio = min_var / max_var
        spread_ratio = np.clip(spread_ratio, 0.0, 1.0)

        diversity = (
            0.25 * spread_norm +
            0.35 * fit_percentile_range +
            0.25 * cond_norm +
            0.15 * spread_ratio
        )

        return float(np.clip(diversity, 1e-15, None))
    
    def _compute_diversity(self):
        """Dispatch to the selected diversity computation strategy."""
        if self.current_diversity_op == 0:
            return self._compute_diversity_original()
        elif self.current_diversity_op == 1:
            return self._compute_diversity_variant_02()
        elif self.current_diversity_op == 2:
            return self._compute_diversity_variant_04()
        elif self.current_diversity_op == 3:
            return self._compute_diversity_variant_06()
        elif self.current_diversity_op == 4:
            return self._compute_diversity_variant_08()
        else:
            return self._compute_diversity_variant_09()
    
    def _check_stagnation(self):
        """Track stagnation counter for restart logic."""
        if self.f_opt < self.f_opt_prev - 1e-12:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        self.f_opt_prev = self.f_opt
    
    def _restart_if_needed(self):
        """Restart population if stagnated or diversity lost."""
        diversity = self._compute_diversity()
        
        if (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            np.any(np.isnan(self.C))):
            
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]
            saved_diversity_op = self.current_diversity_op
            
            self._initialize_population()
            
            self.population[0] = elite
            self.fitness[0] = elite_fit
            self.f_opt = float(np.asarray(elite_fit).flatten()[0])
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0
            
            self.current_diversity_op = saved_diversity_op
