```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 5 covariance adaptation strategies (original + 4 variants)
    - Thompson Sampling for operator selection
    - Sliding window credit assignment
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
        
        # Adaptive operator selection parameters
        self.num_operators = 5
        self.operator_names = ['original', 'variant_01', 'variant_06', 'variant_08', 'variant_09']
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_operators)
        self.beta = np.ones(self.num_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 10
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators)
        self.selection_counts = np.zeros(self.num_operators)
        
        # Runtime state
        self.generation = 0
        self.current_operator = 0
        self.L = None
        
        # ==== ADAPTIVE DIVERSITY SELECTION ====
        self.num_diversity_strategies = 7
        self.diversity_alpha = np.ones(self.num_diversity_strategies)
        self.diversity_beta = np.ones(self.num_diversity_strategies)
        self.diversity_strategy_rewards = {i: [] for i in range(self.num_diversity_strategies)}
        self.diversity_strategy_counts = np.zeros(self.num_diversity_strategies)
        self.current_diversity_strategy = 0
        self.diversity_reward_window = 15
        
        # Track diversity-improvement correlation for reward
        self.prev_diversity = 0.0
        self.diversity_history = {i: [] for i in range(self.num_diversity_strategies)}
        self.improvement_history = []
        
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
            
            # Select and apply covariance adaptation operator
            self._select_operator_thompson()
            self._adapt_covariance()
            
            # Update operator rewards based on improvement
            self._update_operator_rewards()
            
            # Adaptive diversity strategy selection and update
            self._select_diversity_strategy_thompson()
            self._update_diversity_strategy_rewards()
            
            self._check_stagnation()
            self._restart_if_needed()
        
        return self.f_opt, self.x_opt
    
    def _initialize_population(self):
        """Initialize population using Latin Hypercube sampling."""
        samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        
        for d in range(self.dim):
            bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
            perm = np.random.permutation(self.NP)
            samples[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(self.NP)
        
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
        
        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0
        self.current_diversity_strategy = 0
        
        self._reset_operator_state()
        self._reset_diversity_state()
    
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
    
    def _reset_diversity_state(self):
        """Reset diversity tracking state."""
        self.prev_diversity = 0.0
        self.diversity_history = {i: [] for i in range(self.num_diversity_strategies)}
        self.improvement_history = []
    
    def _sample_trials_batch(self):
        """Sample new trial population using Cholesky decomposition."""
        # Ensure positive definiteness before Cholesky
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        # Compute Cholesky factor L where C = L @ L.T
        # More efficient than eigendecomposition for sampling
        try:
            L = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            # Fallback: use diagonal approximation
            diag_C = np.diag(self.C)
            diag_C = np.maximum(diag_C, 1e-10)
            L = np.diag(np.sqrt(diag_C))

        # Sample from standard normal and transform
        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)

        # Clip to bounds
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
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _update_operator_rewards(self):
        """Credit assignment via exponentially-weighted cumulative improvement with stagnation awareness."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)

        # Initialize cumulative tracking if needed
        if not hasattr(self, 'operator_cumulative_reward'):
            self.operator_cumulative_reward = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum'):
            self.operator_decay_sum = np.zeros(self.num_operators)

        # Apply exponential decay to accumulated reward (recent rewards weighted more)
        decay = 0.95
        self.operator_cumulative_reward *= decay
        self.operator_decay_sum *= decay

        # Log-scaled improvement reward (more sensitive to small improvements)
        if improvement > 0:
            reward = float(np.log1p(improvement * 1e10) / 10.0)
        else:
            reward = 0.0

        # Explicit stagnation detection
        is_stagnant = self.stagnation_counter > self.max_stagnation // 2

        # Diversity metric
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        # Exploration bonus when stagnant and diversity is low
        if is_stagnant and diversity < 0.1:
            reward *= 2.0

        # Update cumulative weighted reward
        self.operator_cumulative_reward[self.current_operator] += reward
        self.operator_decay_sum[self.current_operator] += 1.0

        # Normalize by decay sum to get comparable reward values
        norm = max(self.operator_decay_sum[self.current_operator], 1.0)
        normalized_reward = self.operator_cumulative_reward[self.current_operator] / norm
        normalized_reward = float(np.clip(normalized_reward, -10.0, 10.0))

        op = self.current_operator
        self.operator_rewards[op].append(normalized_reward)

        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)

        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            var_reward = float(np.var(self.operator_rewards[op]))
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.beta[op] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = 1.0 + sum_reward
            else:
                self.beta[op] = 1.0 - sum_reward
    
    # ==== ADAPTIVE DIVERSITY STRATEGY SELECTION ====
    
    def _select_diversity_strategy_thompson(self):
        """Select diversity computation strategy using Thompson Sampling."""
        samples = np.random.beta(self.diversity_alpha, self.diversity_beta)
        self.current_diversity_strategy = int(np.argmax(samples))
        self.diversity_strategy_counts[self.current_diversity_strategy] += 1
        return self.current_diversity_strategy
    
    def _compute_diversity_strategy(self, strategy_idx):
        """Compute diversity using the specified strategy."""
        if strategy_idx == 0:
            return self._diversity_original()
        elif strategy_idx == 1:
            return self._diversity_variant_01()
        elif strategy_idx == 2:
            return self._diversity_variant_02()
        elif strategy_idx == 3:
            return self._diversity_variant_04()
        elif strategy_idx == 4:
            return self._diversity_variant_07()
        elif strategy_idx == 5:
            return self._diversity_variant_08()
        else:
            return self._diversity_variant_09()
    
    def _diversity_original(self):
        """Original diversity: mean per-dimension std."""
        return float(np.mean(np.std(self.population, axis=0)))
    
    def _diversity_variant_01(self):
        """Compute diversity as mean pairwise Euclidean distance."""
        pop = self.population
        n = pop.shape[0]
        
        if n < 2:
            return 0.0
        
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        sq_dists = np.sum(diffs ** 2, axis=2)
        distances = np.sqrt(sq_dists)
        
        upper_tri_indices = np.triu_indices(n, k=1)
        pairwise_distances = distances[upper_tri_indices]
        
        if len(pairwise_distances) == 0:
            return 0.0
        
        return float(np.mean(pairwise_distances))
    
    def _diversity_variant_02(self):
        """Compute diversity as normalized effective radius with condition number penalty."""
        pop = self.population
        n, dim = pop.shape
        
        centroid = np.mean(pop, axis=0)
        diffs = pop - centroid
        dists_sq = np.sum(diffs ** 2, axis=1)
        effective_radius = np.sqrt(np.mean(dists_sq) + 1e-15)
        
        bound_range = (self.ub[0] - self.lb[0]) / 2.0
        expected_radius = bound_range * np.sqrt(dim / 3.0)
        
        radius_ratio = effective_radius / (expected_radius + 1e-15)
        radius_ratio = np.clip(radius_ratio, 0.0, 1.0)
        
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.maximum(np.min(eigvals), 1e-15)
        eig_max = np.maximum(np.max(eigvals), 1e-15)
        cond = eig_max / eig_min
        
        cond_penalty = 1.0 / (1.0 + 0.2 * np.log1p(cond))
        
        return float(radius_ratio * cond_penalty)
    
    def _diversity_variant_04(self):
        """Compute diversity as pairwise distance + condition number fusion."""
        pop = self.population
        n = pop.shape[0]
        dim = self.dim
        
        if n < 2:
            return 0.0
        
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        triu_idx = np.triu_indices(n, k=1)
        pairwise_dists = dists[triu_idx]
        
        diameter = np.linalg.norm(self.ub - self.lb)
        if diameter < 1e-10:
            diameter = 1.0
        
        global_div = np.mean(pairwise_dists) / diameter
        global_div = np.clip(global_div, 0.0, 1.0)
        
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        
        if eig_max < 1e-10:
            eig_max = 1.0
        if eig_min < 1e-10:
            eig_min = 1e-10
        
        cond = eig_max / eig_min
        cond_inv = 1.0 / (1.0 + cond)
        cond_inv = np.clip(cond_inv, 0.0, 1.0)
        
        diversity = np.sqrt(global_div * cond_inv)
        return float(diversity)
    
    def _diversity_variant_07(self):
        """Compute diversity using effective sample size of the covariance matrix."""
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
    
    def _diversity_variant_08(self):
        """Compute population diversity as mean pairwise Euclidean distance (sqrt version)."""
        n = self.population.shape[0]
        if n < 2:
            return 0.0
        diffs = self.population[:, np.newaxis, :] - self.population[np.newaxis, :, :]
        sq_dists = np.sum(diffs * diffs, axis=2)
        upper_tri_indices = np.triu_indices(n, k=1)
        pairwise_sq_dists = sq_dists[upper_tri_indices]
        mean_sq_dist = np.mean(pairwise_sq_dists)
        if mean_sq_dist <= 0:
            return 0.0
        return float(np.sqrt(mean_sq_dist))
    
    def _diversity_variant_09(self):
        """Compute diversity using covariance eigenvalue shape (effective sample size)."""
        if self.population.shape[0] < 2:
            return 0.0
        
        centroid = np.mean(self.population, axis=0)
        dists = np.linalg.norm(self.population - centroid, axis=1)
        mean_dist = np.mean(dists)
        
        eigvals = np.linalg.eigvalsh(self.C)
        eigvals = np.maximum(eigvals, 1e-15)
        geometric_mean = np.exp(np.mean(np.log(eigvals)))
        arithmetic_mean = np.mean(eigvals)
        ess = geometric_mean / (arithmetic_mean + 1e-15)
        
        cond = np.min(eigvals) / (np.max(eigvals) + 1e-15)
        cond_score = 1.0 / (1.0 + np.log1p(max(cond, 1.0)))
        
        expected_range = (self.ub[0] - self.lb[0]) / 6.0
        spread_normalized = mean_dist / (expected_range + 1e-10)
        
        diversity = spread_normalized * (0.4 + 0.4 * ess + 0.2 * cond_score)
        return float(np.clip(diversity, 0.0, 1e6))
    
    def _update_diversity_strategy_rewards(self):
        """Update diversity strategy rewards based on prediction accuracy.
        
        Reward is based on how well the diversity metric predicts actual improvement.
        A good diversity metric should correlate with optimization progress.
        """
        # Compute all diversity strategies
        all_diversities = np.array([self._compute_diversity_strategy(i) for i in range(self.num_diversity_strategies)])
        all_diversities = np.clip(all_diversities, 1e-15, None)
        
        # Compute current improvement
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        self.improvement_history.append(improvement)
        
        if len(self.improvement_history) > self.diversity_reward_window:
            self.improvement_history.pop(0)
        
        # Store diversity history for each strategy
        for i in range(self.num_diversity_strategies):
            self.diversity_history[i].append(all_diversities[i])
            if len(self.diversity_history[i]) > self.diversity_reward_window:
                self.diversity_history[i].pop(0)
        
        # Need at least 3 samples to compute meaningful correlation
        if len(self.improvement_history) < 3:
            return
        
        # Compute correlation between each strategy's diversity and improvement
        improv_arr = np.array(self.improvement_history)
        
        for i in range(self.num_diversity_strategies):
            if len(self.diversity_history[i]) < 3:
                continue
            
            div_arr = np.array(self.diversity_history[i])
            
            # Compute Pearson correlation coefficient
            div_mean = np.mean(div_arr)
            imp_mean = np.mean(improv_arr)
            
            num = np.sum((div_arr - div_mean) * (improv_arr - imp_mean))
            den = np.sqrt(np.sum((div_arr - div_mean)**2) * np.sum((improv_arr - imp_mean)**2))
            
            if den > 1e-15:
                correlation = num / den
            else:
                correlation = 0.0
            
            # Also consider predictive power: diversity should decrease when stagnating
            # and increase when making progress
            if len(div_arr) >= 2:
                div_trend = div_arr[-1] - div_arr[0]
                imp_trend = imp_arr[-1] - imp_arr[0]
                trend_alignment = 1.0 if (div_trend * imp_trend >= 0) else 0.5
            else:
                trend_alignment = 1.0
            
            # Combined reward: correlation + trend alignment
            reward = float(np.clip(0.7 * correlation + 0.3 * trend_alignment, -1.0, 1.0))
            
            # Bonus for the currently selected strategy if it led to improvement
            if i == self.current_diversity_strategy:
                if improvement > 0:
                    reward += 0.2
                # Small penalty if stagnant
                if self.stagnation_counter > self.max_stagnation // 3:
                    if all_diversities[i] < 0.01:  # Low diversity during stagnation is bad
                        reward -= 0.3
            
            # Update reward window
            self.diversity_strategy_rewards[i].append(reward)
            if len(self.diversity_strategy_rewards[i]) > self.diversity_reward_window:
                self.diversity_strategy_rewards[i].pop(0)
            
            # Update Beta distribution parameters
            n = len(self.diversity_strategy_rewards[i])
            if n >= 3:
                mean_r = float(np.mean(self.diversity_strategy_rewards[i]))
                var_r = float(np.var(self.diversity_strategy_rewards[i]))
                
                # Transform to [0,1] scale for Beta
                scaled_mean = (mean_r + 1.0) / 2.0
                scaled_mean = np.clip(scaled_mean, 0.01, 0.99)
                
                denom = max(var_r * n + 1e-10, 1e-10)
                self.diversity_alpha[i] = float(max(1.0, scaled_mean * (scaled_mean * (1 - scaled_mean) * (n - 1) / denom + 1)))
                self.diversity_beta[i] = float(max(1.0, (1 - scaled_mean) * ((1 - scaled_mean) * scaled_mean * (n - 1) / denom + 1)))
    
    def _compute_diversity(self):
        """Compute population diversity using the adaptively selected strategy."""
        return self._compute_diversity_strategy(self.current_diversity_strategy)
    
    def _adapt_covariance(self):
        """Dispatch to the selected covariance adaptation strategy."""
        if self.current_operator == 0:
            self._adapt_covariance_original()
        elif self.current_operator == 1:
            self._adapt_covariance_variant_01()
        elif self.current_operator == 2:
            self._adapt_covariance_variant_06()
        elif self.current_operator == 3:
            self._adapt_covariance_variant_08()
        else:
            self._adapt_covariance_variant_09()
    
    def _adapt_covariance_original(self):
        """Original covariance adaptation (baseline)."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_01(self):
        """Archive-guided covariance perturbation for escaping deceptive local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        if not hasattr(self, 'archive_positions'):
            self.archive_positions = []
            self.archive_fitness = []
            self.archive_generations = []

        min_dist_threshold = 1e-3 * (self.ub[0] - self.lb[0])
        is_distinct = True
        for arch_pos in self.archive_positions:
            dist = np.linalg.norm(self.x_opt - arch_pos)
            if dist < min_dist_threshold:
                is_distinct = False
                break

        if is_distinct and len(self.archive_positions) < 10:
            self.archive_positions.append(self.x_opt.copy())
            self.archive_fitness.append(self.f_opt)
            self.archive_generations.append(self.generation)
        elif is_distinct and len(self.archive_positions) >= 10:
            worst_idx = np.argmax(self.archive_fitness)
            self.archive_positions[worst_idx] = self.x_opt.copy()
            self.archive_fitness[worst_idx] = self.f_opt
            self.archive_generations[worst_idx] = self.generation

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        eig_spread = eig_min / (eig_max + 1e-10)
        cond = eig_max / (max(eig_min, 1e-10))

        is_trapped = (rel_var < 1e-3) or (cond > 1e5) or (eig_spread < 1e-5)

        escape_perturb = np.zeros((self.dim, self.dim))
        if is_trapped and len(self.archive_positions) >= 3:
            centroid = np.mean(self.archive_positions, axis=0)
            toward_centroid = centroid - self.mean
            norm_toward = np.linalg.norm(toward_centroid)
            if norm_toward > 1e-10:
                toward_centroid /= norm_toward
                for arch_pos in self.archive_positions:
                    dir_to_arch = arch_pos - self.mean
                    norm_dir = np.linalg.norm(dir_to_arch)
                    if norm_dir > 1e-10:
                        dir_to_arch /= norm_dir
                        escape_perturb += 0.3 * np.outer(dir_to_arch, dir_to_arch)
            escape_perturb += 0.1 * np.eye(self.dim)

        if is_trapped:
            ccov_scale = 0.8
            cc_scale = 0.8
        else:
            ccov_scale = min(1.0, 0.2 + 2.0 * rel_var)
            cc_scale = 1.0

        ccov_1 = ccov_scale * 1.0 / (self.dim + 2.0)
        ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        cc_adapt = cc_scale * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)

        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
                  ccov_1 * rank_one +
                  ccov_mu * rank_mu)

        if is_trapped and len(self.archive_positions) >= 3:
            self.C = self.C + escape_perturb

        self.C = self._ensure_positive_definite(self.C)

        if cond > 1e8 or eig_spread < 1e-8:
            self.C = np.eye(self.dim) * np.mean(eigvals)
            self.pc = np.zeros(self.dim)
            self.L = None
    
    def _adapt_covariance_variant_06(self):
        """Diversity-sensitive learning rate scaling."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)
        
        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
        
        is_converging = (rel_var < 0.01) and (eig_spread < 1e-4)
        
        if is_converging:
            scale = 10.0
        else:
            scale = 1.0 + 5.0 * min(rel_var, 0.1)
        
        ccov_scaled = min(self.ccov * scale, 0.5)
        cc_scaled = min(self.cc * (1.0 + scale * 0.5), 0.3)
        
        self.pc = (1.0 - cc_scaled) * self.pc + np.sqrt(cc_scaled * (2.0 - cc_scaled)) * y_mean
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - ccov_scaled) * self.C + 
                  ccov_scaled * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_scaled * 2.0 * rank_mu))
        
        if is_converging:
            self.C += 0.05 * np.eye(self.dim)
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_08(self):
        """Active CMA-ES: Reduce learning rate along unfavorable eigendirections."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu_pos = np.zeros((self.dim, self.dim))
        rank_mu_neg = np.zeros((self.dim, self.dim))

        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            weight = self.weights[i]
            outer = np.outer(diff, diff)

            if weight > 0:
                rank_mu_pos += weight * outer
            else:
                rank_mu_neg += abs(weight) * outer

        pos_sum = max(np.sum(self.weights[:self.mu][self.weights[:self.mu] > 0]), 1e-10)
        neg_sum = max(np.sum(np.abs(self.weights[:self.mu][self.weights[:self.mu] < 0])), 1e-10)

        rank_mu_pos /= pos_sum
        rank_mu_neg /= neg_sum

        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * rank_one +
                  (1.0 - 1.0 / self.mueff) * self.ccov * rank_mu_pos -
                  self.ccov * 0.4 * rank_mu_neg)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_09(self):
        """Exploration temperature with fitness gradient tracking for escaping local optima."""
        if not hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.exploration_temp = 1.0

        delta_f_opt = self.prev_f_opt - self.f_opt
        self.prev_f_opt = self.f_opt

        fitness_gradient = max(abs(delta_f_opt), 1e-15)
        temp_target = np.clip(1.0 / (1.0 + np.log1p(fitness_gradient * 1e5)), 0.05, 5.0)
        self.exploration_temp = 0.95 * self.exploration_temp + 0.05 * temp_target

        base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        ccov_adaptive = base_ccov * self.exploration_temp
        ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 0.5)

        cc_adaptive = self.cc * (1.0 / max(self.exploration_temp, 0.5))
        cc_adaptive = np.clip(cc_adaptive, 0.01, 0.3)

        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - cc_adaptive) * self.pc + np.sqrt(cc_adaptive * (2.0 - cc_adaptive)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        total_w = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_w += abs(w)

        if total_w > 0:
            rank_mu /= total_w

        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)
    
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
            
            self._initialize_population()
            
            self.population[0] = elite
            self.fitness[0] = elite_fit
            self.f_opt = float(np.asarray(elite_fit).flatten()[0])
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0
```