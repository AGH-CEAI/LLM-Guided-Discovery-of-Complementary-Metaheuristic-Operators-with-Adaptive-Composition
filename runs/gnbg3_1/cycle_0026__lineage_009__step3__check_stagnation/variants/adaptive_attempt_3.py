import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best stagnation detection
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 6 stagnation detection strategies (from benchmark winners)
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
        self.num_operators = 6
        self.operator_names = ['variant_02', 'variant_03', 'variant_04', 'variant_07', 'variant_08', 'variant_10']
        
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
            self._adapt_covariance()
            
            # Select and apply stagnation detection operator
            self._select_operator_thompson()
            self._check_stagnation()
            
            # Update operator rewards based on improvement
            self._update_operator_rewards()
            
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

        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0

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
        # Reset stagnation-specific state
        if hasattr(self, 'stagnation_momentum'):
            self.stagnation_momentum = 0.0
        if hasattr(self, 'fit_history'):
            self.fit_history = []
        if hasattr(self, 'cond_history'):
            self.cond_history = []
        if hasattr(self, 'diversity_stagnation_counter'):
            self.diversity_stagnation_counter = 0
        if hasattr(self, 'condition_stagnation_counter'):
            self.condition_stagnation_counter = 0
        if hasattr(self, 'stagnation_start_f'):
            self.stagnation_start_f = self.f_opt
        if hasattr(self, 'improvement_streak'):
            self.improvement_streak = 0
        if hasattr(self, 'prev_sigma'):
            self.prev_sigma = self.sigma
        if hasattr(self, 'aggressive_restart_pending'):
            self.aggressive_restart_pending = False
    
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
    
    def _adapt_covariance(self):
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
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _update_operator_rewards(self):
        """Credit assignment via exponentially-weighted cumulative improvement with stagnation awareness."""
        improvement = max(0.0, float(self.f_opt_prev) - float(self.f_opt))

        # Initialize cumulative tracking if needed
        if not hasattr(self, 'operator_cumulative_reward'):
            self.operator_cumulative_reward = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum'):
            self.operator_decay_sum = np.zeros(self.num_operators)

        # Apply exponential decay to accumulated reward
        decay = 0.95
        self.operator_cumulative_reward *= decay
        self.operator_decay_sum *= decay

        # Log-scaled improvement reward
        if improvement > 0:
            reward = float(np.log1p(improvement * 1e10) / 10.0)
        else:
            reward = 0.0

        # Diversity metric
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = float(np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0))

        # Exploration bonus when stagnant and diversity is low
        if self.stagnation_counter > self.max_stagnation // 2 and diversity < 0.1:
            reward *= 2.0

        # Bonus for making progress when near convergence
        if improvement > 0 and diversity < 0.3:
            reward *= 1.5

        # Update cumulative weighted reward
        self.operator_cumulative_reward[self.current_operator] += reward
        self.operator_decay_sum[self.current_operator] += 1.0

        # Normalize by decay sum to get comparable reward values
        norm = max(self.operator_decay_sum[self.current_operator], 1.0)
        normalized_reward = float(self.operator_cumulative_reward[self.current_operator]) / norm
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
    
    def _check_stagnation(self):
        """Dispatch to the selected stagnation detection strategy."""
        if self.current_operator == 0:
            self._check_stagnation_variant_02()
        elif self.current_operator == 1:
            self._check_stagnation_variant_03()
        elif self.current_operator == 2:
            self._check_stagnation_variant_04()
        elif self.current_operator == 3:
            self._check_stagnation_variant_07()
        elif self.current_operator == 4:
            self._check_stagnation_variant_08()
        else:
            self._check_stagnation_variant_10()
    
    def _check_stagnation_variant_02(self):
        """Track stagnation counter with scale-adaptive threshold.

        Key insight: The original fixed threshold (1e-12) is orders of magnitude
        too small for high-error tasks (error ~10-500). For Task 17 (error ~538),
        even a 0.001 improvement is meaningful, but 1e-12 is numerically invisible.

        This version uses: threshold = max(1e-6 * |f_opt|, 1e-10)
        - High-error tasks (|f_opt|~100): threshold ~1e-4 → proper stagnation detection
        - Low-error tasks (|f_opt|~1e-8): threshold ~1e-10 → fine-grained detection
        """
        improvement = float(self.f_opt_prev) - float(self.f_opt)

        # Adaptive threshold: scale with problem difficulty
        base_scale = max(abs(float(self.f_opt)), 1.0)
        rel_threshold = 1e-6 * base_scale
        threshold = max(rel_threshold, 1e-10)

        if improvement > threshold:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        # Scale max_stagnation with problem difficulty for harder tasks
        if abs(float(self.f_opt)) > 10.0:
            self.max_stagnation = 150 + self.dim * 5
        elif abs(float(self.f_opt)) > 1.0:
            self.max_stagnation = 100 + self.dim * 4
        else:
            self.max_stagnation = 50 + self.dim * 3

        self.f_opt_prev = self.f_opt
    
    def _check_stagnation_variant_03(self):
        """Track stagnation using momentum-based improvement and covariance condition monitoring."""
        if not hasattr(self, 'stagnation_momentum'):
            self.stagnation_momentum = 0.0
            self.fit_history = []
            self.cond_history = []

        # Track fitness history for momentum calculation
        self.fit_history.append(float(self.f_opt))
        if len(self.fit_history) > 20:
            self.fit_history.pop(0)

        # Track covariance condition number history
        eigvals = np.linalg.eigvalsh(self.C)
        eigvals = np.clip(eigvals, 1e-15, None)
        cond = float(np.max(eigvals)) / float(np.min(eigvals))
        self.cond_history.append(cond)
        if len(self.cond_history) > 20:
            self.cond_history.pop(0)

        # Compute relative improvement momentum
        if len(self.fit_history) >= 5:
            recent_window = self.fit_history[-5:]
            early_window = self.fit_history[:5] if len(self.fit_history) >= 5 else self.fit_history

            # Relative improvement: how much did fitness improve relative to scale?
            scale = max(abs(float(self.f_opt)), 1.0, abs(self.fit_history[0]))
            recent_improvement = (np.mean(early_window) - np.mean(recent_window)) / scale
            recent_improvement = max(recent_improvement, 0.0)

            # Momentum: EMA of improvement rate
            self.stagnation_momentum = 0.7 * self.stagnation_momentum + 0.3 * recent_improvement
        else:
            self.stagnation_momentum = 0.0

        # Stagnation detected if: no momentum AND condition is problematic
        momentum_threshold = 1e-6
        cond_healthy = np.mean(self.cond_history) < 1e7 if self.cond_history else True

        if self.stagnation_momentum < momentum_threshold and not cond_healthy:
            self.stagnation_counter += 3  # Accelerated stagnation on ill-conditioned state
        elif self.stagnation_momentum < momentum_threshold:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)

        self.f_opt_prev = self.f_opt
    
    def _check_stagnation_variant_04(self):
        """Multi-dimensional stagnation detection with diversity and condition awareness."""
        # Track fitness improvement
        if float(self.f_opt) < float(self.f_opt_prev) - 1e-12:
            self.stagnation_counter = 0
            self.stagnation_start_f = float(self.f_opt)
        else:
            self.stagnation_counter += 1

        self.f_opt_prev = self.f_opt

        # Initialize stagnation tracking
        if not hasattr(self, 'stagnation_start_f'):
            self.stagnation_start_f = float(self.f_opt)
        if not hasattr(self, 'diversity_stagnation_counter'):
            self.diversity_stagnation_counter = 0
        if not hasattr(self, 'condition_stagnation_counter'):
            self.condition_stagnation_counter = 0

        # Compute current diversity and condition
        pop_diffs = self.population - self.mean
        max_sq_dist = float(np.max(np.sum(pop_diffs ** 2, axis=1)))
        expected_max_sq = self.dim * ((self.ub[0] - self.lb[0]) / 3.0) ** 2
        current_diversity = float(np.sqrt(max_sq_dist)) / float(np.sqrt(max(expected_max_sq, 1e-10)))
        current_diversity = float(np.clip(current_diversity, 0.0, 2.0))

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = float(np.min(eigvals))
        eig_max = float(np.max(eigvals))
        current_cond = eig_max / max(eig_min, 1e-15)

        # Track diversity stagnation
        if current_diversity < 0.05:
            self.diversity_stagnation_counter += 1
        else:
            self.diversity_stagnation_counter = max(0, self.diversity_stagnation_counter - 1)

        # Track condition stagnation
        if current_cond > 1e6:
            self.condition_stagnation_counter += 1
        else:
            self.condition_stagnation_counter = max(0, self.condition_stagnation_counter - 1)

        # Compute relative improvement rate (for detecting slow progress on multimodal tasks)
        f_range = abs(float(self.stagnation_start_f)) + 1e-10
        relative_stagnation = abs(float(self.f_opt) - float(self.stagnation_start_f)) / f_range

        # Store metrics for restart logic
        self.relative_stagnation = relative_stagnation
        self.current_diversity = current_diversity
        self.current_condition = current_cond
    
    def _check_stagnation_variant_07(self):
        """Multi-signal stagnation detection with covariance health monitoring."""
        # Signal 1: Improvement-based stagnation (adaptive threshold)
        fitness_scale = max(abs(float(self.f_opt)), 1.0)
        adaptive_threshold = 1e-6 * fitness_scale
        if float(self.f_opt) < float(self.f_opt_prev) - adaptive_threshold:
            self.stagnation_counter = 0
            if hasattr(self, 'improvement_streak'):
                self.improvement_streak += 1
            else:
                self.improvement_streak = 1
        else:
            self.stagnation_counter += 1
            if hasattr(self, 'improvement_streak'):
                self.improvement_streak = 0

        # Signal 2: Covariance health (detect ill-conditioning before collapse)
        try:
            eigvals = np.linalg.eigvalsh(self.C)
            eigvals = np.clip(eigvals, 1e-15, None)
            cond = float(np.max(eigvals)) / float(np.min(eigvals))
            cond_stagnation = cond > 1e6
        except:
            cond_stagnation = True

        # Signal 3: Step-size stagnation (sigma not changing)
        if hasattr(self, 'prev_sigma'):
            sigma_ratio = float(self.sigma) / max(float(self.prev_sigma), 1e-15)
            sigma_stagnation = 0.95 < sigma_ratio < 1.05
        else:
            sigma_stagnation = False
        self.prev_sigma = float(self.sigma)

        # Signal 4: Diversity collapse (population converging)
        pop_spread = float(np.mean(np.std(self.population, axis=0)))
        expected_spread = (self.ub[0] - self.lb[0]) / 6.0
        diversity_stagnation = pop_spread < 0.1 * expected_spread

        # Signal 5: Fitness plateau (no improvement in many generations)
        plateau_stagnation = self.stagnation_counter > self.max_stagnation // 3

        # Combined stagnation score (any strong signal triggers restart consideration)
        combined_stagnation = (
            self.stagnation_counter > self.max_stagnation // 2 or
            cond_stagnation or
            (sigma_stagnation and plateau_stagnation) or
            diversity_stagnation
        )

        # If multiple mild signals, also consider stagnant
        mild_signals = int(plateau_stagnation) + int(cond_stagnation) + int(sigma_stagnation)
        if mild_signals >= 2:
            combined_stagnation = True

        # Update f_opt_prev
        self.f_opt_prev = self.f_opt

        # Trigger aggressive restart check in _restart_if_needed
        if combined_stagnation and not hasattr(self, 'aggressive_restart_pending'):
            self.aggressive_restart_pending = True
        elif not combined_stagnation and hasattr(self, 'aggressive_restart_pending'):
            self.aggressive_restart_pending = False
    
    def _check_stagnation_variant_08(self):
        """Track stagnation using adaptive relative threshold and diversity signal."""
        # Adaptive threshold: scale to problem difficulty
        scale = max(1.0, abs(float(self.f_opt)))
        threshold = max(1e-12, 1e-6 * scale)

        # Primary signal: best fitness improvement
        improved = float(self.f_opt) < float(self.f_opt_prev) - threshold

        # Secondary signal: population diversity (avoid restarting when still exploring)
        if hasattr(self, 'population') and len(self.population) > 1:
            pop_diffs = self.population - self.mean
            max_sq_dist = float(np.max(np.sum(pop_diffs ** 2, axis=1)))
            expected_max_sq = self.dim * ((self.ub[0] - self.lb[0]) / 3.0) ** 2
            spread = float(np.sqrt(max_sq_dist)) / float(np.sqrt(max(expected_max_sq, 1e-10)))
            # Still exploring if population has meaningful spread
            still_exploring = spread > 0.05
        else:
            still_exploring = False

        # Reset stagnation if improved OR still exploring
        if improved or still_exploring:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        self.f_opt_prev = self.f_opt
    
    def _check_stagnation_variant_10(self):
        """Multi-signal stagnation detection with adaptive thresholds and failure mode recognition."""
        # Use relative improvement threshold scaled to current fitness magnitude
        rel_threshold = 1e-6 * max(1.0, abs(float(self.f_opt)))
        actual_threshold = max(rel_threshold, 1e-12)

        if float(self.f_opt) < float(self.f_opt_prev) - actual_threshold:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

            # Detect fitness collapse: if population fitness range is tiny relative to best,
            # we have converged to a local optimum and need to restart NOW
            if len(self.fitness) >= 2:
                fitness_range = float(np.max(self.fitness)) - float(np.min(self.fitness))
                # Scale threshold by problem difficulty indicator
                if fitness_range < 1e-8 * max(1.0, abs(float(self.f_opt))):
                    self.stagnation_counter += 2  # Accelerate restart

            # Detect covariance collapse: ill-conditioned C means we've lost exploration ability
            try:
                eigvals = np.linalg.eigvalsh(self.C)
                eig_min = float(np.min(eigvals))
                eig_max = float(np.max(eigvals))
                if eig_max > 0 and eig_min / eig_max < 1e-10:
                    self.stagnation_counter += 2  # Accelerate restart on rank deficiency
            except:
                pass

        self.f_opt_prev = self.f_opt
    
    def _compute_diversity(self):
        """Compute diversity using percentile-robust and condition-aware metrics.

        Key differences from ESS-based approach:
        - Uses MAX-normalized spread (catches catastrophic collapse better)
        - Uses fitness percentile range (detects local optima trapping)
        - Uses condition number directly (detects rank deficiency)
        - Uses population spread ratio (detects severe elongation)

        This makes the metric more sensitive to extreme states that cause
        catastrophic failure on multimodal/ill-conditioned tasks.
        """
        # Max-normalized spread: captures catastrophic collapse better than trace
        pop_diffs = self.population - self.mean
        max_sq_dist = float(np.max(np.sum(pop_diffs ** 2, axis=1)))
        expected_max_sq = self.dim * ((self.ub[0] - self.lb[0]) / 3.0) ** 2
        spread_norm = float(np.sqrt(max_sq_dist)) / float(np.sqrt(max(expected_max_sq, 1e-10)))
        spread_norm = float(np.clip(spread_norm, 0.0, 2.0))

        # Fitness percentile range: detects local optima trapping
        sorted_fit = np.sort(self.fitness)
        fit_range = float(sorted_fit[-1]) - float(sorted_fit[0])
        fit_median = float(sorted_fit[len(sorted_fit) // 2])
        fit_p10 = float(sorted_fit[max(0, len(sorted_fit) // 10 - 1)])
        fit_percentile_range = (fit_median - fit_p10) / (abs(fit_median) + abs(fit_p10) + 1e-10)
        fit_percentile_range = float(np.clip(fit_percentile_range, 0.0, 1.0))

        # Condition number: directly detects rank deficiency
        eigvals = np.linalg.eigvalsh(self.C)
        eigvals = np.clip(eigvals, 1e-15, None)
        eig_min = float(np.min(eigvals))
        eig_max = float(np.max(eigvals))
        cond = eig_max / max(eig_min, 1e-15)
        cond_norm = 1.0 / float(np.log1p(cond))
        cond_norm = float(np.clip(cond_norm, 0.0, 1.0))

        # Spread ratio: detects severe elongation (ratio of max to min axis lengths)
        diag = np.diag(self.C)
        diag = np.maximum(diag, 1e-15)
        max_var = float(np.max(diag))
        min_var = float(np.min(diag))
        spread_ratio = min_var / max_var
        spread_ratio = float(np.clip(spread_ratio, 0.0, 1.0))

        # Weighted combination: emphasize condition and spread ratio for ill-conditioned tasks
        diversity = (
            0.25 * spread_norm +
            0.35 * fit_percentile_range +
            0.25 * cond_norm +
            0.15 * spread_ratio
        )

        return float(np.clip(diversity, 1e-15, None))
    
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
