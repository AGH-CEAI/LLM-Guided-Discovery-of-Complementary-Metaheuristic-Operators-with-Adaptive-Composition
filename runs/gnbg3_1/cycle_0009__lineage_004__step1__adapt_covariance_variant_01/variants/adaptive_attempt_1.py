import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 8 covariance adaptation strategies (based on benchmark results)
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
        
        # Adaptive operator selection parameters - 8 operators from benchmark
        self.num_operators = 8
        self.operator_names = [
            'original', 'variant_01', 'variant_03', 'variant_04',
            'variant_06', 'variant_07', 'variant_09', 'variant_10'
        ]
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_operators, dtype=np.float64)
        self.beta = np.ones(self.num_operators, dtype=np.float64)
        
        # Sliding window for credit assignment
        self.reward_window_size = 15
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators, dtype=np.float64)
        self.selection_counts = np.zeros(self.num_operators, dtype=np.float64)
        
        # Runtime state
        self.generation = 0
        self.current_operator = 0
        self.L = None
        
        # Track per-operator recent fitness improvements
        self.operator_fitness_history = {i: [] for i in range(self.num_operators)}
    
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
        f_best_raw = self.fitness[best_idx]
        self.f_opt = float(np.asarray(f_best_raw).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        
        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0
        
        self._reset_operator_state()
    
    def _reset_operator_state(self):
        """Reset all operator-specific state variables."""
        self.L = None
        
        # Reset all variant-specific attributes
        attrs_to_reset = [
            'y_mean_ema', 'pc_weighted', 'p_cross', 'prev_y_mean',
            'archive_positions', 'archive_fitness', 'archive_generations',
            'eigval_history', 'eigval_ema', 'p_escape'
        ]
        for attr in attrs_to_reset:
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except:
                    pass
        
        if hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.improvement_ema = 1.0
        else:
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
        trial_best_fit_raw = self.trial_fitness[trial_best_idx]
        trial_best_fit = float(np.asarray(trial_best_fit_raw).flatten()[0])
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
                    total_improvement += max(float(np.asarray(diff).flatten()[0]), 0.0)
                recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = recent_improved / recent_total
        success_rate = float(np.clip(success_rate, 0.0, 1.0))

        avg_improvement = total_improvement / max(recent_improved, 1)
        improvement_magnitude = float(np.log1p(max(avg_improvement, 1e-15)))

        if not hasattr(self, 'improvement_ema'):
            self.improvement_ema = improvement_magnitude
        self.improvement_ema = 0.7 * self.improvement_ema + 0.3 * improvement_magnitude

        target_rate = 0.25
        adaptation = (success_rate - target_rate) / target_rate
        adaptation += 0.3 * np.tanh(self.improvement_ema - 1.0)
        adaptation = float(np.clip(adaptation, -0.8, 0.8))

        pop_variance = float(np.mean(np.var(self.population, axis=0)))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = float(np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0))

        diversity_boost = 1.0 + 2.0 * (1.0 - diversity)
        diversity_boost = float(np.clip(diversity_boost, 0.5, 3.0))

        damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim)) / diversity_boost
        damping_adaptive = float(np.clip(damping_adaptive, 0.05, 200.0))

        self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
        self.sigma = float(np.clip(self.sigma, 1e-10, 10.0))
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        self.selection_counts[self.current_operator] += 1
        return self.current_operator
    
    def _update_operator_rewards(self):
        """Update operator rewards using sliding window credit assignment with multi-criteria scoring."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        pop_variance = float(np.mean(np.var(self.population, axis=0)))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = float(np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0))
        
        # Multi-criteria reward
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity)
        
        # Bonus for recent best improvement
        if self.stagnation_counter == 0:
            reward += 0.5
        
        # Penalty for high condition number (poor covariance health)
        try:
            eigvals = np.linalg.eigvalsh(self.C)
            cond = float(np.max(eigvals)) / max(float(np.min(eigvals)), 1e-10)
            if cond > 1e6:
                reward -= 1.0
            elif cond > 1e4:
                reward -= 0.5
        except:
            pass
        
        reward = float(np.clip(reward, -10.0, 10.0))
        
        op = self.current_operator
        self.operator_rewards[op].append(reward)
        
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)
        
        # Track fitness history for this operator
        self.operator_fitness_history[op].append(self.f_opt)
        if len(self.operator_fitness_history[op]) > self.reward_window_size:
            self.operator_fitness_history[op].pop(0)
        
        # Update Beta distribution parameters
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
                self.alpha[op] = float(max(1.0, 1.0 + sum_reward))
            else:
                self.beta[op] = float(max(1.0, 1.0 - sum_reward))
    
    def _adapt_covariance(self):
        """Dispatch to the selected covariance adaptation strategy."""
        if self.current_operator == 0:
            self._adapt_covariance_original()
        elif self.current_operator == 1:
            self._adapt_covariance_variant_01()
        elif self.current_operator == 2:
            self._adapt_covariance_variant_03()
        elif self.current_operator == 3:
            self._adapt_covariance_variant_04()
        elif self.current_operator == 4:
            self._adapt_covariance_variant_06()
        elif self.current_operator == 5:
            self._adapt_covariance_variant_07()
        elif self.current_operator == 6:
            self._adapt_covariance_variant_09()
        else:
            self._adapt_covariance_variant_10()
    
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
        """Momentum-enhanced covariance adaptation with adaptive learning rates (7 wins)."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        beta = 0.5
        if not hasattr(self, 'y_mean_ema'):
            self.y_mean_ema = np.zeros(self.dim)
        self.y_mean_ema = beta * self.y_mean_ema + (1.0 - beta) * y_mean

        eigvals = np.linalg.eigvalsh(self.C)
        eig_ratio = np.max(eigvals) / (np.min(eigvals) + 1e-10)
        cond_factor = min(1.0, 1.0 / np.log1p(eig_ratio + 1.0))

        cc_momentum = self.cc * (0.5 + 0.5 * cond_factor)
        cc_momentum = float(np.clip(cc_momentum, 0.01, 0.2))

        self.pc = (1.0 - cc_momentum) * self.pc + np.sqrt(cc_momentum * (2.0 - cc_momentum)) * self.y_mean_ema

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        ccov_adaptive = self.ccov * (0.5 + 0.5 * cond_factor)
        ccov_adaptive = float(np.clip(ccov_adaptive, 1e-10, 0.5))

        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_03(self):
        """Rank-distribution triggered exploration bursts (5 wins)."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        ranks = np.argsort(np.argsort(self.fitness))
        rank_spread = np.max(ranks) - np.min(ranks)
        rank_diversity = rank_spread / max(len(ranks) - 1, 1)

        expected_spread = 1.0
        rank_ratio = rank_diversity / max(expected_spread, 1e-10)
        rank_ratio = float(np.clip(rank_ratio, 0.01, 10.0))

        ccov_1_base = 1.0 / (self.dim + 2.0)
        ccov_mu_base = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        cc_adapt_base = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)

        if rank_ratio < 0.3:
            exploration_boost = 3.0
            diag_noise = 0.1
        elif rank_ratio < 0.5:
            exploration_boost = 2.0
            diag_noise = 0.05
        elif rank_ratio < 0.7:
            exploration_boost = 1.5
            diag_noise = 0.02
        else:
            exploration_boost = 1.0
            diag_noise = 0.0

        ccov_1 = min(ccov_1_base * exploration_boost, 0.3)
        ccov_mu = ccov_mu_base * min(exploration_boost, 2.0)
        cc_adapt = min(cc_adapt_base * exploration_boost, 0.3)

        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C + 
                  ccov_1 * rank_one + 
                  ccov_mu * rank_mu)

        if diag_noise > 0:
            eigvals = np.linalg.eigvalsh(self.C)
            noise = np.random.randn(self.dim) * diag_noise * np.mean(eigvals)
            self.C += np.diag(noise)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
        if eig_spread < 1e-8 or np.any(np.isnan(self.C)):
            self.C = np.eye(self.dim) * np.mean(eigvals)
            self.pc = np.zeros(self.dim)
            self.L = None

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_04(self):
        """Eigenvalue history smoothing with adaptive regularization (1 win)."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        eigvals = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals) / (np.maximum(np.min(eigvals), 1e-10))
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)

        if not hasattr(self, 'eigval_history'):
            self.eigval_history = []
            self.eigval_ema = eigvals.copy()

        self.eigval_history.append(eigvals.copy())
        if len(self.eigval_history) > 10:
            self.eigval_history.pop(0)

        alpha_smooth = 0.1
        self.eigval_ema = (1.0 - alpha_smooth) * self.eigval_ema + alpha_smooth * eigvals

        if len(self.eigval_history) >= 3:
            drift_ratio = np.median(eigvals) / (np.median(self.eigval_ema) + 1e-10)
            drift_ratio = float(np.clip(drift_ratio, 0.1, 10.0))
        else:
            drift_ratio = 1.0

        reg_strength = 0.0
        if cond > 1e6 or eig_spread < 1e-6:
            reg_strength = 0.1
        elif cond > 1e4 or eig_spread < 1e-4:
            reg_strength = 0.05
        elif drift_ratio < 0.5 or drift_ratio > 2.0:
            reg_strength = 0.02

        if cond > 1e7 or eig_spread < 1e-8:
            target_eigvals = np.full(self.dim, np.exp(np.mean(np.log(eigvals + 1e-10))))
            self.eigval_ema = (1.0 - alpha_smooth) * self.eigval_ema + alpha_smooth * target_eigvals

        if cond > 1e6:
            ccov_scale = 0.1
            cc_scale = 0.1
        elif cond > 1e4:
            ccov_scale = 0.3
            cc_scale = 0.3
        elif cond > 1e2:
            ccov_scale = 0.6
            cc_scale = 0.6
        else:
            ccov_scale = 1.0
            cc_scale = 1.0

        drift_penalty = 0.5 if drift_ratio < 0.5 or drift_ratio > 2.0 else 1.0
        ccov_scale *= drift_penalty
        cc_scale *= drift_penalty

        fit_var = float(np.var(self.fitness))
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        if rel_var < 1e-4:
            ccov_scale *= 0.5
            cc_scale *= 0.5

        if cond > 1e8 or eig_spread < 1e-8:
            self.C = np.eye(self.dim) * np.mean(eigvals)
            self.pc = np.zeros(self.dim)
            self.L = None
            self.eigval_ema = np.full(self.dim, np.mean(eigvals))
            return

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

        if reg_strength > 0:
            current_trace = np.trace(self.C)
            target_trace = np.sum(self.eigval_ema)
            self.C = (1.0 - reg_strength) * self.C + reg_strength * (target_trace / self.dim) * np.eye(self.dim)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_06(self):
        """Archive-guided covariance perturbation (2 wins)."""
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

        fit_var = float(np.var(self.fitness))
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
    
    def _adapt_covariance_variant_07(self):
        """Isotropic reset with aggressive exploration scaling (2 wins)."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        is_stagnant = self.stagnation_counter > self.max_stagnation // 2

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        cond = eig_max / max(eig_min, 1e-10)

        fit_var = float(np.var(self.fitness))
        rel_var = fit_var / (max(abs(self.f_opt), 1.0) ** 2 + 1e-10)
        is_converged = rel_var < 1e-3 and cond > 1e3

        if is_stagnant or is_converged or cond > 1e7:
            C_mean = np.mean(eigvals)
            self.C = np.eye(self.dim) * max(C_mean, self.sigma ** 2)
            self.pc = np.zeros(self.dim)
            self.L = None

            if self.sigma < 0.5 * (self.ub[0] - self.lb[0]) / 3.0:
                self.sigma *= 2.0
                self.sigma = float(np.clip(self.sigma, 1e-10, 10.0))

        if cond > 1e6:
            ccov_scale = 0.2
            cc_scale = 0.2
        elif cond > 1e4:
            ccov_scale = 0.4
            cc_scale = 0.4
        elif cond > 1e2:
            ccov_scale = 0.7
            cc_scale = 0.7
        else:
            ccov_scale = 1.0
            cc_scale = 1.0

        if is_stagnant or is_converged:
            ccov_scale *= 2.0
            cc_scale *= 1.5

        ccov_scale = min(ccov_scale, 2.0)
        cc_scale = min(cc_scale, 1.0)

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

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_09(self):
        """Orthogonal-escape evolution path for multimodal function escape (2 wins)."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        if not hasattr(self, 'p_escape'):
            self.p_escape = None

        if self.stagnation_counter >= self.max_stagnation // 2:
            if self.p_escape is None and np.dot(self.pc, self.pc) > 1e-20:
                v = self.pc / (np.linalg.norm(self.pc) + 1e-20)
                random_vec = np.random.randn(self.dim)
                random_vec -= np.dot(random_vec, v) * v
                norm_r = np.linalg.norm(random_vec)
                if norm_r > 1e-10:
                    self.p_escape = random_vec / norm_r

        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        if self.p_escape is not None:
            rank_escape = np.outer(self.p_escape, self.p_escape)
            self.C = ((1.0 - self.ccov) * self.C + 
                      self.ccov * rank_one + 
                      0.5 * self.ccov * rank_escape +
                      (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)
            self.p_escape *= 0.95
        else:
            self.C = ((1.0 - self.ccov) * self.C + 
                      self.ccov * rank_one + 
                      (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_10(self):
        """Aggressive covariance reset with step-size reduction (1 win)."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
        cond = np.max(eigvals) / (np.maximum(np.min(eigvals), 1e-10))

        recent_improved = 0
        if hasattr(self, 'operator_fitness_history') and self.current_operator in self.operator_fitness_history:
            history = self.operator_fitness_history[self.current_operator]
            for i in range(min(len(history), len(self.fitness))):
                if history[i] < self.fitness[i]:
                    recent_improved += 1
        success_rate = recent_improved / max(len(self.fitness), 1)

        should_reset = (cond > 1e6) or (eig_spread < 1e-6) or \
                       (success_rate < 0.05 and self.stagnation_counter > 20)

        if should_reset:
            self.C = np.eye(self.dim) * np.mean(eigvals)
            self.pc = np.zeros(self.dim)
            self.L = None

            self.sigma = max(self.sigma * 0.3, 1e-10)

            ccov_1 = 0.5 / (self.dim + 2.0)
            ccov_mu = 0.5 * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
            cc_adapt = 0.3 * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        else:
            ccov_1 = self.ccov
            ccov_mu = self.ccov
            cc_adapt = self.cc

        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
                  ccov_1 * rank_one +
                  ccov_mu * rank_mu)

        self.C = self._ensure_positive_definite(self.C)
    
    def _compute_diversity(self):
        """Compute population diversity as mean per-dimension std."""
        return float(np.mean(np.std(self.population, axis=0)))
    
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
        
        needs_restart = False
        if self.stagnation_counter > self.max_stagnation:
            needs_restart = True
        if diversity < self.min_diversity:
            needs_restart = True
        if np.any(np.isnan(self.C)):
            needs_restart = True
        
        if needs_restart:
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit_raw = self.fitness[best_idx]
            elite_fit = float(np.asarray(elite_fit_raw).flatten()[0])
            
            self._initialize_population()
            
            self.population[0] = elite
            self.fitness[0] = elite_fit
            self.f_opt = elite_fit
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0
            
            # Soft reset of operator selection to encourage exploration
            self.alpha = np.ones(self.num_operators, dtype=np.float64)
            self.beta = np.ones(self.num_operators, dtype=np.float64)
