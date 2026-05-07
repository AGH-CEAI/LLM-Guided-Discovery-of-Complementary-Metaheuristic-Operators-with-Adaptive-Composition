import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    AND restart strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 5 covariance adaptation strategies
    - 7 restart strategies (adaptive selection based on benchmark results)
    - Thompson Sampling for both operator selections
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
        
        # Adaptive operator selection parameters (covariance adaptation)
        self.num_operators = 5
        self.operator_names = ['original', 'variant_01', 'variant_06', 'variant_08', 'variant_09']
        
        # Thompson Sampling with Beta distributions for covariance operators
        self.alpha = np.ones(self.num_operators)
        self.beta = np.ones(self.num_operators)
        
        # Sliding window for credit assignment (covariance)
        self.reward_window_size = 10
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators)
        self.selection_counts = np.zeros(self.num_operators)
        
        # Adaptive restart strategy selection
        # Strategies: 0=original, 1=variant_02, 2=variant_03, 3=variant_04, 4=variant_05, 5=variant_06, 6=variant_09
        self.num_restart_strategies = 7
        self.restart_alpha = np.ones(self.num_restart_strategies)
        self.restart_beta = np.ones(self.num_restart_strategies)
        self.restart_rewards = {i: [] for i in range(self.num_restart_strategies)}
        self.restart_counts = np.zeros(self.num_restart_strategies)
        self.restart_reward_window = 5
        self.current_restart_strategy = 0
        
        # Runtime state
        self.generation = 0
        self.current_operator = 0
        self.L = None
        
        # Archive for restart strategies
        self.archive_positions = []
        self.archive_fitness = []
    
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
        """Initialize population using adaptive multi-method sampling with boundary focus."""
        samples_list = []

        # Method 1: Sobol quasi-random (60% of population)
        n_sobol = int(0.6 * self.NP)
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(self.dim, scramble=True)
            sobol_samples = sampler.random(n_sobol)
            sobol_samples = qmc.scale(sobol_samples, self.lb, self.ub)
            samples_list.append(sobol_samples)
        except Exception:
            pass

        # Method 2: Latin Hypercube for better dimension-wise independence (25% of population)
        n_lhs = int(0.25 * self.NP)
        try:
            sampler = qmc.LatinHypercube(self.dim, scramble=True)
            lhs_samples = sampler.random(n_lhs)
            lhs_samples = qmc.scale(lhs_samples, self.lb, self.ub)
            samples_list.append(lhs_samples)
        except Exception:
            pass

        # Method 3: Boundary-focused sampling with adaptive density (15% of population)
        n_boundary = self.NP - sum(len(s) for s in samples_list)
        if n_boundary > 0:
            boundary_samples = np.zeros((n_boundary, self.dim))
            u = np.random.random((n_boundary, self.dim))
            power = 3.0
            scaled = np.power(u, power) if np.random.random() > 0.5 else 1.0 - np.power(u, power)
            dim_mask = np.random.random(self.dim) < 0.3
            boundary_samples[:, dim_mask] = scaled[:, dim_mask]
            boundary_samples[:, ~dim_mask] = u[:, ~dim_mask]
            for d in range(self.dim):
                boundary_samples[:, d] = self.lb[d] + boundary_samples[:, d] * (self.ub[d] - self.lb[d])
            samples_list.append(boundary_samples)

        # Combine all samples
        if samples_list:
            samples = np.vstack(samples_list)
        else:
            samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))

        # Ensure exact size
        if samples.shape[0] > self.NP:
            samples = samples[:self.NP]
        elif samples.shape[0] < self.NP:
            extra = np.random.uniform(self.lb, self.ub, (self.NP - samples.shape[0], self.dim))
            samples = np.vstack([samples, extra])

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
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _select_restart_strategy_thompson(self):
        """Select restart strategy using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.restart_alpha, self.restart_beta)
        self.current_restart_strategy = int(np.argmax(samples))
        self.restart_counts[self.current_restart_strategy] += 1
        return self.current_restart_strategy
    
    def _update_operator_rewards(self):
        """Credit assignment via exponentially-weighted cumulative improvement with stagnation awareness."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)

        if not hasattr(self, 'operator_cumulative_reward'):
            self.operator_cumulative_reward = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum'):
            self.operator_decay_sum = np.zeros(self.num_operators)

        decay = 0.95
        self.operator_cumulative_reward *= decay
        self.operator_decay_sum *= decay

        if improvement > 0:
            reward = float(np.log1p(improvement * 1e10) / 10.0)
        else:
            reward = 0.0

        is_stagnant = self.stagnation_counter > self.max_stagnation // 2

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        if is_stagnant and diversity < 0.1:
            reward *= 2.0

        self.operator_cumulative_reward[self.current_operator] += reward
        self.operator_decay_sum[self.current_operator] += 1.0

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
    
    def _update_restart_rewards(self, restart_reward):
        """Update restart strategy rewards using sliding window credit assignment."""
        op = self.current_restart_strategy
        self.restart_rewards[op].append(float(restart_reward))

        if len(self.restart_rewards[op]) > self.restart_reward_window:
            self.restart_rewards[op].pop(0)

        n = len(self.restart_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.restart_rewards[op]))
            var_reward = float(np.var(self.restart_rewards[op]))
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.restart_alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.restart_beta[op] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.restart_rewards[op]))
            if sum_reward > 0:
                self.restart_alpha[op] = 1.0 + sum_reward
            else:
                self.restart_beta[op] = 1.0 - sum_reward
    
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
    
    def _compute_diversity(self):
        """Compute diversity using percentile-robust and condition-aware metrics."""
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
    
    def _check_stagnation(self):
        """Track stagnation counter with scale-adaptive threshold."""
        improvement = self.f_opt_prev - self.f_opt

        base_scale = max(abs(self.f_opt), 1.0)
        rel_threshold = 1e-6 * base_scale
        threshold = max(rel_threshold, 1e-10)

        if improvement > threshold:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        if abs(self.f_opt) > 10.0:
            self.max_stagnation = 150 + self.dim * 5
        elif abs(self.f_opt) > 1.0:
            self.max_stagnation = 100 + self.dim * 4
        else:
            self.max_stagnation = 50 + self.dim * 3

        self.f_opt_prev = self.f_opt
    
    def _restart_if_needed(self):
        """Adaptive restart strategy selection using Thompson Sampling."""
        diversity = self._compute_diversity()
        
        should_restart = (self.stagnation_counter > self.max_stagnation or 
                          diversity < self.min_diversity or 
                          np.any(np.isnan(self.C)))
        
        if not should_restart:
            return
        
        # Select restart strategy using Thompson Sampling
        self._select_restart_strategy_thompson()
        
        f_opt_before = self.f_opt
        
        # Apply selected restart strategy
        if self.current_restart_strategy == 0:
            self._restart_original()
        elif self.current_restart_strategy == 1:
            self._restart_variant_02()
        elif self.current_restart_strategy == 2:
            self._restart_variant_03()
        elif self.current_restart_strategy == 3:
            self._restart_variant_04()
        elif self.current_restart_strategy == 4:
            self._restart_variant_05()
        elif self.current_restart_strategy == 5:
            self._restart_variant_06()
        else:
            self._restart_variant_09()
        
        # Compute reward: improvement after restart
        restart_improvement = max(0.0, f_opt_before - self.f_opt)
        
        # Bonus for diversity improvement
        new_diversity = self._compute_diversity()
        diversity_bonus = max(0.0, new_diversity - diversity) * 10.0
        
        # Penalty for worsening best
        if self.f_opt > f_opt_before:
            worsening_penalty = min((self.f_opt - f_opt_before) / max(abs(f_opt_before), 1.0), 2.0)
        else:
            worsening_penalty = 0.0
        
        restart_reward = float(np.log1p(restart_improvement * 1e8) / 10.0 + diversity_bonus - worsening_penalty * 0.5)
        restart_reward = float(np.clip(restart_reward, -5.0, 10.0))
        
        # Update restart strategy rewards
        self._update_restart_rewards(restart_reward)
    
    def _restart_original(self):
        """Original restart: single elite preservation with reinitialization."""
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
    
    def _restart_variant_02(self):
        """Archive-guided diversity with adaptive intensity."""
        diversity = self._compute_diversity()

        stagnation_ratio = self.stagnation_counter / max(self.max_stagnation, 1)
        error_magnitude = np.log1p(max(abs(self.f_opt), 1.0))
        intensity = np.clip(0.3 * stagnation_ratio + 0.1 * error_magnitude / 10.0, 0.0, 1.0)

        best_idx = np.argmin(self.fitness)
        elite = self.population[best_idx].copy()
        elite_fit = self.fitness[best_idx]

        # Collect archive solutions
        archive_solutions = []
        if hasattr(self, 'archive_positions') and len(self.archive_positions) >= 2:
            sorted_indices = np.argsort(self.archive_fitness)
            for idx in sorted_indices[:min(5, len(self.archive_positions))]:
                archive_solutions.append(self.archive_positions[idx].copy())

        self._initialize_population()

        n_archive = int(np.floor(self.NP * intensity * 0.5))
        n_archive = min(n_archive, len(archive_solutions), self.NP - 1)

        if n_archive > 0 and archive_solutions:
            for i in range(n_archive):
                if i >= len(archive_solutions):
                    break
                spread = self.sigma * (2.0 + 3.0 * intensity)
                sample = archive_solutions[i] + np.random.randn(self.dim) * spread
                sample = self._clip_to_bounds(sample)
                self.population[i + 1] = sample

        n_boundary = self.NP - 1 - n_archive
        if n_boundary > 0:
            power = 2.0 + 3.0 * intensity
            u = np.random.random((n_boundary, self.dim))
            boundary_samples = np.power(u, power) if np.random.random() > 0.5 else 1.0 - np.power(u, power)

            dim_fraction = 0.2 + 0.4 * intensity
            dim_mask = np.random.random(self.dim) < dim_fraction
            for d in range(self.dim):
                if not dim_mask[d]:
                    boundary_samples[:, d] = np.random.uniform(0, 1, n_boundary)

            for d in range(self.dim):
                boundary_samples[:, d] = self.lb[d] + boundary_samples[:, d] * (self.ub[d] - self.lb[d])

            start_idx = 1 + n_archive
            self.population[start_idx:start_idx + n_boundary] = boundary_samples

        self.population[0] = elite
        self.fitness[0] = elite_fit

        self.f_opt = float(np.asarray(elite_fit).flatten()[0])
        self.x_opt = elite.copy()
        self.f_opt_prev = self.f_opt
        self.stagnation_counter = 0

        if intensity > 0.5:
            self.sigma = max(self.sigma, 0.5 * (self.ub[0] - self.lb[0]) / 6.0)
            self.C = np.eye(self.dim) * (self.sigma ** 2)
            self.pc = np.zeros(self.dim)
            self.ps = np.zeros(self.dim)
    
    def _restart_variant_03(self):
        """Preserve multiple diverse elites with evenly spaced injection."""
        sorted_indices = np.argsort(self.fitness)
        num_elites = min(5, self.NP // 4)
        elite_solutions = self.population[sorted_indices[:num_elites]].copy()
        elite_fitness = self.fitness[sorted_indices[:num_elites]].copy()

        self._initialize_population()

        self.population[0] = elite_solutions[0]
        self.fitness[0] = elite_fitness[0]
        self.f_opt = float(np.asarray(elite_fitness[0]).flatten()[0])
        self.x_opt = elite_solutions[0].copy()
        self.f_opt_prev = self.f_opt
        self.stagnation_counter = 0

        for i in range(1, num_elites):
            idx = (i * self.NP) // num_elites
            if idx < self.NP:
                self.population[idx] = elite_solutions[i]
                self.fitness[idx] = elite_fitness[i]

        if hasattr(self, 'restart_counter'):
            self.restart_counter += 1
            if self.restart_counter > 1:
                n_diverse = min(10, self.NP // 4)
                for _ in range(n_diverse):
                    idx = np.random.randint(self.NP // 4, self.NP)
                    self.population[idx] = np.random.uniform(self.lb, self.ub)
        else:
            self.restart_counter = 1
    
    def _restart_variant_04(self):
        """Covariance-directed multi-point sampling with archive injection."""
        best_idx = np.argmin(self.fitness)
        elite = self.population[best_idx].copy()
        elite_fit = self.fitness[best_idx]

        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-15)
        except np.linalg.LinAlgError:
            eigvals = np.ones(self.dim)
            eigvecs = np.eye(self.dim)

        num_restart_points = min(5, self.NP // 10)
        restart_points = [elite.copy()]

        for i in range(num_restart_points - 1):
            direction = eigvecs[:, -(i + 1)]
            scale = np.sqrt(eigvals[-(i + 1)]) * 2.0
            sign = 1.0 if i % 2 == 0 else -1.0
            offset = sign * scale * direction
            restart_point = self._clip_to_bounds(elite + offset * (0.3 + 0.7 * np.random.random()))
            restart_points.append(restart_point)

        self._initialize_population()

        for i, pt in enumerate(restart_points):
            if i < self.NP:
                self.population[i] = pt

        self.fitness = self.func(self.population)

        self.population[0] = elite
        self.fitness[0] = elite_fit

        best_new_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_new_idx]).flatten()[0])
        self.x_opt = self.population[best_new_idx].copy()
        self.f_opt_prev = self.f_opt
        self.stagnation_counter = 0
    
    def _restart_variant_05(self):
        """Elite-directed focused reinitialization with condition-aware strategy."""
        best_idx = np.argmin(self.fitness)
        elite = self.population[best_idx].copy()
        elite_fit = self.fitness[best_idx]

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        cond = eig_max / max(eig_min, 1e-15)

        is_ill_conditioned = cond > 1e6
        is_deeply_stagnant = self.stagnation_counter > 2 * self.max_stagnation

        sigma_scale = max(self.sigma, 1e-6)

        if not is_ill_conditioned:
            try:
                L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
            except np.linalg.LinAlgError:
                diag_C = np.diag(self.C)
                diag_C = np.maximum(diag_C, 1e-10)
                L = np.diag(np.sqrt(diag_C))

            if is_deeply_stagnant:
                exploration_radius = 5.0 * sigma_scale
            else:
                exploration_radius = 2.0 * sigma_scale

            n_new = self.NP - 1
            z = np.random.randn(n_new, self.dim)
            new_pop = elite + exploration_radius * (z @ L.T)
            new_pop = self._clip_to_bounds(new_pop)
        else:
            self.C = np.eye(self.dim) * np.mean(eigvals)
            self.pc = np.zeros(self.dim)
            self.L = None

            n_new = self.NP - 1
            exploration_radius = max(10.0 * sigma_scale, 0.1 * (self.ub[0] - self.lb[0]))
            u = np.random.randn(n_new, self.dim)
            norms = np.linalg.norm(u, axis=1, keepdims=True)
            radii = np.random.random((n_new, 1)) ** (1.0 / self.dim)
            u = u / (norms + 1e-10) * radii * exploration_radius
            new_pop = elite + u
            new_pop = self._clip_to_bounds(new_pop)

        n_remain = self.NP - 1 - len(new_pop) if len(new_pop) < self.NP else 0
        if n_remain > 0:
            boundary_ratio = 0.4 if is_deeply_stagnant else 0.25
            n_boundary = int(boundary_ratio * n_remain)
            n_random = n_remain - n_boundary

            remainder = []
            if n_random > 0:
                random_samples = np.random.uniform(self.lb, self.ub, (n_random, self.dim))
                remainder.append(random_samples)

            if n_boundary > 0:
                u = np.random.random((n_boundary, self.dim))
                power = 3.0
                scaled = np.power(u, power) if np.random.random() > 0.5 else 1.0 - np.power(u, power)
                dim_mask = np.random.random(self.dim) < 0.3
                boundary = np.zeros((n_boundary, self.dim))
                boundary[:, dim_mask] = scaled[:, dim_mask]
                boundary[:, ~dim_mask] = u[:, ~dim_mask]
                for d in range(self.dim):
                    boundary[:, d] = self.lb[d] + boundary[:, d] * (self.ub[d] - self.lb[d])
                remainder.append(boundary)

            if remainder:
                new_pop = np.vstack([new_pop] + remainder)

        if len(new_pop) < self.NP:
            extra = np.random.uniform(self.lb, self.ub, (self.NP - len(new_pop), self.dim))
            new_pop = np.vstack([new_pop, extra])
        elif len(new_pop) > self.NP:
            new_pop = new_pop[:self.NP]

        self.population = new_pop
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        if not is_ill_conditioned:
            self.C = 0.9 * self.C + 0.1 * np.eye(self.dim) * np.trace(self.C) / self.dim

        self.sigma = np.clip(self.sigma * 1.5, 1e-6, 10.0)
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)

        self.fitness = self.func(self.population)

        best_new_idx = np.argmin(self.fitness)
        if self.fitness[best_new_idx] > elite_fit:
            worst_idx = np.argmax(self.fitness)
            self.population[worst_idx] = elite
            self.fitness[worst_idx] = elite_fit

        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        self.stagnation_counter = 0
    
    def _restart_variant_06(self):
        """Covariance-preserving re-seeding around elite for ill-conditioned tasks."""
        best_idx = np.argmin(self.fitness)
        elite = self.population[best_idx].copy()
        elite_fit = self.fitness[best_idx]

        eigvals, eigvecs = np.linalg.eigh(self.C)

        stagnation_severity = min(self.stagnation_counter / (2.0 * self.max_stagnation), 1.0)
        variance_scale = 1.5 + 1.5 * stagnation_severity

        scaled_eigvals = eigvals * variance_scale
        C_scaled = eigvecs @ np.diag(scaled_eigvals) @ eigvecs.T
        C_scaled = self._ensure_positive_definite(C_scaled)

        n_elite_preserve = min(5, self.NP // 10)
        n_scaled = self.NP - n_elite_preserve

        min_eig = np.min(np.linalg.eigvalsh(C_scaled))
        if min_eig < 1e-8:
            C_scaled += (1e-7 - min_eig) * np.eye(self.dim)

        try:
            L = np.linalg.cholesky(C_scaled)
        except np.linalg.LinAlgError:
            diag_C = np.diag(C_scaled)
            diag_C = np.maximum(diag_C, 1e-10)
            L = np.diag(np.sqrt(diag_C))

        z = np.random.randn(n_scaled, self.dim)
        samples = elite + self.sigma * (z @ L.T)
        samples = self._clip_to_bounds(samples)

        for i in range(n_elite_preserve):
            pert_z = np.random.randn(self.dim)
            pert = self.sigma * 2.0 * (pert_z @ L.T)
            sample = elite + pert
            samples = np.vstack([samples, self._clip_to_bounds(sample)])

        n_boundary = self.NP - samples.shape[0]
        if n_boundary > 0:
            u = np.random.random((n_boundary, self.dim))
            power = 3.0
            boundary = np.where(
                np.random.random((n_boundary, self.dim)) > 0.5,
                np.power(u, power),
                1.0 - np.power(u, power)
            )
            dim_mask = np.random.random((n_boundary, self.dim)) < 0.3
            boundary = np.where(dim_mask, boundary, u)
            for d in range(self.dim):
                boundary[:, d] = self.lb[d] + boundary[:, d] * (self.ub[d] - self.lb[d])
            samples = np.vstack([samples, boundary])

        if samples.shape[0] > self.NP:
            samples = samples[:self.NP]
        elif samples.shape[0] < self.NP:
            extra = np.random.uniform(self.lb, self.ub, (self.NP - samples.shape[0], self.dim))
            samples = np.vstack([samples, extra])

        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = C_scaled
        self.sigma = np.clip(self.sigma * 1.5, 1e-10, 10.0)

        self.fitness = self.func(self.population)

        self.population[0] = elite
        self.fitness[0] = elite_fit

        best_new_idx = np.argmin(self.fitness)
        if self.fitness[best_new_idx] < elite_fit:
            self.f_opt = float(np.asarray(self.fitness[best_new_idx]).flatten()[0])
            self.x_opt = self.population[best_new_idx].copy()
        else:
            self.f_opt = float(np.asarray(elite_fit).flatten()[0])
            self.x_opt = elite.copy()

        self.f_opt_prev = self.f_opt
        self.stagnation_counter = 0
        self.generation = 0
        self.pc = np.zeros(self.dim)
        self._reset_operator_state()
    
    def _restart_variant_09(self):
        """Adaptive restart with hyperbolic step boosting for high-error tasks."""
        best_idx = np.argmin(self.fitness)
        elite = self.population[best_idx].copy()
        elite_fit = self.fitness[best_idx]

        error_mag = max(abs(elite_fit), 1.0)
        log_error = np.log10(error_mag + 1e-15)
        target_log = -8.0
        decades_from_target = max(log_error - target_log, 0.0)

        bound_range = max(self.ub[0] - self.lb[0], 1e-6)
        perturb_factor = np.tanh(decades_from_target / 5.0) * 0.3
        perturb_radius = perturb_factor * bound_range
        perturb_radius = np.clip(perturb_radius, 1e-6, bound_range * 0.5)

        new_pop = np.zeros((self.NP, self.dim))
        for i in range(self.NP):
            if i == 0:
                new_pop[i] = elite.copy()
            else:
                new_pop[i] = elite + np.random.randn(self.dim) * perturb_radius

        new_pop = self._clip_to_bounds(new_pop)

        self.population = new_pop
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()

        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.C = self._ensure_positive_definite(self.C)

        self.sigma = np.clip(perturb_radius / 3.0, 1e-10, 10.0)

        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.L = None

        self.fitness = self.func(self.population)

        new_best_idx = np.argmin(self.fitness)
        new_best_fit = float(np.asarray(self.fitness[new_best_idx]).flatten()[0])
        if new_best_fit < elite_fit:
            self.f_opt = new_best_fit
            self.x_opt = self.population[new_best_idx].copy()
        else:
            self.f_opt = float(np.asarray(elite_fit).flatten()[0])
            self.x_opt = elite.copy()

        self.f_opt_prev = self.f_opt
        self.stagnation_counter = 0
