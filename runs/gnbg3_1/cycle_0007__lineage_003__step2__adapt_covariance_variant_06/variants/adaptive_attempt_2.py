import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 8 covariance adaptation strategies (original + 7 winning variants)
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
        
        # Adaptive operator selection parameters - 8 operators (original + 7 variants)
        self.num_operators = 8
        self.operator_names = ['original', 'variant_01', 'variant_02', 'variant_03',
                               'variant_04', 'variant_05', 'variant_06', 'variant_07']
        
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
            
            # Select and apply covariance adaptation operator
            self._select_operator_thompson()
            self._adapt_covariance()
            
            # Update operator rewards based on improvement
            self._update_operator_rewards()
            
            self._check_stagnation()
            self._restart_if_needed()
            
            self.generation += 1
        
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
        
        self._reset_operator_state()
    
    def _reset_operator_state(self):
        """Reset all operator-specific state variables."""
        self.L = None
        if hasattr(self, 'pc_weighted'):
            delattr(self, 'pc_weighted')
        if hasattr(self, 'p_cross'):
            delattr(self, 'p_cross')
        if hasattr(self, 'prev_y_mean'):
            delattr(self, 'prev_y_mean')
        if hasattr(self, 'prev_f_opt'):
            delattr(self, 'prev_f_opt')
        if hasattr(self, 'exploration_temp'):
            delattr(self, 'exploration_temp')
        if hasattr(self, 'C_pool'):
            delattr(self, 'C_pool')
        if hasattr(self, 'C_scores'):
            delattr(self, 'C_scores')
        if hasattr(self, 'C_pool_updates'):
            delattr(self, 'C_pool_updates')
        if hasattr(self, 'prev_fitness_sum'):
            delattr(self, 'prev_fitness_sum')
        if hasattr(self, 'explosion_strength'):
            delattr(self, 'explosion_strength')
    
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
        """Update operator rewards using sliding window credit assignment."""
        improvement = max(0.0, float(self.f_opt_prev) - float(self.f_opt))
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity)
        reward = float(np.clip(reward, -10.0, 10.0))
        
        op = self.current_operator
        self.operator_rewards[op].append(reward)
        
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
    
    def _adapt_covariance(self):
        """Dispatch to the selected covariance adaptation strategy."""
        if self.current_operator == 0:
            self._adapt_covariance_original()
        elif self.current_operator == 1:
            self._adapt_covariance_variant_01()
        elif self.current_operator == 2:
            self._adapt_covariance_variant_02()
        elif self.current_operator == 3:
            self._adapt_covariance_variant_03()
        elif self.current_operator == 4:
            self._adapt_covariance_variant_04()
        elif self.current_operator == 5:
            self._adapt_covariance_variant_05()
        elif self.current_operator == 6:
            self._adapt_covariance_variant_06()
        else:
            self._adapt_covariance_variant_07()
    
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
        """Active CMA-ES with negative weights for diversity-promoting rank-μ update."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        rank_one = np.outer(self.pc, self.pc)

        n_pos = self.mu // 2
        n_neg = self.mu - n_pos

        sorted_indices = np.argsort(self.fitness)
        sorted_pop = self.population[sorted_indices]

        pos_weights = np.log(n_pos + 0.5) - np.log(np.arange(1, n_pos + 1))
        pos_weights /= np.sum(pos_weights)
        pos_weights = np.maximum(pos_weights, 0.0)

        neg_weights = np.log(n_neg + 0.5) - np.log(np.arange(1, n_neg + 1))
        neg_weights /= np.sum(neg_weights)
        neg_weights = np.maximum(neg_weights, 0.0)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(n_pos):
            diff = (sorted_pop[i] - self.old_mean) / self.sigma
            rank_mu += pos_weights[i] * np.outer(diff, diff)

        for i in range(n_neg):
            idx = self.mu - 1 - i
            diff = (sorted_pop[idx] - self.old_mean) / self.sigma
            rank_mu -= 0.25 * neg_weights[i] * np.outer(diff, diff)

        c_c = 2.0 / ((self.dim + 1.41) ** 2)
        c_1 = 1.0 / (self.dim + 2.0)
        c_mu = self.mueff / (self.dim + 2.0)

        alpha_c = 1.0 + c_1 + c_mu
        c_1_scaled = c_1 / alpha_c
        c_mu_scaled = c_mu / alpha_c
        c_c_scaled = c_c / alpha_c

        self.C = ((1.0 - c_c_scaled) * self.C + 
                  c_c_scaled * rank_one +
                  c_mu_scaled * rank_mu)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_02(self):
        """Forced diversity injection with eigenvalue floor to prevent catastrophic collapse."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        cond = eig_max / max(eig_min, 1e-15)
        eig_spread = eig_min / (eig_max + 1e-15)

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        is_collapsed = (cond > 1e6) or (eig_spread < 1e-6) or (rel_var < 1e-6)

        if is_collapsed:
            ccov_eff = self.ccov * 0.2
            cc_eff = self.cc * 0.3

            target_min = 0.01 * np.mean(eigvals)
            if eig_min < target_min:
                self.C += (target_min - eig_min) * np.eye(self.dim)

            noise_scale = 0.1 * np.mean(eigvals)
            self.C += noise_scale * np.eye(self.dim)
        else:
            damp_factor = 1.0 + min(cond / 1e4, 2.0)
            ccov_eff = self.ccov / damp_factor
            cc_eff = self.cc / max(damp_factor * 0.5, 0.5)

        if is_collapsed:
            self.pc = (1.0 - cc_eff) * self.pc + np.sqrt(cc_eff * (2.0 - cc_eff)) * y_mean
            rank_one = np.outer(self.pc, self.pc)

        self.C = ((1.0 - ccov_eff) * self.C + 
                  ccov_eff * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_eff * 2.0 * rank_mu))

        final_eigvals = np.linalg.eigvalsh(self.C)
        final_min = np.min(final_eigvals)
        if final_min < 1e-12:
            self.C += (1e-10 - final_min) * np.eye(self.dim)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_03(self):
        """Catastrophic diversification with eigenvalue floor for escaping local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity_ratio = pop_variance / (expected_var + 1e-10)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)

        pathological = (rel_var < 1e-6) and (diversity_ratio < 1e-5) and (eig_spread < 1e-6)

        if pathological:
            rand_dir = np.random.randn(self.dim)
            rand_dir /= (np.linalg.norm(rand_dir) + 1e-10)

            explosion_magnitude = np.max(eigvals) * 50.0 * max(self.sigma, 1.0)
            explosion_matrix = explosion_magnitude * np.outer(rand_dir, rand_dir)

            self.C = 0.5 * self.C + 0.5 * explosion_matrix

            min_eigenvalue = (self.ub[0] - self.lb[0]) ** 2 * 1e-6
            for i in range(self.dim):
                if self.C[i, i] < min_eigenvalue:
                    self.C[i, i] = min_eigenvalue

            self.pc = np.zeros(self.dim)

            self.sigma = min(self.sigma * 3.0, 10.0)

            self.C = self._ensure_positive_definite(self.C)
            return

        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        cond = np.max(eigvals) / (np.maximum(np.min(eigvals), 1e-10))
        if cond > 1e6:
            ccov_scale = 0.1
        elif cond > 1e4:
            ccov_scale = 0.3
        elif cond > 1e2:
            ccov_scale = 0.5
        else:
            ccov_scale = 1.0

        ccov_adapt = self.ccov * ccov_scale

        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu))

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_04(self):
        """Multi-hypothesis covariance pooling with performance-weighted fusion."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        if not hasattr(self, 'C_pool'):
            self.C_pool = [self.C.copy() for _ in range(3)]
            self.C_scores = [1.0, 1.0, 1.0]
            self.C_pool_updates = [0, 0, 0]

        C_exploit = ((1.0 - self.ccov) * self.C_pool[0] + 
                     self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        C_history = 0.95 * self.C_pool[1] + 0.05 * self.C
        C_history = self._ensure_positive_definite(C_history)

        eigvals_C, eigvecs = np.linalg.eigh(self.C_pool[2])
        eigvals_C = np.maximum(eigvals_C, 1e-10)
        log_eigen_sum = np.sum(np.log(eigvals_C))
        C_explore = ((1.0 - self.ccov * 0.5) * self.C_pool[2] + 
                     self.ccov * 0.5 * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
        eigvals_explore, _ = np.linalg.eigh(C_explore)
        if len(eigvals_explore) > 0 and np.all(eigvals_explore > 0):
            log_eigen_explore = np.sum(np.log(eigvals_explore))
            if log_eigen_explore < log_eigen_sum - 0.1 * self.dim:
                C_explore = 1.1 * C_explore
        C_explore = self._ensure_positive_definite(C_explore)

        self.C_pool[0] = self._ensure_positive_definite(C_exploit)
        self.C_pool[1] = self._ensure_positive_definite(C_history)
        self.C_pool[2] = self._ensure_positive_definite(C_explore)

        if hasattr(self, 'prev_fitness_sum'):
            fitness_change = self.prev_fitness_sum - np.sum(self.fitness)
            for i in range(3):
                self.C_scores[i] = max(0.1, self.C_scores[i] * 0.9 + 0.1 * max(fitness_change, 0.0))
        self.prev_fitness_sum = np.sum(self.fitness)

        total_score = sum(self.C_scores) + 1e-10
        weights = [s / total_score for s in self.C_scores]

        self.C = weights[0] * self.C_pool[0] + weights[1] * self.C_pool[1] + weights[2] * self.C_pool[2]
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_05(self):
        """Diversity-triggered anisotropic restart with elite preservation."""
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

        fit_var_after = np.var(self.fitness)
        rel_var_after = fit_var_after / (fit_scale ** 2 + 1e-10)
        eigvals_after = np.linalg.eigvalsh(self.C)
        eig_spread_after = np.min(eigvals_after) / (np.max(eigvals_after) + 1e-10)

        needs_restart = (rel_var_after < 0.005) and (eig_spread_after < 1e-5)

        if needs_restart:
            self.pc = np.zeros(self.dim)
            self.C = np.eye(self.dim) * (self.sigma ** 2)

            sorted_idx = np.argsort(self.fitness)
            elites = [self.population[i].copy() for i in sorted_idx[:min(5, self.NP)]]

            spread = 0.3 * self.sigma
            new_pop = self.x_opt + np.random.randn(self.NP, self.dim) * spread
            new_pop = self._clip_to_bounds(new_pop)

            n_elites = min(5, self.NP)
            new_pop[:n_elites] = elites[:n_elites]

            self.population = new_pop
            self.fitness = self.func(self.population)

            self.old_mean = self.mean.copy()
            self.mean = np.mean(self.population, axis=0)

            self.stagnation_counter = 0
    
    def _adapt_covariance_variant_06(self):
        """Stagnation-depth-driven covariance explosion with directional kick."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        stagnation_ratio = min(self.stagnation_counter / max(self.max_stagnation, 1), 1.0)

        explosion_mult = 1.0 + 10.0 * np.tanh(3.0 * stagnation_ratio)

        if not hasattr(self, 'explosion_strength'):
            self.explosion_strength = 1.0
        if stagnation_ratio < 0.1:
            self.explosion_strength *= 0.95
            self.explosion_strength = max(self.explosion_strength, 1.0)
        else:
            self.explosion_strength = explosion_mult

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        ccov_base = min(self.ccov * (1.0 + 5.0 * min(rel_var, 0.1)), 0.5)
        ccov_scaled = ccov_base * self.explosion_strength
        ccov_scaled = min(ccov_scaled, 0.8)

        eigvals, eigvecs = np.linalg.eigh(self.C)
        max_eig = np.max(eigvals)

        if np.linalg.norm(self.pc) > 1e-12:
            kick_dir = self.pc / np.linalg.norm(self.pc)
        else:
            kick_dir = eigvecs[:, -1]
            kick_dir /= (np.linalg.norm(kick_dir) + 1e-12)

        kick_magnitude = 0.5 * max_eig * self.explosion_strength
        kick_matrix = kick_magnitude * np.outer(kick_dir, kick_dir)

        self.C = ((1.0 - ccov_scaled) * self.C + 
                  ccov_scaled * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_scaled * 2.0 * rank_mu) +
                  0.1 * ccov_scaled * kick_matrix)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_07(self):
        """Stagnation-triggered escape with large perturbation and covariance reset."""
        stagnation_threshold = max(15, self.dim)
        is_stagnant = self.stagnation_counter > stagnation_threshold

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

        if is_stagnant:
            escape_dir = np.random.randn(self.dim)
            escape_dir /= (np.linalg.norm(escape_dir) + 1e-10)

            eigvals = np.linalg.eigvalsh(self.C)
            cov_scale = np.sqrt(np.max(eigvals) + 1e-10)
            jump_magnitude = 15.0 * self.sigma * cov_scale

            self.mean = self.mean + jump_magnitude * escape_dir

            self.C = np.eye(self.dim)

            self.pc = np.zeros(self.dim)

            self.sigma *= 2.5

            self.stagnation_counter = 0

            self.mean = self._clip_to_bounds(self.mean)
            self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _compute_diversity(self):
        """Compute population diversity as mean per-dimension std."""
        return np.mean(np.std(self.population, axis=0))
    
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
