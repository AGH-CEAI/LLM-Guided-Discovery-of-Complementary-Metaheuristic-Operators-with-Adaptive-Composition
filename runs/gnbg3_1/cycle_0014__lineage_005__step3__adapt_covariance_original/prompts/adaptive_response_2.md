```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 5 covariance adaptation strategies (based on benchmark winners)
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
        self.operator_names = ['original', 'variant_03', 'variant_08', 'variant_09', 'variant_10']
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_operators, dtype=np.float64)
        self.beta = np.ones(self.num_operators, dtype=np.float64)
        
        # Sliding window for credit assignment
        self.reward_window_size = 15
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators, dtype=np.float64)
        self.selection_counts = np.zeros(self.num_operators, dtype=np.float64)
        
        # Credit assignment parameters
        self.credit_history = {i: [] for i in range(self.num_operators)}
        self.credit_window = 20
        
        # Runtime state
        self.generation = 0
        self.current_operator = 0
        self.L = None
        
        # Performance tracking
        self.best_fitness_history = []
        self.operator_performance = {i: {'rewards': [], 'count': 0} for i in range(self.num_operators)}
        
        # Reward normalization
        self.reward_baseline = 0.0
        self.reward_scale = 1.0
    
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
        f_best = self.fitness[best_idx]
        self.f_opt = float(np.asarray(f_best).flatten()[0])
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
        
        # Reset operator-specific attributes
        if hasattr(self, 'momentum_ema'):
            self.momentum_ema = np.zeros(self.dim)
        if hasattr(self, 'prev_f_opt_trap'):
            self.prev_f_opt_trap = self.f_opt
            self.stagnation_gen = 0
        if hasattr(self, 'prev_f_opt_trap'):
            delattr(self, 'prev_f_opt_trap')
        if hasattr(self, 'stagnation_gen'):
            delattr(self, 'stagnation_gen')
    
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
        self.selection_counts[self.current_operator] += 1
        return self.current_operator
    
    def _compute_reward(self):
        """Compute reward for the current operator based on multiple metrics."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        # Primary reward: normalized improvement
        fitness_scale = max(abs(self.f_opt), 1.0)
        rel_improvement = improvement / (fitness_scale + 1e-10)
        
        # Diversity reward
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        # Condition number reward (lower is better)
        try:
            eigvals = np.linalg.eigvalsh(self.C)
            eig_min = max(np.min(eigvals), 1e-15)
            eig_max = max(np.max(eigvals), 1e-15)
            cond = eig_max / eig_min
            cond_reward = 1.0 / np.log1p(cond + 1.0)
            cond_reward = float(np.clip(cond_reward, 0.0, 1.0))
        except:
            cond_reward = 0.5
        
        # Stagnation penalty
        stagnation_penalty = float(np.clip(self.stagnation_counter / (self.max_stagnation + 1e-10), 0.0, 1.0))
        
        # Combine rewards
        reward = 0.0
        reward += 0.5 * float(np.log1p(rel_improvement * 1e10))
        reward += 0.2 * diversity
        reward += 0.2 * cond_reward
        reward -= 0.1 * stagnation_penalty
        
        # Normalize reward
        reward = float(np.clip(reward, -5.0, 5.0))
        
        # Bonus for finding new best
        if self.f_opt < self.f_opt_prev - 1e-14:
            reward += 0.5
        
        return reward
    
    def _update_operator_rewards(self):
        """Update operator rewards using sliding window credit assignment."""
        reward = self._compute_reward()
        op = self.current_operator
        
        # Store reward in sliding window
        self.operator_rewards[op].append(reward)
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)
        
        # Update credit history
        self.credit_history[op].append(reward)
        if len(self.credit_history[op]) > self.credit_window:
            self.credit_history[op].pop(0)
        
        # Update performance tracking
        self.operator_performance[op]['rewards'].append(reward)
        self.operator_performance[op]['count'] += 1
        
        # Update Beta distribution parameters using empirical Bayes
        n = len(self.operator_rewards[op])
        if n >= 3:
            rewards_arr = np.array(self.operator_rewards[op], dtype=np.float64)
            mean_reward = float(np.mean(rewards_arr))
            var_reward = float(np.var(rewards_arr))
            
            # Avoid division by zero
            var_n = var_reward * n + 1e-10
            mean_clamped = float(np.clip(mean_reward, 0.01, 0.99))
            
            alpha_new = mean_clamped * (mean_clamped * (1.0 - mean_clamped) / var_n - 1.0)
            beta_new = (1.0 - mean_clamped) * (mean_clamped * (1.0 - mean_clamped) / var_n - 1.0)
            
            self.alpha[op] = float(np.clip(1.0 + alpha_new, 0.5, 100.0))
            self.beta[op] = float(np.clip(1.0 + beta_new, 0.5, 100.0))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = float(np.clip(1.0 + sum_reward / n, 0.5, 100.0))
            else:
                self.beta[op] = float(np.clip(1.0 - sum_reward / n, 0.5, 100.0))
        
        # Exploration bonus: slightly favor less-used operators
        total_selections = float(np.sum(self.selection_counts) + 1e-10)
        selection_prob = self.selection_counts[op] / total_selections
        exploration_bonus = 0.1 * (1.0 - selection_prob)
        
        self.alpha[op] += exploration_bonus * 0.1
        self.beta[op] += exploration_bonus * 0.1
    
    def _adapt_covariance(self):
        """Dispatch to the selected covariance adaptation strategy."""
        if self.current_operator == 0:
            self._adapt_covariance_original()
        elif self.current_operator == 1:
            self._adapt_covariance_variant_03()
        elif self.current_operator == 2:
            self._adapt_covariance_variant_08()
        elif self.current_operator == 3:
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
    
    def _adapt_covariance_variant_03(self):
        """Mean-stagnation eigenscale reset: push exploration outward when stuck."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        mean_displacement = np.linalg.norm(self.mean - self.old_mean)
        stagnation_threshold = 1e-3 * self.sigma

        if mean_displacement < stagnation_threshold and mean_displacement > 1e-15:
            try:
                eigvals, eigvecs = np.linalg.eigh(self.C)
                eigvals = np.maximum(eigvals, 1e-10)

                log_eigvals = np.log(eigvals + 1e-10)
                log_scale = log_eigvals - np.min(log_eigvals) + 1.0
                scale_factors = np.power(log_scale, -1.5)
                scale_factors = np.clip(scale_factors, 0.1, 10.0)

                perturbed_eigvals = eigvals * scale_factors
                perturbed_eigvals = np.maximum(perturbed_eigvals, 1e-10)

                self.C = eigvecs @ np.diag(perturbed_eigvals) @ eigvecs.T

            except np.linalg.LinAlgError:
                pass

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_08(self):
        """Covariance adaptation with stagnation-triggered restart for escaping local optima."""
        stagnation_threshold = max(20, self.dim * 2)
        cond_threshold = 1e7
        eigmin_threshold = 1e-8

        need_restart = (self.stagnation_counter > stagnation_threshold)

        try:
            eigvals = np.linalg.eigvalsh(self.C)
            cond = np.max(eigvals) / max(np.min(eigvals), 1e-15)
            need_restart = need_restart or (cond > cond_threshold) or (np.min(eigvals) < eigmin_threshold)
        except:
            pass

        if need_restart:
            self.C = np.eye(self.dim) * (self.sigma ** 2)
            self.pc *= 0.0
            self.stagnation_counter = 0

        y_mean = (self.mean - self.old_mean) / self.sigma

        if not hasattr(self, 'momentum_ema'):
            self.momentum_ema = np.zeros(self.dim)
        self.momentum_ema = 0.7 * self.momentum_ema + 0.3 * y_mean
        y_momentum_norm = np.linalg.norm(y_mean)
        momentum_norm = np.linalg.norm(self.momentum_ema)
        if momentum_norm > 1e-10 and y_momentum_norm > 1e-10:
            y_momentum = self.momentum_ema / momentum_norm * y_momentum_norm
        else:
            y_momentum = y_mean

        cc_adapt = self.cc * (1.5 if need_restart else 1.0)
        cc_adapt = np.clip(cc_adapt, 0.001, 0.5)
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_momentum

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        ccov_boost = 3.0 if need_restart else 1.0
        ccov_adapt = min(self.ccov * ccov_boost, 0.4)

        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

        if need_restart:
            self.C += 0.1 * np.eye(self.dim) * (self.sigma ** 2)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_09(self):
        """Eigenspace restart with orthogonal perturbation for escaping local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        if not hasattr(self, 'prev_f_opt_trap'):
            self.prev_f_opt_trap = self.f_opt
            self.stagnation_gen = 0
        else:
            f_improved = self.prev_f_opt_trap - self.f_opt
            if f_improved > 1e-10:
                self.stagnation_gen = 0
            else:
                self.stagnation_gen += 1
            self.prev_f_opt_trap = self.f_opt

        is_trapped = (rel_var < 1e-3) and (self.stagnation_gen > self.dim * 2)

        if is_trapped:
            try:
                eigvals, eigvecs = np.linalg.eigh(self.C)
                idx = np.argsort(eigvals)[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]

                theta = 0.3 * np.pi
                mix_matrix = np.cos(theta) * np.eye(self.dim) + np.sin(theta) * (np.ones((self.dim, self.dim)) / self.dim - np.eye(self.dim))
                eigvecs_mixed = eigvecs @ mix_matrix

                min_eig = np.min(eigvals)
                max_eig = np.max(eigvals)
                eigvals_scaled = eigvals.copy()

                for i in range(self.dim):
                    ratio = eigvals[i] / (max_eig + 1e-10)
                    if ratio < 0.1:
                        eigvals_scaled[i] = min_eig + 0.2 * (max_eig - min_eig)
                    elif ratio > 0.9:
                        eigvals_scaled[i] = 0.7 * max_eig

                self.C = eigvecs_mixed @ np.diag(eigvals_scaled) @ eigvecs_mixed.T
                self.C = 0.5 * (self.C + self.C.T)

                self.pc *= 0.1
                self.stagnation_gen = 0

                ccov_boost = min(self.ccov * 3.0, 0.5)
                self.C = (1.0 - ccov_boost) * self.C + ccov_boost * np.eye(self.dim)
            except np.linalg.LinAlgError:
                pass
        else:
            self.stagnation_gen = max(0, self.stagnation_gen - 1)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_10(self):
        """Eigenspace perturbation with condition-aware adaptation for escaping local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
        except np.linalg.LinAlgError:
            eigvals = np.diag(self.C)
            eigvecs = np.eye(self.dim)

        eig_min = np.maximum(np.min(eigvals), 1e-12)
        eig_max = np.max(eigvals)
        cond = eig_max / eig_min
        eig_spread = eig_min / (eig_max + 1e-12)

        is_trapped = (cond > 1e4) or (eig_spread < 1e-4) or (eig_min < 1e-8)

        if cond > 1e6:
            ccov_scale = 0.1
            cc_scale = 0.5
        elif cond > 1e3:
            ccov_scale = 0.3
            cc_scale = 0.7
        else:
            ccov_scale = 1.0
            cc_scale = 1.0

        ccov_eff = np.clip(self.ccov * ccov_scale, 1e-10, 0.5)
        cc_eff = np.clip(self.cc * cc_scale, 0.001, 0.3)

        self.pc = (1.0 - cc_eff) * self.pc + np.sqrt(cc_eff * (2.0 - cc_eff)) * y_mean
        rank_one = np.outer(self.pc, self.pc)

        self.C = ((1.0 - ccov_eff) * self.C + 
                  ccov_eff * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_eff * 2.0 * rank_mu))

        if is_trapped:
            try:
                eigvals, eigvecs = np.linalg.eigh(self.C)
                eigvals = np.clip(eigvals, 1e-10, None)

                min_eig = np.min(eigvals)
                max_eig = np.max(eigvals)
                exploration_budget = 0.1 * max_eig

                n_explore = max(1, self.dim // 2)
                sorted_indices = np.argsort(eigvals)
                for idx in sorted_indices[:n_explore]:
                    eigvals[idx] += exploration_budget / n_explore

                self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            except np.linalg.LinAlgError:
                self.C += 0.05 * np.eye(self.dim)

        self.C = self._ensure_positive_definite(self.C)
    
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
```