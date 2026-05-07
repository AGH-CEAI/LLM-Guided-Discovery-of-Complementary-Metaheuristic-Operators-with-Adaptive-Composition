import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 5 covariance adaptation strategies (original + 4 winning variants from benchmarks)
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
        
        # Adaptive operator selection parameters - 5 winning strategies
        self.num_operators = 5
        self.operator_names = ['original', 'variant_03', 'variant_08', 'variant_09', 'variant_10']
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_operators)
        self.beta = np.ones(self.num_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 15
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
        eigenvalues = np.linalg.eigvalsh(C)
        min_eig = float(np.min(eigenvalues))
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
        if hasattr(self, 'momentum_ema'):
            self.momentum_ema = np.zeros(self.dim)
        if hasattr(self, 'prev_f_opt_trap'):
            self.prev_f_opt_trap = self.f_opt
            self.stagnation_gen = 0
        if hasattr(self, 'archive_positions'):
            self.archive_positions = []
            self.archive_fitness = []
        if hasattr(self, 'exploration_temp'):
            self.exploration_temp = 1.0
    
    def _sample_trials_batch(self):
        """Sample new trial population using Cholesky decomposition."""
        # Ensure positive definiteness before Cholesky
        min_eig = float(np.min(np.linalg.eigvalsh(self.C)))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        # Compute Cholesky factor L where C = L @ L.T
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
        success_rate = float(np.clip(success_rate, 0.0, 1.0))

        avg_improvement = total_improvement / max(recent_improved, 1)
        improvement_magnitude = float(np.log1p(max(avg_improvement, 1e-15)))

        if not hasattr(self, 'improvement_ema'):
            self.improvement_ema = improvement_magnitude
        self.improvement_ema = float(0.7 * self.improvement_ema + 0.3 * improvement_magnitude)

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
        """Update operator rewards using sliding window credit assignment."""
        improvement = float(max(0.0, self.f_opt_prev - self.f_opt))
        
        pop_variance = float(np.mean(np.var(self.population, axis=0)))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = float(np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0))
        
        # Compute condition number for reward adjustment
        try:
            eigenvalues = np.linalg.eigvalsh(self.C)
            cond = float(np.max(eigenvalues) / max(float(np.min(eigenvalues)), 1e-15))
        except:
            cond = 1.0
        
        # Higher reward for improvement, adjusted by diversity and condition
        condition_penalty = float(np.log1p(cond) / (np.log1p(cond) + 1.0))
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity - 0.05 * condition_penalty)
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
                self.alpha[op] = float(max(1.0, 1.0 + sum_reward))
            else:
                self.beta[op] = float(max(1.0, 1.0 - sum_reward))
    
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
        """Mean-stagnation eigenscale reset: push exploration outward when stuck.
        (4 wins on benchmark)"""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        # Detect mean stagnation: is the mean barely moving relative to step size?
        mean_displacement = float(np.linalg.norm(self.mean - self.old_mean))
        stagnation_threshold = 1e-3 * self.sigma

        if mean_displacement < stagnation_threshold and mean_displacement > 1e-15:
            # Mean is stagnant — apply exponential eigenscale perturbation to covariance
            try:
                eigvals, eigvecs = np.linalg.eigh(self.C)
                eigvals = np.maximum(eigvals, 1e-10)

                # Exponential scaling: push outward along each principal axis
                log_eigvals = np.log(eigvals + 1e-10)
                log_scale = log_eigvals - np.min(log_eigvals) + 1.0
                scale_factors = np.power(log_scale, -1.5)
                scale_factors = np.clip(scale_factors, 0.1, 10.0)

                # Perturb eigvals: expand narrow directions, contract wide ones
                perturbed_eigvals = eigvals * scale_factors
                perturbed_eigvals = np.maximum(perturbed_eigvals, 1e-10)

                # Reconstruct perturbed covariance
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
        """Covariance adaptation with stagnation-triggered restart for escaping local optima.
        (10 wins on benchmark - most successful)"""
        stagnation_threshold = max(20, self.dim * 2)
        cond_threshold = 1e7
        eigmin_threshold = 1e-8

        need_restart = (self.stagnation_counter > stagnation_threshold)

        try:
            eigvals = np.linalg.eigvalsh(self.C)
            cond = float(np.max(eigvals) / max(float(np.min(eigvals)), 1e-15))
        except:
            cond = 1.0
            
        need_restart = need_restart or (cond > cond_threshold) or (float(np.min(eigvals)) < eigmin_threshold)

        if need_restart:
            self.C = np.eye(self.dim) * (self.sigma ** 2)
            self.pc *= 0.0
            self.stagnation_counter = 0

        y_mean = (self.mean - self.old_mean) / self.sigma

        if not hasattr(self, 'momentum_ema'):
            self.momentum_ema = np.zeros(self.dim)
        self.momentum_ema = 0.7 * self.momentum_ema + 0.3 * y_mean
        y_mean_norm = float(np.linalg.norm(y_mean))
        momentum_norm = float(np.linalg.norm(self.momentum_ema))
        if momentum_norm > 1e-10 and y_mean_norm > 1e-10:
            y_momentum = self.momentum_ema / momentum_norm * y_mean_norm
        else:
            y_momentum = y_mean

        cc_adapt = self.cc * (1.5 if need_restart else 1.0)
        cc_adapt = float(np.clip(cc_adapt, 0.001, 0.5))
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_momentum

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        ccov_boost = 3.0 if need_restart else 1.0
        ccov_adapt = float(min(self.ccov * ccov_boost, 0.4))

        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

        if need_restart:
            self.C += 0.1 * np.eye(self.dim) * (self.sigma ** 2)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_09(self):
        """Eigenspace restart with orthogonal perturbation for escaping local optima.
        (3 wins on benchmark)"""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        # Detect trapping: low fitness variance AND stagnation
        fit_var = float(np.var(self.fitness))
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

        # Trigger eigenspace restart if trapped
        is_trapped = (rel_var < 1e-3) and (self.stagnation_gen > self.dim * 2)

        if is_trapped:
            # Eigendecompose current covariance
            eigvals, eigvecs = np.linalg.eigh(self.C)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            # Mix eigenvectors to create new basis (prevents collapse)
            theta = 0.3 * np.pi
            mix_matrix = np.cos(theta) * np.eye(self.dim) + np.sin(theta) * (np.ones((self.dim, self.dim)) / self.dim - np.eye(self.dim))
            eigvecs_mixed = eigvecs @ mix_matrix

            # Inject exploration noise along weakest eigendirections
            min_eig = float(np.min(eigvals))
            max_eig = float(np.max(eigvals))
            eigvals_scaled = eigvals.copy()

            # Boost weak directions, cap strong directions
            for i in range(self.dim):
                ratio = eigvals[i] / (max_eig + 1e-10)
                if ratio < 0.1:
                    eigvals_scaled[i] = min_eig + 0.2 * (max_eig - min_eig)
                elif ratio > 0.9:
                    eigvals_scaled[i] = 0.7 * max_eig

            # Reconstruct covariance in mixed basis
            self.C = eigvecs_mixed @ np.diag(eigvals_scaled) @ eigvecs_mixed.T
            self.C = 0.5 * (self.C + self.C.T)

            # Reset evolution path to align with new covariance
            self.pc *= 0.1
            self.stagnation_gen = 0

            # Increase learning rate temporarily
            ccov_boost = float(min(self.ccov * 3.0, 0.5))
            self.C = (1.0 - ccov_boost) * self.C + ccov_boost * np.eye(self.dim)
        else:
            self.stagnation_gen = max(0, self.stagnation_gen - 1)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_10(self):
        """Eigenspace perturbation with condition-aware adaptation for escaping local optima.
        (2 wins on benchmark)"""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Compute eigendecomposition for condition monitoring
        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
        except np.linalg.LinAlgError:
            eigvals = np.diag(self.C)
            eigvecs = np.eye(self.dim)

        eig_min = float(np.maximum(np.min(eigvals), 1e-12))
        eig_max = float(np.max(eigvals))
        cond = eig_max / eig_min
        eig_spread = eig_min / (eig_max + 1e-12)

        # Detect catastrophic convergence (primary failure mode on hardest tasks)
        is_trapped = (cond > 1e4) or (eig_spread < 1e-4) or (float(np.min(eigvals)) < 1e-8)

        # Adaptive learning rates based on condition number
        if cond > 1e6:
            ccov_scale = 0.1
            cc_scale = 0.5
        elif cond > 1e3:
            ccov_scale = 0.3
            cc_scale = 0.7
        else:
            ccov_scale = 1.0
            cc_scale = 1.0

        ccov_eff = float(np.clip(self.ccov * ccov_scale, 1e-10, 0.5))
        cc_eff = float(np.clip(self.cc * cc_scale, 0.001, 0.3))

        # Rebuild evolution path with scaled learning rate
        self.pc = (1.0 - cc_eff) * self.pc + np.sqrt(cc_eff * (2.0 - cc_eff)) * y_mean
        rank_one = np.outer(self.pc, self.pc)

        # Standard rank-mu update
        self.C = ((1.0 - ccov_eff) * self.C + 
                  ccov_eff * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_eff * 2.0 * rank_mu))

        # Perturb eigenspectrum if trapped
        if is_trapped:
            try:
                eigvals, eigvecs = np.linalg.eigh(self.C)

                # Clip eigenvalues to prevent collapse
                eigvals = np.clip(eigvals, 1e-10, None)

                # Inject exploration along minor eigendirections
                min_eig = float(np.min(eigvals))
                max_eig = float(np.max(eigvals))
                exploration_budget = 0.1 * max_eig

                # Distribute exploration across bottom 50% of eigendirections
                n_explore = max(1, self.dim // 2)
                sorted_indices = np.argsort(eigvals)
                for idx in sorted_indices[:n_explore]:
                    eigvals[idx] += exploration_budget / n_explore

                # Rebuild covariance from perturbed spectrum
                self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            except np.linalg.LinAlgError:
                # Fallback: add isotropic exploration
                self.C += 0.05 * np.eye(self.dim)

        # Ensure positive definiteness
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
