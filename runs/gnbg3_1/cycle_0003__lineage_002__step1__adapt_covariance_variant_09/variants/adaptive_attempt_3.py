import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 7 covariance adaptation strategies (original + 6 winning variants from benchmarks)
    - Thompson Sampling for operator selection with Beta distributions
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
        
        # Adaptive operator selection parameters - 7 operators (original + 6 winning variants)
        self.num_operators = 7
        self.operator_names = [
            'original', 'variant_01', 'variant_02', 'variant_03',
            'variant_06', 'variant_08', 'variant_09'
        ]
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_operators)
        self.beta = np.ones(self.num_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 15
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators)
        self.selection_counts = np.zeros(self.num_operators)
        
        # Reward history for UCB calculation
        self.reward_history = {i: [] for i in range(self.num_operators)}
        
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
        self.fitness = np.array([float(np.asarray(f).flatten()[0]) for f in self.fitness])
        
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
        
        # Reset variant_02 EMA state
        if hasattr(self, 'ema_rank_one'):
            delattr(self, 'ema_rank_one')
        if hasattr(self, 'ema_rank_mu'):
            delattr(self, 'ema_rank_mu')
        if hasattr(self, 'improvement_ema'):
            delattr(self, 'improvement_ema')
        
        # Reset variant_03 exploration temperature
        if hasattr(self, 'exploration_temp'):
            delattr(self, 'exploration_temp')
        
        # Reset variant_08 eigen history
        if hasattr(self, 'eigen_history'):
            delattr(self, 'eigen_history')
    
    def _sample_trials_batch(self):
        """Sample new trial population using Cholesky decomposition."""
        # Ensure positive definiteness before Cholesky
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        # Compute Cholesky factor L where C = L @ L.T
        try:
            L = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
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
        self.trial_fitness = np.array([float(np.asarray(f).flatten()[0]) for f in self.trial_fitness])
    
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
        """Adapt step-size using short-term success history (no cumulation)."""
        recent_improved = 0
        recent_total = 0

        for i in range(min(self.NP, len(self.trial_fitness), len(self.fitness))):
            if self.trial_fitness[i] < self.fitness[i]:
                recent_improved += 1
            recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = recent_improved / recent_total
        success_rate = np.clip(success_rate, 0.0, 1.0)

        target_rate = 0.25
        adaptation = (success_rate - target_rate) / target_rate
        adaptation = np.clip(adaptation, -0.5, 0.5)

        damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim))
        damping_adaptive = np.clip(damping_adaptive, 0.1, 100.0)

        self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions with UCB exploration."""
        # Thompson Sampling: sample from Beta distributions
        samples = np.random.beta(self.alpha, self.beta)
        
        # Add UCB-based exploration bonus to avoid never selecting under-explored operators
        exploration_bonus = np.zeros(self.num_operators)
        total_selections = np.sum(self.selection_counts)
        
        if total_selections > 0:
            for i in range(self.num_operators):
                if self.selection_counts[i] > 0:
                    # UCB bonus for operators with few selections
                    ucb_bonus = np.sqrt(2.0 * np.log(total_selections) / max(1, self.selection_counts[i]))
                    exploration_bonus[i] = 0.1 * ucb_bonus
        
        # Combine Thompson samples with exploration bonus
        combined_scores = samples + exploration_bonus
        self.current_operator = int(np.argmax(combined_scores))
        self.operator_counts[self.current_operator] += 1
        self.selection_counts[self.current_operator] += 1
        return self.current_operator
    
    def _update_operator_rewards(self):
        """Update operator rewards using sliding window credit assignment."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        # Compute reward based on improvement and diversity
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity)
        reward = float(np.clip(reward, -10.0, 10.0))
        
        op = self.current_operator
        self.operator_rewards[op].append(reward)
        
        # Keep sliding window
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)
        
        # Also track in reward history for UCB
        self.reward_history[op].append(reward)
        if len(self.reward_history[op]) > 100:
            self.reward_history[op].pop(0)
        
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
            self._adapt_covariance_variant_06()
        elif self.current_operator == 5:
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
        """Cholesky factor update - maintains C = LL^T directly."""
        if self.L is None:
            try:
                self.L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
            except np.linalg.LinAlgError:
                diag_C = np.diag(self.C)
                diag_C = np.maximum(diag_C, 1e-10)
                self.L = np.diag(np.sqrt(diag_C))
        
        y_mean = (self.mean - self.old_mean) / self.sigma
        norm_y = np.linalg.norm(y_mean)
        z_1 = y_mean / max(norm_y, 1e-10)
        
        rank_mu = np.zeros((self.dim, self.dim))
        total_weight = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_weight += w
        
        if abs(total_weight) > 1e-10:
            rank_mu /= total_weight
        
        ccov_1 = 1.0 / (self.dim + 2.0)
        ccov_mu = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        
        C_new = (1.0 - ccov_1 - ccov_mu) * self.C
        C_new += ccov_1 * np.outer(z_1, z_1)
        C_new += ccov_mu * rank_mu * np.sum(self.weights[:self.mu])
        
        self.C = self._ensure_positive_definite(C_new)
        
        try:
            v = np.linalg.solve(self.L.T, z_1)
            alpha = ccov_1 / (1.0 + ccov_1 * np.dot(v, v))
            self.L = self.L @ (np.eye(self.dim) + alpha * np.outer(v, v))
        except np.linalg.LinAlgError:
            try:
                self.L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
            except np.linalg.LinAlgError:
                diag_C = np.diag(self.C)
                diag_C = np.maximum(diag_C, 1e-10)
                self.L = np.diag(np.sqrt(diag_C))
    
    def _adapt_covariance_variant_02(self):
        """Exponential moving average covariance with natural gradient damping (8 wins)."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Compute natural gradient direction (C^-1 @ y_mean)
        try:
            y_cov_norm = np.linalg.solve(self.C, y_mean)
        except np.linalg.LinAlgError:
            try:
                y_cov_norm = np.linalg.solve(self.C + 1e-6 * np.eye(self.dim), y_mean)
            except np.linalg.LinAlgError:
                y_cov_norm = y_mean / (np.diag(self.C) + 1e-6)

        nat_grad_norm = np.linalg.norm(y_cov_norm)
        damping_factor = min(1.0 + nat_grad_norm * 0.1, 2.0)

        cc_damped = self.cc / damping_factor
        self.pc = (1.0 - cc_damped) * self.pc + np.sqrt(cc_damped * (2.0 - cc_damped)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Initialize EMA tracking for smooth covariance evolution
        if not hasattr(self, 'ema_rank_one'):
            self.ema_rank_one = np.zeros((self.dim, self.dim))
            self.ema_rank_mu = np.zeros((self.dim, self.dim))
            self.ema_beta = 0.95

        self.ema_rank_one = self.ema_beta * self.ema_rank_one + (1.0 - self.ema_beta) * rank_one
        self.ema_rank_mu = self.ema_beta * self.ema_rank_mu + (1.0 - self.ema_beta) * rank_mu

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.1, 1.0)

        # Stagnation detection for adaptive scaling
        if not hasattr(self, 'prev_f_opt_v2'):
            self.prev_f_opt_v2 = self.f_opt
            self.improvement_ema = 1.0

        improvement = max(1e-10, self.prev_f_opt_v2 - self.f_opt)
        self.improvement_ema = 0.9 * self.improvement_ema + 0.1 * improvement
        self.prev_f_opt_v2 = self.f_opt

        stagnation = self.improvement_ema < 1e-10 * max(1.0, abs(self.f_opt))

        # Adaptive learning rate based on state
        base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)

        ccov_scale = 1.0
        if stagnation:
            ccov_scale *= 2.0
        if diversity < 0.3:
            ccov_scale *= 1.5

        ccov_adaptive = np.clip(base_ccov * ccov_scale, 1e-10, 0.5)

        # Blend current and EMA updates for stability
        ema_weight = 0.3 if stagnation else 0.5
        rank_one_blend = ema_weight * self.ema_rank_one + (1.0 - ema_weight) * rank_one
        rank_mu_blend = ema_weight * self.ema_rank_mu + (1.0 - ema_weight) * rank_mu

        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one_blend + 
                  ccov_adaptive * (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu_blend)

        # Eigenvalue bounds for conditioning control
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min, eig_max = np.min(eigvals), np.max(eigvals)
        cond = eig_max / (eig_min + 1e-10)

        if cond > 1e6:
            shrink = np.sqrt(1e6 / cond)
            self.C *= shrink

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_03(self):
        """Exploration temperature with fitness gradient tracking (8 wins)."""
        if not hasattr(self, 'prev_f_opt_v3'):
            self.prev_f_opt_v3 = self.f_opt
            self.exploration_temp = 1.0

        delta_f_opt = self.prev_f_opt_v3 - self.f_opt
        self.prev_f_opt_v3 = self.f_opt

        fitness_gradient = max(abs(float(delta_f_opt)), 1e-15)
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

        if abs(total_w) > 1e-10:
            rank_mu /= total_w

        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_06(self):
        """Eigenvalue floor with active conditioning - prevents premature collapse."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Base covariance update
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        # Enforce minimum eigenvalue floor to prevent collapse
        eigvals, eigvecs = np.linalg.eigh(self.C)
        min_eig = np.min(eigvals)
        eig_floor = 1e-6 * np.max(eigvals)

        if min_eig < eig_floor:
            eigvals[eigvals < eig_floor] = eig_floor
            self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Active condition number control
        eigvals = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals) / (np.min(eigvals) + 1e-10)
        max_cond = 1e6

        if cond > max_cond:
            target_spread = max_cond
            min_target = np.min(eigvals)
            max_target = min_target * target_spread

            mask_large = eigvals > max_target
            if np.any(mask_large):
                eigvals[mask_large] = max_target

            if np.min(eigvals) < min_target:
                eigvals[eigvals < min_target] = min_target

            self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_08(self):
        """Eigenvalue-drift control with active conditioning number management (1 win)."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)

        pop_std = np.std(self.population, axis=0)
        pop_spread = np.max(pop_std) / (np.min(pop_std) + 1e-10)
        target_cond = min(1e6, self.dim * 10.0)

        if not hasattr(self, 'eigen_history'):
            self.eigen_history = []

        eigvals, eigvecs = np.linalg.eigh(self.C)
        self.eigen_history.append(eigvals.copy())
        if len(self.eigen_history) > 5:
            self.eigen_history.pop(0)

        if len(self.eigen_history) >= 3:
            eigen_drift = np.max(np.abs(self.eigen_history[-1] / (self.eigen_history[0] + 1e-10) - 1.0))
        else:
            eigen_drift = 0.0

        cond_C = np.max(eigvals) / (np.min(eigvals) + 1e-10)

        if cond_C > target_cond:
            min_eig = np.min(eigvals)
            max_eig = np.max(eigvals)
            target_min = max_eig / target_cond

            if min_eig < target_min * 0.9:
                self.C += eigvecs @ np.diag(np.maximum(0, target_min - eigvals)) @ eigvecs.T

        ccov_adaptive = base_ccov
        if eigen_drift > 0.5:
            ccov_adaptive *= 0.7
        elif eigen_drift < 0.1 and cond_C < target_cond:
            ccov_adaptive *= 1.3

        if pop_spread < 0.1:
            ccov_adaptive *= 1.5
        elif pop_spread > 10.0:
            ccov_adaptive *= 0.8

        ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 1.0)

        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  ccov_adaptive * (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)

        eigvals_final = np.linalg.eigvalsh(self.C)
        if np.min(eigvals_final) < 1e-12:
            self.C += (1e-10 - np.min(eigvals_final)) * np.eye(self.dim)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_09(self):
        """Diversity-sensitive and stagnation-aware scaling (most robust)."""
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.1, 1.0)
        
        if not hasattr(self, 'prev_f_opt_v9'):
            self.prev_f_opt_v9 = self.f_opt
            self.improvement_ema = 1.0
        
        improvement = max(1e-10, self.prev_f_opt_v9 - self.f_opt)
        self.improvement_ema = 0.9 * self.improvement_ema + 0.1 * improvement
        self.prev_f_opt_v9 = self.f_opt
        
        stagnation = self.improvement_ema < 1e-10 * max(1.0, abs(self.f_opt))
        
        base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        ccov_adaptive = base_ccov * (1.5 if stagnation else 1.0) * (1.5 if diversity < 0.3 else 1.0)
        ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 1.0)
        
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        rank_mu_scale = 2.0 if (stagnation or diversity < 0.3) else 1.0
        
        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  ccov_adaptive * rank_mu_scale * (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)
        
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
