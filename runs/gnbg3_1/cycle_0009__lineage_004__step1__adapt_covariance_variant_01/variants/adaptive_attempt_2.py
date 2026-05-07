import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 8 covariance adaptation strategies (all winning variants from benchmark)
    - Thompson Sampling for operator selection
    - Sliding window credit assignment
    - Automatic restart on stagnation
    - Robust handling of edge cases (NaN, Inf, crashes)
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
        self.num_operators = 9
        self.operator_names = [
            'original', 'variant_01', 'variant_03', 'variant_04',
            'variant_06', 'variant_07', 'variant_09', 'variant_10', 'variant_08'
        ]
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_operators)
        self.beta = np.ones(self.num_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 15
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators)
        self.selection_counts = np.zeros(self.num_operators)
        
        # Performance tracking for adaptive selection
        self.generational_rewards = np.zeros(self.num_operators)
        self.generational_counts = np.zeros(self.num_operators)
        
        # Runtime state
        self.generation = 0
        self.current_operator = 0
        self.L = None
    
    def _clip_to_bounds(self, x):
        """Clip solution to bounds."""
        x = np.asarray(x, dtype=np.float64)
        return np.clip(x, self.lb, self.ub)
    
    def _ensure_positive_definite(self, C):
        """Ensure matrix is symmetric and positive definite."""
        C = np.asarray(C, dtype=np.float64)
        C = 0.5 * (C + C.T)
        min_eig = float(np.min(np.linalg.eigvalsh(C)))
        if min_eig < 1e-10:
            C += (1e-7 - min_eig) * np.eye(self.dim)
        return C
    
    def _safe_float(self, value):
        """Safely convert value to finite float scalar."""
        value = float(np.asarray(value).flatten()[0])
        if np.isnan(value) or np.isinf(value):
            return 1e10
        return value
    
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
        self.fitness = np.array([self._safe_float(f) for f in self.fitness])
        
        best_idx = np.argmin(self.fitness)
        self.f_opt = self._safe_float(self.fitness[best_idx])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        
        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0
        
        self._reset_operator_state()
    
    def _reset_operator_state(self):
        """Reset all operator-specific state variables."""
        self.L = None
        
        # Variant 01 state
        if hasattr(self, 'y_mean_ema'):
            self.y_mean_ema = np.zeros(self.dim)
        
        # Variant 03 state
        # (no persistent state)
        
        # Variant 04 state
        if hasattr(self, 'eigval_history'):
            self.eigval_history = []
        if hasattr(self, 'eigval_ema'):
            self.eigval_ema = np.zeros(self.dim)
        
        # Variant 06 state
        if hasattr(self, 'archive_positions'):
            self.archive_positions = []
        if hasattr(self, 'archive_fitness'):
            self.archive_fitness = []
        if hasattr(self, 'archive_generations'):
            self.archive_generations = []
        
        # Variant 07 state
        # (uses self.stagnation_counter)
        
        # Variant 09 state
        if hasattr(self, 'p_escape'):
            self.p_escape = None
        
        # Variant 10 state
        # (uses self.trial_fitness)
        
        # Variant 08 state
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
        # Ensure positive definiteness before Cholesky
        min_eig = float(np.min(np.linalg.eigvalsh(self.C)))
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
        self.trial_fitness = np.array([self._safe_float(f) for f in self.trial_fitness])
    
    def _update_best(self):
        """Update best solution if trial population contains improvement."""
        trial_best_idx = np.argmin(self.trial_fitness)
        trial_best_fit = self._safe_float(self.trial_fitness[trial_best_idx])
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
        
        for i in range(min(self.NP, len(self.trial_fitness), len(self.fitness))):
            if self.trial_fitness[i] < self.fitness[i]:
                recent_improved += 1
                diff = self.fitness[i] - self.trial_fitness[i]
                total_improvement += max(float(diff), 0.0)
            recent_total += 1
        
        if recent_total == 0:
            recent_total = max(1, self.NP // 4)
        
        success_rate = float(recent_improved) / float(recent_total)
        success_rate = float(np.clip(success_rate, 0.0, 1.0))
        
        avg_improvement = float(total_improvement) / float(max(recent_improved, 1))
        improvement_magnitude = float(np.log1p(max(avg_improvement, 1e-15)))
        
        if not hasattr(self, 'improvement_ema'):
            self.improvement_ema = improvement_magnitude
        self.improvement_ema = 0.7 * self.improvement_ema + 0.3 * improvement_magnitude
        
        target_rate = 0.25
        adaptation = (success_rate - target_rate) / target_rate
        adaptation += 0.3 * float(np.tanh(self.improvement_ema - 1.0))
        adaptation = float(np.clip(adaptation, -0.8, 0.8))
        
        pop_variance = float(np.mean(np.var(self.population, axis=0)))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = float(np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0))
        
        diversity_boost = 1.0 + 2.0 * (1.0 - diversity)
        diversity_boost = float(np.clip(diversity_boost, 0.5, 3.0))
        
        damping_adaptive = float(self.damping) * (0.5 + 0.5 * float(np.log1p(self.dim))) / diversity_boost
        damping_adaptive = float(np.clip(damping_adaptive, 0.05, 200.0))
        
        self.sigma *= float(np.exp(adaptation * self.cs / damping_adaptive))
        self.sigma = float(np.clip(self.sigma, 1e-10, 10.0))
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions with UCB bonus."""
        # Thompson Sampling samples
        samples = np.random.beta(self.alpha, self.beta)
        
        # Add small UCB bonus to encourage exploration of under-sampled operators
        exploration_bonus = 0.1 * np.sqrt(np.log(self.generation + 1) / (self.selection_counts + 1))
        samples += exploration_bonus
        
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        self.selection_counts[self.current_operator] += 1
        return self.current_operator
    
    def _update_operator_rewards(self):
        """Update operator rewards using sliding window credit assignment."""
        improvement = max(0.0, float(self.f_opt_prev) - float(self.f_opt))
        
        pop_variance = float(np.mean(np.var(self.population, axis=0)))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = float(np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0))
        
        # Compute fitness variance as additional reward signal
        fit_var = float(np.var(self.fitness))
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = float(fit_var) / (fit_scale ** 2 + 1e-10)
        
        # Reward components: improvement + diversity + exploration signal
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity + 0.05 * min(rel_var, 1.0))
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
        
        # Track generational performance
        self.generational_rewards[op] += reward
        self.generational_counts[op] += 1
    
    def _adapt_covariance(self):
        """Dispatch to the selected covariance adaptation strategy."""
        dispatch_table = {
            0: self._adapt_covariance_original,
            1: self._adapt_covariance_variant_01,
            2: self._adapt_covariance_variant_03,
            3: self._adapt_covariance_variant_04,
            4: self._adapt_covariance_variant_06,
            5: self._adapt_covariance_variant_07,
            6: self._adapt_covariance_variant_09,
            7: self._adapt_covariance_variant_10,
            8: self._adapt_covariance_variant_08,
        }
        
        if self.current_operator in dispatch_table:
            dispatch_table[self.current_operator]()
        else:
            self._adapt_covariance_original()
    
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
        
        # Momentum term: EMA of mean shift direction for stable adaptation
        beta = 0.5
        if not hasattr(self, 'y_mean_ema'):
            self.y_mean_ema = np.zeros(self.dim)
        self.y_mean_ema = beta * self.y_mean_ema + (1.0 - beta) * y_mean
        
        # Adaptive learning rate based on eigenvalue spread
        eigvals = np.linalg.eigvalsh(self.C)
        eig_ratio = float(np.max(eigvals)) / (float(np.min(eigvals)) + 1e-10)
        cond_factor = min(1.0, 1.0 / float(np.log1p(eig_ratio + 1.0)))
        
        # Momentum-enhanced evolution path
        cc_momentum = float(self.cc) * (0.5 + 0.5 * cond_factor)
        cc_momentum = float(np.clip(cc_momentum, 0.01, 0.2))
        
        self.pc = (1.0 - cc_momentum) * self.pc + np.sqrt(cc_momentum * (2.0 - cc_momentum)) * self.y_mean_ema
        
        # Rank-one update with momentum
        rank_one = np.outer(self.pc, self.pc)
        
        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        # Adaptive covariance learning rate
        ccov_adaptive = float(self.ccov) * (0.5 + 0.5 * cond_factor)
        ccov_adaptive = float(np.clip(ccov_adaptive, 1e-10, 0.5))
        
        # Combine with momentum-weighted original covariance
        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_03(self):
        """Rank-distribution triggered exploration bursts for escaping local optima (5 wins)."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        
        # Compute rank-based diversity: how spread out are solutions in fitness space?
        ranks = np.argsort(np.argsort(self.fitness))  # rank of each individual
        rank_spread = float(np.max(ranks)) - float(np.min(ranks))
        rank_diversity = rank_spread / max(float(len(ranks)) - 1, 1)
        
        # Normalize by expected spread for uniform distribution
        expected_spread = 1.0
        rank_ratio = rank_diversity / max(expected_spread, 1e-10)
        rank_ratio = float(np.clip(rank_ratio, 0.01, 10.0))
        
        # Base learning rates
        ccov_1_base = 1.0 / (self.dim + 2.0)
        ccov_mu_base = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        cc_adapt_base = (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        
        # Exploration boost when ranks are clustered (low diversity)
        if rank_ratio < 0.3:  # Clustered ranks = need exploration
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
        
        # Apply exploration boost to rank-one update
        ccov_1 = min(ccov_1_base * exploration_boost, 0.3)
        ccov_mu = ccov_mu_base * min(exploration_boost, 2.0)
        cc_adapt = min(cc_adapt_base * exploration_boost, 0.3)
        
        # Evolution path update
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
        
        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)
        
        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        # Combine updates
        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C + 
                  ccov_1 * rank_one + 
                  ccov_mu * rank_mu)
        
        # Inject diagonal noise for exploration when needed
        if diag_noise > 0:
            eigvals = np.linalg.eigvalsh(self.C)
            noise = np.random.randn(self.dim) * diag_noise * float(np.mean(eigvals))
            self.C += np.diag(noise)
        
        # Restart when severely converged (extreme case)
        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = float(np.min(eigvals)) / (float(np.max(eigvals)) + 1e-10)
        if eig_spread < 1e-8 or np.any(np.isnan(self.C)):
            self.C = np.eye(self.dim) * float(np.mean(eigvals))
            self.pc = np.zeros(self.dim)
            self.L = None
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_04(self):
        """Eigenvalue history smoothing with adaptive regularization (1 win)."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        
        # Compute current eigenvalues and condition number
        eigvals = np.linalg.eigvalsh(self.C)
        cond = float(np.max(eigvals)) / (max(float(np.min(eigvals)), 1e-10))
        eig_spread = float(np.min(eigvals)) / (float(np.max(eigvals)) + 1e-10)
        
        # Track eigenvalue history for drift detection
        if not hasattr(self, 'eigval_history'):
            self.eigval_history = []
            self.eigval_ema = eigvals.copy()
        
        self.eigval_history.append(eigvals.copy())
        if len(self.eigval_history) > 10:
            self.eigval_history.pop(0)
        
        # Exponential smoothing of eigenvalues
        alpha_smooth = 0.1
        self.eigval_ema = (1.0 - alpha_smooth) * self.eigval_ema + alpha_smooth * eigvals
        
        # Detect eigenvalue drift
        if len(self.eigval_history) >= 3:
            drift_ratio = float(np.median(eigvals)) / (float(np.median(self.eigval_ema)) + 1e-10)
            drift_ratio = float(np.clip(drift_ratio, 0.1, 10.0))
        else:
            drift_ratio = 1.0
        
        # Adaptive regularization based on eigenvalue health
        reg_strength = 0.0
        if cond > 1e6 or eig_spread < 1e-6:
            reg_strength = 0.1
        elif cond > 1e4 or eig_spread < 1e-4:
            reg_strength = 0.05
        elif drift_ratio < 0.5 or drift_ratio > 2.0:
            reg_strength = 0.02
        
        # Force eigenvalues toward geometric mean for badly conditioned cases
        if cond > 1e7 or eig_spread < 1e-8:
            target_eigvals = np.full(self.dim, np.exp(np.mean(np.log(eigvals + 1e-10))))
            self.eigval_ema = (1.0 - alpha_smooth) * self.eigval_ema + alpha_smooth * target_eigvals
        
        # Adaptive learning rates with eigenvalue-health scaling
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
        
        # Scale down when eigenvalues are drifting
        drift_penalty = 0.5 if drift_ratio < 0.5 or drift_ratio > 2.0 else 1.0
        ccov_scale *= drift_penalty
        cc_scale *= drift_penalty
        
        # Fitness-based adaptation signal
        fit_var = float(np.var(self.fitness))
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)
        
        if rel_var < 1e-4:
            ccov_scale *= 0.5
            cc_scale *= 0.5
        
        # Trigger covariance restart on extreme condition number
        if cond > 1e8 or eig_spread < 1e-8:
            self.C = np.eye(self.dim) * float(np.mean(eigvals))
            self.pc = np.zeros(self.dim)
            self.L = None
            self.eigval_ema = np.full(self.dim, float(np.mean(eigvals)))
            return
        
        # Adaptive learning rates
        ccov_1 = ccov_scale * 1.0 / (self.dim + 2.0)
        ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        cc_adapt = cc_scale * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        
        # Evolution path with adaptive rate
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
        
        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)
        
        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        # Combine updates
        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C + 
                  ccov_1 * rank_one + 
                  ccov_mu * rank_mu)
        
        # Apply eigenvalue regularization
        if reg_strength > 0:
            current_trace = float(np.trace(self.C))
            target_trace = float(np.sum(self.eigval_ema))
            self.C = (1.0 - reg_strength) * self.C + reg_strength * (target_trace / self.dim) * np.eye(self.dim)
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_06(self):
        """Archive-guided covariance perturbation for escaping deceptive local optima (2 wins)."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        
        # Initialize archive for tracking distinct best solutions
        if not hasattr(self, 'archive_positions'):
            self.archive_positions = []
            self.archive_fitness = []
            self.archive_generations = []
        
        # Add current best to archive if distinct enough
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
            worst_idx = int(np.argmax(self.archive_fitness))
            self.archive_positions[worst_idx] = self.x_opt.copy()
            self.archive_fitness[worst_idx] = self.f_opt
            self.archive_generations[worst_idx] = self.generation
        
        # Compute diversity and condition metrics
        fit_var = float(np.var(self.fitness))
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)
        
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = float(np.min(eigvals))
        eig_max = float(np.max(eigvals))
        eig_spread = eig_min / (eig_max + 1e-10)
        cond = eig_max / (max(eig_min, 1e-10))
        
        # Detect trapping: low variance OR ill-conditioned OR collapsed spectrum
        is_trapped = (rel_var < 1e-3) or (cond > 1e5) or (eig_spread < 1e-5)
        
        # Compute archive-based escape direction
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
        
        # Adaptive learning rates with exploration boost when trapped
        if is_trapped:
            ccov_scale = 0.8
            cc_scale = 0.8
        else:
            ccov_scale = min(1.0, 0.2 + 2.0 * rel_var)
            cc_scale = 1.0
        
        ccov_1 = ccov_scale * 1.0 / (self.dim + 2.0)
        ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        cc_adapt = cc_scale * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        
        # Evolution path update
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
        
        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)
        
        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        # Combine updates with archive-based perturbation
        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
                  ccov_1 * rank_one +
                  ccov_mu * rank_mu)
        
        # Inject escape perturbation when trapped
        if is_trapped and len(self.archive_positions) >= 3:
            self.C = self.C + escape_perturb
        
        self.C = self._ensure_positive_definite(self.C)
        
        # Full restart on extreme ill-conditioning
        if cond > 1e8 or eig_spread < 1e-8:
            self.C = np.eye(self.dim) * float(np.mean(eigvals))
            self.pc = np.zeros(self.dim)
            self.L = None
    
    def _adapt_covariance_variant_07(self):
        """Isotropic reset with aggressive exploration scaling for escaping local optima (2 wins)."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        
        # Detect stagnation: no improvement in recent generations
        is_stagnant = self.stagnation_counter > self.max_stagnation // 2
        
        # Also trigger reset if covariance is too elongated
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = float(np.min(eigvals))
        eig_max = float(np.max(eigvals))
        cond = eig_max / max(eig_min, 1e-10)
        
        # Fitness-based trigger
        fit_var = float(np.var(self.fitness))
        rel_var = fit_var / (max(abs(self.f_opt), 1.0) ** 2 + 1e-10)
        is_converged = rel_var < 1e-3 and cond > 1e3
        
        # Force isotropic reset on stagnation or extreme conditioning
        if is_stagnant or is_converged or cond > 1e7:
            C_mean = float(np.mean(eigvals))
            self.C = np.eye(self.dim) * max(C_mean, self.sigma ** 2)
            self.pc = np.zeros(self.dim)
            self.L = None
            
            # Increase step size to encourage exploration
            if self.sigma < 0.5 * (self.ub[0] - self.lb[0]) / 3.0:
                self.sigma *= 2.0
                self.sigma = float(np.clip(self.sigma, 1e-10, 10.0))
        
        # Adaptive learning rates based on condition number
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
        
        # Additional scaling for stagnant/poorly-conditioned cases
        if is_stagnant or is_converged:
            ccov_scale *= 2.0
            cc_scale *= 1.5
        
        ccov_scale = min(ccov_scale, 2.0)
        cc_scale = min(cc_scale, 1.0)
        
        # Adaptive learning rates
        ccov_1 = ccov_scale * 1.0 / (self.dim + 2.0)
        ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        cc_adapt = cc_scale * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        
        # Evolution path update
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
        
        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)
        
        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        # Combine updates
        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C + 
                  ccov_1 * rank_one + 
                  ccov_mu * rank_mu)
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_09(self):
        """Orthogonal-escape evolution path for multimodal function escape (2 wins)."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        
        # Compute perpendicular direction to pc for escape
        if not hasattr(self, 'p_escape'):
            self.p_escape = None
        
        # Trigger escape mechanism at 50% of stagnation threshold
        if self.stagnation_counter >= self.max_stagnation // 2:
            if self.p_escape is None and np.dot(self.pc, self.pc) > 1e-20:
                v = self.pc / (np.linalg.norm(self.pc) + 1e-20)
                random_vec = np.random.randn(self.dim)
                random_vec -= np.dot(random_vec, v) * v
                norm_r = np.linalg.norm(random_vec)
                if norm_r > 1e-10:
                    self.p_escape = random_vec / norm_r
        
        # Apply main evolution path update
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        # Rank-one and rank-mu updates
        rank_one = np.outer(self.pc, self.pc)
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        # Combine main and escape updates
        if self.p_escape is not None:
            rank_escape = np.outer(self.p_escape, self.p_escape)
            self.C = ((1.0 - self.ccov) * self.C + 
                      self.ccov * rank_one + 
                      0.5 * self.ccov * rank_escape +
                      (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)
            # Decay escape path
            self.p_escape *= 0.95
        else:
            self.C = ((1.0 - self.ccov) * self.C + 
                      self.ccov * rank_one + 
                      (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_10(self):
        """Aggressive covariance reset with step-size reduction for escaping local optima (1 win)."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        
        # Compute condition number and eigenvalue spread
        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = float(np.min(eigvals)) / (float(np.max(eigvals)) + 1e-10)
        cond = float(np.max(eigvals)) / (max(float(np.min(eigvals)), 1e-10))
        
        # Check for stagnation indicators
        recent_improved = 0
        for i in range(min(len(self.trial_fitness), len(self.fitness))):
            if self.trial_fitness[i] < self.fitness[i]:
                recent_improved += 1
        success_rate = recent_improved / max(len(self.trial_fitness), 1)
        
        # Trigger aggressive reset when:
        # 1. Extreme condition number (ill-conditioned landscape)
        # 2. Very small eigenvalue spread (collapsed covariance)
        # 3. Stagnation with low success rate (trapped in local optimum)
        should_reset = (cond > 1e6) or (eig_spread < 1e-6) or \
                       (success_rate < 0.05 and self.stagnation_counter > 20)
        
        if should_reset:
            # Reset to spherical covariance (isotropic search)
            self.C = np.eye(self.dim) * float(np.mean(eigvals))
            self.pc = np.zeros(self.dim)
            self.L = None
            
            # Shrink step-size to refine search after reset
            self.sigma = max(self.sigma * 0.3, 1e-10)
            
            # Use conservative learning rates after reset
            ccov_1 = 0.5 / (self.dim + 2.0)
            ccov_mu = 0.5 * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
            cc_adapt = 0.3 * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)
        else:
            # Normal adaptation when not in trap
            ccov_1 = self.ccov
            ccov_mu = self.ccov
            cc_adapt = self.cc
        
        # Evolution path update
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
        
        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)
        
        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        # Combine updates
        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
                  ccov_1 * rank_one +
                  ccov_mu * rank_mu)
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_08(self):
        """Dual active evolution paths with exponential history weighting."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        alpha = 0.05
        if not hasattr(self, 'pc_weighted'):
            self.pc_weighted = np.zeros(self.dim)
        self.pc_weighted = (1.0 - alpha) * self.pc_weighted + alpha * y_mean
        
        beta = 0.02
        if not hasattr(self, 'p_cross'):
            self.p_cross = np.zeros(self.dim)
        if hasattr(self, 'prev_y_mean'):
            cross_contribution = np.sqrt(beta) * (y_mean - self.prev_y_mean)
            self.p_cross = (1.0 - beta) * self.p_cross + cross_contribution
        self.prev_y_mean = y_mean.copy()
        
        rank_one_primary = np.outer(self.pc, self.pc)
        rank_one_weighted = np.outer(self.pc_weighted, self.pc_weighted)
        rank_one_cross = np.outer(self.p_cross, self.p_cross)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        ccov_primary = self.ccov
        ccov_weighted = 0.3 * self.ccov
        ccov_cross = 0.1 * self.ccov
        
        self.C = ((1.0 - ccov_primary) * self.C + 
                  ccov_primary * rank_one_primary +
                  ccov_weighted * rank_one_weighted +
                  ccov_cross * rank_one_cross +
                  (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)
        
        self.C = self._ensure_positive_definite(self.C)
        
        eigvals = np.linalg.eigvalsh(self.C)
        cond = float(np.max(eigvals)) / float(np.min(eigvals))
        if cond > 1e7:
            self.C *= 0.5
    
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
        
        # Check for NaN in covariance matrix
        has_nan = np.any(np.isnan(self.C)) if hasattr(self, 'C') else False
        
        if (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            has_nan):
            
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]
            
            self._initialize_population()
            
            self.population[0] = elite
            self.fitness[0] = elite_fit
            self.f_opt = self._safe_float(elite_fit)
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0
