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
        """Adapt step-size using short-term success history (no cumulation)."""
        recent_improved = 0
        recent_total = 0

        for i in range(self.NP):
            if i < len(self.trial_fitness) and i < len(self.fitness):
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
        """Select operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _update_operator_rewards(self):
        """Update operator rewards using sliding window credit assignment."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
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
        """Cholesky factor update - maintains C = LL^T directly."""
        if self.L is None:
            self.L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
        
        y_mean = (self.mean - self.old_mean) / self.sigma
        z_1 = y_mean / np.maximum(np.linalg.norm(y_mean), 1e-10)
        
        rank_mu = np.zeros((self.dim, self.dim))
        total_weight = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_weight += w
        
        if total_weight > 0:
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
            self.L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
    
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
        cond = np.max(eigvals) / np.min(eigvals)
        if cond > 1e7:
            self.C *= 0.5
    
    def _adapt_covariance_variant_09(self):
        """Block-diagonal covariance with stagnation-triggered differential exploration."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        # Track stagnation
        if not hasattr(self, 'stagnation_gentle'):
            self.stagnation_gentle = 0

        improvement = self.prev_f_opt - self.f_opt if hasattr(self, 'prev_f_opt') else 0
        if improvement > 1e-10:
            self.stagnation_gentle = 0
        else:
            self.stagnation_gentle += 1
        self.prev_f_opt = self.f_opt

        if not hasattr(self, 'restart_counter'):
            self.restart_counter = 0
        self.restart_counter += 1

        # Block-diagonal structure: independent adaptation per dimension block
        num_blocks = max(2, min(self.dim // 4, 16))
        block_size = max(1, self.dim // num_blocks)

        if not hasattr(self, 'C_blocks'):
            self.C_blocks = []
            self.pc_blocks = []
            for b in range(num_blocks):
                block_dim = min(block_size, self.dim - b * block_size)
                self.C_blocks.append(np.eye(block_dim) * 0.01)
                self.pc_blocks.append(np.zeros(block_dim))

        # Update each block independently
        for b in range(num_blocks):
            start = b * block_size
            end = min((b + 1) * block_size, self.dim)
            block_dim = end - start

            if block_dim <= 0:
                continue

            pc_block = self.pc[start:end]
            cc_block = min(self.cc * (1 + 0.1 * block_dim), 0.5)

            self.pc_blocks[b] = (1 - cc_block) * self.pc_blocks[b] + np.sqrt(cc_block * (2 - cc_block)) * y_mean[start:end]

            rank_one_block = np.outer(self.pc_blocks[b], self.pc_blocks[b])

            rank_mu_block = np.zeros((block_dim, block_dim))
            for i in range(self.mu):
                diff_block = (self.population[i, start:end] - self.old_mean[start:end]) / self.sigma
                rank_mu_block += self.weights[i] * np.outer(diff_block, diff_block)

            # Boost learning rate during stagnation
            ccov_block = min(self.ccov * (1 + 0.5 * self.stagnation_gentle / max(1, self.max_stagnation)), 0.3)

            self.C_blocks[b] = ((1 - ccov_block) * self.C_blocks[b] + 
                               ccov_block * rank_one_block + 
                               ccov_block * (1 - 1 / self.mueff) * ccov_block * 2 * rank_mu_block)

            self.C_blocks[b] = self._ensure_positive_definite(self.C_blocks[b])

        # Reconstruct full covariance from blocks
        self.C = np.zeros((self.dim, self.dim))
        for b in range(num_blocks):
            start = b * block_size
            end = min((b + 1) * block_size, self.dim)
            block_dim = end - start
            if block_dim > 0:
                self.C[start:end, start:end] = self.C_blocks[b]

        # Inject differential mutation when stagnating (DE-style exploration)
        if self.stagnation_gentle > self.max_stagnation // 3:
            for _ in range(3):
                b1, b2 = np.random.randint(num_blocks), np.random.randint(num_blocks)
                start1, end1 = b1 * block_size, min((b1 + 1) * block_size, self.dim)
                start2, end2 = b2 * block_size, min((b2 + 1) * block_size, self.dim)

                if end1 > start1 and end2 > start2:
                    diff1 = np.random.randn(end1 - start1)
                    diff2 = np.random.randn(end2 - start2)

                    perturb = np.zeros(self.dim)
                    perturb[start1:end1] = diff1
                    perturb[start2:end2] = diff2 - diff1

                    self.C += (0.1 * self.sigma ** 2) * np.outer(perturb, perturb)

        # Periodic full restart of covariance to escape deep local optima
        if self.restart_counter > self.max_stagnation * 2:
            self.C = np.eye(self.dim) * ((self.ub[0] - self.lb[0]) / 12.0) ** 2
            self.C_blocks = [np.eye(min(block_size, self.dim - b * block_size)) * 0.01 
                            for b in range(num_blocks)]
            self.pc = np.zeros(self.dim)
            self.pc_blocks = [np.zeros(min(block_size, self.dim - b * block_size)) 
                             for b in range(num_blocks)]
            self.restart_counter = 0
            self.stagnation_gentle = 0

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
