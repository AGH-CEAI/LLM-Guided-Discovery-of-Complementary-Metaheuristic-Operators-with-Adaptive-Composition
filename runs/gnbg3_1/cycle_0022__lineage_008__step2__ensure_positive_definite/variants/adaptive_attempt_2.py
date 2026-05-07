import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 10 covariance adaptation strategies (original + 9 variants)
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
        
        # Adaptive operator selection parameters for _ensure_positive_definite
        self.num_pd_operators = 10
        self.pd_operator_names = [
            'original', 'v01_eigclip', 'v02', 'v03_adaptive_floor', 'v04_dim_adaptive',
            'v05_cond_adaptive', 'v06_log_rescale', 'v07_rescale', 'v08', 'v09_dim_scaled', 'v10_trace_preserve'
        ]
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_pd_operators)
        self.beta = np.ones(self.num_pd_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 15
        self.operator_rewards = {i: [] for i in range(self.num_pd_operators)}
        self.operator_counts = np.zeros(self.num_pd_operators)
        self.selection_counts = np.zeros(self.num_pd_operators)
        
        # Reward tracking
        self.cumulative_rewards = np.zeros(self.num_pd_operators)
        self.decay_sums = np.zeros(self.num_pd_operators)
        
        # Runtime state
        self.generation = 0
        self.current_operator = 0
        self.L = None
        
        # Track recent fitness for reward computation
        self.fitness_history = []
        self.max_fitness_history = 20
        
    def _clip_to_bounds(self, x):
        """Clip solution to bounds."""
        return np.clip(x, self.lb, self.ub)
    
    def _ensure_positive_definite(self, C):
        """Dispatch to the selected positive definiteness enforcement strategy."""
        if self.current_operator == 0:
            return self._ensure_pd_original(C)
        elif self.current_operator == 1:
            return self._ensure_pd_variant_01(C)
        elif self.current_operator == 2:
            return self._ensure_pd_variant_02(C)
        elif self.current_operator == 3:
            return self._ensure_pd_variant_03(C)
        elif self.current_operator == 4:
            return self._ensure_pd_variant_04(C)
        elif self.current_operator == 5:
            return self._ensure_pd_variant_05(C)
        elif self.current_operator == 6:
            return self._ensure_pd_variant_06(C)
        elif self.current_operator == 7:
            return self._ensure_pd_variant_07(C)
        elif self.current_operator == 8:
            return self._ensure_pd_variant_08(C)
        elif self.current_operator == 9:
            return self._ensure_pd_variant_09(C)
        else:
            return self._ensure_pd_variant_10(C)
    
    def _ensure_pd_original(self, C):
        """Original: Ensure matrix is symmetric and positive definite."""
        C = 0.5 * (C + C.T)
        min_eig = np.min(np.linalg.eigvalsh(C))
        if min_eig < 1e-10:
            C += (1e-7 - min_eig) * np.eye(self.dim)
        return C
    
    def _ensure_pd_variant_01(self, C):
        """Eigenvalue clipping with fixed 1e-10 threshold."""
        C = 0.5 * (C + C.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-10)
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return C
    
    def _ensure_pd_variant_02(self, C):
        """Diagonal shift with adaptive strength."""
        C = 0.5 * (C + C.T)
        min_eig = np.min(np.linalg.eigvalsh(C))
        if min_eig < 1e-12:
            shift = 1e-8 * (1.0 - min_eig / 1e-12)
            C += shift * np.eye(self.dim)
        return C
    
    def _ensure_pd_variant_03(self, C):
        """Eigendecomposition with adaptive flooring based on condition number."""
        C = 0.5 * (C + C.T)
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        eig_min = np.min(eigenvalues)
        if eig_min < 1e-12:
            eigenvalues = np.clip(eigenvalues, 1e-12, None)
            C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        eig_max = np.max(eigenvalues)
        cond_threshold = 1e6
        if eig_max / max(eig_min, 1e-12) > cond_threshold:
            eigenvalues = np.clip(eigenvalues, eig_max / cond_threshold, None)
            C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return C
    
    def _ensure_pd_variant_04(self, C):
        """Adaptive eigenvalue flooring based on max eigenvalue and dimension."""
        C = 0.5 * (C + C.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        min_eig = np.min(eigvals)
        max_eig = np.max(eigvals)
        if max_eig == 0:
            max_eig = 1.0
        eig_thresh = max(1e-10, 1e-6 * max_eig / self.dim)
        if min_eig < eig_thresh:
            eigvals = np.maximum(eigvals, eig_thresh)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return C
    
    def _ensure_pd_variant_05(self, C):
        """Adaptive flooring based on condition number (5 wins - best overall)."""
        C = 0.5 * (C + C.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        min_eig = np.min(eigvals)
        max_eig = np.max(eigvals)
        if min_eig < 1e-10:
            cond = max_eig / max(min_eig, 1e-30)
            if cond > 1e6:
                floor = max(min_eig, max_eig * 1e-6)
            elif cond > 1e4:
                floor = max(min_eig, max_eig * 1e-5)
            elif cond > 1e2:
                floor = max(min_eig, max_eig * 1e-4)
            else:
                floor = max(min_eig, 1e-10)
            eigvals = np.maximum(eigvals, floor)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return C
    
    def _ensure_pd_variant_06(self, C):
        """Eigenvalue clipping with condition number control via logarithmic rescaling."""
        C = 0.5 * (C + C.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-10)
        cond = eigvals[-1] / eigvals[0]
        max_cond = 1e6
        if cond > max_cond:
            log_eigvals = np.log(eigvals)
            log_min = log_eigvals[0]
            log_max = log_eigvals[-1]
            target_range = np.log(max_cond * eigvals[0]) - log_min
            current_range = log_max - log_min
            if current_range > 1e-10:
                scale = target_range / current_range
                log_eigvals[1:] = log_min + scale * (log_eigvals[1:] - log_min)
            eigvals = np.exp(np.clip(log_eigvals, -50, 50))
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        C = 0.5 * (C + C.T)
        return C
    
    def _ensure_pd_variant_07(self, C):
        """Adaptive eigenvalue rescaling with condition number check."""
        C = 0.5 * (C + C.T)
        if np.any(np.isnan(C)) or np.any(np.isinf(C)):
            return np.eye(self.dim) * 1e-6
        try:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.nan_to_num(eigvals, nan=1e-10, posinf=1e10, neginf=-1e10)
            eig_min = np.min(eigvals)
            eig_max = np.max(eigvals)
            cond = eig_max / max(abs(eig_min), 1e-30)
            if cond > 1e6 or eig_min < 1e-10:
                target_min = max(1e-10, eig_min)
                if cond > 1e6:
                    log_scale = np.log1p(eigvals - eig_min + 1e-10)
                    max_log = np.max(log_scale) + 1e-10
                    eigvals = target_min * np.exp(log_scale / max_log * np.log(1e6))
                else:
                    eigvals = np.maximum(eigvals, target_min)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            min_eig = np.min(np.linalg.eigvalsh(C))
            if min_eig < 1e-12:
                mean_eig = np.mean(np.abs(eigvals))
                offset = max(1e-10, 1e-8 * mean_eig - min_eig)
                C = C + offset * np.eye(self.dim)
            return C
        except np.linalg.LinAlgError:
            diag_C = np.diag(C)
            diag_C = np.maximum(np.abs(diag_C), 1e-10)
            return np.diag(diag_C)
    
    def _ensure_pd_variant_08(self, C):
        """Spectral normalization with soft thresholding."""
        C = 0.5 * (C + C.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        eig_max = np.max(eigvals)
        eig_min = np.min(eigvals)
        if eig_min < 1e-12:
            eigvals = np.clip(eigvals, 1e-12, None)
        if eig_max > 1e12:
            eigvals = np.clip(eigvals, None, 1e12)
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return C
    
    def _ensure_pd_variant_09(self, C):
        """Eigenvalue clipping with dimension scaling."""
        C = 0.5 * (C + C.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        min_eig = np.min(eigvals)
        max_eig = np.max(eigvals)
        cond = max_eig / max(min_eig, 1e-15)
        dim_scale = max(1.0, self.dim / 10.0)
        eig_thresh = 1e-10 * dim_scale
        max_cond = 1e6 * dim_scale
        if cond > max_cond:
            target_min = max_eig / max_cond
            eigvals = np.clip(eigvals, target_min, None)
        elif min_eig < eig_thresh:
            eigvals = np.clip(eigvals, eig_thresh, None)
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        C = 0.5 * (C + C.T)
        return C
    
    def _ensure_pd_variant_10(self, C):
        """Eigenvalue clipping with trace preservation."""
        C = 0.5 * (C + C.T)
        eigvals, eigvecs = np.linalg.eigh(C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        min_thresh = max(1e-12, abs(eig_min) * 0.1 + 1e-12)
        eigvals_clipped = np.clip(eigvals, min_thresh, None)
        max_thresh = max(eigvals_clipped) * 1e8
        eigvals_clipped = np.clip(eigvals_clipped, None, max_thresh)
        original_trace = np.sum(eigvals)
        clipped_trace = np.sum(eigvals_clipped)
        if clipped_trace > 0 and abs(original_trace - clipped_trace) > 1e-10:
            eigvals_clipped *= (original_trace / clipped_trace)
        C = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
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
        
        # Reset fitness history
        self.fitness_history = []
    
    def _sample_trials_batch(self):
        """Sample new trial population using Cholesky decomposition."""
        # Ensure positive definiteness before Cholesky using selected strategy
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C = self._ensure_positive_definite(self.C)

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
        # Track fitness history for reward computation
        self.fitness_history.append(self.f_opt)
        if len(self.fitness_history) > self.max_fitness_history:
            self.fitness_history.pop(0)
        
        # Compute improvement over recent history
        improvement = 0.0
        if len(self.fitness_history) >= 2:
            # Use exponential improvement rate
            f_start = self.fitness_history[0]
            f_end = self.fitness_history[-1]
            if f_start > f_end:
                # Relative improvement
                rel_imp = (f_start - f_end) / max(abs(f_start), 1e-10)
                improvement = float(rel_imp)
        
        # Add recent step improvement
        recent_imp = max(0.0, self.f_opt_prev - self.f_opt)
        improvement = max(improvement, recent_imp)

        # Apply exponential decay to accumulated reward
        decay = 0.95
        self.cumulative_rewards *= decay
        self.decay_sums *= decay

        # Log-scaled improvement reward
        if improvement > 1e-15:
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
        self.cumulative_rewards[self.current_operator] += reward
        self.decay_sums[self.current_operator] += 1.0

        # Normalize by decay sum
        norm = max(self.decay_sums[self.current_operator], 1.0)
        normalized_reward = self.cumulative_rewards[self.current_operator] / norm
        normalized_reward = float(np.clip(normalized_reward, -10.0, 10.0))

        op = self.current_operator
        self.operator_rewards[op].append(normalized_reward)

        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)

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
