```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 10 covariance adaptation strategies (original + 9 variants from benchmark)
    - Thompson Sampling for operator selection
    - Sliding window credit assignment
    - Automatic restart on stagnation
    - Robust edge case handling
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
        
        # Adaptive operator selection parameters (10 operators)
        self.num_operators = 10
        self.operator_names = [
            'original', 'variant_01', 'variant_02', 'variant_04', 'variant_05',
            'variant_06', 'variant_07', 'variant_08', 'variant_09', 'variant_10'
        ]
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_operators)
        self.beta = np.ones(self.num_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 15
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators)
        self.selection_counts = np.zeros(self.num_operators)
        self.operator_fitness_history = {i: [] for i in range(self.num_operators)}
        
        # Runtime state
        self.generation = 0
        self.current_operator = 0
        self.L = None
        self.diag_C = None
    
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
        
        # Initialize diagonal covariance for variant_01
        self.diag_C = np.ones(self.dim) * np.mean(np.diag(self.C))
        
        self.fitness = self.func(self.population)
        
        best_idx = np.argmin(self.fitness)
        f_best = np.asarray(self.fitness[best_idx]).flatten()[0]
        self.f_opt = float(f_best) if not np.isinf(f_best) else 1e10
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
        if hasattr(self, 'stagnation_history'):
            self.stagnation_history = []
        if hasattr(self, 'best_known_fitness'):
            self.best_known_fitness = float('inf')
        if hasattr(self, 'stagnation_depth'):
            self.stagnation_depth = 0.0
    
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
        f_trial = np.asarray(self.trial_fitness[trial_best_idx]).flatten()[0]
        trial_best_fit = float(f_trial) if not np.isinf(f_trial) and not np.isnan(f_trial) else 1e10
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
                f_trial = float(np.asarray(self.trial_fitness[i]).flatten()[0])
                f_curr = float(np.asarray(self.fitness[i]).flatten()[0])
                if f_trial < f_curr:
                    recent_improved += 1
                    diff = f_curr - f_trial
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
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        # Compute condition number for additional reward signal
        try:
            eigvals = np.linalg.eigvalsh(self.C)
            cond = float(np.max(eigvals) / max(np.min(eigvals), 1e-15))
        except:
            cond = 1.0
        
        # Reward components: improvement, diversity, condition number health
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity)
        
        # Penalize high condition numbers (indicates potential convergence issues)
        if cond > 1e6:
            reward -= 0.5
        elif cond > 1e4:
            reward -= 0.2
        
        reward = float(np.clip(reward, -10.0, 10.0))
        
        op = self.current_operator
        self.operator_rewards[op].append(reward)
        
        # Store fitness history for this operator
        self.operator_fitness_history[op].append(self.f_opt)
        
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)
        if len(self.operator_fitness_history[op]) > self.reward_window_size:
            self.operator_fitness_history[op].pop(0)
        
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
        
        # Exploration bonus for under-sampled operators
        min_selections = float(np.min(self.selection_counts[self.selection_counts > 0])) if np.any(self.selection_counts > 0) else 1.0
        if min_selections > 0:
            for i in range(self.num_operators):
                if self.selection_counts[i] < min_selections * 0.5:
                    self.alpha[i] *= 0.9
                    self.beta[i] *= 1.1
    
    def _adapt_covariance(self):
        """Dispatch to the selected covariance adaptation strategy."""
        dispatch_table = {
            0: self._adapt_covariance_original,
            1: self._adapt_covariance_variant_01,
            2: self._adapt_covariance_variant_02,
            3: self._adapt_covariance_variant_04,
            4: self._adapt_covariance_variant_05,
            5: self._adapt_covariance_variant_06,
            6: self._adapt_covariance_variant_07,
            7: self._adapt_covariance_variant_08,
            8: self._adapt_covariance_variant_09,
            9: self._adapt_covariance_variant_10,
        }
        
        op = self.current_operator
        if op in dispatch_table:
            dispatch_table[op]()
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
        """Diagonal-only (sep-CMA) covariance adaptation with per-dimension learning rates."""
        # Initialize diagonal covariance if needed
        if not hasattr(self, 'diag_C') or self.diag_C is None:
            self.diag_C = np.ones(self.dim)

        # Compute mean shift in normalized space
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Per-dimension rank-1 update using evolution path
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        # Separate learning rates for rank-1 and rank-mu updates
        cc_1 = 1.0 / (self.dim + 2.0)
        cc_mu = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)

        # Rank-1 update for diagonal (squared values)
        rank_one_contrib = self.pc ** 2

        # Rank-mu update for diagonal
        rank_mu_contrib = np.zeros(self.dim)
        total_weight = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu_contrib += self.weights[i] * (diff ** 2)
            total_weight += abs(self.weights[i])

        if total_weight > 0:
            rank_mu_contrib /= total_weight

        # Adaptive learning rate based on dimensionality
        dim_adapt = np.log(self.dim + 1.0) / np.log(100.0)
        cc_1_adapt = cc_1 * (0.5 + 0.5 * dim_adapt)
        cc_mu_adapt = cc_mu * (1.5 - 0.5 * dim_adapt)

        # Clip learning rates for stability
        cc_1_adapt = np.clip(cc_1_adapt, 1e-10, 0.5)
        cc_mu_adapt = np.clip(cc_mu_adapt, 1e-10, 0.5)

        # Update diagonal covariance
        self.diag_C = (1.0 - cc_1_adapt - cc_mu_adapt) * self.diag_C
        self.diag_C += cc_1_adapt * rank_one_contrib
        self.diag_C += cc_mu_adapt * rank_mu_contrib

        # Ensure positive diagonal with minimum variance
        min_var = 1e-10 * ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        self.diag_C = np.maximum(self.diag_C, min_var)

        # Build full covariance matrix from diagonal
        self.C = np.diag(self.diag_C)

        # Update Cholesky factor for sampling (diagonal case)
        self.L = np.diag(np.sqrt(self.diag_C))
    
    def _adapt_covariance_variant_02(self):
        """Eigendecomposition-based anisotropic expansion for escaping local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        total_weight = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_weight += abs(w)

        if total_weight > 0:
            rank_mu /= total_weight

        ccov_1 = 1.0 / (self.dim + 2.0)
        ccov_mu = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)

        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C + 
                  ccov_1 * rank_one + 
                  ccov_mu * rank_mu * np.sum(self.weights[:self.mu]))

        # Detect narrow covariance (high condition number) indicating potential local optima trap
        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-15)
            cond = np.max(eigvals) / np.min(eigvals)

            # Threshold for "too narrow" covariance
            narrow_threshold = 1e5

            if cond > narrow_threshold:
                # Anisotropic expansion: scale up smallest eigenvalues significantly
                min_eig = np.min(eigvals)
                expansion_factor = 10.0

                # Create expansion vector: scale smallest eigvals more
                expansion = np.ones_like(eigvals)
                expansion[eigvals < np.median(eigvals)] = expansion_factor

                new_eigvals = eigvals * expansion
                new_eigvals = np.maximum(new_eigvals, min_eig * 0.1)

                # Reconstruct covariance with expanded eigenvalues
                self.C = eigvecs @ np.diag(new_eigvals) @ eigvecs.T
                self.C = 0.5 * (self.C + self.C.T)

            # Also trigger expansion if population diversity is very low
            pop_variance = np.mean(np.var(self.population, axis=0))
            expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
            diversity_ratio = pop_variance / (expected_var + 1e-10)

            if diversity_ratio < 1e-4:
                # Broad isotropic expansion when diversity collapses
                min_eig = np.min(eigvals)
                self.C += 0.5 * min_eig * np.eye(self.dim)

        except np.linalg.LinAlgError:
            pass

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_04(self):
        """Eigenspace-adaptive covariance with explosive exploration."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Eigendecomposition of current covariance
        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-15)
        except np.linalg.LinAlgError:
            self.C = 1e-6 * np.eye(self.dim)
            return

        # Project mean change into eigenspace
        z_1 = eigvecs.T @ y_mean
        z_1_norm = np.linalg.norm(z_1)
        if z_1_norm > 1e-10:
            z_1 = z_1 / z_1_norm

        # Adaptive learning rates based on eigenvalue spread
        cond_C = np.max(eigvals) / np.min(eigvals)
        cond_log = np.log1p(cond_C)

        # More aggressive rates for ill-conditioned problems
        ccov_1 = min(2.0 / (self.dim + 2.0), 0.3)
        ccov_mu = min(self.mueff / (self.dim + 2.0) / (self.dim + 4.0), 0.2)

        # Rank-one update in eigenspace
        rank_one = z_1 ** 2

        # Rank-mu update in eigenspace
        rank_mu = np.zeros(self.dim)
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            z_i = eigvecs.T @ diff
            rank_mu += self.weights[i] * (z_i ** 2)

        # Update eigenvalues directly
        eigvals_new = (1.0 - ccov_1 - ccov_mu) * eigvals
        eigvals_new += ccov_1 * z_1_norm * z_1_norm * eigvals * rank_one
        eigvals_new += ccov_mu * eigvals * rank_mu

        # Explosive exploration: elongate axes for ill-conditioned tasks
        if cond_log > 2.0:
            elongation = min(cond_log * 0.3, 2.0)
            max_idx = np.argmax(eigvals)
            eigvals_new[max_idx] *= (1.0 + elongation)

        # Clamp eigenvalues for numerical stability
        eigvals_new = np.clip(eigvals_new, 1e-14, 1e+10)

        # Reconstruct covariance matrix
        self.C = eigvecs @ np.diag(eigvals_new) @ eigvecs.T
        self.C = 0.5 * (self.C + self.C.T)

        # Condition number control
        new_cond = np.max(eigvals_new) / np.min(eigvals_new)
        if new_cond > 1e+8:
            min_eig = np.min(eigvals_new)
            max_eig = min_eig * 1e+8
            eigvals_new = np.clip(eigvals_new, None, max_eig)
            self.C = eigvecs @ np.diag(eigvals_new) @ eigvecs.T
    
    def _adapt_covariance_variant_05(self):
        """Aggressive restart-driven covariance with stagnation escape."""
        stagnation_threshold = max(20, self.dim * 2)

        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        if self.stagnation_counter > stagnation_threshold:
            explosion_factor = min(50.0, 1.0 + self.stagnation_counter * 0.5)
            self.sigma *= explosion_factor
            self.sigma = np.clip(self.sigma, 1e-10, 10.0)

            self.C *= explosion_factor

            pop_variance = np.mean(np.var(self.population, axis=0))
            expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
            diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

            if diversity < 0.1:
                self.C += 0.5 * np.diag(np.abs(np.diag(self.C)) + 1e-8)

            ccov_escape = min(0.8, self.ccov * explosion_factor)
            self.C = ((1.0 - ccov_escape) * self.C + 
                      ccov_escape * rank_one +
                      (1.0 - 1.0 / self.mueff) * ccov_escape * 2.0 * rank_mu)

            self.stagnation_counter = 0
        else:
            self.C = ((1.0 - self.ccov) * self.C + 
                      self.ccov * rank_one +
                      (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_06(self):
        """Stagnation-triggered eigenvalue perturbation for escaping local optima."""
        # Detect stagnation by monitoring fitness improvement
        if not hasattr(self, 'stagnation_history'):
            self.stagnation_history = []

        self.stagnation_history.append(self.f_opt)
        if len(self.stagnation_history) > 20:
            self.stagnation_history.pop(0)

        # Compute stagnation metrics
        is_stagnant = False
        if len(self.stagnation_history) >= 20:
            recent_improvement = self.stagnation_history[0] - self.stagnation_history[-1]
            relative_improvement = recent_improvement / (abs(self.stagnation_history[0]) + 1e-10)
            is_stagnant = (relative_improvement < 0.01) or (self.stagnation_counter > self.max_stagnation // 2)

        # Evolution path update
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        # Stagnation-triggered perturbation along minor eigenvectors
        if is_stagnant:
            try:
                eigvals, eigvecs = np.linalg.eigh(self.C)
                eigvals = np.maximum(eigvals, 1e-15)

                # Perturb along 3 minor eigenvectors (highest uncertainty directions)
                num_perturb = min(3, self.dim)
                for k in range(num_perturb):
                    minor_dir = eigvecs[:, k]
                    scale = np.sqrt(eigvals[k]) * self.sigma * 0.3
                    perturbation = minor_dir * np.random.randn() * scale
                    self.mean = self.mean + perturbation

                # Reset evolution path to encourage new direction
                self.pc *= 0.1

                # Inject fresh diversity into population
                for i in range(min(10, self.NP)):
                    idx = self.NP - 1 - i
                    random_dir = np.random.randn(self.dim)
                    random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-15)
                    self.population[idx] = self._clip_to_bounds(
                        self.mean + random_dir * np.random.uniform(0.5, 2.0) * self.sigma * np.sqrt(self.dim)
                    )
            except np.linalg.LinAlgError:
                pass

        # Rank-one and rank-μ updates
        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Adaptive learning rate based on stagnation
        if is_stagnant:
            ccov_effective = min(self.ccov * 3.0, 0.5)
        else:
            ccov_effective = self.ccov

        self.C = ((1.0 - ccov_effective) * self.C + 
                  ccov_effective * (rank_one + (1.0 - 1.0 / self.mueff) * 2.0 * rank_mu))

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_07(self):
        """Stagnation-triggered covariance explosions for escaping local optima."""
        # Track best known fitness for stagnation detection
        if not hasattr(self, 'best_known_fitness'):
            self.best_known_fitness = float('inf')

        f_opt = max(abs(self.f_opt), 1e-10)
        if self.f_opt < self.best_known_fitness:
            self.best_known_fitness = self.f_opt
            self.stagnation_depth = 0.0
        else:
            self.stagnation_depth = np.log1p(max(self.best_known_fitness - self.f_opt, 0)) + 1.0

        stagnation_factor = np.clip(self.stagnation_depth / 10.0, 0.0, 1.0)

        ccov_base = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)

        ccov = ccov_base * (1.0 + 5.0 * stagnation_factor)
        ccov = np.clip(ccov, 1e-8, 0.5)

        cc_adaptive = self.cc * (1.0 + 2.0 * stagnation_factor)
        cc_adaptive = np.clip(cc_adaptive, 0.01, 0.5)

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

        C_new = ((1.0 - ccov) * self.C + 
                 ccov * rank_one + 
                 (1.0 - 1.0 / self.mueff) * ccov * 2.0 * rank_mu)

        # Stagnation escape: inject large anisotropic perturbations
        if stagnation_factor > 0.3:
            escape_strength = stagnation_factor * 0.5
            n_directions = min(self.dim, max(3, int(self.dim * 0.3)))

            eigvals, eigvecs = np.linalg.eigh(self.C)
            idx = np.argsort(eigvals)
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            for k in range(n_directions):
                direction = eigvecs[:, -(k + 1)]
                perturbation = escape_strength * np.outer(direction, direction)
                C_new += perturbation

            self.sigma *= (1.0 + stagnation_factor * 2.0)

        self.C = self._ensure_positive_definite(C_new)

        # Maintain Cholesky factor L for sampling
        try:
            self.L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
        except np.linalg.LinAlgError:
            diag_C = np.diag(self.C)
            diag_C = np.maximum(diag_C, 1e-10)
            self.L = np.diag(np.sqrt(diag_C))
    
    def _adapt_covariance_variant_08(self):
        """Eigendecomposition-directed covariance with adaptive eigenvalue control."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Compute population diversity to detect premature convergence
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        # Compute rank-mu update matrix
        rank_mu = np.zeros((self.dim, self.dim))
        total_weight = 0.0
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            w = self.weights[i]
            rank_mu += w * np.outer(diff, diff)
            total_weight += abs(w)

        if total_weight > 1e-30:
            rank_mu /= total_weight

        # Build target covariance as weighted combination
        ccov_1 = 1.0 / (self.dim + 2.0)
        ccov_mu = self.mueff / (self.dim + 2.0) / (self.dim + 4.0)

        rank_one = np.outer(y_mean, y_mean) / max(np.dot(y_mean, y_mean), 1e-30)

        C_target = (1.0 - ccov_1 - ccov_mu) * self.C
        C_target += ccov_1 * rank_one
        C_target += ccov_mu * rank_mu * np.sum(self.weights[:self.mu])

        # Ensure symmetry and positive definiteness
        C_target = 0.5 * (C_target + C_target.T)
        min_eig = np.min(np.linalg.eigvalsh(C_target))
        if min_eig < 1e-12:
            C_target += (1e-11 - min_eig) * np.eye(self.dim)

        # Perform eigendecomposition for direct eigenvalue control
        try:
            eigvals, eigvecs = np.linalg.eigh(C_target)
        except np.linalg.LinAlgError:
            C_target = np.eye(self.dim) * np.mean(np.diag(self.C))
            eigvals, eigvecs = np.linalg.eigh(C_target)

        # Stability check
        eigvals = np.maximum(eigvals, 1e-15)

        # Adaptive eigenvalue modification based on convergence state
        cond = np.max(eigvals) / (np.min(eigvals) + 1e-15)

        if cond > 1e6 or diversity < 0.05:
            # Escape mode: boost smallest eigenvalues significantly
            eigvals = eigvals ** 0.5
            min_eig_val = np.min(eigvals)
            boost_factor = 2.0 + 3.0 * (1.0 - diversity)
            eigvals = eigvals + boost_factor * min_eig_val
            eigvals = np.clip(eigvals, 1e-14, None)
        elif cond < 1e2 and diversity > 0.2:
            # Refinement mode: shrink largest eigenvalues for precision
            max_eig_val = np.max(eigvals)
            shrink_factor = 0.7
            eigvals = np.where(eigvals > 0.5 * max_eig_val, eigvals * shrink_factor, eigvals)
            eigvals = np.maximum(eigvals, 1e-15)
        else:
            # Balanced mode: moderate regularization
            eigvals = eigvals * 0.95 + 0.05 * np.mean(eigvals)

        # Reconstruct covariance from modified eigenvalues
        self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T
        self.C = 0.5 * (self.C + self.C.T)

        # Inject orthogonal random exploration when diversity is critically low
        if diversity < 0.02:
            for _ in range(3):
                z = np.random.randn(self.dim)
                z = z / max(np.linalg.norm(z), 1e-15)
                random_dir = np.outer(z, z)
                self.C += 0.1 * np.mean(eigvals) * random_dir

        # Final safety check
        self.C = self._ensure_positive_definite(self.C)

        # Update Cholesky factor for sampling
        try:
            self.L = np.linalg.cholesky(self.C + 1e-10 * np.eye(self.dim))
        except np.linalg.LinAlgError:
            diag_C = np.diag(self.C)
            diag_C = np.maximum(diag_C, 1e-10)
            self.L = np.diag(np.sqrt(diag_C))
    
    def _adapt_covariance_variant_09(self):
        """Eigenvalue spread control with adaptive learning rates."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Compute eigenvalue spread of covariance matrix
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.maximum(eigvals.min(), 1e-15)
        eig_max = np.maximum(eigvals.max(), 1e-15)
        cond = eig_max / eig_min
        eig_spread = eig_min / (eig_max + 1e-15)

        # Adaptive learning rates based on condition number
        log_cond = np.log10(max(cond, 1.0))
        spread_factor = 1.0 + max(0.0, log_cond - 2.0) * 0.5

        # Base learning rates from CMA-ES theory
        ccov_base = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)

        # Scale up learning rates when condition number is large
        ccov_adaptive = min(ccov_base * spread_factor, 0.5)
        cc_adaptive = min(self.cc * spread_factor, 0.3)

        # Evolution path update
        self.pc = (1.0 - cc_adaptive) * self.pc + \
                  np.sqrt(cc_adaptive * (2.0 - cc_adaptive)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Combined update with adaptive rate
        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * (rank_one + (1.0 - 1.0 / self.mueff) * 2.0 * rank_mu))

        # Inject exploration when condition number is problematic
        if log_cond > 3.0:
            # Add small random perturbation to prevent eigenvalue collapse
            noise_scale = 0.01 * ccov_adaptive * np.mean(eigvals)
            self.C += noise_scale * np.eye(self.dim)

        # Hard reset if covariance becomes pathological
        if cond > 1e8 or np.any(np.isnan(self.C)):
            self.C = np.eye(self.dim) * (self.sigma ** 2)
            self.pc = np.zeros(self.dim)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_10(self):
        """Condition-number triggered restart with adaptive learning rate scaling."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Compute condition number as diversity proxy
        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
        cond = np.max(eigvals) / (np.maximum(np.min(eigvals), 1e-10))

        # Adaptive learning rate based on condition number
        if cond > 1e6:
            ccov_scale = 0.1
            cc_scale = 0.1
        elif cond > 1e4:
            ccov_scale = 0.3
            cc_scale = 0.3
        elif cond > 1e2:
            ccov_scale = 0.5
            cc_scale = 0.5
        else:
            ccov_scale = 1.0
            cc_scale = 1.0

        # Fitness variance for additional adaptation signal
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        if rel_var < 1e-4:
            ccov_scale *= 0.5
            cc_scale *= 0.5

        # Trigger covariance restart on extreme condition number
        if cond > 1e8 or eig_spread < 1e-8:
            self.C = np.eye(self.dim) * np.mean(eigvals)
            self.pc = np.zeros(self.dim)
            self.L = None

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
        
        # Check for NaN or Inf in covariance
        has_nan = np.any(np.isnan(self.C)) if hasattr(self, 'C') else False
        has_inf = np.any(np.isinf(self.C)) if hasattr(self, 'C') else False
        
        if (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            has_nan or has_inf):
            
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]
            
            self._initialize_population()
            
            self.population[0] = elite
            self.fitness[0] = elite_fit
            f_elite = np.asarray(elite_fit).flatten()[0]
            self.f_opt = float(f_elite) if not np.isinf(f_elite) and not np.isnan(f_elite) else 1e10
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0
```