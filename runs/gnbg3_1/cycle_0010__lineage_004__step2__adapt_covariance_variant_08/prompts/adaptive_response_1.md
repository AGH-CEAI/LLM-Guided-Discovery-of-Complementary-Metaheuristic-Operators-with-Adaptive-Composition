```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 11 covariance adaptation strategies (original + 10 winning variants)
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
        
        # Adaptive operator selection parameters - 11 operators (original + 10 variants)
        self.num_operators = 11
        self.operator_names = [
            'original', 'variant_01', 'variant_02', 'variant_03', 'variant_04',
            'variant_05', 'variant_06', 'variant_07', 'variant_08', 'variant_09', 'variant_10'
        ]
        
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
        # Reset variant-specific state
        if hasattr(self, 'rotation_stagnation'):
            self.rotation_stagnation = 0
        if hasattr(self, 'restart_fitness_history'):
            self.restart_fitness_history = []
        if hasattr(self, 'archive_positions'):
            self.archive_positions = []
            self.archive_fitness = []
            self.archive_generations = []
        if hasattr(self, 'improvement_history'):
            self.improvement_history = []
        if hasattr(self, 'prev_f_opt_rot'):
            delattr(self, 'prev_f_opt_rot')
        if hasattr(self, 'exploration_temp'):
            delattr(self, 'exploration_temp')
    
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
            self._adapt_covariance_variant_02()
        elif self.current_operator == 3:
            self._adapt_covariance_variant_03()
        elif self.current_operator == 4:
            self._adapt_covariance_variant_04()
        elif self.current_operator == 5:
            self._adapt_covariance_variant_05()
        elif self.current_operator == 6:
            self._adapt_covariance_variant_06()
        elif self.current_operator == 7:
            self._adapt_covariance_variant_07()
        elif self.current_operator == 8:
            self._adapt_covariance_variant_08()
        elif self.current_operator == 9:
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
        """Cholesky-direct active covariance adaptation with diagonal perturbation."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        cc_adapt = min(self.cc * 1.5, 0.3)
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        ccov_1 = min(2.0 / (self.dim + 2.0), self.ccov * 2.0)
        ccov_mu = min(self.mueff / (self.dim + 2.0) / (self.dim + 4.0), ccov_1 * 0.5)

        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
                  ccov_1 * rank_one +
                  ccov_mu * rank_mu)

        diag_noise = 1e-6 * (self.ub[0] - self.lb[0])
        self.C += diag_noise * np.eye(self.dim)

        self.C = self._ensure_positive_definite(self.C)

        eigvals = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals) / max(np.min(eigvals), 1e-15)

        if cond > 1e7:
            self.C *= 0.9
            self.C += 1e-4 * np.eye(self.dim)
        elif cond < 1e3 and self.generation > 20:
            self.C += 0.01 * np.eye(self.dim)
    
    def _adapt_covariance_variant_02(self):
        """Variant 02 - baseline variant for completeness."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        ccov_scale = min(1.5, 0.5 + 0.5 * self.generation / 100.0)
        ccov_adapted = min(self.ccov * ccov_scale, 0.5)
        
        self.C = ((1.0 - ccov_adapted) * self.C + 
                  ccov_adapted * rank_one +
                  (1.0 - 1.0 / self.mueff) * ccov_adapted * 2.0 * rank_mu)
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_03(self):
        """Variant 03 - safe fallback with minimal adaptation."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        # Conservative update
        ccov_conservative = min(self.ccov * 0.5, 0.1)
        self.C = ((1.0 - ccov_conservative) * self.C + 
                  ccov_conservative * rank_one +
                  (1.0 - 1.0 / self.mueff) * ccov_conservative * 2.0 * rank_mu)
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_04(self):
        """Restart-triggered aggressive covariance injection for escaping catastrophic stagnation."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        eigvals = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals) / np.maximum(np.min(eigvals), 1e-15)

        stagnation_ratio = self.stagnation_counter / max(self.max_stagnation, 1)

        if stagnation_ratio > 0.7 or cond > 1e5:
            inject_dir = np.random.randn(self.dim)
            inject_dir /= max(np.linalg.norm(inject_dir), 1e-10)
            random_rank_one = np.outer(inject_dir, inject_dir)
            inject_strength = 0.5
            ccov_boost = 10.0
            cc_boost = 5.0
        elif stagnation_ratio > 0.4 or cond > 1e4:
            inject_dir = np.random.randn(self.dim)
            inject_dir /= max(np.linalg.norm(inject_dir), 1e-10)
            random_rank_one = np.outer(inject_dir, inject_dir)
            inject_strength = 0.2
            ccov_boost = 4.0
            cc_boost = 2.5
        elif stagnation_ratio > 0.2:
            inject_dir = np.random.randn(self.dim)
            inject_dir /= max(np.linalg.norm(inject_dir), 1e-10)
            random_rank_one = np.outer(inject_dir, inject_dir)
            inject_strength = 0.1
            ccov_boost = 2.0
            cc_boost = 1.5
        else:
            random_rank_one = np.zeros((self.dim, self.dim))
            inject_strength = 0.0
            ccov_boost = 1.0
            cc_boost = 1.0

        cc_adapted = min(self.cc * cc_boost, 0.5)
        self.pc = (1.0 - cc_adapted) * self.pc + np.sqrt(cc_adapted * (2.0 - cc_adapted)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        ccov_adapted = min(self.ccov * ccov_boost, 0.6)
        self.C = ((1.0 - ccov_adapted) * self.C +
                  ccov_adapted * rank_one +
                  inject_strength * random_rank_one +
                  (1.0 - 1.0 / self.mueff) * ccov_adapted * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)

        eigvals_new = np.linalg.eigvalsh(self.C)
        cond_new = np.max(eigvals_new) / np.maximum(np.min(eigvals_new), 1e-15)
        if cond_new > 1e8:
            self.C = np.eye(self.dim) * np.mean(eigvals_new)
            self.pc = np.zeros(self.dim)
            self.L = None
    
    def _adapt_covariance_variant_05(self):
        """Variant 05 - moderate adaptation with restart detection."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        ccov_moderate = min(self.ccov * 1.2, 0.3)
        self.C = ((1.0 - ccov_moderate) * self.C + 
                  ccov_moderate * rank_one +
                  (1.0 - 1.0 / self.mueff) * ccov_moderate * 2.0 * rank_mu)
        
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_06(self):
        """Coordinate system rotation for non-separable and multi-modal landscapes."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        if not hasattr(self, 'rotation_stagnation'):
            self.rotation_stagnation = 0

        is_improving = self.f_opt < getattr(self, 'prev_f_opt_rot', self.f_opt) - 1e-10
        self.prev_f_opt_rot = self.f_opt

        if is_improving:
            self.rotation_stagnation = 0
        else:
            self.rotation_stagnation += 1

        if self.rotation_stagnation >= 3 and np.linalg.norm(self.pc) > 1e-10:
            pc_norm = np.linalg.norm(self.pc)
            u = self.pc / pc_norm

            v = u / (1.0 + abs(np.dot(u, np.ones(self.dim) / np.sqrt(self.dim)) + 1e-10))
            v = v - (np.dot(v, np.ones(self.dim)) / self.dim) * np.ones(self.dim)
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-10:
                v = v / v_norm
                self.C = self.C - 2.0 * np.outer(v, np.dot(v, self.C))
                self.C = self.C - 2.0 * np.outer(np.dot(self.C, v), v) + 4.0 * np.outer(np.dot(v, self.C), v)

            self.C += 0.1 * np.eye(self.dim)
            self.rotation_stagnation = 0

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * rank_one +
                  (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)

        if self.rotation_stagnation >= 2:
            self.C += 0.05 * np.eye(self.dim)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)

        if eig_min < 1e-10:
            self.C += (1e-9 - eig_min) * np.eye(self.dim)

        cond = eig_max / max(eig_min, 1e-10)
        if cond > 1e7:
            self.C *= 0.5
    
    def _adapt_covariance_variant_07(self):
        """Restart-triggered covariance reset for escaping local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        if not hasattr(self, 'restart_fitness_history'):
            self.restart_fitness_history = []
        self.restart_fitness_history.append(float(self.f_opt))
        max_history = max(20, self.dim * 2)
        if len(self.restart_fitness_history) > max_history:
            self.restart_fitness_history.pop(0)

        should_restart = False
        if len(self.restart_fitness_history) >= max_history:
            fit_range = max(self.restart_fitness_history) - min(self.restart_fitness_history)
            fit_scale = max(abs(self.f_opt), 1.0, np.median(self.restart_fitness_history))
            if fit_range < 1e-6 * fit_scale:
                should_restart = True

        if self.stagnation_counter > self.max_stagnation // 2:
            should_restart = True

        if should_restart:
            self.C = np.eye(self.dim)
            self.pc = np.zeros(self.dim)

            bound_range = max(self.ub[0] - self.lb[0], 1.0)
            self.sigma = min(self.sigma * 3.0, 0.1 * bound_range)
            self.sigma = max(self.sigma, 1e-8)

            self.restart_fitness_history = []
            should_restart = False

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_08(self):
        """Eigenvalue-balanced restart escape for severe multimodality."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

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

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        cond = eig_max / max(eig_min, 1e-15)
        eig_spread = eig_min / (eig_max + 1e-15)

        is_stagnant = hasattr(self, 'stagnation_counter') and self.stagnation_counter > self.dim * 2
        is_ill_conditioned = cond > 1e6 or eig_spread < 1e-6

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)
        is_converged_trap = rel_var < 1e-3 and cond > 1e4

        needs_escape = is_stagnant or is_ill_conditioned or is_converged_trap

        if needs_escape:
            target_cond = min(cond, 1e4)
            eigvals_balanced = np.exp(np.linspace(np.log(eig_min), np.log(eig_min * target_cond), self.dim))
            eigvals_balanced = np.clip(eigvals_balanced, 1e-10, None)

            Q = np.linalg.eig(self.C)[1]
            self.C = Q @ np.diag(eigvals_balanced) @ Q.T
            self.C = 0.5 * (self.C + self.C.T)

            self.C += 0.2 * np.mean(eigvals) * np.eye(self.dim)

            ccov_escape = min(self.ccov * 5.0, 0.4)
            cc_escape = min(self.cc * 3.0, 0.25)
        else:
            ccov_escape = self.ccov
            cc_escape = self.cc

        if needs_escape:
            self.pc = (1.0 - cc_escape) * self.pc + np.sqrt(cc_escape * (2.0 - cc_escape)) * y_mean
            rank_one = np.outer(self.pc, self.pc)

        self.C = ((1.0 - ccov_escape) * self.C + 
                  ccov_escape * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_escape * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)

        eigvals_final = np.linalg.eigvalsh(self.C)
        cond_final = np.max(eigvals_final) / np.min(eigvals_final)
        if cond_final > 1e8 or np.any(np.isnan(self.C)):
            self.C = np.eye(self.dim) * np.mean(eigvals_final)
            self.pc = np.zeros(self.dim)
            self.L = None
    
    def _adapt_covariance_variant_09(self):
        """Condition-aware covariance adaptation with stagnation-triggered exploration bursts."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        if not hasattr(self, 'improvement_history'):
            self.improvement_history = []
        current_improvement = max(0.0, float(self.f_opt_prev) - float(self.f_opt))
        self.improvement_history.append(current_improvement)
        if len(self.improvement_history) > 5:
            self.improvement_history.pop(0)
        avg_improvement = np.mean(self.improvement_history) if self.improvement_history else 1e-15

        is_stagnant = (avg_improvement < 1e-8 and len(self.improvement_history) >= 5)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.maximum(np.min(eigvals), 1e-15)
        eig_max = np.max(eigvals)
        cond = eig_max / eig_min

        cond_factor = np.clip(np.log10(cond + 1.0) / 10.0, 0.01, 1.0)

        if is_stagnant:
            self.improvement_history = []

            burst_strength = np.clip(0.5 * np.log10(cond + 1.0), 0.1, 2.0)
            self.C = (1.0 - burst_strength) * self.C + burst_strength * np.eye(self.dim)

            self.pc = np.zeros(self.dim)

        cc_adapt = np.clip(self.cc * cond_factor, 0.001, 0.3)
        ccov_adapt = np.clip(self.ccov * cond_factor, 1e-10, 0.5)

        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * rank_one +
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

        if cond > 1e6:
            self.C += 0.01 * np.eye(self.dim)

        self.C = self._ensure_positive_definite(self.C)

        eigvals_check = np.linalg.eigvalsh(self.C)
        cond_check = np.max(eigvals_check) / np.maximum(np.min(eigvals_check), 1e-15)
        if cond_check > 1e7:
            self.C *= 0.5
    
    def _adapt_covariance_variant_10(self):
        """Explosive eigendirection perturbation on stagnation for multimodal escape."""
        stagnation_threshold = max(30, self.dim * 2)
        is_stagnant = self.stagnation_counter > stagnation_threshold

        if is_stagnant:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)

            k = min(self.dim, max(3, self.dim // 4))
            for i in range(k):
                strength = 2.0 * (k - i) / k
                self.C += strength * np.outer(eigvecs[:, i], eigvecs[:, i])

            self.C += 0.5 * np.eye(self.dim)

            self.C = self._ensure_positive_definite(self.C)
            self.pc *= 0.1
            self.stagnation_counter = 0
        else:
            y_mean = (self.mean - self.old_mean) / self.sigma

            improvement = max(0.0, float(self.f_opt_prev) - float(self.f_opt))
            improv_scale = np.clip(1.0 + 10.0 * np.log1p(improvement * 1e6), 0.1, 5.0)

            cc_adapt = np.clip(self.cc * improv_scale, 0.001, 0.3)
            self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

            rank_one = np.outer(self.pc, self.pc)

            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                diff = (self.population[i] - self.old_mean) / self.sigma
                rank_mu += self.weights[i] * np.outer(diff, diff)

            ccov_adapt = np.clip(self.ccov * improv_scale, 1e-10, 0.5)
            self.C = ((1.0 - ccov_adapt) * self.C + 
                      ccov_adapt * rank_one +
                      (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

            self.C = self._ensure_positive_definite(self.C)

            eigvals_check = np.linalg.eigvalsh(self.C)
            cond = np.max(eigvals_check) / np.min(eigvals_check)
            if cond > 1e7:
                self.C *= 0.5
    
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