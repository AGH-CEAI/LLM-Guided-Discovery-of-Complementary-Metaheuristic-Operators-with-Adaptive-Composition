```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 8 covariance adaptation strategies (original + 7 best-performing variants)
    - Thompson Sampling for operator selection with Beta distributions
    - Sliding window credit assignment for reward tracking
    - Automatic restart on stagnation with elite preservation
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
        
        # Adaptive operator selection parameters - 8 operators from benchmark
        self.num_operators = 8
        self.operator_names = [
            'original', 'variant_01', 'variant_02', 'variant_05',
            'variant_06', 'variant_07', 'variant_09', 'variant_10'
        ]
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_operators)
        self.beta = np.ones(self.num_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 15
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators)
        self.selection_counts = np.zeros(self.num_operators)
        
        # Per-operator performance tracking
        self.operator_best_fitness = {i: float('inf') for i in range(self.num_operators)}
        self.operator_generations = {i: 0 for i in range(self.num_operators)}
        
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
        if hasattr(self, 'exploration_temp'):
            self.exploration_temp = 1.0
        if hasattr(self, 'archive_positions'):
            self.archive_positions = []
            self.archive_fitness = []
            self.archive_generations = []
    
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

        for i in range(min(self.NP, len(self.trial_fitness), len(self.fitness))):
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
        self.operator_generations[self.current_operator] += 1
        return self.current_operator
    
    def _update_operator_rewards(self):
        """Update operator rewards using sliding window credit assignment."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        # Compute fitness variance component
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)
        
        # Compute condition number component
        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = max(np.min(eigvals), 1e-15)
        eig_max = np.max(eigvals)
        cond = eig_max / eig_min
        cond_factor = np.clip(1.0 / np.log10(cond + 10.0), 0.1, 1.0)
        
        # Combined reward with multiple signals
        reward = float(np.log1p(improvement * 1e10) / 10.0)
        reward += 0.1 * float(diversity)
        reward += 0.05 * float(rel_var)
        reward += 0.02 * float(cond_factor)
        
        reward = float(np.clip(reward, -10.0, 10.0))
        
        op = self.current_operator
        self.operator_rewards[op].append(float(reward))
        
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)
        
        # Update best fitness tracking for this operator
        if self.f_opt < self.operator_best_fitness[op]:
            self.operator_best_fitness[op] = float(self.f_opt)
        
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
        
        # Apply small reward bonus for operators that haven't been tried much
        exploration_bonus = max(0.0, 1.0 - self.operator_counts[op] / max(self.generation + 1, 1))
        self.alpha[op] += 0.1 * exploration_bonus
    
    def _adapt_covariance(self):
        """Dispatch to the selected covariance adaptation strategy."""
        if self.current_operator == 0:
            self._adapt_covariance_original()
        elif self.current_operator == 1:
            self._adapt_covariance_variant_01()
        elif self.current_operator == 2:
            self._adapt_covariance_variant_02()
        elif self.current_operator == 3:
            self._adapt_covariance_variant_05()
        elif self.current_operator == 4:
            self._adapt_covariance_variant_06()
        elif self.current_operator == 5:
            self._adapt_covariance_variant_07()
        elif self.current_operator == 6:
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
    
    def _adapt_covariance_variant_02(self):
        """Multi-trigger adaptive eigenspace rescaling with minor-axis emphasis."""
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-12)
        cond = np.max(eigvals) / np.min(eigvals)

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = pop_variance / (expected_var + 1e-10)

        is_stagnant = self.stagnation_counter > max(20, self.dim)
        is_ill_cond = cond > 1e5
        is_collapsed = rel_var < 1e-3
        is_low_diversity = diversity < 1e-3
        needs_rescue = is_stagnant or is_ill_cond or is_collapsed or is_low_diversity

        if needs_rescue:
            log_cond = np.log10(cond + 1.0)
            rescue_strength = np.clip(0.5 + 0.5 * log_cond, 0.5, 5.0)

            sorted_indices = np.argsort(eigvals)
            for idx, i in enumerate(sorted_indices):
                strength = rescue_strength * (idx + 1) / self.dim
                self.C += strength * np.outer(eigvecs[:, i], eigvecs[:, i])

            self.C += 0.3 * np.eye(self.dim)

            diag_C = np.diag(self.C)
            diag_sqrt = np.sqrt(np.maximum(diag_C, 1e-10))
            corr = self.C / np.outer(diag_sqrt, diag_sqrt)
            corr = np.clip(corr, -0.95, 0.95)
            self.C = np.diag(diag_C) + 0.2 * (corr - np.diag(np.diag(corr)))

            self.C = self._ensure_positive_definite(self.C)
            self.pc *= 0.2
            self.stagnation_counter = 0
        else:
            y_mean = (self.mean - self.old_mean) / self.sigma

            improvement = max(0.0, self.f_opt_prev - self.f_opt)
            improv_scale = np.clip(1.0 + 5.0 * np.log1p(improvement * 1e6), 0.2, 3.0)

            cond_factor = np.clip(1.0 / np.log10(cond + 10.0), 0.1, 1.0)

            cc_adapt = np.clip(self.cc * improv_scale * cond_factor, 0.001, 0.3)
            self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

            rank_one = np.outer(self.pc, self.pc)

            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                diff = (self.population[i] - self.old_mean) / self.sigma
                rank_mu += self.weights[i] * np.outer(diff, diff)

            ccov_adapt = np.clip(self.ccov * improv_scale * cond_factor, 1e-10, 0.5)
            self.C = ((1.0 - ccov_adapt) * self.C +
                      ccov_adapt * rank_one +
                      (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

            self.C = self._ensure_positive_definite(self.C)

            eigvals_check = np.linalg.eigvalsh(self.C)
            cond_check = np.max(eigvals_check) / np.min(eigvals_check)
            if cond_check > 1e7:
                self.C *= 0.5
    
    def _adapt_covariance_variant_05(self):
        """Forced perturbation with mean relocation for escaping deep local optima."""
        stagnation_threshold = max(30, self.dim * 2)
        is_stagnant = self.stagnation_counter > stagnation_threshold

        if is_stagnant:
            fit_min = np.min(self.fitness)
            fit_max = np.max(self.fitness)
            fit_range = max(fit_max - fit_min, 1e-10)
            weights_reloc = np.exp(-2.0 * (self.fitness - fit_min) / fit_range)
            weights_reloc /= np.sum(weights_reloc)

            candidate_idx = np.random.choice(self.NP, p=weights_reloc)
            candidate = self.population[candidate_idx]

            pop_spread = np.mean(np.std(self.population, axis=0))
            perturb_scale = max(pop_spread * 3.0, (self.ub[0] - self.lb[0]) * 0.1)

            random_dir = np.random.randn(self.dim)
            random_dir /= (np.linalg.norm(random_dir) + 1e-10)
            self.mean = candidate + perturb_scale * random_dir

            self.mean = self._clip_to_bounds(self.mean)

            self.sigma = min(self.sigma * 5.0, (self.ub[0] - self.lb[0]) * 0.2)

            eigvals = np.linalg.eigvalsh(self.C)
            avg_eig = np.mean(eigvals)
            self.C = np.eye(self.dim) * max(avg_eig, 1e-4)

            self.pc *= 0.1
            self.stagnation_counter = 0
        else:
            y_mean = (self.mean - self.old_mean) / self.sigma
            self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

            rank_one = np.outer(self.pc, self.pc)

            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                diff = (self.population[i] - self.old_mean) / self.sigma
                rank_mu += self.weights[i] * np.outer(diff, diff)

            self.C = ((1.0 - self.ccov) * self.C +
                      self.ccov * rank_one +
                      (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)

            self.C = self._ensure_positive_definite(self.C)

            eigvals_check = np.linalg.eigvalsh(self.C)
            cond = np.max(eigvals_check) / np.min(eigvals_check)
            if cond > 1e7:
                self.C *= 0.5
    
    def _adapt_covariance_variant_06(self):
        """Fitness variance + condition-triggered multi-direction exploration."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        cond = np.max(eigvals) / np.min(eigvals)

        is_converging = (rel_var < 1e-3) and (cond > 1e4)

        if is_converging:
            k_perturb = min(self.dim, max(5, self.dim // 2))

            for i in range(k_perturb):
                strength = 3.0 * (i + 1) / k_perturb
                self.C += strength * np.outer(eigvecs[:, i], eigvecs[:, i])

            self.C += 1.0 * np.eye(self.dim)

            self.pc = np.zeros(self.dim)

            self.C = self._ensure_positive_definite(self.C)
        else:
            ccov_scale = 1.0 + 20.0 * min(rel_var, 0.1)
            ccov_scale = np.clip(ccov_scale, 0.1, 10.0)

            cc_adapt = np.clip(self.cc * ccov_scale, 0.001, 0.5)
            self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

            rank_one = np.outer(self.pc, self.pc)

            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                diff = (self.population[i] - self.old_mean) / self.sigma
                rank_mu += self.weights[i] * np.outer(diff, diff)

            ccov_adapt = np.clip(self.ccov * ccov_scale, 1e-10, 0.5)
            self.C = ((1.0 - ccov_adapt) * self.C + 
                      ccov_adapt * rank_one +
                      (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

            self.C = self._ensure_positive_definite(self.C)

            eigvals_check = np.linalg.eigvalsh(self.C)
            cond_check = np.max(eigvals_check) / np.min(eigvals_check)
            if cond_check > 1e7:
                self.C *= 0.5
    
    def _adapt_covariance_variant_07(self):
        """Continuous landscape-adaptive dual-path covariance for multimodal escape."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        C_exploit = ((1.0 - self.ccov) * self.C + 
                     self.ccov * rank_one + 
                     (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu)

        pop_range = self.ub[0] - self.lb[0]
        explore_radius = 0.05 * pop_range * self.sigma / np.sqrt(self.dim)
        C_explore = np.eye(self.dim) * (explore_radius ** 2 + 1e-10)

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)

        is_trapped = (self.stagnation_counter > 10 and rel_var > 1e-4)

        multimodality_factor = np.clip(rel_var * 10.0, 0.0, 1.0)
        illcond_factor = np.clip(1.0 - eig_spread, 0.0, 1.0)
        trap_factor = 1.0 if is_trapped else 0.0

        mix = 0.05 + 0.15 * multimodality_factor + 0.30 * illcond_factor + 0.30 * trap_factor
        mix = np.clip(mix, 0.01, 0.7)

        self.C = (1.0 - mix) * C_exploit + mix * C_explore

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_09(self):
        """Adaptive diversity injection via convergence pressure and eigendirection restart."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = max(np.min(eigvals), 1e-15)
        eig_max = np.max(eigvals)
        eig_spread = eig_min / (eig_max + 1e-15)
        cond = eig_max / eig_min

        pressure = min(1.0, rel_var * 10.0 + eig_spread * 5.0)
        pressure = max(0.0, pressure)

        adaptive_threshold = max(15, self.dim * 1.5) * (1.0 + 0.1 * np.log1p(cond))
        is_stagnant = self.stagnation_counter > adaptive_threshold
        is_converged = pressure < 0.05 or cond > 1e6

        if is_stagnant or is_converged:
            eigvecs = np.linalg.eigh(self.C)[1]

            k = min(self.dim, max(2, self.dim // 3))
            for i in range(k):
                strength = 3.0 * (k - i) / k
                self.C += strength * np.outer(eigvecs[:, i], eigvecs[:, i])

            self.C += 0.8 * np.eye(self.dim)

            self.C = self._ensure_positive_definite(self.C)
            self.pc *= 0.05
            self.stagnation_counter = 0
        else:
            improv_scale = 1.0 + 5.0 * (1.0 - pressure)
            improv_scale = np.clip(improv_scale, 0.2, 10.0)

            cc_adapt = np.clip(self.cc * improv_scale, 0.001, 0.5)
            self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

            rank_one = np.outer(self.pc, self.pc)

            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(self.mu):
                diff = (self.population[i] - self.old_mean) / self.sigma
                rank_mu += self.weights[i] * np.outer(diff, diff)

            ccov_adapt = np.clip(self.ccov * improv_scale, 1e-10, 0.6)
            self.C = ((1.0 - ccov_adapt) * self.C +
                      ccov_adapt * rank_one +
                      (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

            self.C = self._ensure_positive_definite(self.C)

            eigvals_check = np.linalg.eigvalsh(self.C)
            cond_check = np.max(eigvals_check) / max(np.min(eigvals_check), 1e-15)
            if cond_check > 1e6:
                self.C *= 0.5
                self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_10(self):
        """Anticipatory diversity-driven restart with exponential eigendirection perturbation."""
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.maximum(np.min(eigvals), 1e-10)
        eig_max = np.max(eigvals)
        cond = eig_max / eig_min
        eig_spread = eig_min / (eig_max + 1e-10)

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        restart_prob = 0.0
        restart_prob += 0.3 * max(0.0, 1.0 - rel_var / 1e-4) if rel_var < 1e-4 else 0.0
        restart_prob += 0.4 * max(0.0, 1.0 - np.log10(cond + 1) / 7.0) if cond > 1e3 else 0.0
        restart_prob += 0.3 * max(0.0, 1.0 - diversity / 1e-3) if diversity < 1e-3 else 0.0
        restart_prob *= np.clip(1.0 + np.log1p(abs(self.f_opt)), 0.1, 10.0)

        if np.random.random() < restart_prob:
            eigvecs = np.linalg.eigh(self.C)[1]

            k = min(self.dim, max(2, self.dim // 3))
            for i in range(k):
                strength = 3.0 * (i + 1) / k
                self.C += strength * np.outer(eigvecs[:, i], eigvecs[:, i])

            self.C += np.eye(self.dim)

            self.C = self._ensure_positive_definite(self.C)
            self.pc *= 0.05
            self.stagnation_counter = 0

        y_mean = (self.mean - self.old_mean) / self.sigma

        diversity_factor = np.clip(1.0 / (diversity + 1e-4), 0.1, 10.0)
        ccov_scale = np.clip(0.5 + 0.5 * diversity_factor, 0.3, 3.0)
        cc_scale = np.clip(0.5 + 0.5 * diversity_factor, 0.5, 2.0)

        cc_adapt = np.clip(self.cc * cc_scale, 0.01, 0.3)
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        ccov_adapt = np.clip(self.ccov * ccov_scale, 1e-10, 0.5)
        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * rank_one +
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)

        eigvals_check = np.linalg.eigvalsh(self.C)
        cond_check = np.max(eigvals_check) / np.maximum(np.min(eigvals_check), 1e-10)
        if cond_check > 1e7:
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
        
        # Check for NaN in covariance matrix
        has_nan = np.any(np.isnan(self.C)) if hasattr(self, 'C') else False
        has_inf = np.any(np.isinf(self.C)) if hasattr(self, 'C') else False
        
        if (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            has_nan or has_inf):
            
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]
            
            self._initialize_population()
            
            # Preserve elite
            self.population[0] = elite
            self.fitness[0] = elite_fit
            self.f_opt = float(np.asarray(elite_fit).flatten()[0])
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0
            
            # Ensure covariance is valid
            if np.any(np.isnan(self.C)) or np.any(np.isinf(self.C)):
                self.C = np.eye(self.dim) * 1.0
```