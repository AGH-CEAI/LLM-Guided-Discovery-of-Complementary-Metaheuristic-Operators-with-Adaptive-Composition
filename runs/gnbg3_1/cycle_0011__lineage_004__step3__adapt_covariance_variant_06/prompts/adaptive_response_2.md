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
        self.operator_names = ['original', 'variant_01', 'variant_02', 'variant_03', 'variant_07']
        
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
        x = np.asarray(x, dtype=np.float64)
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
        self.fitness = np.asarray(self.fitness, dtype=np.float64).flatten()
        
        best_idx = np.argmin(self.fitness)
        f_opt_val = float(np.asarray(self.fitness[best_idx]).flatten()[0])
        if np.isfinite(f_opt_val):
            self.f_opt = f_opt_val
        else:
            self.f_opt = 1e100
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        
        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0
        
        self._reset_operator_state()
    
    def _reset_operator_state(self):
        """Reset all operator-specific state variables."""
        self.L = None
        self.best_history = []
        self.archive_positions = []
        self.archive_fitness = []
        self.archive_generations = []
        self.prev_f_opt = self.f_opt
        self.exploration_temp = 1.0
        self.improvement_ema = 1.0
    
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
        self.trial_fitness = np.asarray(self.trial_fitness, dtype=np.float64).flatten()
        self.trial_fitness = np.where(np.isfinite(self.trial_fitness), self.trial_fitness, 1e100)
    
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
        """Active CMA-ES with negative weights for improved exploration."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        rank_one = np.outer(self.pc, self.pc)

        rank_mu_pos = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu_pos += self.weights[i] * np.outer(diff, diff)

        neg_mu = max(1, self.mu // 2)
        neg_weights = np.zeros(neg_mu)
        sum_neg = 0.0
        for i in range(neg_mu):
            neg_weights[i] = -(np.log((self.NP + i) / 2.0 + 0.5) - np.log(self.NP / 2.0 + 1.0))
            sum_neg += neg_weights[i]

        if sum_neg > 1e-10:
            neg_weights /= sum_neg

        rank_mu_neg = np.zeros((self.dim, self.dim))
        for i in range(neg_mu):
            idx = self.NP - 1 - i
            if idx < len(self.population):
                diff = (self.population[idx] - self.old_mean) / self.sigma
                rank_mu_neg += neg_weights[i] * np.outer(diff, diff)

        eigvals = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals) / (np.min(eigvals) + 1e-10)

        if cond < 1e3:
            ccov_scale = 1.0
        elif cond < 1e5:
            ccov_scale = 0.5
        else:
            ccov_scale = 0.25

        ccov_1 = ccov_scale / (self.dim + 2.0)
        ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        ccov_mu_neg = 0.5 * ccov_scale / (self.dim + 2.0) / (self.dim + 4.0)

        self.C = ((1.0 - ccov_1 - ccov_mu - ccov_mu_neg) * self.C +
                  ccov_1 * rank_one +
                  ccov_mu * rank_mu_pos +
                  ccov_mu_neg * rank_mu_neg)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_02(self):
        """Multi-direction ensemble covariance: blend diverse search directions by predicted quality."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        k_directions = min(self.mu, max(3, self.dim // 2))
        candidate_dirs = []
        candidate_scores = []

        for i in range(k_directions):
            diff = (self.population[i] - self.old_mean) / self.sigma
            dir_norm = np.linalg.norm(diff)
            if dir_norm > 1e-10:
                pop_dist = np.linalg.norm(self.population[i] - self.mean)
                fitness_rank = (self.mu - i) / max(self.mu, 1)
                score = fitness_rank * np.log1p(pop_dist / max(dir_norm, 1e-10))
                candidate_dirs.append(diff)
                candidate_scores.append(score)

        if len(candidate_dirs) >= 2:
            max_score = max(abs(s) for s in candidate_scores)
            if max_score > 1e-10:
                weights_dir = [s / max_score for s in candidate_scores]
            else:
                weights_dir = [1.0 / len(candidate_dirs)] * len(candidate_dirs)

            ensemble_pc = np.zeros(self.dim)
            for d, w in zip(candidate_dirs, weights_dir):
                ensemble_pc += w * d
            ensemble_pc /= sum(weights_dir)

            ensemble_rank_one = np.zeros((self.dim, self.dim))
            for d, w in zip(candidate_dirs, weights_dir):
                d_normalized = d / max(np.linalg.norm(d), 1e-10)
                ensemble_rank_one += w * np.outer(d_normalized, d_normalized)
            ensemble_rank_one /= sum(weights_dir)
        else:
            self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
            ensemble_rank_one = np.outer(self.pc, self.pc)

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
        eig_spread = np.min(eigvals) / (np.max(eigvals) + 1e-10)
        cond = np.max(eigvals) / max(np.min(eigvals), 1e-10)

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        is_ill_cond = cond > 1e5
        is_low_var = rel_var < 1e-3
        is_narrow = eig_spread < 1e-5

        if is_ill_cond or is_low_var or is_narrow:
            ccov_eff = min(self.ccov * 5.0, 0.5)
            cc_eff = min(self.cc * 2.0, 0.3)
            rank_mu += 0.2 * np.eye(self.dim)
            ensemble_rank_one += 0.1 * np.eye(self.dim)
        else:
            ccov_eff = self.ccov
            cc_eff = self.cc

        self.C = ((1.0 - ccov_eff) * self.C + 
                  ccov_eff * ensemble_rank_one +
                  (1.0 - 1.0 / self.mueff) * ccov_eff * 2.0 * rank_mu)

        if cond > 1e8:
            self.C = 0.5 * self.C + 0.5 * np.eye(self.dim) * np.mean(eigvals)
            self.pc *= 0.1

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_03(self):
        """Proactive stagnation-guided escaping with directional perturbations."""
        stagnation_threshold = max(40, self.dim * 3)
        is_stagnant = self.stagnation_counter > stagnation_threshold

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        cond = eig_max / (max(eig_min, 1e-10))

        if not hasattr(self, 'best_history'):
            self.best_history = []

        min_dist_threshold = 1e-3 * (self.ub[0] - self.lb[0])
        is_distinct = True
        for hist_pos in self.best_history:
            if np.linalg.norm(self.x_opt - hist_pos) < min_dist_threshold:
                is_distinct = False
                break

        if is_distinct and len(self.best_history) < 15:
            self.best_history.append(self.x_opt.copy())
        elif is_distinct and len(self.best_history) >= 15:
            worst_idx = np.argmax([np.linalg.norm(self.x_opt - p) for p in self.best_history])
            self.best_history[worst_idx] = self.x_opt.copy()

        if is_stagnant or cond > 1e6:
            escape_dir = np.zeros(self.dim)
            if len(self.best_history) >= 3:
                centroid = np.mean(self.best_history, axis=0)
                toward_centroid = centroid - self.mean
                norm_toward = np.linalg.norm(toward_centroid)
                if norm_toward > 1e-10:
                    toward_centroid /= norm_toward
                    escape_dir = toward_centroid
                else:
                    escape_dir = self.mean - centroid
                    norm_dir = np.linalg.norm(escape_dir)
                    if norm_dir > 1e-10:
                        escape_dir /= norm_dir
            else:
                escape_dir = np.random.randn(self.dim)
                norm_dir = np.linalg.norm(escape_dir)
                if norm_dir > 1e-10:
                    escape_dir /= norm_dir

            escape_strength = 0.5 * (1.0 + self.stagnation_counter / max(stagnation_threshold, 1))
            escape_perturb = escape_strength * np.outer(escape_dir, escape_dir)
            self.C += escape_perturb
            self.C += 0.3 * np.eye(self.dim)
            self.C = self._ensure_positive_definite(self.C)
            self.pc *= 0.1
            self.stagnation_counter = 0

        y_mean = (self.mean - self.old_mean) / self.sigma

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        if is_stagnant:
            cc_scale = 1.5
            ccov_scale = 2.0
        else:
            cc_scale = 1.0
            ccov_scale = min(1.0 + 3.0 * rel_var, 3.0)

        cc_adapt = np.clip(self.cc * cc_scale, 0.001, 0.5)
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        ccov_adapt = np.clip(self.ccov * ccov_scale, 1e-10, 0.6)
        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu))

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_07(self):
        """Fitness gradient-directed covariance adaptation with stagnation escape."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        fit_diff = self.fitness - np.mean(self.fitness)
        std_fit = np.std(fit_diff)
        if std_fit > 1e-10:
            fit_diff = fit_diff / std_fit
        else:
            fit_diff = np.zeros_like(fit_diff)

        grad_matrix = np.zeros((self.dim, self.dim))
        for i in range(min(self.mu, len(fit_diff))):
            diff = (self.population[i] - self.old_mean) / self.sigma
            grad_matrix += fit_diff[i] * np.outer(diff, diff)

        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)

        grad_proj = np.zeros(self.dim)
        for j in range(self.dim):
            proj = 0.0
            for i in range(min(self.mu, len(fit_diff))):
                diff = (self.population[i] - self.old_mean) / self.sigma
                proj += fit_diff[i] * np.dot(eigvecs[:, j], diff)
            grad_proj[j] = proj

        dir_weights = np.zeros(self.dim)
        for j in range(self.dim):
            if eigvals[j] > 1e-10:
                dir_weights[j] = np.tanh(grad_proj[j] / (np.sqrt(eigvals[j]) * self.mu + 1e-10))

        eig_scale = np.ones(self.dim)
        for j in range(self.dim):
            if dir_weights[j] < -0.2:
                eig_scale[j] = 0.5 + 0.5 * dir_weights[j]
            elif dir_weights[j] > 0.2:
                eig_scale[j] = 1.0 + 0.3 * dir_weights[j]

        grad_strength = np.linalg.norm(grad_proj) / np.sqrt(self.dim)
        cc_adapt = self.cc * np.clip(1.0 + 0.5 * np.tanh(grad_strength - 1.0), 0.01, 0.3)

        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean
        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        ccov_adapt = self.ccov * np.clip(1.0 + 0.5 * np.tanh(grad_strength - 0.5), 0.1, 5.0)
        ccov_adapt = np.clip(ccov_adapt, 1e-10, 0.5)

        self.C = ((1.0 - ccov_adapt) * self.C + 
                  ccov_adapt * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

        for j in range(self.dim):
            self.C += (eig_scale[j] - 1.0) * eigvals[j] * np.outer(eigvecs[:, j], eigvecs[:, j])

        stagnation_threshold = max(20, self.dim * 2)
        is_stagnant = self.stagnation_counter > stagnation_threshold

        if is_stagnant:
            self.stagnation_counter = 0

            for j in range(self.dim):
                if grad_proj[j] > 0:
                    strength = 1.0 + 0.5 * np.tanh(grad_proj[j])
                else:
                    strength = 1.0 - 0.3 * np.tanh(-grad_proj[j])
                self.C += strength * eigvals[j] * np.outer(eigvecs[:, j], eigvecs[:, j])

            self.C += 0.1 * np.eye(self.dim)

            self.pc *= 0.1
            self.L = None

        self.C = self._ensure_positive_definite(self.C)

        eigvals_check = np.linalg.eigvalsh(self.C)
        cond = np.max(eigvals_check) / np.min(eigvals_check)
        if cond > 1e8:
            self.C = np.eye(self.dim) * np.mean(eigvals_check)
            self.pc = np.zeros(self.dim)
    
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
            f_opt_val = float(np.asarray(elite_fit).flatten()[0])
            if np.isfinite(f_opt_val):
                self.f_opt = f_opt_val
            else:
                self.f_opt = 1e100
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0
```