```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best _adapt_step_size
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 8 covariance adaptation strategies (all winning variants + original)
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
        
        # Adaptive operator selection parameters for _adapt_step_size
        self.num_step_operators = 8
        self.step_operator_names = [
            'original', 'variant_01', 'variant_02', 'variant_04',
            'variant_05', 'variant_07', 'variant_09', 'variant_10'
        ]
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_step_operators)
        self.beta = np.ones(self.num_step_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 12
        self.operator_rewards = {i: [] for i in range(self.num_step_operators)}
        self.operator_counts = np.zeros(self.num_step_operators)
        self.selection_counts = np.zeros(self.num_step_operators)
        
        # Runtime state
        self.generation = 0
        self.current_step_operator = 0
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
            
            # Select and apply step-size adaptation operator
            self._select_step_operator_thompson()
            self._adapt_step_size()
            
            # Update operator rewards based on improvement
            self._update_step_operator_rewards()
            
            self._adapt_covariance()
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
        self.current_step_operator = 0
        
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
        # Reset step operator state
        if hasattr(self, 'improvement_ema'):
            self.improvement_ema = 1.0
        if hasattr(self, 'p_sigma'):
            self.p_sigma = None
        if hasattr(self, 'ps_path'):
            self.ps_path = None
        if hasattr(self, 'sigma_momentum'):
            self.sigma_momentum = 0.0
        if hasattr(self, 'sigma_history'):
            self.sigma_history = []
        if hasattr(self, 'f_mean_prev'):
            self.f_mean_prev = float(np.mean(self.fitness))
    
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
    
    def _select_step_operator_thompson(self):
        """Select step-size adaptation operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_step_operator = int(np.argmax(samples))
        self.operator_counts[self.current_step_operator] += 1
        return self.current_step_operator
    
    def _update_step_operator_rewards(self):
        """Update operator rewards using sliding window credit assignment."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = float(np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0))
        
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.15 * diversity)
        reward = float(np.clip(reward, -10.0, 10.0))
        
        op = self.current_step_operator
        self.operator_rewards[op].append(reward)
        
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)
        
        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            var_reward = float(np.var(self.operator_rewards[op]))
            
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.beta[op] = float(max(1.0, (1.0 - mean_reward) * ((n - 1) * (1.0 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = float(max(1.0, 1.0 + sum_reward))
            else:
                self.beta[op] = float(max(1.0, 1.0 - sum_reward))
    
    def _adapt_step_size(self):
        """Dispatch to the selected step-size adaptation strategy."""
        if self.current_step_operator == 0:
            self._adapt_step_size_original()
        elif self.current_step_operator == 1:
            self._adapt_step_size_variant_01()
        elif self.current_step_operator == 2:
            self._adapt_step_size_variant_02()
        elif self.current_step_operator == 3:
            self._adapt_step_size_variant_04()
        elif self.current_step_operator == 4:
            self._adapt_step_size_variant_05()
        elif self.current_step_operator == 5:
            self._adapt_step_size_variant_07()
        elif self.current_step_operator == 6:
            self._adapt_step_size_variant_09()
        else:
            self._adapt_step_size_variant_10()
    
    def _adapt_step_size_original(self):
        """Original step-size adaptation using simple success rate."""
        recent_improved = 0
        recent_total = 0

        for i in range(self.NP):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                if self.trial_fitness[i] < self.fitness[i]:
                    recent_improved += 1
                recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = float(recent_improved / recent_total)
        success_rate = float(np.clip(success_rate, 0.0, 1.0))

        target_rate = 0.25
        adaptation = (success_rate - target_rate) / target_rate
        adaptation = float(np.clip(adaptation, -0.5, 0.5))

        damping_adaptive = float(self.damping * (0.5 + 0.5 * np.log1p(self.dim)))
        damping_adaptive = float(np.clip(damping_adaptive, 0.1, 100.0))

        self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
        self.sigma = float(np.clip(self.sigma, 1e-10, 10.0))
    
    def _adapt_step_size_variant_01(self):
        """CMA-ES Cumulative Step-size Adaptation (CSA)."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        if not hasattr(self, 'ps'):
            self.ps = np.zeros(self.dim)
        
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * y_mean

        chi_d = self.dim ** 0.5 * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim ** 2))

        ps_norm = float(np.linalg.norm(self.ps))
        ps_norm_safe = max(ps_norm, 1e-10)
        chi_d_safe = max(chi_d, 1e-10)

        adaptation = (ps_norm_safe / chi_d_safe) - 1.0
        adaptation = float(np.clip(adaptation, -1.0, 1.0))

        self.sigma *= np.exp((self.cs / self.damping) * adaptation)
        self.sigma = float(np.clip(self.sigma, 1e-10, 10.0))
    
    def _adapt_step_size_variant_02(self):
        """Momentum-weighted improvement with diversity correction."""
        recent_improved = 0
        recent_total = 0
        improvement_sum = 0.0

        for i in range(self.NP):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                if self.trial_fitness[i] < self.fitness[i]:
                    recent_improved += 1
                    improvement_sum += float(self.fitness[i] - self.trial_fitness[i])
                recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = float(recent_improved / recent_total)
        success_rate = float(np.clip(success_rate, 0.0, 1.0))

        if not hasattr(self, 'improvement_ema'):
            self.improvement_ema = 1.0

        avg_improvement = improvement_sum / max(recent_improved, 1)
        self.improvement_ema = 0.7 * self.improvement_ema + 0.3 * avg_improvement

        pop_variance = float(np.mean(np.var(self.population, axis=0)))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity_ratio = float(np.clip(pop_variance / (expected_var + 1e-10), 0.01, 1.0))

        target_rate = 0.25 * (1.0 + 0.5 * np.log1p(1.0 / (diversity_ratio + 1e-10)))
        target_rate = float(np.clip(target_rate, 0.05, 0.6))

        rate_ratio = success_rate / max(target_rate, 1e-10)
        if rate_ratio < 0.5:
            adaptation = -0.5 * (1.0 + np.log1p(0.5 / (rate_ratio + 1e-10)))
        elif rate_ratio > 1.5:
            adaptation = 0.5 * np.log1p(rate_ratio)
        else:
            adaptation = (success_rate - target_rate) / target_rate

        adaptation = float(np.clip(adaptation, -0.8, 0.8))

        momentum_factor = float(np.clip(np.log1p(self.improvement_ema * 1e6) / 10.0, 0.2, 2.0))
        damping_adaptive = self.damping * momentum_factor * (0.3 + 0.7 * np.log1p(self.dim) / diversity_ratio)
        damping_adaptive = float(np.clip(damping_adaptive, 0.05, 150.0))

        generation_factor = np.exp(-self.generation / (50 + self.dim * 2))
        cs_adaptive = self.cs * (1.0 + generation_factor)

        self.sigma *= np.exp(adaptation * cs_adaptive / damping_adaptive)
        self.sigma = float(np.clip(self.sigma, 1e-10, 10.0))
    
    def _adapt_step_size_variant_04(self):
        """Momentum-enhanced improvement tracking with diversity-based damping."""
        recent_improved = 0
        recent_total = 0
        total_improvement = 0.0

        for i in range(self.NP):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                if self.trial_fitness[i] < self.fitness[i]:
                    recent_improved += 1
                    diff = float(self.fitness[i] - self.trial_fitness[i])
                    total_improvement += max(diff, 0.0)
                recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = float(recent_improved / recent_total)
        success_rate = float(np.clip(success_rate, 0.0, 1.0))

        avg_improvement = total_improvement / max(recent_improved, 1)
        improvement_magnitude = float(np.log1p(max(avg_improvement, 1e-15)))

        if not hasattr(self, 'improvement_ema'):
            self.improvement_ema = improvement_magnitude
        self.improvement_ema = 0.7 * self.improvement_ema + 0.3 * improvement_magnitude

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
    
    def _adapt_step_size_variant_05(self):
        """Evolution path length control (CMA-ES style)."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        if not hasattr(self, 'p_sigma') or self.p_sigma is None:
            self.p_sigma = np.zeros(self.dim)

        self.p_sigma = (1.0 - self.cs) * self.p_sigma + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean

        path_norm = float(np.linalg.norm(self.p_sigma))
        expected_norm = float(np.sqrt(self.dim) * (1.0 - (1.0 - self.cs) ** (2.0 * max(1, self.generation))))
        expected_norm = max(expected_norm, 0.1)

        control = path_norm / expected_norm - 1.0
        control = float(np.clip(control, -0.5, 0.5))

        stagnation_boost = 1.0 + 0.5 * min(self.stagnation_counter / max(1, self.max_stagnation), 1.0)
        damping_eff = self.damping * stagnation_boost
        damping_eff = float(np.clip(damping_eff, 0.1, 200.0))

        self.sigma *= np.exp(control * self.cs / damping_eff)
        self.sigma = float(np.clip(self.sigma, 1e-10, 10.0))
    
    def _adapt_step_size_variant_07(self):
        """Momentum, fitness gradient, and diversity awareness."""
        f_mean_prev = getattr(self, 'f_mean_prev', float(np.mean(self.fitness)))
        f_mean_curr = float(np.mean(self.fitness))
        f_improvement = f_mean_prev - f_mean_curr

        if not hasattr(self, 'sigma_momentum'):
            self.sigma_momentum = 0.0
            self.sigma_history = []

        self.sigma_history.append(float(self.sigma))
        if len(self.sigma_history) > 10:
            self.sigma_history.pop(0)

        improvement_rate = f_improvement / (abs(f_mean_prev) + 1e-10)
        target_momentum = float(np.clip(improvement_rate * 10.0, -1.0, 1.0))

        momentum_alpha = 0.3
        self.sigma_momentum = (1.0 - momentum_alpha) * self.sigma_momentum + momentum_alpha * target_momentum
        self.sigma_momentum = float(np.clip(self.sigma_momentum, -0.8, 0.8))

        pop_spread = float(np.mean(np.std(self.population, axis=0)))
        expected_spread = (self.ub[0] - self.lb[0]) / 6.0
        diversity_ratio = pop_spread / (expected_spread + 1e-10)
        diversity_ratio = float(np.clip(diversity_ratio, 0.0, 1.0))

        recent_improved = 0
        recent_total = 0
        for i in range(min(self.NP, len(self.trial_fitness), len(self.fitness))):
            if self.trial_fitness[i] < self.fitness[i]:
                recent_improved += 1
            recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = float(recent_improved / recent_total)
        success_rate = float(np.clip(success_rate, 0.0, 1.0))

        min_diversity_thresh = 0.05
        if diversity_ratio < min_diversity_thresh:
            diversity_factor = 0.3
        else:
            diversity_factor = 1.0

        adaptation = self.sigma_momentum + 0.2 * (success_rate - 0.25)

        if diversity_ratio < min_diversity_thresh:
            adaptation = max(adaptation, 0.3)

        adaptation = float(np.clip(adaptation, -0.6, 0.6))

        damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim)) * diversity_factor
        damping_adaptive = float(np.clip(damping_adaptive, 0.05, 100.0))

        self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
        self.sigma = float(np.clip(self.sigma, 1e-12, 15.0))

        if pop_spread > 1e-10 and self.sigma < pop_spread * 1e-4:
            self.sigma = pop_spread * 1e-3

        self.f_mean_prev = f_mean_curr
    
    def _adapt_step_size_variant_09(self):
        """CMA-ES-style cumulative evolution path with momentum."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        if not hasattr(self, 'ps_path') or self.ps_path is None:
            self.ps_path = np.zeros(self.dim)

        self.ps_path = (1.0 - self.cs) * self.ps_path + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean

        path_norm = float(np.linalg.norm(self.ps_path))

        expected_path = float(np.sqrt(self.dim) * (1.0 - (1.0 - self.cs) ** (2.0 * self.generation + 1)))
        expected_path = max(expected_path, 0.1)

        success_ratio = path_norm / expected_path if expected_path > 1e-15 else 1.0
        success_ratio = float(np.clip(success_ratio, 0.01, 10.0))

        adaptation = np.log(success_ratio) / np.log(2.0)
        adaptation = float(np.clip(adaptation, -0.8, 0.8))

        damping_adaptive = self.damping * (0.3 + 0.7 * np.log1p(self.dim) / np.log1p(path_norm + 1e-10))
        damping_adaptive = float(np.clip(damping_adaptive, 0.05, 200.0))

        self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
        self.sigma = float(np.clip(self.sigma, 1e-12, 20.0))
    
    def _adapt_step_size_variant_10(self):
        """Magnitude-aware improvement with stagnation detection."""
        if len(self.trial_fitness) > 0 and len(self.fitness) > 0:
            improvements = np.maximum(0.0, self.fitness - self.trial_fitness)
            recent_improved = int(np.sum(improvements > 0))
            total_improvement = float(np.sum(improvements))
            max_improvement = float(np.max(improvements))
        else:
            recent_improved = 0
            total_improvement = 0.0
            max_improvement = 0.0

        recent_total = max(1, len(self.trial_fitness))
        success_rate = float(recent_improved / recent_total)

        target_rate = 0.2 + 0.1 * np.tanh(self.dim / 20.0)

        expected_improvement = max(abs(self.f_opt), 1.0) * 0.01
        if total_improvement > 0 and expected_improvement > 0:
            improvement_ratio = total_improvement / (expected_improvement * recent_total + 1e-15)
            magnitude_factor = float(np.tanh(np.log1p(improvement_ratio) / 5.0))
        else:
            magnitude_factor = -0.5

        is_stagnant = (max_improvement < abs(self.f_opt) * 1e-6 + 1e-10) or (self.f_opt > 1e3)

        if is_stagnant:
            stagnation_factor = 2.0
            self.sigma = float(np.clip(self.sigma * stagnation_factor, 1e-10, 50.0))
        else:
            rate_adaptation = (success_rate - target_rate) / (target_rate + 1e-15)
            combined_adaptation = 0.7 * rate_adaptation + 0.3 * magnitude_factor
            combined_adaptation = float(np.clip(combined_adaptation, -0.8, 0.8))

            damping_adaptive = self.damping * (0.3 + 0.7 * np.log1p(self.dim) / np.log1p(100))
            damping_adaptive = float(np.clip(damping_adaptive, 0.05, 50.0))

            self.sigma *= np.exp(combined_adaptation * self.cs / damping_adaptive)
            self.sigma = float(np.clip(self.sigma, 1e-10, 50.0))
    
    def _adapt_covariance(self):
        """Dispatch to the selected covariance adaptation strategy."""
        if self.current_step_operator == 0:
            self._adapt_covariance_original()
        elif self.current_step_operator == 1:
            self._adapt_covariance_variant_01()
        elif self.current_step_operator == 2:
            self._adapt_covariance_variant_06()
        elif self.current_step_operator == 3:
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
```