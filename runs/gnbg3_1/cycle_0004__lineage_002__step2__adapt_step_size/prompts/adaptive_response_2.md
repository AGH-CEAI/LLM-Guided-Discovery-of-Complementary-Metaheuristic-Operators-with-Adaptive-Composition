```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best step-size adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 8 step-size adaptation strategies (based on benchmark results)
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
        self.num_operators = 8
        self.operator_names = [
            'original', 'variant_01', 'variant_02', 'variant_04',
            'variant_05', 'variant_07', 'variant_09', 'variant_10'
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
            
            # Select and apply step-size adaptation operator
            self._select_operator_thompson()
            self._adapt_step_size()
            
            # Update operator rewards based on improvement
            self._update_operator_rewards()
            
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
        # Reset step-size adaptation state variables
        if hasattr(self, 'ps_path'):
            self.ps_path = np.zeros(self.dim)
        else:
            self.ps_path = np.zeros(self.dim)
        if hasattr(self, 'p_sigma'):
            self.p_sigma = None
        else:
            self.p_sigma = None
        if hasattr(self, 'sigma_history'):
            self.sigma_history = []
        else:
            self.sigma_history = []
        if hasattr(self, 'sigma_momentum'):
            self.sigma_momentum = 0.0
        else:
            self.sigma_momentum = 0.0
        if hasattr(self, 'f_mean_prev'):
            self.f_mean_prev = float(np.mean(self.fitness))
        else:
            self.f_mean_prev = float(np.mean(self.fitness))
        if hasattr(self, 'improvement_ema'):
            self.improvement_ema = 1.0
        else:
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
        
        # Compute success rate as additional reward signal
        recent_improved = 0
        recent_total = 0
        for i in range(min(self.NP, len(self.trial_fitness), len(self.fitness))):
            if float(self.trial_fitness[i]) < float(self.fitness[i]):
                recent_improved += 1
            recent_total += 1
        success_rate = float(recent_improved / max(recent_total, 1))
        
        # Combined reward with diversity and success rate
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity + 0.2 * success_rate)
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
    
    def _adapt_step_size(self):
        """Dispatch to the selected step-size adaptation strategy."""
        if self.current_operator == 0:
            self._adapt_step_size_original()
        elif self.current_operator == 1:
            self._adapt_step_size_variant_01()
        elif self.current_operator == 2:
            self._adapt_step_size_variant_02()
        elif self.current_operator == 3:
            self._adapt_step_size_variant_04()
        elif self.current_operator == 4:
            self._adapt_step_size_variant_05()
        elif self.current_operator == 5:
            self._adapt_step_size_variant_07()
        elif self.current_operator == 6:
            self._adapt_step_size_variant_09()
        else:
            self._adapt_step_size_variant_10()
    
    def _adapt_step_size_original(self):
        """Original step-size adaptation (baseline)."""
        recent_improved = 0
        recent_total = 0

        for i in range(self.NP):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                if float(self.trial_fitness[i]) < float(self.fitness[i]):
                    recent_improved += 1
                recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = float(np.clip(recent_improved / recent_total, 0.0, 1.0))
        target_rate = 0.25
        adaptation = float(np.clip((success_rate - target_rate) / target_rate, -0.5, 0.5))

        damping_adaptive = float(np.clip(self.damping * (0.5 + 0.5 * np.log1p(self.dim)), 0.1, 100.0))

        self.sigma = float(np.clip(self.sigma * np.exp(adaptation * self.cs / damping_adaptive), 1e-10, 10.0))
    
    def _adapt_step_size_variant_01(self):
        """Adapt step-size using CMA-ES Cumulative Step-size Adaptation (CSA)."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        if not hasattr(self, 'ps_path'):
            self.ps_path = np.zeros(self.dim)
        self.ps_path = (1.0 - self.cs) * self.ps_path + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * y_mean

        chi_d = self.dim ** 0.5 * (1.0 - 1.0 / (4.0 * self.dim) + 1.0 / (21.0 * self.dim ** 2))
        ps_norm = float(np.linalg.norm(self.ps_path))
        ps_norm_safe = max(ps_norm, 1e-10)
        chi_d_safe = max(chi_d, 1e-10)

        adaptation = float(np.clip((ps_norm_safe / chi_d_safe) - 1.0, -1.0, 1.0))
        self.sigma = float(np.clip(self.sigma * np.exp((self.cs / self.damping) * adaptation), 1e-10, 10.0))
    
    def _adapt_step_size_variant_02(self):
        """Adapt step-size using momentum-weighted improvement with diversity correction."""
        recent_improved = 0
        recent_total = 0
        improvement_sum = 0.0

        for i in range(self.NP):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                if float(self.trial_fitness[i]) < float(self.fitness[i]):
                    recent_improved += 1
                    improvement_sum += float(self.fitness[i] - self.trial_fitness[i])
                recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = float(np.clip(recent_improved / recent_total, 0.0, 1.0))

        if not hasattr(self, 'improvement_ema'):
            self.improvement_ema = 1.0

        avg_improvement = improvement_sum / max(recent_improved, 1)
        improvement_magnitude = float(np.log1p(max(avg_improvement, 1e-15)))
        self.improvement_ema = float(0.7 * self.improvement_ema + 0.3 * improvement_magnitude)

        pop_variance = float(np.mean(np.var(self.population, axis=0)))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity_ratio = float(np.clip(pop_variance / (expected_var + 1e-10), 0.01, 1.0))

        target_rate = float(np.clip(0.25 * (1.0 + 0.5 * np.log1p(1.0 / (diversity_ratio + 1e-10))), 0.05, 0.6))

        rate_ratio = success_rate / max(target_rate, 1e-10)
        if rate_ratio < 0.5:
            adaptation = -0.5 * (1.0 + np.log1p(0.5 / (rate_ratio + 1e-10)))
        elif rate_ratio > 1.5:
            adaptation = 0.5 * np.log1p(rate_ratio)
        else:
            adaptation = (success_rate - target_rate) / target_rate

        adaptation = float(np.clip(adaptation, -0.8, 0.8))
        momentum_factor = float(np.clip(np.log1p(self.improvement_ema * 1e6) / 10.0, 0.2, 2.0))
        damping_adaptive = float(np.clip(self.damping * momentum_factor * (0.3 + 0.7 * np.log1p(self.dim) / diversity_ratio), 0.05, 150.0))

        generation_factor = np.exp(-self.generation / (50 + self.dim * 2))
        cs_adaptive = self.cs * (1.0 + generation_factor)

        self.sigma = float(np.clip(self.sigma * np.exp(adaptation * cs_adaptive / damping_adaptive), 1e-10, 10.0))
    
    def _adapt_step_size_variant_04(self):
        """Adapt step-size using momentum-enhanced improvement tracking with diversity-based damping."""
        recent_improved = 0
        recent_total = 0
        total_improvement = 0.0

        for i in range(self.NP):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                if float(self.trial_fitness[i]) < float(self.fitness[i]):
                    recent_improved += 1
                    diff = float(self.fitness[i] - self.trial_fitness[i])
                    total_improvement += max(diff, 0.0)
                recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = float(np.clip(recent_improved / recent_total, 0.0, 1.0))

        avg_improvement = total_improvement / max(recent_improved, 1)
        improvement_magnitude = float(np.log1p(max(avg_improvement, 1e-15)))

        if not hasattr(self, 'improvement_ema'):
            self.improvement_ema = improvement_magnitude
        self.improvement_ema = float(0.7 * self.improvement_ema + 0.3 * improvement_magnitude)

        target_rate = 0.25
        adaptation = float((success_rate - target_rate) / target_rate)
        adaptation += float(0.3 * np.tanh(self.improvement_ema - 1.0))
        adaptation = float(np.clip(adaptation, -0.8, 0.8))

        pop_variance = float(np.mean(np.var(self.population, axis=0)))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = float(np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0))

        diversity_boost = float(np.clip(1.0 + 2.0 * (1.0 - diversity), 0.5, 3.0))
        damping_adaptive = float(np.clip(self.damping * (0.5 + 0.5 * np.log1p(self.dim)) / diversity_boost, 0.05, 200.0))

        self.sigma = float(np.clip(self.sigma * np.exp(adaptation * self.cs / damping_adaptive), 1e-10, 10.0))
    
    def _adapt_step_size_variant_05(self):
        """Adapt step-size using evolution path length control (CMA-ES style)."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        if not hasattr(self, 'p_sigma') or self.p_sigma is None:
            self.p_sigma = np.zeros(self.dim)

        self.p_sigma = (1.0 - self.cs) * self.p_sigma + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean

        path_norm = float(np.linalg.norm(self.p_sigma))
        expected_norm = float(np.sqrt(self.dim) * (1.0 - (1.0 - self.cs) ** (2.0 * max(1, self.generation))))
        expected_norm = max(expected_norm, 0.1)

        control = float(np.clip(path_norm / expected_norm - 1.0, -0.5, 0.5))

        stagnation_boost = 1.0 + 0.5 * min(self.stagnation_counter / max(1, self.max_stagnation), 1.0)
        damping_eff = float(np.clip(self.damping * stagnation_boost, 0.1, 200.0))

        self.sigma = float(np.clip(self.sigma * np.exp(control * self.cs / damping_eff), 1e-10, 10.0))
    
    def _adapt_step_size_variant_07(self):
        """Adapt step-size using momentum, fitness gradient, and diversity awareness."""
        f_mean_prev = float(getattr(self, 'f_mean_prev', float(np.mean(self.fitness))))
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
        self.sigma_momentum = float(np.clip((1.0 - momentum_alpha) * self.sigma_momentum + momentum_alpha * target_momentum, -0.8, 0.8))

        pop_spread = float(np.mean(np.std(self.population, axis=0)))
        expected_spread = (self.ub[0] - self.lb[0]) / 6.0
        diversity_ratio = float(np.clip(pop_spread / (expected_spread + 1e-10), 0.0, 1.0))

        recent_improved = 0
        recent_total = 0
        for i in range(min(self.NP, len(self.trial_fitness), len(self.fitness))):
            if float(self.trial_fitness[i]) < float(self.fitness[i]):
                recent_improved += 1
            recent_total += 1

        if recent_total == 0:
            recent_total = max(1, self.NP // 4)

        success_rate = float(np.clip(recent_improved / recent_total, 0.0, 1.0))

        min_diversity_thresh = 0.05
        if diversity_ratio < min_diversity_thresh:
            diversity_factor = 0.3
        else:
            diversity_factor = 1.0

        adaptation = self.sigma_momentum + 0.2 * (success_rate - 0.25)

        if diversity_ratio < min_diversity_thresh:
            adaptation = max(adaptation, 0.3)

        adaptation = float(np.clip(adaptation, -0.6, 0.6))

        damping_adaptive = float(np.clip(self.damping * (0.5 + 0.5 * np.log1p(self.dim)) * diversity_factor, 0.05, 100.0))

        self.sigma = float(np.clip(self.sigma * np.exp(adaptation * self.cs / damping_adaptive), 1e-12, 15.0))

        if pop_spread > 1e-10 and self.sigma < pop_spread * 1e-4:
            self.sigma = pop_spread * 1e-3

        self.f_mean_prev = f_mean_curr
    
    def _adapt_step_size_variant_09(self):
        """Adapt step-size using CMA-ES-style cumulative evolution path with momentum."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        if not hasattr(self, 'ps_path'):
            self.ps_path = np.zeros(self.dim)

        self.ps_path = (1.0 - self.cs) * self.ps_path + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean

        path_norm = float(np.linalg.norm(self.ps_path))
        expected_path = float(np.sqrt(self.dim) * (1.0 - (1.0 - self.cs) ** (2.0 * self.generation + 1)))
        expected_path = max(expected_path, 0.1)

        success_ratio = float(np.clip(path_norm / expected_path if expected_path > 1e-15 else 1.0, 0.01, 10.0))

        adaptation = float(np.clip(np.log(success_ratio) / np.log(2.0), -0.8, 0.8))

        damping_adaptive = float(np.clip(self.damping * (0.3 + 0.7 * np.log1p(self.dim) / np.log1p(path_norm + 1e-10)), 0.05, 200.0))

        self.sigma = float(np.clip(self.sigma * np.exp(adaptation * self.cs / damping_adaptive), 1e-12, 20.0))
    
    def _adapt_step_size_variant_10(self):
        """Adapt step-size using magnitude-aware improvement with stagnation detection."""
        if len(self.trial_fitness) > 0 and len(self.fitness) > 0:
            improvements = np.maximum(0.0, np.array(self.fitness) - np.array(self.trial_fitness))
            recent_improved = int(np.sum(improvements > 0))
            total_improvement = float(np.sum(improvements))
            max_improvement = float(np.max(improvements))
        else:
            recent_improved = 0
            total_improvement = 0.0
            max_improvement = 0.0

        recent_total = max(1, len(self.trial_fitness))
        success_rate = float(recent_improved / recent_total)

        target_rate = float(0.2 + 0.1 * np.tanh(self.dim / 20.0))

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
            combined_adaptation = float(np.clip(0.7 * rate_adaptation + 0.3 * magnitude_factor, -0.8, 0.8))

            damping_adaptive = float(np.clip(self.damping * (0.3 + 0.7 * np.log1p(self.dim) / np.log1p(100)), 0.05, 50.0))

            self.sigma = float(np.clip(self.sigma * np.exp(combined_adaptation * self.cs / damping_adaptive), 1e-10, 50.0))
    
    def _adapt_covariance(self):
        """Dispatch to the selected covariance adaptation strategy."""
        if not hasattr(self, '_cov_operator'):
            self._cov_operator = 0
        
        if self._cov_operator == 0:
            self._adapt_covariance_original()
        elif self._cov_operator == 1:
            self._adapt_covariance_variant_01()
        elif self._cov_operator == 2:
            self._adapt_covariance_variant_06()
        elif self._cov_operator == 3:
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
        
        y_mean = (self.mean -