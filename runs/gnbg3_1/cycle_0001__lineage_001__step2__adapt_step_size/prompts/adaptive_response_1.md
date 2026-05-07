```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best step-size adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 6 step-size adaptation strategies (based on benchmark results)
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
        
        # Step-size operator selection parameters (based on benchmark winning strategies)
        self.num_step_operators = 6
        self.step_operator_names = [
            'original',      # 0: baseline CSA
            'variant_02',    # 1: short-term success history (6 wins)
            'variant_04',    # 2: multi-feedback signals (1 win)
            'variant_05',    # 3: success history + diversity-weighted (1 win)
            'variant_06',    # 4: explicit success-rate tracking (6 wins)
            'variant_08'     # 5: mirrored sampling variance ratio (6 wins)
        ]
        
        # Thompson Sampling with Beta distributions for step-size operators
        self.step_alpha = np.ones(self.num_step_operators)
        self.step_beta = np.ones(self.num_step_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 10
        self.step_operator_rewards = {i: [] for i in range(self.num_step_operators)}
        self.step_operator_counts = np.zeros(self.num_step_operators)
        
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
            
            # Select step-size operator BEFORE adaptation
            self._select_step_operator_thompson()
            
            # Apply selected step-size adaptation strategy
            self._adapt_step_size()
            
            self._select_survivors_batch()
            self._adapt_covariance()
            
            # Update operator rewards based on improvement
            self._update_step_operator_rewards()
            
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
        
        self._reset_step_operator_state()
    
    def _reset_step_operator_state(self):
        """Reset all step-size operator-specific state variables."""
        self.L = None
        # State for variant_04
        if hasattr(self, 'pc_weighted'):
            self.pc_weighted = np.zeros(self.dim)
        if hasattr(self, 'p_cross'):
            self.p_cross = np.zeros(self.dim)
        if hasattr(self, 'prev_y_mean'):
            delattr(self, 'prev_y_mean')
        # State for variant_05
        if hasattr(self, 'success_history'):
            self.success_history = []
        if hasattr(self, 'sigma_history'):
            self.sigma_history = []
        # State for variant_06
        if hasattr(self, '_step_history'):
            self._step_history = []
        # State for variant_09
        if hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.improvement_ema = 1.0
    
    def _sample_trials_batch(self):
        """Sample new trial population from multivariate normal."""
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
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
        """Select step-size operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.step_alpha, self.step_beta)
        self.current_step_operator = int(np.argmax(samples))
        self.step_operator_counts[self.current_step_operator] += 1
        return self.current_step_operator
    
    def _update_step_operator_rewards(self):
        """Update step-size operator rewards using sliding window credit assignment."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        # Reward based on improvement and diversity
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity)
        reward = float(np.clip(reward, -10.0, 10.0))
        
        op = self.current_step_operator
        self.step_operator_rewards[op].append(reward)
        
        if len(self.step_operator_rewards[op]) > self.reward_window_size:
            self.step_operator_rewards[op].pop(0)
        
        n = len(self.step_operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.step_operator_rewards[op]))
            var_reward = float(np.var(self.step_operator_rewards[op]))
            
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.step_alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.step_beta[op] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.step_operator_rewards[op]))
            if sum_reward > 0:
                self.step_alpha[op] = 1.0 + sum_reward
            else:
                self.step_beta[op] = 1.0 - sum_reward
    
    def _adapt_step_size(self):
        """Dispatch to the selected step-size adaptation strategy."""
        if self.current_step_operator == 0:
            self._adapt_step_size_original()
        elif self.current_step_operator == 1:
            self._adapt_step_size_variant_02()
        elif self.current_step_operator == 2:
            self._adapt_step_size_variant_04()
        elif self.current_step_operator == 3:
            self._adapt_step_size_variant_05()
        elif self.current_step_operator == 4:
            self._adapt_step_size_variant_06()
        else:
            self._adapt_step_size_variant_08()
    
    def _adapt_step_size_original(self):
        """Adapt step-size using evolution path (CSA - Cumulation)."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean
        
        self.sigma *= np.exp((np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1.0) * self.cs / self.damping)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _adapt_step_size_variant_02(self):
        """Adapt step-size using short-term success history (no cumulation) - 6 wins."""
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
    
    def _adapt_step_size_variant_04(self):
        """Adapt step-size using multi-feedback signals with diversity-sensitive damping - 1 win."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Signal 1: Evolution path (CSA baseline)
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean
        path_norm = np.linalg.norm(self.ps) / np.sqrt(self.dim)

        # Signal 2: Fitness gradient direction
        if hasattr(self, 'prev_f_opt'):
            f_improved = self.f_opt < self.f_opt_prev - 1e-12
            fit_grad = np.sign(self.f_opt_prev - self.f_opt) if hasattr(self, 'prev_f_opt') else 0.0
        else:
            f_improved = False
            fit_grad = 0.0

        # Signal 3: Diversity ratio for adaptive damping
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity_ratio = pop_variance / (expected_var + 1e-10)
        diversity_ratio = np.clip(diversity_ratio, 1e-4, 10.0)

        # Stagnation detection
        stagnation = self.stagnation_counter > self.max_stagnation // 2

        # Adaptive damping: increase when diverse (explore more), decrease when converged (exploit)
        base_damping = self.damping
        if stagnation:
            adaptive_damping = base_damping * 0.3  # Lower damping = faster sigma growth
        elif diversity_ratio < 0.1:
            adaptive_damping = base_damping * 0.5  # Low diversity: encourage exploration
        elif diversity_ratio > 0.5:
            adaptive_damping = base_damping * 1.5  # High diversity: be more conservative
        else:
            adaptive_damping = base_damping

        # Combined step-size update with multiple feedback signals
        csa_term = (path_norm - 1.0) * self.cs / adaptive_damping

        # Secondary: fitness-based adjustment
        if f_improved:
            fitness_adjustment = -0.1 / adaptive_damping  # Small reward for improvement
        else:
            fitness_adjustment = 0.05 / adaptive_damping  # Small penalty for no improvement

        # Tertiary: diversity-based adjustment (prevent collapse or explosion)
        if diversity_ratio < 0.05:
            diversity_adjustment = 0.2 / adaptive_damping  # Prevent collapse
        elif diversity_ratio > 2.0:
            diversity_adjustment = -0.1 / adaptive_damping  # Slow down if too spread
        else:
            diversity_adjustment = 0.0

        # Combine all signals
        total_adjustment = csa_term + fitness_adjustment + diversity_adjustment

        # Apply with safeguards
        self.sigma *= np.exp(np.clip(total_adjustment, -2.0, 2.0))
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _adapt_step_size_variant_05(self):
        """Adapt step-size using success history and diversity-weighted control - 1 win."""
        improvement = self.f_opt_prev - self.f_opt

        if not hasattr(self, 'success_history'):
            self.success_history = []
        if not hasattr(self, 'sigma_history'):
            self.sigma_history = []

        is_success = improvement > 0
        self.success_history.append(1.0 if is_success else 0.0)
        self.sigma_history.append(self.sigma)

        max_history = max(5, self.dim // 10 + 5)
        if len(self.success_history) > max_history:
            self.success_history.pop(0)
            self.sigma_history.pop(0)

        success_rate = np.mean(self.success_history)

        pop_spread = np.mean(np.std(self.population, axis=0))
        expected_spread = (self.ub[0] - self.lb[0]) / 6.0
        diversity_ratio = np.clip(pop_spread / (expected_spread + 1e-10), 0.1, 3.0)

        target_success = 0.25

        if success_rate > target_success:
            adjustment = 1.0 + 0.3 * (success_rate - target_success) / (1.0 - target_success + 1e-10)
        else:
            adjustment = 0.7 + 0.3 * success_rate / (target_success + 1e-10)

        if diversity_ratio < 0.3:
            adjustment *= 1.5
        elif diversity_ratio < 0.6:
            adjustment *= 1.2

        if len(self.sigma_history) >= 3:
            sigma_trend = self.sigma_history[-1] / (np.mean(self.sigma_history[-3:]) + 1e-10)
            if sigma_trend < 0.8:
                adjustment *= 1.1
            elif sigma_trend > 1.2:
                adjustment *= 0.9

        self.sigma *= adjustment
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _adapt_step_size_variant_06(self):
        """Adapt step-size using explicit success-rate tracking (no CSA evolution path) - 6 wins."""
        if not hasattr(self, '_step_history'):
            self._step_history = []

        n_successful = 0
        for f in self.trial_fitness:
            f_val = float(np.asarray(f).flatten()[0])
            if f_val < self.f_opt_prev:
                n_successful += 1
        
        current_success_rate = n_successful / max(len(self.trial_fitness), 1)
        self._step_history.append(current_success_rate)

        if len(self._step_history) > 5:
            self._step_history.pop(0)

        smoothed_success = sum(self._step_history) / len(self._step_history)

        expected_success = 0.2
        success_diff = smoothed_success - expected_success

        damping_adjusted = self.damping * np.sqrt(self.dim)

        adapt_rate = self.cs * 1.5
        sigma_multiplier = np.exp(adapt_rate * success_diff / damping_adjusted)

        self.sigma *= sigma_multiplier

        pop_spread = np.mean(np.std(self.population, axis=0))
        if pop_spread > 1e-10:
            target_spread = (self.ub[0] - self.lb[0]) * 0.05
            if self.sigma * pop_spread > target_spread * 10:
                self.sigma *= 0.9
            elif self.sigma * pop_spread < target_spread * 0.1:
                self.sigma *= 1.1

        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _adapt_step_size_variant_08(self):
        """Adapt step-size using mirrored sampling variance ratio - 6 wins."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        # Mirrored sampling: generate paired samples in opposite directions
        z_mirror = np.random.randn(self.dim)
        z_mirror = z_mirror / np.maximum(np.linalg.norm(z_mirror), 1e-10)

        # Compute variance ratio: current vs expected under isotropic normal
        z_var = np.var(z_mirror)
        expected_var = 1.0 / self.dim
        variance_ratio = np.clip(z_var / (expected_var + 1e-10), 0.1, 10.0)

        # Scale evolution path by variance ratio for adaptive search
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean * np.sqrt(variance_ratio)

        # Compute adaptive sigma change using combined norm and variance ratio
        norm_ps = np.linalg.norm(self.ps) / np.sqrt(self.dim)
        sigma_factor = np.exp((norm_ps - 1.0) * self.cs / self.damping * (1.0 / np.sqrt(variance_ratio) + 0.5 * (variance_ratio - 1.0)))

        self.sigma *= sigma_factor
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _adapt_covariance(self):
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