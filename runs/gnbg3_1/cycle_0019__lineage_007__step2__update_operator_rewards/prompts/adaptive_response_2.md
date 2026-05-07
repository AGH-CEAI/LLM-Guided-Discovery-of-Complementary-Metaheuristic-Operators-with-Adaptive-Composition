```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy AND operator reward strategy during optimization using multi-level
    Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 5 covariance adaptation strategies
    - 10 operator reward update strategies (adaptive selection)
    - Multi-level Thompson Sampling for both operator and strategy selection
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
        
        # Thompson Sampling with Beta distributions for operator selection
        self.alpha = np.ones(self.num_operators)
        self.beta = np.ones(self.num_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 10
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators)
        self.selection_counts = np.zeros(self.num_operators)
        
        # Adaptive reward strategy selection (meta-level)
        self.num_reward_strategies = 10
        self.strategy_alpha = np.ones(self.num_reward_strategies)
        self.strategy_beta = np.ones(self.num_reward_strategies)
        self.current_reward_strategy = 0
        self.strategy_rewards = {i: [] for i in range(self.num_reward_strategies)}
        self.strategy_counts = np.zeros(self.num_reward_strategies)
        self.strategy_window_size = 8
        
        # Track strategy performance for meta-level credit assignment
        self.strategy_improvement_history = {i: [] for i in range(self.num_reward_strategies)}
        
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
            
            # Select reward strategy using meta-level Thompson Sampling
            self._select_reward_strategy_thompson()
            
            # Update operator rewards using the selected strategy
            self._update_operator_rewards()
            
            # Update meta-level strategy rewards based on improvement
            self._update_strategy_rewards()
            
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
    
    def _select_reward_strategy_thompson(self):
        """Select reward update strategy using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.strategy_alpha, self.strategy_beta)
        self.current_reward_strategy = int(np.argmax(samples))
        self.strategy_counts[self.current_reward_strategy] += 1
        return self.current_reward_strategy
    
    def _update_strategy_rewards(self):
        """Update meta-level strategy rewards based on improvement contributions."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        # Compute normalized improvement signal
        scale = max(abs(self.f_opt), abs(self.f_opt_prev), 1.0)
        normalized_improvement = improvement / scale
        
        # Log-scaled reward for the strategy
        strategy_reward = float(np.log1p(improvement * 1e10) / 10.0)
        strategy_reward = float(np.clip(strategy_reward, -10.0, 10.0))
        
        strategy = self.current_reward_strategy
        self.strategy_rewards[strategy].append(strategy_reward)
        self.strategy_improvement_history[strategy].append(improvement)
        
        # Maintain window size
        if len(self.strategy_rewards[strategy]) > self.strategy_window_size:
            self.strategy_rewards[strategy].pop(0)
        if len(self.strategy_improvement_history[strategy]) > self.strategy_window_size:
            self.strategy_improvement_history[strategy].pop(0)
        
        n = len(self.strategy_rewards[strategy])
        if n >= 3:
            mean_reward = float(np.mean(self.strategy_rewards[strategy]))
            var_reward = float(np.var(self.strategy_rewards[strategy]))
            
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.strategy_alpha[strategy] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.strategy_beta[strategy] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.strategy_rewards[strategy]))
            if sum_reward > 0:
                self.strategy_alpha[strategy] = float(max(1.0, 1.0 + sum_reward))
            else:
                self.strategy_beta[strategy] = float(max(1.0, 1.0 - sum_reward))
    
    def _update_operator_rewards(self):
        """Dispatch to the selected reward update strategy."""
        strategy = self.current_reward_strategy
        if strategy == 0:
            self._reward_strategy_original()
        elif strategy == 1:
            self._reward_strategy_variant_01()
        elif strategy == 2:
            self._reward_strategy_variant_02()
        elif strategy == 3:
            self._reward_strategy_variant_04()
        elif strategy == 4:
            self._reward_strategy_variant_05()
        elif strategy == 5:
            self._reward_strategy_variant_06()
        elif strategy == 6:
            self._reward_strategy_variant_07()
        elif strategy == 7:
            self._reward_strategy_variant_08()
        elif strategy == 8:
            self._reward_strategy_variant_09()
        else:
            self._reward_strategy_variant_10()
    
    def _reward_strategy_original(self):
        """Original reward strategy: sliding window credit assignment."""
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
    
    def _reward_strategy_variant_01(self):
        """Strategy 1: Sigmoid reward with EMA tracking."""
        improvement = self.f_opt_prev - self.f_opt
        scale = max(abs(self.f_opt), abs(self.f_opt_prev), 1.0)
        relative_improvement = improvement / scale
        reward = 2.0 / (1.0 + np.exp(-5.0 * np.clip(relative_improvement, -3.0, 3.0))) - 1.0
        reward = float(np.clip(reward, -1.0, 1.0))
        
        op = self.current_operator
        
        if not hasattr(self, 'reward_ema'):
            self.reward_ema = np.zeros(self.num_operators)
            self.reward_count = np.zeros(self.num_operators)
        
        if self.reward_count[op] == 0:
            self.reward_ema[op] = reward
        else:
            self.reward_ema[op] = (1.0 - 0.3) * self.reward_ema[op] + 0.3 * reward
        
        self.reward_count[op] += 1
        
        reward_mapped = (reward + 1.0) / 2.0
        reward_mapped = float(np.clip(reward_mapped, 1e-10, 1.0 - 1e-10))
        
        self.alpha[op] = float(np.clip(self.alpha[op] + reward_mapped, 1e-10, 1e6))
        self.beta[op] = float(np.clip(self.beta[op] + (1.0 - reward_mapped), 1e-10, 1e6))
    
    def _reward_strategy_variant_02(self):
        """Strategy 2: Rank-based credit assignment with stagnation escape detection."""
        pop_range = max(np.max(self.fitness) - np.min(self.fitness), 1e-10)
        rank_improvement = max(0.0, self.f_opt_prev - self.f_opt) / pop_range
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        is_escaping = (self.stagnation_counter == 0 and 
                       hasattr(self, 'prev_stagnation') and 
                       self.prev_stagnation >= self.max_stagnation // 2)
        escape_bonus = 5.0 if is_escaping else 0.0
        self.prev_stagnation = self.stagnation_counter
        
        op = self.current_operator
        
        if not hasattr(self, 'op_best_history'):
            self.op_best_history = {i: [] for i in range(self.num_operators)}
        
        if len(self.trial_fitness) >= self.NP // 4:
            sorted_trial_fitness = np.sort(self.trial_fitness)
            best_trial_rank = np.searchsorted(sorted_trial_fitness, self.f_opt, side='left') / len(sorted_trial_fitness)
            rank_reward = float(1.0 - best_trial_rank)
        else:
            rank_reward = 0.5
        
        reward = float(2.0 * rank_reward + 0.5 * diversity + escape_bonus)
        reward = float(np.clip(reward, -10.0, 20.0))
        
        window_size = max(self.reward_window_size, 15)
        self.operator_rewards[op].append(reward)
        if len(self.operator_rewards[op]) > window_size:
            self.operator_rewards[op].pop(0)
        
        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            if hasattr(self, 'ema_reward'):
                self.ema_reward[op] = 0.7 * self.ema_reward[op] + 0.3 * reward
            else:
                self.ema_reward = {i: 0.0 for i in range(self.num_operators)}
                self.ema_reward[op] = reward
            
            ema = self.ema_reward[op]
            var_reward = float(np.var(self.operator_rewards[op]))
            
            scaled_mean = np.clip(ema * 2.0, 0.0, 1.0)
            scaled_var = np.clip(var_reward * 4.0, 0.01, 1.0)
            
            self.alpha[op] = float(max(1.0, scaled_mean * (scaled_mean * (n - 1) / (scaled_var + 1e-10) + 1)))
            self.beta[op] = float(max(1.0, (1 - scaled_mean) * ((n - 1) * (1 - scaled_mean) / (scaled_var + 1e-10) + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = 1.0 + sum_reward
            else:
                self.beta[op] = 1.0 - sum_reward
    
    def _reward_strategy_variant_04(self):
        """Strategy 3: Trend-aware multi-signal credit assignment."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        if not hasattr(self, 'ema_improvement_v4'):
            self.ema_improvement_v4 = 0.0
            self.ema_reward_v4 = 0.0
            self.op_trend_count = {i: 0 for i in range(self.num_operators)}
            self.op_last_reward = {i: 0.0 for i in range(self.num_operators)}
        
        self.ema_improvement_v4 = 0.7 * self.ema_improvement_v4 + 0.3 * improvement
        trend_signal = np.log1p(max(self.ema_improvement_v4 * 1e10, 1e-15))
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        ranks = np.argsort(np.argsort(self.fitness))
        rank_probs = (ranks + 1) / (self.NP + 1)
        rank_probs = np.clip(rank_probs, 1e-10, 1.0)
        rank_entropy = -np.mean(rank_probs * np.log(rank_probs))
        rank_entropy /= np.log(self.NP + 1) + 1e-10
        
        op = self.current_operator
        if improvement > 1e-15:
            if self.op_last_reward[op] > 0.0 or self.op_trend_count[op] > 0:
                self.op_trend_count[op] += 1
            else:
                self.op_trend_count[op] = 1
        else:
            self.op_trend_count[op] = max(0, self.op_trend_count[op] - 1)
        self.op_last_reward[op] = improvement
        
        consistency = np.tanh(self.op_trend_count[op] / 5.0)
        
        base_reward = trend_signal / 10.0
        diversity_reward = 0.15 * diversity
        ruggedness_penalty = 0.05 * (1.0 - rank_entropy)
        consistency_reward = 0.3 * consistency
        
        reward = base_reward + diversity_reward + ruggedness_penalty + consistency_reward
        reward = float(np.clip(reward, -10.0, 10.0))
        
        self.ema_reward_v4 = 0.9 * self.ema_reward_v4 + 0.1 * reward
        
        self.operator_rewards[op].append(reward)
        
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)
        
        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            ema_reward = float(self.ema_reward_v4)
            combined = 0.6 * mean_reward + 0.4 * ema_reward
            
            var_reward = float(np.var(self.operator_rewards[op]))
            denom = max(var_reward * n + 1e-10, 1e-10)
            
            self.alpha[op] = float(max(1.0, combined * (combined * (n - 1) / denom + 1)))
            self.beta[op] = float(max(1.0, (1 - combined) * ((n - 1) * (1 - combined) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = 1.0 + sum_reward
            else:
                self.beta[op] = 1.0 - sum_reward
    
    def _reward_strategy_variant_05(self):
        """Strategy 4: Exponentially-weighted cumulative improvement with stagnation awareness."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        if not hasattr(self, 'operator_cumulative_reward'):
            self.operator_cumulative_reward = np.zeros(self.num_operators)
            self.operator_decay_sum = np.zeros(self.num_operators)
        
        decay = 0.95
        self.operator_cumulative_reward *= decay
        self.operator_decay_sum *= decay
        
        if improvement > 0:
            reward = float(np.log1p(improvement * 1e10) / 10.0)
        else:
            reward = 0.0
        
        is_stagnant = self.stagnation_counter > self.max_stagnation // 2
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        if is_stagnant and diversity < 0.1:
            reward *= 2.0
        
        self.operator_cumulative_reward[self.current_operator] += reward
        self.operator_decay_sum[self.current_operator] += 1.0
        
        norm = max(self.operator_decay_sum[self.current_operator], 1.0)
        normalized_reward = self.operator_cumulative_reward[self.current_operator] / norm
        normalized_reward = float(np.clip(normalized_reward, -10.0, 10.0))
        
        op = self.current_operator
        self.operator_rewards[op].append(normalized_reward)
        
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
    
    def _reward_strategy_variant_06(self):
        """Strategy 5: Max-reward tracking with EMA smoothing."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        current_reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity)
        current_reward = float(np.clip(current_reward, -10.0, 10.0))
        
        op = self.current_operator
        
        if not hasattr(self, 'op_max_reward_ema'):
            self.op_max_reward_ema = np.ones(self.num_operators) * 0.5
            self.op_max_reward_count = np.zeros(self.num_operators)
        
        prev_max_ema = self.op_max_reward_ema[op]
        if current_reward > prev_max_ema:
            self.op_max_reward_ema[op] = 0.7 * prev_max_ema + 0.3 * current_reward
        else:
            self.op_max_reward_ema[op] = 0.98 * prev_max_ema + 0.02 * current_reward
        
        if current_reward >= prev_max_ema and self.op_max_reward_count[op] > 0:
            self.op_max_reward_count[op] += 1
        elif self.op_max_reward_count[op] == 0 and current_reward > 0:
            self.op_max_reward_count[op] = 1
        
        max_component = 0.5 * max(self.op_max_reward_ema[op], 0.0)
        reward = 0.6 * current_reward + 0.4 * max_component
        reward = float(np.clip(reward, -10.0, 10.0))
        
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
    
    def _reward_strategy_variant_07(self):
        """Strategy 6: Per-operator rank-normalized credit assignment."""
        if not hasattr(self, 'op_success_history_v7'):
            self.op_success_history_v7 = {i: [] for i in range(self.num_operators)}
            self.op_improvement_history_v7 = {i: [] for i in range(self.num_operators)}
        
        op = self.current_operator
        self.selection_counts[op] += 1
        
        if len(self.trial_fitness) >= self.NP:
            individual_improvement = max(0.0, self.f_opt_prev - np.min(self.trial_fitness))
        else:
            individual_improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        self.op_improvement_history_v7[op].append(individual_improvement)
        if len(self.op_improvement_history_v7[op]) > self.reward_window_size:
            self.op_improvement_history_v7[op].pop(0)
        
        recent_improved = 0
        recent_total = min(self.NP, len(self.trial_fitness))
        if recent_total > 0:
            for i in range(recent_total):
                if i < len(self.trial_fitness) and i < len(self.fitness):
                    if self.trial_fitness[i] < self.fitness[i]:
                        recent_improved += 1
        op_success_rate = recent_improved / max(recent_total, 1)
        
        self.op_success_history_v7[op].append(op_success_rate)
        if len(self.op_success_history_v7[op]) > self.reward_window_size:
            self.op_success_history_v7[op].pop(0)
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        all_improvements = []
        for i in range(self.NP):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                imp = max(0.0, self.fitness[i] - self.trial_fitness[i])
                all_improvements.append(imp)
        
        roughness = 0.0
        if len(all_improvements) > 1:
            roughness = np.var(all_improvements) / (np.mean(all_improvements) ** 2 + 1e-10)
        roughness = np.clip(roughness, 0.0, 10.0)
        
        op_improvement_mean = np.mean(self.op_improvement_history_v7[op]) if self.op_improvement_history_v7[op] else 0.0
        op_success_mean = np.mean(self.op_success_history_v7[op]) if self.op_success_history_v7[op] else 0.0
        
        improvement_ratio = 1.0
        if op_improvement_mean > 1e-15:
            improvement_ratio = np.clip(individual_improvement / (op_improvement_mean + 1e-15), 0.0, 5.0)
        
        success_ratio = 1.0
        if op_success_mean > 1e-15:
            success_ratio = np.clip(op_success_rate / (op_success_mean + 1e-15), 0.0, 5.0)
        
        reward = 0.0
        reward += np.log1p(individual_improvement * 1e10) / 10.0
        reward += 0.5 * (improvement_ratio - 1.0)
        reward += 0.3 * (success_ratio - 1.0)
        reward += 0.2 * diversity
        reward += 0.1 * np.tanh(roughness - 1.0)
        
        reward = float(np.clip(reward, -10.0, 10.0))
        
        self.operator_rewards[op].append(reward)
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)
        
        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            var_reward = float(np.var(self.operator_rewards[op]))
            
            cv_sq = var_reward / (mean_reward ** 2 + 1e-10)
            lr_scale = 1.0 / (1.0 + np.sqrt(max(cv_sq, 0.01)))
            lr_scale = np.clip(lr_scale, 0.3, 3.0)
            
            denom = max(var_reward * n + 1e-10, 1e-10)
            alpha_update = mean_reward * (mean_reward * (n - 1) / denom + 1) * lr_scale
            beta_update = (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1) * lr_scale
            
            self.alpha[op] = float(max(1.0, alpha_update))
            self.beta[op] = float(max(1.0, beta_update))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = max(1.0, 1.0 + sum_reward)
            else:
                self.beta[op] = max(1.0, 1.0 - sum_reward)
    
    def _reward_strategy_variant_08(self):
        """Strategy 7: Inverse-error task-difficulty weighting."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        error_magnitude = max(abs(self.f_opt), 1.0)
        difficulty_weight = np.log1p(error_magnitude) + 1.0
        difficulty_weight = float(np.clip(difficulty_weight, 1.0, 100.0))
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        base_reward = float(np.log1p(improvement * 1e10) / 10.0)
        reward = base_reward * difficulty_weight + 0.1 * diversity
        reward = float(np.clip(reward, -20.0, 20.0))
        
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
    
    def _reward_strategy_variant_09(self):
        """Strategy 8: Error-scaled exploration-weighted credit assignment."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        error_scale = np.log1p(max(self.f_opt, 1e-15))
        error_scale = float(np.clip(error_scale / 10.0, 0.1, 5.0))
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        mean_dist = np.mean(np.linalg.norm(self.population - self.mean, axis=1))
        expected_dist = (self.ub[0] - self.lb[0]) / np.sqrt(12.0)
        exploration_ratio = np.clip(mean_dist / (expected_dist + 1e-10), 0.0, 2.0)
        
        if not hasattr(self, 'progress_history'):
            self.progress_history = []
        self.progress_history.append(self.f_opt)
        if len(self.progress_history) > 20:
            self.progress_history.pop(0)
        
        progress_rate = 0.0
        if len(self.progress_history) >= 5:
            recent = np.array(self.progress_history[-5:])
            older = np.array(self.progress_history[:len(recent)])
            if len(older) >= len(recent):
                progress_rate = float(np.mean(recent - older[:len(recent)]))
        
        if not hasattr(self, 'operator_ema'):
            self.operator_ema = {i: 0.0 for i in range(self.num_operators)}
        
        base_reward = float(np.log1p(improvement * 1e10 + 1e-15) / 10.0)
        exploration_bonus = 0.5 * diversity * exploration_ratio
        progress_reward = -0.3 * np.tanh(progress_rate)
        
        reward = error_scale * (base_reward + exploration_bonus + progress_reward)
        reward += 0.1 * diversity
        reward = float(np.clip(reward, -10.0, 10.0))
        
        op = self.current_operator
        decay = 0.85
        self.operator_ema[op] = decay * self.operator_ema[op] + (1.0 - decay) * reward
        self.operator_rewards[op].append(self.operator_ema[op])
        
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)
        
        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            var_reward = float(np.var(self.operator_rewards[op]))
            
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1) + 0.5 * diversity * n))
            self.beta[op] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = 1.0 + sum_reward
            else:
                self.beta[op] = 1.0 - sum_reward
    
    def _reward_strategy_variant_10(self):
        """Strategy 9: Multi-signal trajectory tracking."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        reward_immediate = float(np.log1p(improvement * 1e10) / 10.0)
        reward_immediate = np.clip(reward_immediate, -10.0, 10.0)
        
        if not hasattr(self, 'reward_ema_v10'):
            self.reward_ema_v10 = 0.0
        self.reward_ema_v10 = 0.7 * self.reward_ema_v10 + 0.3 * reward_immediate
        
        if not hasattr(self, 'reward_cumsum'):
            self.reward_cumsum = 0.0
            self.cumsum_buffer = 0.0
        self.reward_cumsum += improvement
        self.cumsum_buffer += 1
        if self.cumsum_buffer > 20:
            self.reward_cumsum *= 0.95
            self.cumsum_buffer *= 0.95
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        if not hasattr(self, 'stagnation_for_reward'):
            self.stagnation_for_reward = 0
        if improvement > 1e-15:
            self.stagnation_for_reward = 0
        else:
            self.stagnation_for_reward += 1
        
        reward = (0.30 * reward_immediate + 
                  0.40 * self.reward_ema_v10 + 
                  0.15 * diversity + 
                  0.15 * np.clip(np.log1p(self.reward_cumsum * 1e5) / 10.0, -5.0, 5.0))
        
        if self.stagnation_for_reward > 10:
            stagnation_factor = max(0.0, 1.0 - (self.stagnation_for_reward - 10) / 50.0)
            reward *= stagnation_factor
        
        if self.stagnation_for_reward == 0 and self.reward_ema_v10 > 1.0:
            reward += 0.5
        
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
```