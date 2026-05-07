```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    AND reward update strategy during optimization using hierarchical Thompson Sampling.
    
    Features:
    - 5 covariance adaptation strategies
    - 10 reward update strategies (auto-selected via meta-bandit)
    - Hierarchical Thompson Sampling for both operator and reward strategy selection
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
        
        # Thompson Sampling with Beta distributions for covariance operators
        self.alpha = np.ones(self.num_operators)
        self.beta = np.ones(self.num_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 10
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators)
        self.selection_counts = np.zeros(self.num_operators)
        
        # === NEW: Meta-level reward strategy selection ===
        self.num_reward_strategies = 11  # 10 variants + original
        self.reward_strategy_names = [
            'original', 'v01_ucb', 'v02_rank', 'v03_personal_best', 'v04_rank_gen',
            'v05_diversity', 'v06_diversity_2', 'v07_multi_component', 'v08_comparative',
            'v09_difficulty', 'v10_elite'
        ]
        
        # Meta-level Thompson Sampling for reward strategies
        self.meta_alpha = np.ones(self.num_reward_strategies)
        self.meta_beta = np.ones(self.num_reward_strategies)
        
        # Track performance of each reward strategy
        self.reward_strategy_rewards = {i: [] for i in range(self.num_reward_strategies)}
        self.reward_strategy_improvements = {i: [] for i in range(self.num_reward_strategies)}
        self.reward_strategy_selection_counts = np.zeros(self.num_reward_strategies)
        
        # Meta reward window
        self.meta_reward_window = 15
        self.current_reward_strategy = 0
        
        # Runtime state
        self.generation = 0
        self.current_operator = 0
        self.L = None
        self.f_opt_prev = 0.0
    
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
            
            # === NEW: Select reward strategy using meta-level Thompson Sampling ===
            self._select_reward_strategy_thompson()
            
            # Update operator rewards based on improvement using selected strategy
            self._update_operator_rewards_adaptive()
            
            # Update meta-level reward based on improvement quality
            self._update_meta_rewards()
            
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
    
    def _select_reward_strategy_thompson(self):
        """Select reward strategy using meta-level Thompson Sampling."""
        samples = np.random.beta(self.meta_alpha, self.meta_beta)
        self.current_reward_strategy = int(np.argmax(samples))
        self.reward_strategy_selection_counts[self.current_reward_strategy] += 1
        return self.current_reward_strategy
    
    def _update_meta_rewards(self):
        """Update meta-level reward strategy performance based on improvement quality."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        # Compute reward quality for this generation
        if improvement > 0:
            problem_scale = max(abs(self.f_opt), 1.0)
            relative_reward = float(np.log1p(improvement * 1e10) / 10.0)
        else:
            relative_reward = 0.0
        
        # Add diversity bonus for exploration
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        # Stagnation awareness
        is_stagnant = self.stagnation_counter > self.max_stagnation // 2
        if is_stagnant:
            relative_reward *= 1.5
        
        # Update meta-level reward tracking for current strategy
        rs = self.current_reward_strategy
        self.reward_strategy_improvements[rs].append(float(relative_reward))
        
        # Keep window bounded
        if len(self.reward_strategy_improvements[rs]) > self.meta_reward_window:
            self.reward_strategy_improvements[rs].pop(0)
        
        # Update meta Beta distribution based on smoothed performance
        n = len(self.reward_strategy_improvements[rs])
        if n >= 3:
            mean_imp = float(np.mean(self.reward_strategy_improvements[rs]))
            var_imp = float(np.var(self.reward_strategy_improvements[rs]))
            
            # Map to [0, 1] scale for Beta distribution
            mapped_reward = float(np.clip(mean_imp / 10.0, 0.01, 0.99))
            
            lr = 0.2
            self.meta_alpha[rs] = float(max(1.0, (1 - lr) * self.meta_alpha[rs] + lr * (1.0 + mapped_reward * 10.0)))
            self.meta_beta[rs] = float(max(1.0, (1 - lr) * self.meta_beta[rs] + lr * (1.0 + (1 - mapped_reward) * 10.0)))
    
    def _update_operator_rewards_adaptive(self):
        """Dispatch to the selected reward update strategy."""
        strategy = self.current_reward_strategy
        
        if strategy == 0:
            self._reward_strategy_original()
        elif strategy == 1:
            self._reward_strategy_v01_ucb()
        elif strategy == 2:
            self._reward_strategy_v02_rank()
        elif strategy == 3:
            self._reward_strategy_v03_personal_best()
        elif strategy == 4:
            self._reward_strategy_v04_rank_gen()
        elif strategy == 5:
            self._reward_strategy_v05_diversity()
        elif strategy == 6:
            self._reward_strategy_v06_diversity_2()
        elif strategy == 7:
            self._reward_strategy_v07_multi_component()
        elif strategy == 8:
            self._reward_strategy_v08_comparative()
        elif strategy == 9:
            self._reward_strategy_v09_difficulty()
        else:
            self._reward_strategy_v10_elite()
    
    # ==================== REWARD STRATEGY IMPLEMENTATIONS ====================
    
    def _reward_strategy_original(self):
        """Original reward strategy (baseline)."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        if not hasattr(self, 'operator_cumulative_reward'):
            self.operator_cumulative_reward = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum'):
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
    
    def _reward_strategy_v01_ucb(self):
        """UCB-inspired success rate tracking with immediate credit assignment."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        success = 1.0 if improvement > 0 else 0.0
        
        if not hasattr(self, 'operator_successes'):
            self.operator_successes = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_trials'):
            self.operator_trials = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_ema_reward'):
            self.operator_ema_reward = np.zeros(self.num_operators)
        
        op = self.current_operator
        self.operator_trials[op] += 1.0
        self.operator_ema_reward[op] = 0.9 * self.operator_ema_reward[op] + 0.1 * success
        
        if success > 0:
            self.operator_successes[op] += success
        
        ucb_scores = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            trials_i = max(self.operator_trials[i], 1.0)
            success_rate = self.operator_successes[i] / trials_i
            exploration_bonus = np.sqrt(2.0 * np.log(max(self.generation + 1, 1)) / trials_i)
            ucb_scores[i] = success_rate + exploration_bonus
        
        exploration_temp = getattr(self, 'exploration_temp', 1.0)
        ucb_bonus_scaled = ucb_scores[op] - (self.operator_successes[op] / max(self.operator_trials[op], 1.0))
        ucb_bonus_scaled *= exploration_temp
        
        reward = float(np.clip(self.operator_ema_reward[op] + ucb_bonus_scaled, 0.0, 1.0))
        
        self.operator_rewards.setdefault(op, [])
        self.operator_rewards[op].append(reward)
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)
        
        total_trials = max(self.operator_trials[op], 1.0)
        empirical_success_rate = self.operator_successes[op] / total_trials
        combined_rate = 0.7 * empirical_success_rate + 0.3 * self.operator_ema_reward[op]
        combined_rate = float(np.clip(combined_rate, 0.0, 1.0))
        
        self.alpha[op] = 1.0 + self.operator_successes[op]
        self.beta[op] = 1.0 + (self.operator_trials[op] - self.operator_successes[op])
        
        reward_adjustment = reward * 5.0
        self.alpha[op] = max(1.0, self.alpha[op] * (1.0 + reward_adjustment))
        self.beta[op] = max(1.0, self.beta[op] * (1.0 + (1.0 - reward) * 2.0))
        
        total_ab = self.alpha[op] + self.beta[op]
        if total_ab > 1000.0:
            scale = 1000.0 / total_ab
            self.alpha[op] *= scale
            self.beta[op] *= scale
    
    def _reward_strategy_v02_rank(self):
        """Rank-based multi-dimensional credit assignment (variant_02)."""
        if not hasattr(self, 'operator_rank_scores'):
            self.operator_rank_scores = {i: [] for i in range(self.num_operators)}
        if not hasattr(self, 'operator_spread_scores'):
            self.operator_spread_scores = {i: [] for i in range(self.num_operators)}
        if not hasattr(self, 'operator_improvement_scores'):
            self.operator_improvement_scores = {i: [] for i in range(self.num_operators)}
        
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        if improvement > 0:
            temp = max(abs(self.f_opt), 1.0)
            imp_reward = float(np.log1p(improvement / (temp + 1e-10)) / np.log(11.0))
        else:
            imp_reward = 0.0
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        spread_ratio = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        spread_reward = float(spread_ratio)
        
        fit_mean = np.mean(self.fitness)
        fit_std = max(np.std(self.fitness), 1e-10)
        if fit_std > 1e-10:
            best_z = (self.f_opt - fit_mean) / fit_std
            rank_reward = float(np.clip(0.5 + best_z / 4.0, 0.0, 1.0))
        else:
            rank_reward = 0.5
        
        is_stagnant = self.stagnation_counter > self.max_stagnation // 2
        is_very_stagnant = self.stagnation_counter > self.max_stagnation
        
        if is_very_stagnant:
            w_improve, w_spread, w_rank = 0.2, 0.5, 0.3
        elif is_stagnant:
            w_improve, w_spread, w_rank = 0.4, 0.3, 0.3
        else:
            w_improve, w_spread, w_rank = 0.6, 0.2, 0.2
        
        combined_reward = w_improve * imp_reward + w_spread * spread_reward + w_rank * rank_reward
        combined_reward = float(np.clip(combined_reward, 0.0, 1.0))
        
        if is_stagnant and spread_ratio < 0.1:
            combined_reward *= 0.5
        
        op = self.current_operator
        self.operator_improvement_scores[op].append(imp_reward)
        self.operator_spread_scores[op].append(spread_reward)
        self.operator_rank_scores[op].append(rank_reward)
        
        window = self.reward_window_size
        for scores in [self.operator_improvement_scores, self.operator_spread_scores, self.operator_rank_scores]:
            if len(scores[op]) > window:
                scores[op].pop(0)
        
        n = len(self.operator_improvement_scores[op])
        if n >= 3:
            mean_improve = np.mean(self.operator_improvement_scores[op])
            mean_spread = np.mean(self.operator_spread_scores[op])
            mean_rank = np.mean(self.operator_rank_scores[op])
            smoothed_reward = w_improve * mean_improve + w_spread * mean_spread + w_rank * mean_rank
        elif n >= 1:
            smoothed_reward = combined_reward
        else:
            smoothed_reward = 0.5
        
        smoothed_reward = float(np.clip(smoothed_reward, 0.0, 1.0))
        
        prior_strength = 2.0
        if n >= 1:
            obs_weight = min(n / (n + 2.0), 0.8)
            effective_reward = (1 - obs_weight) * 0.5 + obs_weight * smoothed_reward
        else:
            effective_reward = smoothed_reward
        
        effective_reward = float(np.clip(effective_reward, 0.01, 0.99))
        
        lr = 0.3
        self.alpha[op] = float(max(1.0, (1 - lr) * self.alpha[op] + lr * (1.0 + effective_reward * prior_strength)))
        self.beta[op] = float(max(1.0, (1 - lr) * self.beta[op] + lr * (1.0 + (1 - effective_reward) * prior_strength)))
        
        self.alpha[op] = float(np.clip(self.alpha[op], 1.0, 50.0))
        self.beta[op] = float(np.clip(self.beta[op], 1.0, 50.0))
    
    def _reward_strategy_v03_personal_best(self):
        """Personal-best centric credit assignment (variant_03)."""
        op = self.current_operator
        
        if not hasattr(self, 'operator_best_fitness'):
            self.operator_best_fitness = np.full(self.num_operators, float('inf'))
        if not hasattr(self, 'operator_cumulative_reward_v3'):
            self.operator_cumulative_reward_v3 = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum_v3'):
            self.operator_decay_sum_v3 = np.zeros(self.num_operators)
        
        current_fitness = float(np.min(self.fitness))
        
        is_new_best = False
        if current_fitness < self.operator_best_fitness[op]:
            improvement = self.operator_best_fitness[op] - current_fitness
            self.operator_best_fitness[op] = current_fitness
            is_new_best = True
        else:
            improvement = 0.0
        
        best_op = self.operator_best_fitness[op]
        worst_op = float(np.max(self.fitness))
        fitness_range = max(worst_op - best_op, 1e-10)
        relative_improvement = improvement / fitness_range
        improvement_reward = float(np.clip(relative_improvement * 10.0, 0.0, 5.0))
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = float(np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0))
        diversity_reward = float(2.0 * (1.0 - diversity))
        
        reward = 0.6 * improvement_reward + 0.4 * diversity_reward
        
        if is_new_best:
            new_best_bonus = 5.0 * float(np.log1p(max(improvement, 1e-10) / max(abs(best_op), 1e-10)))
            reward += new_best_bonus
        
        decay = 0.95
        self.operator_cumulative_reward_v3 *= decay
        self.operator_decay_sum_v3 *= decay
        self.operator_cumulative_reward_v3[op] += reward
        self.operator_decay_sum_v3[op] += 1.0
        
        norm = max(self.operator_decay_sum_v3[op], 1.0)
        normalized_reward = float(np.clip(self.operator_cumulative_reward_v3[op] / norm, -5.0, 10.0))
        
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
    
    def _reward_strategy_v04_rank_gen(self):
        """Rank-based credit assignment with generational memory (variant_04)."""
        if not hasattr(self, 'operator_cumulative_reward_v4'):
            self.operator_cumulative_reward_v4 = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum_v4'):
            self.operator_decay_sum_v4 = np.zeros(self.num_operators)
        if not hasattr(self, 'gen_rewards_history'):
            self.gen_rewards_history = []
        
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        fit_scale = max(abs(self.f_opt), 1.0)
        difficulty = max(1.0, np.log10(fit_scale + 1.0))
        
        self.gen_rewards_history.append((improvement, self.current_operator))
        if len(self.gen_rewards_history) > self.reward_window_size:
            self.gen_rewards_history.pop(0)
        
        if len(self.gen_rewards_history) >= 3:
            all_improvements = [g[0] for g in self.gen_rewards_history]
            sorted_imps = sorted(all_improvements, reverse=True)
            rank = sorted_imps.index(improvement) if improvement > 0 else len(sorted_imps) - 1
            max_rank = max(len(sorted_imps) - 1, 1)
            rank_pct = 1.0 - (rank / max_rank)
            reward = float(np.log1p(rank_pct * difficulty * 10.0))
        elif improvement > 0:
            reward = float(np.log1p(improvement * 1e8) / 20.0)
        else:
            reward = 0.0
        
        decay = 0.9
        self.operator_cumulative_reward_v4 *= decay
        self.operator_decay_sum_v4 *= decay
        
        self.operator_cumulative_reward_v4[self.current_operator] += reward
        self.operator_decay_sum_v4[self.current_operator] += 1.0
        
        norm = max(self.operator_decay_sum_v4[self.current_operator], 1.0)
        normalized_reward = self.operator_cumulative_reward_v4[self.current_operator] / norm
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
    
    def _reward_strategy_v05_diversity(self):
        """Reward operators for diversity contribution (variant_05)."""
        if not hasattr(self, 'operator_cumulative_reward_v5'):
            self.operator_cumulative_reward_v5 = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum_v5'):
            self.operator_decay_sum_v5 = np.zeros(self.num_operators)
        if not hasattr(self, 'prev_population'):
            self.prev_population = self.population.copy()
        if not hasattr(self, 'prev_mean'):
            self.prev_mean = self.mean.copy()
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        current_diversity = pop_variance / (expected_var + 1e-10)
        
        prev_variance = np.mean(np.var(self.prev_population, axis=0))
        prev_diversity = prev_variance / (expected_var + 1e-10)
        
        diversity_delta = current_diversity - prev_diversity
        diversity_reward = np.log1p(max(diversity_delta, 1e-15)) - np.log1p(max(-diversity_delta, 1e-15))
        diversity_reward = float(np.clip(diversity_reward, -3.0, 3.0))
        
        pop_distances = np.linalg.norm(self.population - self.mean, axis=1)
        mean_movement = np.linalg.norm(self.mean - self.prev_mean)
        exploration_score = float(np.mean(pop_distances) / (mean_movement + 1e-10))
        exploration_reward = float(np.clip(np.log1p(exploration_score) / 3.0, -2.0, 2.0))
        
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        improvement_reward = 0.0
        if improvement > 0:
            improvement_reward = float(np.clip(np.log1p(improvement * 1e10) / 15.0, 0.0, 5.0))
        
        is_stagnant = self.stagnation_counter > self.max_stagnation // 2
        stagnation_bonus = 1.5 if is_stagnant else 1.0
        
        if is_stagnant:
            total_reward = (0.15 * improvement_reward + 0.45 * diversity_reward + 0.40 * exploration_reward) * stagnation_bonus
        else:
            total_reward = (0.50 * improvement_reward + 0.25 * diversity_reward + 0.25 * exploration_reward)
        
        total_reward = float(np.clip(total_reward, -5.0, 5.0))
        
        decay = 0.95
        self.operator_cumulative_reward_v5 *= decay
        self.operator_decay_sum_v5 *= decay
        
        self.operator_cumulative_reward_v5[self.current_operator] += total_reward
        self.operator_decay_sum_v5[self.current_operator] += 1.0
        
        norm = max(self.operator_decay_sum_v5[self.current_operator], 1.0)
        normalized_reward = self.operator_cumulative_reward_v5[self.current_operator] / norm
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
        
        self.prev_population = self.population.copy()
        self.prev_mean = self.mean.copy()
    
    def _reward_strategy_v06_diversity_2(self):
        """Diversity-sensitive reward with exploration bonus."""
        if not hasattr(self, 'operator_cumulative_reward_v6'):
            self.operator_cumulative_reward_v6 = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum_v6'):
            self.operator_decay_sum_v6 = np.zeros(self.num_operators)
        
        decay = 0.9
        self.operator_cumulative_reward_v6 *= decay
        self.operator_decay_sum_v6 *= decay
        
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        if improvement > 0:
            base_reward = float(np.log1p(improvement * 1e10) / 8.0)
        else:
            base_reward = 0.0
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        is_stagnant = self.stagnation_counter > self.max_stagnation // 2
        
        if is_stagnant:
            diversity_bonus = (1.0 - diversity) * 3.0
            reward = base_reward + diversity_bonus
        else:
            reward = base_reward + diversity * 0.5
        
        self.operator_cumulative_reward_v6[self.current_operator] += reward
        self.operator_decay_sum_v6[self.current_operator] += 1.0
        
        norm = max(self.operator_decay_sum_v6[self.current_operator], 1.0)
        normalized_reward = self.operator_cumulative_reward_v6[self.current_operator] / norm
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
    
    def _reward_strategy_v07_multi_component(self):
        """Rank-based multi-component credit assignment (variant_07)."""
        if not hasattr(self, 'operator_cumulative_reward_v7'):
            self.operator_cumulative_reward_v7 = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_selection_quality'):
            self.operator_selection_quality = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_diversity_contribution'):
            self.operator_diversity_contribution = np.zeros(self.num_operators)
        
        if len(self.trial_fitness) >= 2:
            ranks = np.argsort(np.argsort(self.trial_fitness))
            n_trials = len(self.trial_fitness)
            rank_fraction = ranks / max(n_trials - 1, 1)
            avg_rank_fraction = float(np.mean(rank_fraction))
            selection_quality = 1.0 - avg_rank_fraction
        else:
            selection_quality = 0.5
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        if len(self.population) >= 2:
            pop_center = np.mean(self.population, axis=0)
            dists = np.linalg.norm(self.population - pop_center, axis=1)
            diversity_contribution = float(np.mean(dists) / (expected_var ** 0.5 + 1e-10))
            diversity_contribution = np.clip(diversity_contribution, 0.0, 1.0)
        else:
            diversity_contribution = 0.5
        
        if not hasattr(self, 'random_baseline_fitness'):
            self.random_baseline_fitness = float(np.mean(self.func(np.random.uniform(
                self.lb, self.ub, (min(1000, 10 * self.dim), self.dim))))
            )
        if not hasattr(self, 'regret_ema'):
            self.regret_ema = max(0.0, self.random_baseline_fitness - self.f_opt)
        
        regret = max(0.0, self.random_baseline_fitness - self.f_opt)
        self.regret_ema = 0.9 * self.regret_ema + 0.1 * regret
        regret_score = 1.0 / (1.0 + np.log1p(max(self.regret_ema, 1e-10)))
        
        is_stagnant = self.stagnation_counter > self.max_stagnation // 3
        stagnation_bonus = 2.0 if is_stagnant else 1.0
        
        if is_stagnant:
            reward = 0.2 * selection_quality + 0.6 * diversity_contribution + 0.2 * regret_score
        else:
            reward = 0.5 * selection_quality + 0.3 * diversity_contribution + 0.2 * regret_score
        
        reward *= stagnation_bonus
        reward = float(np.clip(reward, 0.0, 5.0))
        
        op = self.current_operator
        self.operator_cumulative_reward_v7[op] += reward
        self.operator_selection_quality[op] = 0.9 * self.operator_selection_quality[op] + 0.1 * selection_quality
        self.operator_diversity_contribution[op] = 0.9 * self.operator_diversity_contribution[op] + 0.1 * diversity_contribution
        
        total_reward = float(np.sum(self.operator_cumulative_reward_v7))
        if total_reward > 1e-10:
            normalized_reward = self.operator_cumulative_reward_v7[op] / total_reward
        else:
            normalized_reward = 1.0 / self.num_operators
        
        composite_score = 0.5 * normalized_reward + 0.3 * self.operator_selection_quality[op] + 0.2 * self.operator_diversity_contribution[op]
        composite_score = float(np.clip(composite_score, 0.0, 1.0))
        
        self.operator_rewards[op].append(composite_score)
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
            self.alpha[op] = 1.0 + sum_reward
            self.beta[op] = 1.0 + (n - sum_reward)
    
    def _reward_strategy_v08_comparative(self):
        """Comparative rank-based multi-objective credit assignment (variant_08)."""
        if not hasattr(self, 'operator_rank_history'):
            self.operator_rank_history = {i: [] for i in range(self.num_operators)}
        if not hasattr(self, 'operator_streak'):
            self.operator_streak = np.zeros(self.num_operators)
        if not hasattr(self, 'generation_counter'):
            self.generation_counter = 0
        
        self.generation_counter += 1
        
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        problem_scale = max(abs(self.f_opt), abs(self.f_opt_prev), 1.0)
        relative_improvement = improvement / problem_scale if problem_scale > 0 else 0.0
        
        if self.f_opt > 1.0:
            difficulty_mult = min(10.0, np.log10(self.f_opt + 1.0) + 1.0)
        elif self.f_opt > 1e-3:
            difficulty_mult = 2.0
        elif self.f_opt > 1e-6:
            difficulty_mult = 1.5
        else:
            difficulty_mult = 1.0
        
        op = self.current_operator
        recent_rewards = []
        for i in range(self.num_operators):
            if len(self.operator_rewards[i]) > 0:
                recent_rewards.append((i, np.mean(self.operator_rewards[i][-5:])))
            else:
                recent_rewards.append((i, 0.0))
        
        sorted_by_perf = sorted(recent_rewards, key=lambda x: x[1], reverse=True)
        rank_map = {op_idx: rank + 1 for rank, (op_idx, _) in enumerate(sorted_by_perf)}
        current_rank = rank_map[op]
        
        rank_score = (self.num_operators - current_rank) / max(self.num_operators - 1, 1)
        rank_score = 2.0 * rank_score - 1.0
        
        if improvement > 0:
            improvement_score = np.sqrt(improvement) / (np.sqrt(problem_scale) + 1.0)
            improvement_score = min(float(improvement_score), 5.0)
        else:
            improvement_score = 0.0
        
        if improvement > 1e-15:
            self.operator_streak[op] += 1
        else:
            self.operator_streak[op] = max(0, self.operator_streak[op] - 1)
        
        streak_bonus = np.tanh(self.operator_streak[op] / 10.0) * 0.5
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        diversity_bonus = 0.0
        if hasattr(self, 'prev_diversity'):
            if diversity > self.prev_diversity * 1.1:
                diversity_bonus = 0.3 * (diversity - self.prev_diversity) / (self.prev_diversity + 1e-10)
        self.prev_diversity = diversity
        
        combined_reward = 0.4 * rank_score + 0.4 * improvement_score * difficulty_mult + 0.1 * streak_bonus + 0.1 * diversity_bonus
        combined_reward = float(np.clip(combined_reward, -5.0, 10.0))
        
        self.operator_rank_history[op].append(current_rank)
        if len(self.operator_rank_history[op]) > 20:
            self.operator_rank_history[op].pop(0)
        
        self.operator_rewards[op].append(combined_reward)
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)
        
        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            var_reward = float(np.var(self.operator_rewards[op]))
            avg_rank = np.mean(self.operator_rank_history[op])
            rank_confidence = 1.0 / (1.0 + avg_rank / self.num_operators)
            effective_mean = mean_reward * (0.5 + 0.5 * rank_confidence)
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.alpha[op] = float(max(0.5, effective_mean * (effective_mean * (n - 1) / denom + 1)))
            self.beta[op] = float(max(0.5, (1 - effective_mean) * ((n - 1) * (1 - effective_mean) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = 1.0 + sum_reward
                self.beta[op] = 1.0
            else:
                self.alpha[op] = 1.0
                self.beta[op] = 1.0 - sum_reward + 1.0
    
    def _reward_strategy_v09_difficulty(self):
        """Difficulty-aware scaling with exploration boost."""
        if not hasattr(self, 'operator_cumulative_reward_v9'):
            self.operator_cumulative_reward_v9 = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum_v9'):
            self.operator_decay_sum_v9 = np.zeros(self.num_operators)
        
        decay = 0.9
        self.operator_cumulative_reward_v9 *= decay
        self.operator_decay_sum_v9 *= decay
        
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        if improvement > 0:
            base_reward = float(np.log1p(improvement * 1e10) / 10.0)
        else:
            base_reward = 0.0
        
        task_difficulty = max(1.0, np.log10(max(abs(self.f_opt), 1e-3) + 1.0))
        difficulty_scale = 1.0 + 0.5 * min(task_difficulty, 5.0)
        reward = base_reward * difficulty_scale
        
        is_stagnant = self.stagnation_counter > self.max_stagnation // 2
        if is_stagnant:
            reward *= 1.5
        
        self.operator_cumulative_reward_v9[self.current_operator] += reward
        self.operator_decay_sum_v9[self.current_operator] += 1.0
        
        norm = max(self.operator_decay_sum_v9[self.current_operator], 1.0)
        normalized_reward = self.operator_cumulative_reward_v9[self.current_operator] / norm
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
    
    def _reward_strategy_v10_elite(self):
        """Credit assignment via elite-relative progress (variant_10)."""
        if not hasattr(self, 'operator_cumulative_reward_v10'):
            self.operator_cumulative_reward_v10 = np.zeros(self.num_operators)
        if not hasattr(self, 'operator_decay_sum_v10'):
            self.operator_decay_sum_v10 = np.zeros(self.num_operators)
        
        decay = 0.9
        self.operator_cumulative_reward_v10 *= decay
        self.operator_decay_sum_v10 *= decay
        
        elite = self.f_opt
        trial_array = np.asarray(self.trial_fitness).flatten()
        n_trials = len(trial_array)
        if n_trials > 0:
            beaten = np.sum(trial_array < elite)
            elite_progress = beaten / n_trials
        else:
            elite_progress = 0.0
        
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        if improvement > 0:
            base_reward = float(np.log1p(improvement * 1e10) / 10.0)
        else:
            base_reward = 0.0
        
        reward = 0.7 * elite_progress * 5.0 + 0.3 * base_reward
        
        task_difficulty = max(1.0, np.log10(max(abs(self.f_opt), 1e-3) + 1.0))
        difficulty_scale = 1.0 + 0.5 * min(task_difficulty, 5.0)
        reward *= difficulty_scale
        
        is_stagnant = self.stagnation_counter > self.max_stagnation // 2
        if is_stagnant:
            reward *= 1.5
        
        if improvement > 0:
            reward += 2.0
        
        self.operator_cumulative_reward_v10[self.current_operator] += reward
        self.operator_decay_sum_v10[self.current_operator] += 1.0
        
        norm = max(self.operator_decay_sum_v10[self.current_operator], 1.0)
        normalized_reward = self.operator_cumulative_reward_v10[self.current_operator] / norm
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
    
    # ==================== COVARIANCE ADAPTATION STRATEGIES ====================
    
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