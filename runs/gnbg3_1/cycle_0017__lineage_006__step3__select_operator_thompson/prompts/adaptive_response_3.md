```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using meta-adaptive operator selection.
    
    Features:
    - 5 covariance adaptation strategies (original + 4 variants)
    - Meta-adaptive Thompson Sampling for operator selection
    - Sliding window credit assignment
    - Automatic restart on stagnation
    - Dynamic strategy switching based on performance
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
        
        # Meta-adaptive selection parameters
        self.num_selector_strategies = 9
        self.selector_strategy_names = [
            'thompson_beta', 'ucb_optimistic', 'context_aware', 
            'long_term_memory', 'entropy_regularized', 'variance_weighted',
            'round_robin', 'empirical_weighted', 'softmax_best'
        ]
        
        # Meta-selector state (Thompson Sampling for selecting selector strategy)
        self.meta_alpha = np.ones(self.num_selector_strategies)
        self.meta_beta = np.ones(self.num_selector_strategies)
        self.meta_rewards = {i: [] for i in range(self.num_selector_strategies)}
        self.meta_counts = np.zeros(self.num_selector_strategies)
        self.meta_window_size = 15
        
        # Current selector strategy
        self.current_selector_strategy = 0
        
        # Per-generation reward tracking for meta-selector
        self.generation_rewards = []
        self.generation_strategies = []
        
        # Context tracking for context-aware selector
        self.context_history = {op: [] for op in range(self.num_selector_strategies)}
        self.context_rewards_meta = {op: [] for op in range(self.num_selector_strategies)}
        
        # Long-term reward tracking for long-term memory selector
        self.lt_alpha_local = np.ones(self.num_selector_strategies)
        self.lt_beta_local = np.ones(self.num_selector_strategies)
        self.lt_reward_sum = np.zeros(self.num_selector_strategies)
        self.lt_reward_count = np.zeros(self.num_selector_strategies)
    
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
            
            # Update meta-selector based on overall performance
            self._update_meta_selector()
            
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
        """Select operator using meta-adaptive strategy selection."""
        # First, select which selector strategy to use
        self._select_meta_strategy()
        
        # Then apply the selected strategy
        if self.current_selector_strategy == 0:
            self._select_operator_thompson_beta()
        elif self.current_selector_strategy == 1:
            self._select_operator_ucb_optimistic()
        elif self.current_selector_strategy == 2:
            self._select_operator_context_aware()
        elif self.current_selector_strategy == 3:
            self._select_operator_long_term_memory()
        elif self.current_selector_strategy == 4:
            self._select_operator_entropy_regularized()
        elif self.current_selector_strategy == 5:
            self._select_operator_variance_weighted()
        elif self.current_selector_strategy == 6:
            self._select_operator_round_robin()
        elif self.current_selector_strategy == 7:
            self._select_operator_empirical_weighted()
        else:
            self._select_operator_softmax_best()
        
        return self.current_operator
    
    def _select_meta_strategy(self):
        """Select meta-strategy using Thompson Sampling with adaptive parameters."""
        # Use Thompson Sampling from Beta distributions with adaptive mixing
        total_selections = float(np.sum(self.meta_counts))
        
        # Adaptive mixing: more exploration early, exploitation later
        exploration_rate = 0.3 / (1.0 + 0.05 * total_selections)
        
        # Mix between Thompson Sampling and UCB-style selection
        if np.random.random() < exploration_rate:
            # Random exploration
            self.current_selector_strategy = int(np.random.randint(0, self.num_selector_strategies))
        else:
            # Thompson Sampling from Beta distributions
            samples = np.random.beta(self.meta_alpha, self.meta_beta)
            self.current_selector_strategy = int(np.argmax(samples))
        
        self.meta_counts[self.current_selector_strategy] += 1
    
    def _update_meta_selector(self):
        """Update meta-selector rewards based on overall improvement."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
        # Log-scaled reward (handles both large and small improvements)
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity)
        reward = float(np.clip(reward, -10.0, 10.0))
        
        # Store reward for current selector strategy
        strategy = self.current_selector_strategy
        self.meta_rewards[strategy].append(reward)
        
        # Maintain window size
        if len(self.meta_rewards[strategy]) > self.meta_window_size:
            self.meta_rewards[strategy].pop(0)
        
        # Update Beta parameters using empirical Bayes
        n = len(self.meta_rewards[strategy])
        if n >= 3:
            mean_reward = float(np.mean(self.meta_rewards[strategy]))
            var_reward = float(np.var(self.meta_rewards[strategy]))
            
            mean_reward = float(np.clip(mean_reward, 0.01, 0.99))
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.meta_alpha[strategy] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.meta_beta[strategy] = float(max(1.0, (1 - mean_reward) * ((n - 1) * (1 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.meta_rewards[strategy]))
            if sum_reward > 0:
                self.meta_alpha[strategy] = 1.0 + max(sum_reward, 0.0)
                self.meta_beta[strategy] = 1.0
            else:
                self.meta_alpha[strategy] = 1.0
                self.meta_beta[strategy] = 1.0 - min(sum_reward, 0.0)
        
        # Small reward for all strategies to encourage exploration
        for i in range(self.num_selector_strategies):
            if i != strategy and len(self.meta_rewards[i]) > 0:
                self.meta_rewards[i].append(0.01)
                if len(self.meta_rewards[i]) > self.meta_window_size:
                    self.meta_rewards[i].pop(0)
    
    # ============== Selector Strategy Implementations ==============
    
    def _select_operator_thompson_beta(self):
        """Standard Thompson Sampling with Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _select_operator_ucb_optimistic(self):
        """UCB with optimistic initialization."""
        total_selections = np.sum(self.selection_counts)

        if total_selections == 0:
            self.current_operator = int(np.random.randint(0, self.num_operators))
        else:
            mean_rewards = np.zeros(self.num_operators)
            valid_ops = np.zeros(self.num_operators, dtype=bool)

            for i in range(self.num_operators):
                if len(self.operator_rewards[i]) > 0:
                    mean_rewards[i] = np.mean(self.operator_rewards[i])
                    valid_ops[i] = True

            exploration = np.zeros(self.num_operators)
            nonzero_mask = self.selection_counts > 0
            exploration[nonzero_mask] = np.sqrt(
                2.0 * np.log(max(total_selections, 1)) / np.maximum(self.selection_counts[nonzero_mask], 1)
            )

            exploration[~nonzero_mask] = np.sqrt(2.0 * np.log(max(total_selections, 1)) + 1)

            ucb_scores = mean_rewards + exploration

            if np.any(~valid_ops):
                unvisited = np.where(~valid_ops)[0]
                ucb_scores[unvisited] += np.max(ucb_scores[valid_ops]) + 1.0

            self.current_operator = int(np.argmax(ucb_scores))

        self.selection_counts[self.current_operator] += 1
        return self.current_operator
    
    def _select_operator_context_aware(self):
        """Context-aware Thompson Sampling using problem structure features."""
        fitness_scale = max(abs(self.f_opt), 1.0)
        fitness_scale_log = np.log1p(fitness_scale)

        pc_norm = float(np.linalg.norm(self.pc))

        eigvals = np.linalg.eigvalsh(self.C)
        cond = float(eigvals[-1] / max(float(eigvals[0]), 1e-10))
        cond_log = np.log1p(cond)

        pop_diversity = float(np.mean(np.std(self.population, axis=0)))
        expected_range = float(self.ub[0] - self.lb[0])
        diversity_ratio = float(np.clip(pop_diversity / (expected_range + 1e-10), 0.0, 1.0))

        improvement = float(getattr(self, 'improvement_ema', 0.0))

        gen = float(self.generation)
        phase = float(np.clip(gen / max(1.0, float(self.max_stagnation)), 0.0, 1.0))

        meta_features = np.array([fitness_scale_log, pc_norm, cond_log, diversity_ratio, improvement, phase])

        operator_context_scores = np.zeros(self.num_operators)

        for op in range(self.num_operators):
            if len(self.context_history.get(op, [])) >= 3:
                contexts = np.array(self.context_history[op])
                rewards = np.array(self.context_rewards_meta.get(op, []))

                diffs = contexts - meta_features
                feature_ranges = np.array([5.0, 10.0, 10.0, 0.5, 5.0, 1.0])
                similarities = np.exp(-np.sum((diffs / (feature_ranges + 1e-10))**2, axis=1) * 0.5)

                n = len(rewards)
                recency_weights = np.array([0.5 ** ((n - 1 - i) / 5.0) for i in range(n)])

                weighted_rewards = similarities * recency_weights * np.maximum(rewards, 0.0)
                operator_context_scores[op] = float(np.sum(weighted_rewards))

        context_boost = np.zeros(self.num_operators)
        max_score = float(np.max(operator_context_scores))
        if max_score > 1e-10:
            context_boost = operator_context_scores / (max_score + 1e-10) * 2.0

        boosted_alpha = self.alpha + context_boost
        boosted_beta = self.beta.copy()

        samples = np.random.beta(boosted_alpha, boosted_beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1

        op = self.current_operator
        reward = float(max(0.0, getattr(self, 'f_opt_prev', self.f_opt) - self.f_opt))

        if op not in self.context_history:
            self.context_history[op] = []
            self.context_rewards_meta[op] = []
        self.context_history[op].append(meta_features)
        self.context_rewards_meta[op].append(reward)

        max_contexts = 50
        if len(self.context_history[op]) > max_contexts:
            self.context_history[op] = self.context_history[op][-max_contexts:]
            self.context_rewards_meta[op] = self.context_rewards_meta[op][-max_contexts:]

        return self.current_operator
    
    def _select_operator_long_term_memory(self):
        """Thompson Sampling with long-term memory and entropy-promoting exploration."""
        n_ops = self.num_operators

        if not hasattr(self, 'lt_alpha'):
            self.lt_alpha = np.ones(n_ops)
            self.lt_beta = np.ones(n_ops)
            self.lt_reward_sum = np.zeros(n_ops)
            self.lt_reward_count = np.zeros(n_ops)

        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.15 * diversity)
        reward = float(np.clip(reward, -5.0, 5.0))

        op = self.current_operator
        self.lt_reward_sum[op] += reward
        self.lt_reward_count[op] += 1.0

        epsilon = 1e-6
        for i in range(n_ops):
            if self.lt_reward_count[i] >= 3.0:
                mean_r = self.lt_reward_sum[i] / max(self.lt_reward_count[i], 1.0)
                var_r = 1.0 / max(self.lt_reward_count[i], 1.0)
                mean_r = np.clip(mean_r, epsilon, 1.0 - epsilon)

                denom = max(var_r * self.lt_reward_count[i] + epsilon, epsilon)
                shape1 = max(mean_r * (mean_r * (self.lt_reward_count[i] - 1.0) / denom + 1.0), 1.0)
                shape2 = max((1.0 - mean_r) * ((self.lt_reward_count[i] - 1.0) * (1.0 - mean_r) / denom + 1.0), 1.0)

                self.lt_alpha[i] = float(shape1)
                self.lt_beta[i] = float(shape2)
            elif self.lt_reward_count[i] >= 1.0:
                total = float(np.sum(self.lt_reward_sum))
                if total > 0:
                    self.lt_alpha[i] = 1.0 + max(self.lt_reward_sum[i], 0.0)
                    self.lt_beta[i] = 1.0 + max(-self.lt_reward_sum[i], 0.0)
                else:
                    self.lt_alpha[i] = 1.0
                    self.lt_beta[i] = 1.5

        exploration_bonus = np.zeros(n_ops)
        total_selections = float(np.sum(self.operator_counts) + 1.0)
        for i in range(n_ops):
            sel_frac = self.operator_counts[i] / total_selections
            exploration_bonus[i] = 2.0 * np.exp(-5.0 * sel_frac)

        samples = np.random.beta(self.lt_alpha, self.beta)
        samples += exploration_bonus

        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _select_operator_entropy_regularized(self):
        """Thompson Sampling with entropy-regularized exploration bonus."""
        log_rewards = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if len(self.operator_rewards[i]) >= 3:
                log_rewards[i] = np.log1p(max(np.mean(self.operator_rewards[i]), 1e-15))

        total_selections = np.sum(self.operator_counts) + 1.0
        entropy_bonus = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            p_select = (self.operator_counts[i] + 0.1) / (total_selections + 0.5 * self.num_operators)
            p_select = np.clip(p_select, 1e-6, 1.0)
            entropy_bonus[i] = -np.log(p_select + 1e-10) / max(np.log(total_selections + 1), 1.0)

        scores = log_rewards + 0.5 * entropy_bonus

        scores_shifted = scores - np.max(scores)
        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / (np.sum(exp_scores) + 1e-10)
        probs = np.clip(probs, 1e-6, 1.0)
        probs /= np.sum(probs)

        self.current_operator = int(np.random.choice(self.num_operators, p=probs))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _select_operator_variance_weighted(self):
        """Select operator using variance-weighted adaptive selection."""
        rewards = np.array([np.mean(r) if len(r) > 0 else 0.0 for r in self.operator_rewards.values()])
        reward_vars = np.array([np.var(r) if len(r) > 1 else 1.0 for r in self.operator_rewards.values()])

        reward_vars = np.clip(reward_vars, 1e-6, None)

        best_reward = np.max(rewards)
        temp = 5.0 if best_reward < 1.0 else max(0.5, 2.0 - best_reward)

        var_bonus = 0.5 * np.sqrt(reward_vars)

        scores = rewards + var_bonus

        scores_shifted = scores - np.max(scores)
        exp_scores = np.exp(scores_shifted / temp)
        probs = exp_scores / np.sum(exp_scores)

        probs = np.clip(probs, 1e-10, 1.0)
        probs /= np.sum(probs)

        if np.max(probs) > 0.8:
            probs = 0.7 * probs + 0.3 / self.num_operators

        self.current_operator = int(np.random.choice(self.num_operators, p=probs))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _select_operator_round_robin(self):
        """Select operator using deterministic round-robin with periodic random perturbation."""
        cycle_position = self.generation % self.num_operators

        if hasattr(self, 'total_selections'):
            self.total_selections += 1
        else:
            self.total_selections = 0

        if np.random.random() < 0.1:
            self.current_operator = int(np.random.randint(0, self.num_operators))
        else:
            self.current_operator = int(cycle_position)

        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _select_operator_empirical_weighted(self):
        """Select operator using empirical performance-weighted sampling."""
        num_ops = self.num_operators

        scores = np.zeros(num_ops)
        for op in range(num_ops):
            rewards = self.operator_rewards[op]
            if len(rewards) >= 3:
                mean_r = np.mean(rewards)
                std_r = np.std(rewards) + 1e-10
                consistency = 1.0 / (1.0 + std_r)
                scores[op] = mean_r * consistency
            elif len(rewards) >= 1:
                scores[op] = np.mean(rewards) * 0.5
            else:
                scores[op] = 0.0

        total_counts = np.sum(self.operator_counts) + 1e-10
        exploitation_bonus = 0.3 * (self.operator_counts / total_counts)
        scores = scores + exploitation_bonus

        scores_shifted = scores - np.min(scores) + 0.1
        probs = scores_shifted / np.sum(scores_shifted)

        cumsum = np.cumsum(probs)
        r = np.random.random()

        selected = 0
        for op in range(num_ops):
            if r <= cumsum[op]:
                selected = op
                break

        self.current_operator = int(selected)
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _select_operator_softmax_best(self):
        """Select operator using softmax over absolute best-fitness per operator."""
        if not hasattr(self, 'operator_best_fitness'):
            self.operator_best_fitness = np.full(self.num_operators, np.inf)

        op = self.current_operator
        if self.f_opt < self.operator_best_fitness[op]:
            self.operator_best_fitness[op] = self.f_opt

        neg_best = -self.operator_best_fitness

        noise = np.random.uniform(0, 1e-10, self.num_operators)
        neg_best = neg_best + noise

        stagnation_ratio = self.stagnation_counter / max(self.max_stagnation, 1)
        temperature = max(0.01, 1.0 - 0.9 * stagnation_ratio)

        neg_best_shifted = neg_best - np.max(neg_best)
        exp_scores = np.exp(neg_best_shifted / temperature)
        probs = exp_scores / np.sum(exp_scores)

        cumsum = np.cumsum(probs)
        r = np.random.random()
        self.current_operator = int(np.searchsorted(cumsum, r))

        self.operator_counts[self.current_operator] += 1

        return self.current_operator
    
    # ============== End Selector Strategies ==============
    
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