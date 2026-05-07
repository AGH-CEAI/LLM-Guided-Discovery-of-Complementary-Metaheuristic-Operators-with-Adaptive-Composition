```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer with automatic operator selection strategy adaptation.
    Uses a meta-bandit to learn which operator selection strategy works best
    during a run, switching between different Thompson Sampling variants.
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
        
        # ===== META-ADAPTIVE OPERATOR SELECTION STRATEGY =====
        # Track performance of different operator selection strategies
        self.strategy_names = [
            'epsilon_greedy',      # variant_01 style - adaptive exploration
            'softmax_temp',        # variant_07 style - stagnation-based temperature
            'progress_adaptive',   # variant_03 style - difficulty-adaptive
            'entropy_thompson',    # variant_04 style - entropy-driven
            'regime_detection',    # variant_05 style - diversity-guided regimes
            'diversity_boost',     # variant_06 style - entropy-weighted
            'difficulty_aware',    # variant_08 style - task difficulty proxy
            'diversity_bonus',     # variant_09 style - stuck population detection
            'variance_adjusted',   # variant_10 style - Wilson score CI
            'basic_thompson'       # original - basic Thompson Sampling
        ]
        self.num_strategies = len(self.strategy_names)
        
        # Strategy performance tracking (meta-bandit)
        self.strategy_scores = np.zeros(self.num_strategies)
        self.strategy_counts = np.zeros(self.num_strategies)
        self.strategy_rewards = {i: [] for i in range(self.num_strategies)}
        self.strategy_window_size = 15
        
        # Current strategy
        self.current_strategy = np.random.randint(0, self.num_strategies)
        self.strategy_switch_cooldown = 0
        
        # Per-strategy operator state (to allow smooth transitions)
        self._init_strategy_state()
    
    def _init_strategy_state(self):
        """Initialize per-strategy tracking variables."""
        # Epsilon-greedy state
        self.epsilon_greedy_state = {'last_switch': 0}
        
        # Softmax temperature state
        self.softmax_state = {'temperature': 1.0}
        
        # Progress adaptive state
        self.progress_state = {'history': [], 'window_size': 5}
        
        # Entropy Thompson state
        self.entropy_state = {'last_selected_gen': np.zeros(self.num_operators, dtype=int)}
        
        # Regime detection state
        self.regime_state = {'regime_history': []}
        
        # Diversity boost state
        self.diversity_state = {'exploration_scale': 1.5}
        
        # Difficulty aware state
        self.difficulty_state = {'op_reward_buffer': [[] for _ in range(self.num_operators)]}
        
        # Diversity bonus state
        self.diversity_bonus_state = {
            'operator_last_used': {},
            'operator_diversity_scores': np.ones(self.num_operators)
        }
        
        # Variance adjusted state
        self.variance_state = {'op_last_selected': np.zeros(self.num_operators, dtype=int)}
    
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
            
            # Update strategy performance tracking
            self._update_strategy_tracking()
            
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
        """Select operator using the current adaptive strategy."""
        strategy_funcs = [
            self._strategy_epsilon_greedy,
            self._strategy_softmax_temp,
            self._strategy_progress_adaptive,
            self._strategy_entropy_thompson,
            self._strategy_regime_detection,
            self._strategy_diversity_boost,
            self._strategy_difficulty_aware,
            self._strategy_diversity_bonus,
            self._strategy_variance_adjusted,
            self._strategy_basic_thompson
        ]
        
        # Execute current strategy
        self.current_operator = strategy_funcs[self.current_strategy]()
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _update_strategy_tracking(self):
        """Track performance of current strategy and adapt if needed."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        if improvement > 1e-15:
            reward = float(np.log1p(improvement * 1e10) / 10.0)
        else:
            is_stagnant = self.stagnation_counter > self.max_stagnation // 2
            reward = -0.5 if is_stagnant else 0.0
        
        strategy_id = self.current_strategy
        self.strategy_rewards[strategy_id].append(reward)
        
        if len(self.strategy_rewards[strategy_id]) > self.strategy_window_size:
            self.strategy_rewards[strategy_id].pop(0)
        
        # Update strategy score using exponential moving average
        if len(self.strategy_rewards[strategy_id]) >= 3:
            recent_avg = float(np.mean(self.strategy_rewards[strategy_id][-3:]))
            decay = 0.9
            self.strategy_scores[strategy_id] = decay * self.strategy_scores[strategy_id] + (1.0 - decay) * recent_avg
            self.strategy_counts[strategy_id] += 1
        
        # Adaptive strategy switching
        self.strategy_switch_cooldown = max(0, self.strategy_switch_cooldown - 1)
        
        if self.strategy_switch_cooldown == 0:
            is_stuck = self.stagnation_counter > self.max_stagnation // 3
            
            # Check if current strategy is underperforming
            current_score = self.strategy_scores[self.current_strategy]
            best_other_score = float(np.max(np.delete(self.strategy_scores, self.current_strategy)))
            
            should_switch = False
            
            if is_stuck and current_score < best_other_score - 0.1:
                # Stuck and current strategy not best - switch
                should_switch = True
            elif self.strategy_counts[self.current_strategy] > self.dim * 2 and current_score < -0.3:
                # Strategy has been used extensively but consistently negative - switch
                should_switch = True
            
            if should_switch:
                # Select best performing strategy
                scores = self.strategy_scores.copy()
                
                # Add small random noise to prevent lock-in
                noise = np.random.uniform(-0.05, 0.05, self.num_strategies)
                scores += noise
                
                # Prefer strategies with some track record
                for i in range(self.num_strategies):
                    if self.strategy_counts[i] < 3:
                        scores[i] -= 0.5
                
                new_strategy = int(np.argmax(scores))
                
                if new_strategy != self.current_strategy:
                    self.current_strategy = new_strategy
                    self.strategy_switch_cooldown = max(5, self.dim // 2)
    
    # ===== OPERATOR SELECTION STRATEGIES =====
    
    def _strategy_epsilon_greedy(self):
        """Adaptive Epsilon-Greedy strategy (variant_01 style)."""
        operator_avg = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if len(self.operator_rewards[i]) > 0:
                operator_avg[i] = float(np.mean(self.operator_rewards[i]))

        total_selections = float(np.sum(self.operator_counts))
        epsilon = 1.0 / (1.0 + total_selections / max(self.dim, 10.0))
        epsilon = float(np.clip(epsilon, 0.01, 0.5))

        if np.random.random() < epsilon:
            selection_probs = 1.0 / (self.operator_counts + 1.0)
            selection_probs /= np.sum(selection_probs)
            selection_probs = np.clip(selection_probs, 1e-10, 1.0 - 1e-10)
            op = int(np.random.choice(self.num_operators, p=selection_probs))
        else:
            op = int(np.argmax(operator_avg))

        return op
    
    def _strategy_softmax_temp(self):
        """Adaptive softmax with stagnation-driven temperature (variant_07 style)."""
        qualities = np.zeros(self.num_operators)
        for op in range(self.num_operators):
            if len(self.operator_rewards[op]) > 0:
                qualities[op] = float(np.mean(self.operator_rewards[op]))
            else:
                qualities[op] = -5.0

        stagnation_ratio = float(self.stagnation_counter) / max(float(self.max_stagnation), 1.0)
        stagnation_ratio = np.clip(stagnation_ratio, 0.0, 1.0)
        temp = 0.01 + 4.99 * stagnation_ratio
        temp = float(np.clip(temp, 0.01, 5.0))

        max_q = float(np.max(qualities))
        exp_q = np.zeros(self.num_operators)
        for op in range(self.num_operators):
            exp_q[op] = float(np.exp(np.clip(qualities[op] - max_q, -700.0, 700.0)))

        sum_exp = float(np.sum(exp_q))
        if sum_exp <= 0.0 or np.isnan(sum_exp) or np.isinf(sum_exp):
            probs = np.ones(self.num_operators) / self.num_operators
        else:
            probs = exp_q / sum_exp

        probs = np.clip(probs, 1e-10, 1.0 - 1e-10)
        probs /= np.sum(probs)

        return int(np.random.choice(self.num_operators, p=probs))
    
    def _strategy_progress_adaptive(self):
        """Difficulty-adaptive with progress tracking (variant_03 style)."""
        current_improvement = max(0.0, self.f_opt_prev - self.f_opt)
        self.progress_state['history'].append(current_improvement)

        window_size = self.progress_state['window_size'] * self.dim
        if len(self.progress_state['history']) > max(10, window_size):
            self.progress_state['history'].pop(0)

        if len(self.progress_state['history']) >= 3:
            recent = self.progress_state['history'][-3:]
            avg_progress = np.mean(recent)
        else:
            avg_progress = current_improvement

        is_stagnant = (avg_progress < 1e-12 * max(abs(self.f_opt), 1.0)) and len(self.progress_state['history']) >= self.dim

        quality_scores = np.zeros(self.num_operators)
        for op in range(self.num_operators):
            if len(self.operator_rewards[op]) > 0:
                quality_scores[op] = np.mean(self.operator_rewards[op])

        if is_stagnant:
            if np.sum(self.operator_counts) > self.dim * 3:
                inv_counts = 1.0 / (self.operator_counts + 1.0)
                op = int(np.argmax(inv_counts))
            else:
                op = int(np.random.randint(0, self.num_operators))
        else:
            if np.max(quality_scores) > 0:
                op = int(np.argmax(quality_scores))
            else:
                op = int(np.argmin(self.operator_counts))

        return op
    
    def _strategy_entropy_thompson(self):
        """Entropy-driven Thompson Sampling (variant_04 style)."""
        entropy = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            a, b = max(self.alpha[i], 1e-10), max(self.beta[i], 1e-10)
            total = a + b
            e = np.log1p(total) - (a * np.log1p(a) + b * np.log1p(b)) / total
            entropy[i] = max(e, 0.0)

        total_entropy = np.sum(entropy)
        if total_entropy > 1e-10:
            explore_weights = entropy / total_entropy
        else:
            explore_weights = np.ones(self.num_operators) / self.num_operators

        total_counts = np.sum(self.operator_counts) + 1e-10
        count_penalty = 1.0 - (self.operator_counts / total_counts)
        count_penalty = np.clip(count_penalty, 0.0, 1.0)

        samples = np.random.beta(np.maximum(self.alpha, 1e-10), np.maximum(self.beta, 1e-10))
        combined_score = (
            0.4 * samples / (np.max(samples) + 1e-10) +
            0.4 * explore_weights / (np.max(explore_weights) + 1e-10) +
            0.2 * count_penalty / (np.max(count_penalty) + 1e-10)
        )

        gens_since_selection = self.generation - self.entropy_state['last_selected_gen']
        forced_threshold = max(20, self.dim * 2)
        for i in range(self.num_operators):
            if gens_since_selection[i] > forced_threshold:
                combined_score[i] *= 3.0
                if gens_since_selection[i] > forced_threshold * 2:
                    combined_score[i] *= 2.0

        op = int(np.argmax(combined_score))
        self.entropy_state['last_selected_gen'][op] = self.generation
        return op
    
    def _strategy_regime_detection(self):
        """Diversity-guided regime detection (variant_05 style)."""
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min, eig_max = np.min(eigvals), np.max(eigvals)
        cond = eig_max / (max(eig_min, 1e-10))
        eig_spread = eig_min / (eig_max + 1e-10)

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        is_severe_trap = (diversity < 0.05) or (cond > 1e6) or (eig_spread < 1e-6)
        is_converging = (rel_var < 0.001) or (diversity < 0.1)
        is_normal = not (is_severe_trap or is_converging)

        if is_severe_trap:
            preferred = 1
        elif is_converging:
            preferred = np.random.choice([2, 3])
        else:
            preferred = None

        if preferred is not None:
            op = int(preferred)
        else:
            regime_prior = np.ones(self.num_operators)
            regime_prior[1] = 2.0
            combined_alpha = self.alpha * regime_prior
            samples = np.random.beta(combined_alpha, self.beta)
            op = int(np.argmax(samples))

        return op
    
    def _strategy_diversity_boost(self):
        """Entropy-weighted Thompson Sampling (variant_06 style)."""
        total_reward = np.sum(self.alpha) + np.sum(self.beta)
        if total_reward > self.num_operators * 2:
            alpha_sum = np.sum(self.alpha)
            probs = self.alpha / (alpha_sum + 1e-10)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(self.num_operators)
            entropy_ratio = entropy / (max_entropy + 1e-10)
            exploration_scale = 0.5 + 2.0 * (1.0 - entropy_ratio)
            exploration_scale = np.clip(exploration_scale, 0.5, 3.0)
        else:
            exploration_scale = 1.5

        samples = np.random.beta(self.alpha * exploration_scale, self.beta * exploration_scale)

        if np.sum(self.selection_counts) > self.num_operators * 3:
            min_selections = np.min(self.selection_counts)
            selection_deficit = min_selections / (self.selection_counts + 1e-10)
            diversity_boost = 0.1 * selection_deficit
            samples = samples + diversity_boost * np.random.random(self.num_operators)

        op = int(np.argmax(samples))
        self.selection_counts[op] += 1
        return op
    
    def _strategy_difficulty_aware(self):
        """Task-difficulty-aware Thompson Sampling (variant_08 style)."""
        current_error = abs(self.f_opt)
        log_error = np.log10(max(current_error, 1e-15))
        target_log = -8.0
        error_gap = max(0.0, log_error - target_log)

        is_stagnant = self.stagnation_counter > self.max_stagnation // 3

        if error_gap > 8.0:
            difficulty = 0.0
        elif error_gap > 4.0:
            difficulty = 0.25
        elif error_gap > 1.0:
            difficulty = 0.5
        else:
            difficulty = 0.75

        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        if improvement > 0:
            base_reward = np.log1p(improvement * 1e10) / 10.0
        else:
            base_reward = -0.5 if is_stagnant else 0.0

        if difficulty < 0.5:
            exploration_bonus = 2.0 * (0.5 - difficulty)
            base_reward += exploration_bonus

        op_count = max(self.operator_counts[self.current_operator], 1)
        normalized_reward = base_reward / np.sqrt(op_count)
        normalized_reward = np.clip(normalized_reward, -5.0, 5.0)

        self.difficulty_state['op_reward_buffer'][self.current_operator].append(normalized_reward)
        buffer_size = 15
        if len(self.difficulty_state['op_reward_buffer'][self.current_operator]) > buffer_size:
            self.difficulty_state['op_reward_buffer'][self.current_operator].pop(0)

        op_means = np.zeros(self.num_operators)
        op_vars = np.zeros(self.num_operators)
        op_counts_snapshot = self.operator_counts.copy()

        for op in range(self.num_operators):
            if len(self.difficulty_state['op_reward_buffer'][op]) >= 3:
                op_means[op] = np.mean(self.difficulty_state['op_reward_buffer'][op])
                op_vars[op] = np.var(self.difficulty_state['op_reward_buffer'][op]) + 1e-10
            else:
                op_means[op] = 0.0
                op_vars[op] = 1.0

        alpha_samples = np.zeros(self.num_operators)
        beta_samples = np.zeros(self.num_operators)

        for op in range(self.num_operators):
            count = max(op_counts_snapshot[op], 1)
            confidence = 1.0 / np.sqrt(count)
            alpha_base = max(1.0, 1.0 + op_means[op] * 5.0)
            beta_base = max(1.0, 1.0 - op_means[op] * 5.0)
            alpha_param = alpha_base * (1.0 + confidence)
            beta_param = beta_base * (1.0 + confidence)
            alpha_samples[op] = alpha_param
            beta_samples[op] = beta_param

        samples = np.array([np.random.beta(alpha_samples[op], beta_samples[op]) 
                            for op in range(self.num_operators)])

        if difficulty < 0.5 or is_stagnant:
            explore_prob = 0.3 if difficulty < 0.25 else 0.15
            if np.random.random() < explore_prob:
                samples = np.random.uniform(size=self.num_operators)

        return int(np.argmax(samples))
    
    def _strategy_diversity_bonus(self):
        """Thompson Sampling with diversity bonus (variant_09 style)."""
        if not hasattr(self, 'operator_diversity_scores'):
            self.operator_diversity_scores = np.ones(self.num_operators)

        expected_random_improvement = max(1e-10, 0.1 * (self.ub[0] - self.lb[0]) / self.sigma)
        success_rates = np.zeros(self.num_operators)
        
        for i in range(self.num_operators):
            if self.operator_counts[i] > 0 and hasattr(self, 'operator_cumulative_reward'):
                reward_sum = self.operator_cumulative_reward[i] / max(getattr(self, 'operator_decay_sum', np.ones(self.num_operators))[i], 1.0)
                success_rates[i] = reward_sum / expected_random_improvement

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)

        time_since_used = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            time_since_used[i] = self.generation - self.diversity_bonus_state['operator_last_used'].get(i, 0)

        is_stuck = (diversity < 0.05) or (self.stagnation_counter > self.max_stagnation // 4)
        if is_stuck:
            diversity_bonus = time_since_used / max(np.max(time_since_used), 1.0)
            diversity_bonus = np.clip(diversity_bonus, 0.0, 3.0)
        else:
            diversity_bonus = np.zeros(self.num_operators)

        alpha_posterior = np.ones(self.num_operators)
        beta_posterior = np.ones(self.num_operators)

        for i in range(self.num_operators):
            sr = np.clip(success_rates[i], 0.0, 1.0)
            base_alpha = 1.0 + 10.0 * sr
            base_beta = 1.0 + 10.0 * (1.0 - sr)
            alpha_posterior[i] = max(1.0, base_alpha + diversity_bonus[i] * 5.0)
            beta_posterior[i] = max(1.0, base_beta - diversity_bonus[i] * 2.0)

        samples = np.random.beta(alpha_posterior, beta_posterior)

        self.diversity_bonus_state['operator_last_used'][self.current_operator] = self.generation

        op = int(np.argmax(samples))
        self.operator_counts[op] += 1

        self.operator_diversity_scores *= 0.95
        self.operator_diversity_scores[op] += 0.05 * diversity

        return op
    
    def _strategy_variance_adjusted(self):
        """Variance-adjusted Thompson Sampling (variant_10 style)."""
        n_ops = self.num_operators
        mean_rewards = np.zeros(n_ops)
        var_rewards = np.zeros(n_ops)
        valid_counts = np.zeros(n_ops)

        for i in range(n_ops):
            rewards = self.operator_rewards[i]
            if len(rewards) >= 3:
                mean_rewards[i] = np.mean(rewards)
                var_rewards[i] = np.var(rewards)
                valid_counts[i] = len(rewards)
            elif len(rewards) >= 1:
                mean_rewards[i] = np.mean(rewards)
                var_rewards[i] = 1.0
                valid_counts[i] = len(rewards)
            else:
                mean_rewards[i] = 0.0
                var_rewards[i] = 1.0
                valid_counts[i] = 0.0

        wilson_scores = np.zeros(n_ops)
        z = 1.96
        for i in range(n_ops):
            if valid_counts[i] >= 3:
                p = np.clip(mean_rewards[i], 0.0, 1.0)
                n = valid_counts[i]
                denominator = 1.0 + z**2 / n
                center = (p + z**2 / (2*n)) / denominator
                margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
                wilson_scores[i] = center + margin
            elif valid_counts[i] >= 1:
                log_reward = np.log1p(max(mean_rewards[i], 1e-10))
                uncertainty = 2.0 / max(valid_counts[i], 1.0)
                wilson_scores[i] = log_reward + uncertainty
            else:
                wilson_scores[i] = 2.0 + np.random.uniform(0, 0.5)

        total_samples = np.sum(self.operator_counts) + 1.0
        base_exploration = min(2.0, 20.0 / np.sqrt(total_samples))

        max_var = max(np.max(var_rewards), 1e-10)
        variance_bonus = 1.5 * (var_rewards / (max_var + 1e-10))

        inv_counts = 1.0 / (valid_counts + 1.0)
        inv_counts_normalized = inv_counts / (np.sum(inv_counts) + 1e-10)

        exploration_bonus = base_exploration * inv_counts_normalized + 0.5 * variance_bonus

        log_rewards = np.log1p(np.clip(mean_rewards, 0.0, 10.0))
        scores = log_rewards + exploration_bonus

        temperature = max(0.1, 1.0 / (1.0 + np.mean(var_rewards) + 1e-10))
        exp_scores = np.exp((scores - np.max(scores)) / temperature)
        probs = exp_scores / (np.sum(exp_scores) + 1e-10)

        return int(np.random.choice(n_ops, p=probs))
    
    def _strategy_basic_thompson(self):
        """Basic Thompson Sampling (original)."""
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))
    
    def _update_operator_rewards(self):
        """Credit assignment via exponentially-weighted cumulative improvement."""
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
        """Archive-guided covariance perturbation."""
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
        """Active CMA-ES."""
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

        if pos_sum > 0:
            rank_mu_pos /= pos_sum
        if neg_sum > 0:
            rank_mu_neg /= neg_sum

        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * rank_one +
                  (1.0 - 1.0 / self.mueff) * self.ccov * rank_mu_pos -
                  self.ccov * 0.4 * rank_mu_neg)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_09(self):
        """Exploration temperature with fitness gradient tracking."""
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
    
    def _clip_to_bounds(self, x):
        """Clip