```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using a meta-bandit approach with multiple
    Thompson Sampling strategies.
    
    Features:
    - 5 covariance adaptation strategies (original + 4 variants)
    - Meta-bandit for adaptive operator selection strategy
    - 5 different operator selection strategies (Thompson Sampling variants)
    - Automatic strategy switching based on problem phase
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
        
        # Meta-bandit for adaptive strategy selection
        # 5 selection strategies: basic, UCB, context-aware, entropy, best-fitness
        self.num_selection_strategies = 5
    
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
        """Adaptive operator selection combining multiple strategies via meta-bandit."""
        # Initialize meta-bandit if needed
        if not hasattr(self, 'meta_alpha'):
            self.meta_alpha = np.ones(self.num_selection_strategies)
            self.meta_beta = np.ones(self.num_selection_strategies)
            self.meta_rewards = {i: [] for i in range(self.num_selection_strategies)}
            self.meta_counts = np.zeros(self.num_selection_strategies)
            self.meta_window = 15
            self.meta_selection_history = []
            self.strategy_switches = 0
            self.last_strategy = -1
        
        # Context features for meta-strategy selection
        fitness_scale_log = float(np.log1p(max(abs(self.f_opt), 1.0)))
        eigvals = np.linalg.eigvalsh(self.C)
        cond = float(np.max(eigvals) / max(float(np.min(eigvals)), 1e-10))
        cond_log = float(np.log1p(cond))
        pop_diversity = float(np.mean(np.std(self.population, axis=0)))
        expected_range = float(self.ub[0] - self.lb[0])
        diversity_ratio = float(np.clip(pop_diversity / (expected_range + 1e-10), 0.0, 1.0))
        improvement = float(getattr(self, 'improvement_ema', 0.0))
        stagnation_ratio = float(self.stagnation_counter / max(self.max_stagnation, 1))
        gen = float(self.generation)
        
        # Compute context-weighted scores for each meta-strategy
        meta_scores = np.zeros(self.num_selection_strategies)
        for s in range(self.num_selection_strategies):
            if len(self.meta_rewards[s]) >= 2:
                recent_rewards = self.meta_rewards[s][-self.meta_window:]
                mean_r = float(np.mean(recent_rewards))
                std_r = float(np.std(recent_rewards)) + 1e-10
                consistency = 1.0 / (1.0 + std_r)
                meta_scores[s] = mean_r * consistency
        
        # Thompson sampling from meta-level Beta distributions
        meta_samples = np.random.beta(self.meta_alpha, self.meta_beta)
        
        # Boost under-explored strategies
        total_meta = float(np.sum(self.meta_counts) + 1e-10)
        for s in range(self.num_selection_strategies):
            sel_frac = self.meta_counts[s] / total_meta
            exploration_bonus = 1.5 * np.exp(-3.0 * sel_frac)
            meta_samples[s] += exploration_bonus
        
        # Context-based adjustment: prefer strategies suited for current phase
        if stagnation_ratio > 0.5:
            # Stagnation: prefer exploration strategies
            meta_samples[1] *= 1.3  # UCB
            meta_samples[2] *= 1.2  # Context-aware
        elif diversity_ratio < 0.1:
            # Low diversity: prefer diversity-promoting strategies
            meta_samples[3] *= 1.3  # Entropy
            meta_samples[4] *= 1.2  # Best-fitness
        elif gen < 10:
            # Early generations: prefer balanced exploration
            meta_samples[0] *= 1.2  # Basic
            meta_samples[4] *= 1.2  # Best-fitness
        
        self.current_selection_strategy = int(np.argmax(meta_samples))
        self.meta_counts[self.current_selection_strategy] += 1
        
        if self.last_strategy != -1 and self.last_strategy != self.current_selection_strategy:
            self.strategy_switches += 1
        self.last_strategy = self.current_selection_strategy
        self.meta_selection_history.append(self.current_selection_strategy)
        
        # Execute the selected strategy
        if self.current_selection_strategy == 0:
            self._select_operator_basic()
        elif self.current_selection_strategy == 1:
            self._select_operator_ucb()
        elif self.current_selection_strategy == 2:
            self._select_operator_context_aware()
        elif self.current_selection_strategy == 3:
            self._select_operator_entropy()
        else:
            self._select_operator_best_fitness()
        
        return self.current_operator
    
    def _select_operator_basic(self):
        """Basic Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        self.selection_counts[self.current_operator] += 1
        return self.current_operator
    
    def _select_operator_ucb(self):
        """UCB with optimistic initialization."""
        total_selections = float(np.sum(self.selection_counts))
        
        if total_selections == 0:
            self.current_operator = int(np.random.randint(0, self.num_operators))
        else:
            mean_rewards = np.zeros(self.num_operators)
            valid_ops = np.zeros(self.num_operators, dtype=bool)
            
            for i in range(self.num_operators):
                if len(self.operator_rewards[i]) > 0:
                    mean_rewards[i] = float(np.mean(self.operator_rewards[i]))
                    valid_ops[i] = True
            
            exploration = np.zeros(self.num_operators)
            nonzero_mask = self.selection_counts > 0
            exploration[nonzero_mask] = np.sqrt(2.0 * np.log(max(total_selections, 1)) / np.maximum(self.selection_counts[nonzero_mask], 1))
            exploration[~nonzero_mask] = np.sqrt(2.0 * np.log(max(total_selections, 1)) + 1)
            
            ucb_scores = mean_rewards + exploration
            
            if np.any(~valid_ops):
                unvisited = np.where(~valid_ops)[0]
                ucb_scores[unvisited] += float(np.max(ucb_scores[valid_ops]) + 1.0)
            
            self.current_operator = int(np.argmax(ucb_scores))
        
        self.selection_counts[self.current_operator] += 1
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _select_operator_context_aware(self):
        """Context-aware selection using problem structure features."""
        fitness_scale_log = float(np.log1p(max(abs(self.f_opt), 1.0)))
        pc_norm = float(np.linalg.norm(self.pc))
        eigvals = np.linalg.eigvalsh(self.C)
        cond = float(np.max(eigvals) / max(float(np.min(eigvals)), 1e-10))
        cond_log = float(np.log1p(cond))
        pop_diversity = float(np.mean(np.std(self.population, axis=0)))
        expected_range = float(self.ub[0] - self.lb[0])
        diversity_ratio = float(np.clip(pop_diversity / (expected_range + 1e-10), 0.0, 1.0))
        improvement = float(getattr(self, 'improvement_ema', 0.0))
        gen = float(self.generation)
        phase = float(np.clip(gen / max(1.0, float(self.max_stagnation)), 0.0, 1.0))
        
        meta_features = np.array([fitness_scale_log, pc_norm, cond_log, diversity_ratio, improvement, phase])
        
        if not hasattr(self, 'context_history'):
            self.context_history = {op: [] for op in range(self.num_operators)}
            self.context_rewards = {op: [] for op in range(self.num_operators)}
        
        operator_context_scores = np.zeros(self.num_operators)
        
        for op in range(self.num_operators):
            if len(self.context_history[op]) >= 3:
                contexts = np.array(self.context_history[op])
                rewards = np.array(self.context_rewards[op])
                
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
        self.selection_counts[self.current_operator] += 1
        
        op = self.current_operator
        reward = float(max(0.0, getattr(self, 'f_opt_prev', self.f_opt) - self.f_opt))
        
        self.context_history[op].append(meta_features)
        self.context_rewards[op].append(reward)
        
        max_contexts = 50
        if len(self.context_history[op]) > max_contexts:
            self.context_history[op] = self.context_history[op][-max_contexts:]
            self.context_rewards[op] = self.context_rewards[op][-max_contexts:]
        
        return self.current_operator
    
    def _select_operator_entropy(self):
        """Entropy-regularized selection with softmax probabilities."""
        log_rewards = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            if len(self.operator_rewards[i]) >= 3:
                log_rewards[i] = float(np.log1p(max(float(np.mean(self.operator_rewards[i])), 1e-15)))
        
        total_selections = float(np.sum(self.operator_counts)) + 1.0
        entropy_bonus = np.zeros(self.num_operators)
        for i in range(self.num_operators):
            p_select = (self.operator_counts[i] + 0.1) / (total_selections + 0.5 * self.num_operators)
            p_select = float(np.clip(p_select, 1e-6, 1.0))
            entropy_bonus[i] = -np.log(p_select + 1e-10) / max(np.log(total_selections + 1), 1.0)
        
        scores = log_rewards + 0.5 * entropy_bonus
        
        scores_shifted = scores - np.max(scores)
        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / (np.sum(exp_scores) + 1e-10)
        probs = np.clip(probs, 1e-6, 1.0)
        probs /= np.sum(probs)
        
        self.current_operator = int(np.random.choice(self.num_operators, p=probs))
        self.operator_counts[self.current_operator] += 1
        self.selection_counts[self.current_operator] += 1
        return self.current_operator
    
    def _select_operator_best_fitness(self):
        """Softmax over absolute best-fitness per operator."""
        if not hasattr(self, 'operator_best_fitness'):
            self.operator_best_fitness = np.full(self.num_operators, np.inf)
        
        op = self.current_operator
        if self.f_opt < self.operator_best_fitness[op]:
            self.operator_best_fitness[op] = float(self.f_opt)
        
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
        self.selection_counts[self.current_operator] += 1
        
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
                self.alpha[op] = 1.0 + sum_reward
            else:
                self.beta[op] = 1.0 - sum_reward
        
        # Update meta-bandit with delayed reward
        if len(self.meta_selection_history) > 0:
            recent_meta = self.meta_selection_history[-1]
            self.meta_rewards[recent_meta].append(reward)
            if len(self.meta_rewards[recent_meta]) > self.meta_window:
                self.meta_rewards[recent_meta].pop(0)
            
            # Update meta Beta parameters
            n_meta = len(self.meta_rewards[recent_meta])
            if n_meta >= 3:
                mean_r = float(np.mean(self.meta_rewards[recent_meta]))
                var_r = float(np.var(self.meta_rewards[recent_meta]))
                denom = max(var_r * n_meta + 1e-10, 1e-10)
                self.meta_alpha[recent_meta] = float(max(1.0, mean_r * (mean_r * (n_meta - 1) / denom + 1)))
                self.meta_beta[recent_meta] = float(max(1.0, (1 - mean_r) * ((n_meta - 1) * (1 - mean_r) / denom + 1)))
            elif n_meta >= 1:
                sum_r = float(np.sum(self.meta_rewards[recent_meta]))
                if sum_r > 0:
                    self.meta_alpha[recent_meta] = 1.0 + sum_r
                else:
                    self.meta_beta[recent_meta] = 1.0 - sum_r
    
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
            self