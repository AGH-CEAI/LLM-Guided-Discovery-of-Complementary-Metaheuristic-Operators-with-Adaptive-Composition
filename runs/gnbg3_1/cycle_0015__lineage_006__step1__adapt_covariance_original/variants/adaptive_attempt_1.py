import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 9 covariance adaptation strategies (original + 8 variants)
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
        self.num_operators = 9
        self.operator_names = [
            'original', 'variant_01', 'variant_02', 'variant_03', 
            'variant_05', 'variant_06', 'variant_07', 'variant_08', 'variant_09'
        ]
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_operators)
        self.beta = np.ones(self.num_operators)
        
        # Sliding window for credit assignment
        self.reward_window_size = 15
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators)
        self.selection_counts = np.zeros(self.num_operators)
        self.operator_fitness_history = {i: [] for i in range(self.num_operators)}
        
        # Runtime state
        self.generation = 0
        self.current_operator = 0
        self.L = None
        self.trial_fitness = None
        self.trials = None
        self.population = None
        self.fitness = None
    
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
        
        # Reset variant-specific state
        for attr in ['step_memory', 'step_memory_fitness', 'recent_improvements', 
                     'cov_momentum', 'archive_positions', 'archive_fitness', 
                     'archive_generations', 'exploration_temp']:
            if hasattr(self, attr):
                delattr(self, attr)
    
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
        
        # Compute fitness variance reward
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = np.clip(fit_var / (fit_scale ** 2 + 1e-10), 0.0, 1.0)
        
        # Reward components: improvement, diversity, fitness spread
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity + 0.05 * rel_var)
        reward = float(np.clip(reward, -10.0, 10.0))
        
        op = self.current_operator
        self.operator_rewards[op].append(reward)
        
        # Store fitness history for this operator
        self.operator_fitness_history[op].append(self.f_opt)
        if len(self.operator_fitness_history[op]) > self.reward_window_size:
            self.operator_fitness_history[op].pop(0)
        
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
        
        # Penalize operators that lead to stagnation
        if len(self.operator_fitness_history[op]) >= 5:
            recent_fitness = self.operator_fitness_history[op][-5:]
            fitness_change = abs(recent_fitness[-1] - recent_fitness[0])
            if fitness_change < 1e-6 * max(abs(self.f_opt), 1.0):
                self.alpha[op] = float(max(1.0, self.alpha[op] * 0.95))
    
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
        elif self.current_operator == 4:
            self._adapt_covariance_variant_05()
        elif self.current_operator == 5:
            self._adapt_covariance_variant_06()
        elif self.current_operator == 6:
            self._adapt_covariance_variant_07()
        elif self.current_operator == 7:
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
    
    def _adapt_covariance_variant_02(self):
        """Directional memory with eigenspace injection for escaping local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        if not hasattr(self, 'step_memory'):
            self.step_memory = []
            self.step_memory_fitness = []
            self.step_memory_size = 5

        for i in range(min(3, self.NP)):
            if i < len(self.trial_fitness) and i < len(self.fitness):
                if self.trial_fitness[i] < self.fitness[i]:
                    step = (self.trials[i] - self.population[i]) / self.sigma
                    step_norm = np.linalg.norm(step)
                    if step_norm > 1e-10:
                        step_normalized = step / step_norm
                        self.step_memory.append(step_normalized)
                        self.step_memory_fitness.append(float(np.asarray(self.fitness[i] - self.trial_fitness[i]).flatten()[0]))

        while len(self.step_memory) > self.step_memory_size * 3:
            if self.step_memory_fitness:
                min_idx = np.argmin(self.step_memory_fitness)
                self.step_memory.pop(min_idx)
                self.step_memory_fitness.pop(min_idx)
            else:
                self.step_memory.pop(0)

        C_dir = np.zeros((self.dim, self.dim))
        if len(self.step_memory) >= 3:
            steps = np.array(self.step_memory)
            weights = np.array(self.step_memory_fitness) if self.step_memory_fitness else np.ones(len(self.step_memory))
            weights = np.maximum(weights, 1e-10)
            weights /= np.sum(weights)
            centered = steps - np.mean(steps, axis=0)
            C_dir = np.dot(centered.T * weights, centered) + 0.1 * np.eye(self.dim)

        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        improvement_threshold = 1e-8 * max(1.0, abs(self.f_opt))
        is_stagnant = improvement < improvement_threshold

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        ccov_base = min(0.5, 0.01 / (self.dim + 1.0))
        ccov_adapt = ccov_base * max(0.1, min(10.0, rel_var * 100.0 + 1.0))
        ccov_dir_weight = min(0.3, ccov_adapt * 2.0) if is_stagnant else ccov_adapt * 0.5

        cc_adapt = min(self.cc * 2.0, (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim))
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - ccov_adapt - ccov_dir_weight) * self.C +
                  ccov_adapt * rank_one +
                  (1.0 - 1.0 / self.mueff) * ccov_adapt * 2.0 * rank_mu)

        if len(self.step_memory) >= 3:
            self.C = (1.0 - ccov_dir_weight) * self.C + ccov_dir_weight * C_dir

        if is_stagnant:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)
            perturbation = np.zeros((self.dim, self.dim))
            for k in range(min(3, self.dim)):
                if k < len(eigvals):
                    v = eigvecs[:, k]
                    perturbation += 0.05 * eigvals[k] * np.outer(v, v)
            self.C = self.C + perturbation

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_03(self):
        """Stagnation-triggered covariance expansion with condition-based damping."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        if not hasattr(self, 'recent_improvements'):
            self.recent_improvements = []
        self.recent_improvements.append(self.f_opt)
        if len(self.recent_improvements) > 10:
            self.recent_improvements.pop(0)

        if len(self.recent_improvements) >= 5:
            recent_change = abs(self.recent_improvements[-1] - self.recent_improvements[0])
            is_stagnant = recent_change < 1e-8 * max(abs(self.f_opt), 1.0)
        else:
            is_stagnant = False

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = max(np.min(eigvals), 1e-10)
        eig_max = np.max(eigvals)
        cond_C = eig_max / eig_min

        cond_damping = np.sqrt(np.log1p(cond_C)) / np.sqrt(np.log1p(self.dim))
        cond_damping = np.clip(cond_damping, 0.3, 3.0)

        if cond_C > 1e6:
            is_stagnant = True

        if is_stagnant:
            stagnation_boost = 5.0
            diag_injection = 0.2 * self.sigma * np.eye(self.dim)
        else:
            stagnation_boost = 1.0
            diag_injection = np.zeros((self.dim, self.dim))

        cc_eff = min(self.cc * stagnation_boost, 0.5)
        ccov_eff = min(self.ccov * stagnation_boost / cond_damping, 0.5)
        ccov_eff = max(ccov_eff, 1e-8)

        self.pc = (1.0 - cc_eff) * self.pc + np.sqrt(cc_eff * (2.0 - cc_eff)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - ccov_eff) * self.C + 
                  ccov_eff * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_eff * 2.0 * rank_mu))

        self.C = self.C + diag_injection

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_05(self):
        """Original covariance adaptation with restart on stagnation."""
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

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        cond = eig_max / (max(eig_min, 1e-10))

        should_restart = False
        restart_reason = ""

        if cond > 1e6:
            should_restart = True
            restart_reason = "ill_conditioned"

        if hasattr(self, 'stagnation_counter') and self.stagnation_counter > self.max_stagnation:
            should_restart = True
            restart_reason = "stagnation"

        if eig_max > 1e-10 and eig_min / eig_max < 1e-8:
            should_restart = True
            restart_reason = "spectrum_collapse"

        if should_restart:
            self.C = np.eye(self.dim)
            self.pc = np.zeros(self.dim)
            self.ps = np.zeros(self.dim)
            self.sigma = min(self.sigma * 5.0, (self.ub[0] - self.lb[0]) * 0.1)
            self.sigma = max(self.sigma, 1e-6)
    
    def _adapt_covariance_variant_06(self):
        """Natural Evolution Strategy (NES) gradient-based covariance adaptation."""
        fitness_array = np.array([float(np.asarray(f).flatten()[0]) for f in self.fitness], dtype=np.float64)
        f_min = np.min(fitness_array)
        f_max = np.max(fitness_array)
        fit_range = max(f_max - f_min, 1e-10)

        sorted_indices = np.argsort(fitness_array)
        u_weights = np.zeros(self.NP)
        for rank, idx in enumerate(sorted_indices[:self.mu]):
            u_weights[idx] = max(0.0, np.log(self.mu / 2.0 + 1.0) - np.log(rank + 1.0))

        u_sum = np.sum(u_weights)
        if u_sum > 1e-10:
            u_weights /= u_sum
        else:
            u_weights = self.weights[:self.NP]
            u_weights /= np.sum(u_weights)

        min_eig = np.min(np.linalg.eigvalsh(self.C))
        C_reg = self.C.copy()
        if min_eig < 1e-8:
            C_reg += (1e-7 - min_eig) * np.eye(self.dim)

        try:
            C_inv = np.linalg.inv(C_reg)
        except np.linalg.LinAlgError:
            diag_C = np.diag(C_reg)
            diag_C = np.maximum(diag_C, 1e-10)
            C_inv = np.diag(1.0 / diag_C)

        centered = self.population[:self.NP] - self.mean

        eigvals, eigvecs = np.linalg.eigh(C_reg)
        eigvals = np.maximum(eigvals, 1e-10)
        C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        y_coords = centered @ C_inv_sqrt.T

        grad_mean = np.zeros(self.dim)
        for i in range(self.NP):
            grad_mean += u_weights[i] * y_coords[i]
        grad_mean = C_inv_sqrt @ grad_mean

        grad_cov = np.zeros((self.dim, self.dim))
        for i in range(self.NP):
            outer_y = np.outer(y_coords[i], y_coords[i])
            grad_cov += u_weights[i] * (outer_y - np.eye(self.dim))

        eta_mean = 1.0
        eta_cov = 0.5 / (self.dim + 2.0)

        self.mean = self.mean + eta_mean * self.sigma * (C_inv_sqrt @ grad_mean)

        delta_cov = eta_cov * grad_cov

        if not hasattr(self, 'cov_momentum'):
            self.cov_momentum = np.zeros((self.dim, self.dim))
        self.cov_momentum = 0.7 * self.cov_momentum + 0.3 * delta_cov

        fitness_improvement = abs(f_min - f_max) / max(abs(f_min), 1.0)
        lr_scale = np.clip(1.0 + np.log1p(fitness_improvement * 10.0), 0.1, 5.0)

        self.C = self.C + lr_scale * self.cov_momentum

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_07(self):
        """Eigenvalue floor with stagnation detection for escaping local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma

        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-12)

        y_mean_scaled = y_mean @ eigvecs
        progress_per_dir = np.abs(y_mean_scaled)

        min_eig = 1e-6 * np.max(eigvals)
        eigvals_clamped = np.maximum(eigvals, min_eig)

        progress_threshold = 1e-4 * np.linalg.norm(y_mean_scaled) + 1e-10
        stagnation_mask = progress_per_dir < progress_threshold

        if np.any(stagnation_mask):
            boost_factor = 5.0
            eigvals_boosted = eigvals_clamped.copy()
            eigvals_boosted[stagnation_mask] *= boost_factor
            eigvals_clamped = eigvals_boosted

        self.C = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T
        self.C = 0.5 * (self.C + self.C.T)

        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        blend_weight = 0.7
        self.C = ((1.0 - blend_weight * self.ccov) * self.C + 
                  blend_weight * self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_08(self):
        """Success-ratio covariance adaptation with dynamic rank weighting."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        median_fit = np.median(self.fitness)

        success_weights = np.zeros(self.mu)
        for i in range(self.mu):
            fitness_i = self.fitness[i]
            improvement_ratio = (median_fit - fitness_i) / (abs(median_fit) + 1e-10)
            if improvement_ratio > 0:
                success_weights[i] = np.log1p(improvement_ratio * 10.0)

        sum_sw = np.sum(success_weights)
        if sum_sw > 1e-15:
            success_weights /= sum_sw
            final_weights = 0.6 * success_weights + 0.4 * self.weights[:self.mu]
            final_weights /= np.sum(final_weights)
        else:
            final_weights = self.weights[:self.mu]

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += final_weights[i] * np.outer(diff, diff)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.maximum(np.min(eigvals), 1e-15)
        eig_max = np.max(eigvals)
        cond = eig_max / eig_min

        if cond > 1e4:
            ccov_eff = np.clip(self.ccov * 2.0, 1e-8, 0.5)
            cc_eff = np.clip(self.cc * 1.5, 0.001, 0.3)
        elif cond > 1e2:
            ccov_eff = np.clip(self.ccov * 1.2, 1e-8, 0.5)
            cc_eff = self.cc
        else:
            ccov_eff = self.ccov
            cc_eff = self.cc

        self.pc = (1.0 - cc_eff) * self.pc + np.sqrt(cc_eff * (2.0 - cc_eff)) * y_mean
        rank_one = np.outer(self.pc, self.pc)

        self.C = ((1.0 - ccov_eff) * self.C + 
                  ccov_eff * (rank_one + (1.0 - 1.0 / self.mueff) * ccov_eff * 2.0 * rank_mu))

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_09(self):
        """Progressive restart with normalized intensity for escaping severe stagnation."""
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

        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        threshold = max(fit_scale * 1e-6, 1e-8)
        is_stagnant = fit_var < threshold

        if is_stagnant:
            gen_norm = max(self.generation, 1)
            restart_intensity = min(1.0 + np.log1p(gen_norm) / gen_norm, 5.0)

            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]

            spread = restart_intensity * 0.5 * (self.ub[0] - self.lb[0])
            samples = np.random.randn(self.NP, self.dim)
            for d in range(self.dim):
                bins = np.linspace(elite[d] - spread, elite[d] + spread, self.NP + 1)
                bins = np.clip(bins, self.lb[d], self.ub[d])
                perm = np.random.permutation(self.NP)
                samples[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(self.NP)

            self.population = self._clip_to_bounds(samples)
            self.fitness = self.func(self.population)
            self.mean = np.mean(self.population, axis=0)
            self.old_mean = self.mean.copy()

            self.sigma = min(self.sigma * (1.0 + restart_intensity * 0.5), 0.3 * (self.ub[0] - self.lb[0]))

            eigvals = np.linalg.eigvalsh(self.C)
            self.C = np.eye(self.dim) * np.mean(eigvals)
            self.C += restart_intensity * 0.2 * np.random.randn(self.dim, self.dim)
            self.C = 0.5 * (self.C + self.C.T)
            self.C = self._ensure_positive_definite(self.C)

            self.pc = np.zeros(self.dim)

            self.cc = min(self.cc * (1.0 + restart_intensity * 0.3), 0.3)
            self.ccov = min(self.ccov * (1.0 + restart_intensity * 0.5), 0.5)

            if elite_fit < self.f_opt:
                self.f_opt = elite_fit
                self.x_opt = elite.copy()
    
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
