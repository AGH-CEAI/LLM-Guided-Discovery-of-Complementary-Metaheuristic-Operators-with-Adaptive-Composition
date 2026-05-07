import numpy as np


class CovarianceGuidedEvolutionStrategy:
    """
    A novel optimizer combining CMA-ES-style covariance adaptation
    with a particle-swarm-inspired mean tracking mechanism.
    
    Key features:
    - Simplified full covariance matrix adaptation (not diagonal-only)
    - Dual evolution paths for step-size and covariance
    - Rank-based weighted recombination
    - Automatic restart on stagnation
    - Adaptive operator selection for covariance adaptation strategies
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        # Population size: 8 * dim (balanced for dim=30 -> NP=240)
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
        self.ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        self.damping = 1.0 + np.maximum(0.0, np.sqrt(self.mueff) - 1.0)
        
        # Restart parameters
        self.max_stagnation = 50 + self.dim * 3
        self.min_diversity = 1e-6 * (self.ub[0] - self.lb[0])
        
        # === Adaptive Operator Selection for Covariance Adaptation ===
        self.num_operators = 5
        self.operator_names = [
            'original', 'variant_01', 'variant_06', 'variant_08', 'variant_09'
        ]
        
        # Sliding window for reward tracking (per operator)
        self.reward_window_size = 10 + self.dim // 5
        self.rewards = {i: [] for i in range(self.num_operators)}
        
        # Thompson Sampling: Beta distribution parameters
        self.alpha = np.ones(self.num_operators)
        self.beta = np.ones(self.num_operators)
        
        # Prior weights based on benchmark win counts
        benchmark_priors = np.array([1.0, 5.0, 4.0, 1.0, 13.0])
        benchmark_priors /= benchmark_priors.sum()
        self.alpha = 1.0 + 5.0 * benchmark_priors
        self.beta = 1.0 + 5.0 * (1.0 - benchmark_priors)
        
        # Track which operator is currently selected
        self.current_operator = self.num_operators - 1  # Start with best from benchmarks
        
        # Performance tracking for credit assignment
        self.fitness_history = []
        self.operator_history = []
        self.improvement_window = 20
        
        # Diversity tracking for adaptive selection
        self.pop_diversity_history = []
        
    def _initialize_population(self):
        """Initialize population using Latin Hypercube sampling."""
        samples = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        
        # Apply LHS for better spread
        for d in range(self.dim):
            bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
            perm = np.random.permutation(self.NP)
            samples[:, d] = bins[perm] + (bins[1] - bins[0]) * np.random.random(self.NP)
        
        self.population = samples
        self.mean = np.mean(self.population, axis=0)
        self.old_mean = self.mean.copy()
        
        # Initialize covariance from population spread
        self.C = np.cov(self.population.T) + 1e-8 * np.eye(self.dim)
        self.sigma = np.mean(np.std(self.population, axis=0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        
        # Evaluate initial population
        self.fitness = self.func(self.population)
        
        # Track best solution
        best_idx = np.argmin(self.fitness)
        self.f_opt = self.fitness[best_idx]
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = self.f_opt
        
        self.stagnation_counter = 0
        self.generation = 0
        
        # Reset adaptive tracking on new run
        self.fitness_history = [float(self.f_opt)]
        self.pop_diversity_history = [float(self._compute_diversity())]
        
    def _sample_trials_batch(self):
        """Sample new trial population from multivariate normal."""
        # Eigendecomposition for stable sampling
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        # Sample from N(mean, sigma^2 * C)
        z = np.random.randn(self.NP, self.dim)
        self.trials = self.mean + self.sigma * (z @ L.T)
        self.trials = np.clip(self.trials, self.lb, self.ub)
    
    def _evaluate_batch(self):
        """Evaluate all trial candidates in single batch call."""
        self.trial_fitness = self.func(self.trials)
    
    def _update_best(self):
        """Update best solution if trial population contains improvement."""
        trial_best_idx = np.argmin(self.trial_fitness)
        if self.trial_fitness[trial_best_idx] < self.f_opt:
            self.f_opt = self.trial_fitness[trial_best_idx]
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
        """
        Adapt step-size using evolution path (CSA - Cumulation).
        """
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs)) * y_mean
        
        self.sigma *= np.exp((np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1.0) * self.cs / self.damping)
        self.sigma = np.clip(self.sigma, 1e-10, 10.0)
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions."""
        # Sample from Beta distributions
        samples = np.random.beta(self.alpha, self.beta)
        
        # Add small exploration bonus based on uncertainty
        uncertainty = 1.0 / (self.alpha + self.beta)
        exploration_bonus = 0.1 * uncertainty * np.random.rand(self.num_operators)
        
        selected = np.argmax(samples + exploration_bonus)
        return int(selected)
    
    def _compute_reward(self, op_idx, prev_fitness, curr_fitness, diversity_change):
        """
        Compute reward for an operator based on improvement and diversity.
        Uses normalized improvement and diversity contribution.
        """
        # Fitness improvement (normalized, negative is better)
        if len(prev_fitness) > 1:
            prev_best = min(prev_fitness)
            curr_best = min(curr_fitness)
            fit_scale = max(abs(prev_best), abs(curr_best), 1.0)
            fit_improvement = (prev_best - curr_best) / fit_scale
        else:
            fit_improvement = 0.0
        
        # Diversity contribution (positive reward for maintaining diversity)
        diversity_reward = np.clip(diversity_change / (abs(diversity_change) + 1e-10), -1.0, 1.0) * 0.2
        
        # Combined reward with bounded range
        reward = np.clip(fit_improvement + diversity_reward, -2.0, 2.0)
        return float(reward)
    
    def _update_operator_statistics(self, op_idx, reward):
        """Update reward history and Beta distribution parameters for an operator."""
        # Store reward in sliding window
        self.rewards[op_idx].append(float(reward))
        if len(self.rewards[op_idx]) > self.reward_window_size:
            self.rewards[op_idx].pop(0)
        
        # Update Beta distribution parameters based on sliding window
        if len(self.rewards[op_idx]) > 0:
            window_mean = np.mean(self.rewards[op_idx])
            window_std = np.std(self.rewards[op_idx]) + 1e-10
            
            # Increase alpha for positive rewards, beta for negative
            if window_mean > 0:
                self.alpha[op_idx] += 0.1 * window_mean
            else:
                self.beta[op_idx] += 0.1 * abs(window_mean)
            
            # Ensure valid parameters
            self.alpha[op_idx] = np.clip(self.alpha[op_idx], 1e-6, 100.0)
            self.beta[op_idx] = np.clip(self.beta[op_idx], 1e-6, 100.0)
    
    def _adapt_covariance(self):
        """
        Adapt covariance matrix using adaptive operator selection.
        Selects between 5 different covariance adaptation strategies.
        """
        # Store state before adaptation for reward computation
        prev_fitness = list(self.fitness)
        prev_diversity = float(self._compute_diversity())
        
        # Select operator using Thompson Sampling
        if self.generation > 5:  # Warm up with variant_09 initially
            selected_op = self._select_operator_thompson()
        else:
            selected_op = self.num_operators - 1  # variant_09 is best
        
        self.current_operator = int(selected_op)
        self.operator_history.append(self.current_operator)
        
        # Execute selected covariance adaptation strategy
        if selected_op == 0:
            self._adapt_covariance_original()
        elif selected_op == 1:
            self._adapt_covariance_variant_01()
        elif selected_op == 2:
            self._adapt_covariance_variant_06()
        elif selected_op == 3:
            self._adapt_covariance_variant_08()
        else:  # selected_op == 4
            self._adapt_covariance_variant_09()
        
        # Compute and store reward after adaptation
        curr_diversity = float(self._compute_diversity())
        reward = self._compute_reward(selected_op, prev_fitness, list(self.fitness), 
                                       curr_diversity - prev_diversity)
        
        # Update operator statistics (delayed to avoid immediate bias)
        if self.generation > 2:
            self._update_operator_statistics(selected_op, reward)
        
        # Ensure positive definiteness (universal safety check)
        self._ensure_covariance_validity()
    
    def _ensure_covariance_validity(self):
        """Ensure covariance matrix is symmetric and positive definite."""
        self.C = 0.5 * (self.C + self.C.T)
        eigvals = np.linalg.eigvalsh(self.C)
        min_eig = np.min(eigvals)
        if min_eig < 1e-10:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)
        # Clip condition number
        max_eig = np.max(eigvals)
        cond = max_eig / (min_eig + 1e-10)
        if cond > 1e7:
            self.C *= 0.5
    
    def _adapt_covariance_original(self):
        """Original covariance adaptation strategy."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))
    
    def _adapt_covariance_variant_01(self):
        """Variant 01: Cholesky factor update approach."""
        if not hasattr(self, 'L') or self.L is None:
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
        
        C_new = 0.5 * (C_new + C_new.T)
        min_eig = np.min(np.linalg.eigvalsh(C_new))
        if min_eig < 1e-10:
            C_new += (1e-8 - min_eig) * np.eye(self.dim)
        
        self.C = C_new
        
        try:
            v = np.linalg.solve(self.L.T, z_1)
            alpha = ccov_1 / (1.0 + ccov_1 * np.dot(v, v))
            self.L = self.L @ (np.eye(self.dim) + alpha * np.outer(v, v))
        except np.linalg.LinAlgError:
            self.L = np.linalg.cholesky(self.C + 1e-8 * np.eye(self.dim))
    
    def _adapt_covariance_variant_06(self):
        """Variant 06: Diversity-sensitive learning rate scaling."""
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
    
    def _adapt_covariance_variant_08(self):
        """Variant 08: Dual active evolution paths with exponential history."""
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
    
    def _adapt_covariance_variant_09(self):
        """Variant 09: Diversity-sensitive and stagnation-aware scaling (best overall)."""
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.1, 1.0)
        
        if not hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.improvement_ema = 1.0
        improvement = max(1e-10, self.prev_f_opt - self.f_opt)
        self.improvement_ema = 0.9 * self.improvement_ema + 0.1 * improvement
        self.prev_f_opt = self.f_opt
        
        stagnation = self.improvement_ema < 1e-10 * max(1.0, abs(self.f_opt))
        
        base_ccov = (1.0 / self.mueff) * (2.0 / (self.dim + 1.41) ** 2) + \
                    (1.0 - 1.0 / self.mueff) * (2.0 * self.cc - 1.0 / self.mueff)
        ccov_adaptive = base_ccov * (1.5 if stagnation else 1.0) * (1.5 if diversity < 0.3 else 1.0)
        ccov_adaptive = np.clip(ccov_adaptive, 1e-10, 1.0)
        
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean
        
        rank_one = np.outer(self.pc, self.pc)
        
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)
        
        rank_mu_scale = 2.0 if (stagnation or diversity < 0.3) else 1.0
        
        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  ccov_adaptive * rank_mu_scale * (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)
    
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
        
        # Track history for adaptive selection
        self.fitness_history.append(float(self.f_opt))
        if len(self.fitness_history) > self.improvement_window:
            self.fitness_history.pop(0)
        
        curr_diversity = self._compute_diversity()
        self.pop_diversity_history.append(float(curr_diversity))
        if len(self.pop_diversity_history) > self.improvement_window:
            self.pop_diversity_history.pop(0)
    
    def _restart_if_needed(self):
        """Restart population if stagnated or diversity lost."""
        diversity = self._compute_diversity()
        
        if (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            np.any(np.isnan(self.C))):
            
            # Keep best individual, reinitialize rest
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]
            
            # Reset operator selection on restart
            self.alpha = 1.0 + 5.0 * np.array([0.05, 0.25, 0.2, 0.05, 0.45])
            self.beta = 1.0 + 5.0 * np.array([0.95, 0.75, 0.8, 0.95, 0.55])
            self.rewards = {i: [] for i in range(self.num_operators)}
            self.current_operator = self.num_operators - 1
            
            # Reset covariance-specific state
            if hasattr(self, 'L'):
                self.L = None
            if hasattr(self, 'pc_weighted'):
                self.pc_weighted = np.zeros(self.dim)
            if hasattr(self, 'p_cross'):
                self.p_cross = np.zeros(self.dim)
            
            self._initialize_population()
            
            # Inject elite into new population
            self.population[0] = elite
            self.fitness[0] = elite_fit
            self.f_opt = elite_fit
            self.x_opt = elite.copy()
            self.f_opt_prev = elite_fit
    
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
            self._adapt_covariance()
            self._check_stagnation()
            self._restart_if_needed()
        
        return self.f_opt, self.x_opt
