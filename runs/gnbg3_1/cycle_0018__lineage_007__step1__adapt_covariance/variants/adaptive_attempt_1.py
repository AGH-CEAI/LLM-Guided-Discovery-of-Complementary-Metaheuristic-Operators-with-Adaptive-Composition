import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 5 covariance adaptation strategies (original + 4 best-performing variants)
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
        
        # Adaptive operator selection parameters - 5 best variants from benchmarks
        self.num_operators = 5
        self.operator_names = ['original', 'variant_01', 'variant_02', 'variant_05', 'variant_07']
        
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
        """Ensure matrix is symmetric and positive definite with safe fallback."""
        C = 0.5 * (C + C.T)
        
        # Safe eigenvalue computation with error handling
        try:
            eigvals = np.linalg.eigvalsh(C)
            min_eig = float(np.min(eigvals))
        except (np.linalg.LinAlgError, ValueError):
            min_eig = -1.0
        
        if min_eig < 1e-10:
            C += (1e-7 - min_eig) * np.eye(self.dim)
        
        # Final safety check
        if np.any(np.isnan(C)) or np.any(np.isinf(C)):
            C = np.eye(self.dim) * max(1e-8, float(np.mean(np.abs(C))) + 1e-8)
        
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
        
        # Safe fitness extraction
        best_idx = np.argmin(self.fitness)
        best_fit = self.fitness[best_idx]
        if hasattr(best_fit, '__len__'):
            self.f_opt = float(np.asarray(best_fit).flatten()[0])
        else:
            self.f_opt = float(best_fit)
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
        # Reset operator-specific attributes
        if hasattr(self, 'mirror_improvement_history'):
            self.mirror_improvement_history = []
        if hasattr(self, 'exploration_temp'):
            self.exploration_temp = 1.0
        if hasattr(self, 'reconstruction_count'):
            self.reconstruction_count = 0
            self.last_reconstruction_gen = -1
    
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
        trial_best_fit = self.trial_fitness[trial_best_idx]
        if hasattr(trial_best_fit, '__len__'):
            trial_best_fit = float(np.asarray(trial_best_fit).flatten()[0])
        else:
            trial_best_fit = float(trial_best_fit)
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

        for i in range(min(self.NP, len(self.trial_fitness), len(self.fitness))):
            if self.trial_fitness[i] < self.fitness[i]:
                recent_improved += 1
                diff = self.fitness[i] - self.trial_fitness[i]
                total_improvement += max(float(diff), 0.0)
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
        adaptation += 0.3 * float(np.tanh(self.improvement_ema - 1.0))
        adaptation = float(np.clip(adaptation, -0.8, 0.8))

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = float(np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0))

        diversity_boost = 1.0 + 2.0 * (1.0 - diversity)
        diversity_boost = float(np.clip(diversity_boost, 0.5, 3.0))

        damping_adaptive = self.damping * (0.5 + 0.5 * np.log1p(self.dim)) / diversity_boost
        damping_adaptive = float(np.clip(damping_adaptive, 0.05, 200.0))

        self.sigma *= np.exp(adaptation * self.cs / damping_adaptive)
        self.sigma = float(np.clip(self.sigma, 1e-10, 10.0))
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _update_operator_rewards(self):
        """Update operator rewards using sliding window credit assignment."""
        improvement = max(0.0, float(self.f_opt_prev) - float(self.f_opt))
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = float(np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0))
        
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
        
        self.f_opt_prev = float(self.f_opt)
    
    def _adapt_covariance(self):
        """Dispatch to the selected covariance adaptation strategy."""
        if self.current_operator == 0:
            self._adapt_covariance_original()
        elif self.current_operator == 1:
            self._adapt_covariance_variant_01()
        elif self.current_operator == 2:
            self._adapt_covariance_variant_02()
        elif self.current_operator == 3:
            self._adapt_covariance_variant_05()
        else:
            self._adapt_covariance_variant_07()
    
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
        """Mirrored sampling with temperature-modulated covariance adaptation."""
        if not hasattr(self, 'mirror_improvement_history'):
            self.mirror_improvement_history = []

        mirror_improvement = 0.0
        trial_len = len(self.trial_fitness)
        if trial_len >= 2:
            half = trial_len // 2
            for i in range(half):
                if i + half < trial_len:
                    diff_fit = float(self.trial_fitness[i]) - float(self.trial_fitness[i + half])
                    mirror_improvement += max(0.0, diff_fit)

        self.mirror_improvement_history.append(float(mirror_improvement))
        if len(self.mirror_improvement_history) > 10:
            self.mirror_improvement_history.pop(0)

        avg_mirror = float(np.mean(self.mirror_improvement_history)) if self.mirror_improvement_history else 0.0
        target_temp = float(np.clip(1.0 + 0.3 * np.tanh(avg_mirror * 0.01), 0.3, 3.0))

        if not hasattr(self, 'exploration_temp'):
            self.exploration_temp = 1.0
        self.exploration_temp = 0.9 * self.exploration_temp + 0.1 * target_temp

        y_mean = (self.mean - self.old_mean) / self.sigma
        cc_adaptive = float(np.clip(self.cc * max(self.exploration_temp, 0.5), 0.01, 0.3))

        self.pc = (1.0 - cc_adaptive) * self.pc + np.sqrt(cc_adaptive * (2.0 - cc_adaptive)) * y_mean
        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        ccov_adaptive = float(np.clip(self.ccov * self.exploration_temp, 1e-10, 0.5))

        self.C = ((1.0 - ccov_adaptive) * self.C + 
                  ccov_adaptive * rank_one + 
                  (1.0 - 1.0 / self.mueff) * ccov_adaptive * 2.0 * rank_mu)

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_02(self):
        """Eigenvalue floor with orthogonal perturbation for escaping deceptive local optima."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        # Force exploration along underexplored eigendirections
        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
        except np.linalg.LinAlgError:
            eigvals = np.full(self.dim, np.mean(np.diag(self.C)))
            eigvecs = np.eye(self.dim)

        # Enforce eigenvalue floor to prevent spectrum collapse
        min_eig_target = max(1e-8, float(np.median(eigvals)) * 1e-6)
        eigvals_clipped = np.maximum(eigvals, min_eig_target)

        # Detect under-explored eigendirections (smallest eigenvalues)
        median_eig = float(np.median(eigvals_clipped))
        small_mask = eigvals_clipped < (median_eig * 0.1)

        if np.any(small_mask):
            # Add structured perturbation along underexplored orthogonal directions
            small_indices = np.where(small_mask)[0]
            for idx in small_indices:
                # Perturb along this eigenvector direction
                perturbation = min_eig_target * 10.0 * np.outer(eigvecs[:, idx], eigvecs[:, idx])
                self.C = self.C + perturbation

        # Also add small isotropic perturbation to prevent full collapse
        pop_variance = float(np.mean(np.var(self.population, axis=0)))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = pop_variance / (expected_var + 1e-10)

        if diversity < 0.05:
            # Severe diversity loss - force exploration
            self.C = self.C + 0.01 * float(np.mean(np.diag(self.C))) * np.eye(self.dim)

        # If covariance is extremely ill-conditioned, reset along random orthogonal basis
        cond = float(np.max(eigvals_clipped)) / (float(np.min(eigvals_clipped)) + 1e-10)
        if cond > 1e7 or np.any(np.isnan(self.C)):
            # Reset to diagonal with floor
            diag_val = max(float(np.mean(np.diag(self.C))), min_eig_target)
            self.C = np.eye(self.dim) * diag_val
            self.pc = np.zeros(self.dim)
            self.L = None

        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_05(self):
        """Eigendecomposition-based condition number control for ill-conditioned tasks."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        rank_one = np.outer(self.pc, self.pc)

        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Eigendecomposition-based condition number control
        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.clip(eigvals, 1e-30, None)

            cond = float(np.max(eigvals)) / (float(np.min(eigvals)) + 1e-30)

            # Exponential rescaling of eigenvalues based on condition number
            if cond > 1e2:
                log_eig = np.log(eigvals + 1e-30)
                log_mean = float(np.mean(log_eig))
                # Exponential rescaling: push eigenvalues back toward geometric mean
                alpha = min(1.0, float(np.log(cond)) / 20.0)
                log_eig_scaled = log_mean + (log_eig - log_mean) * (1.0 - alpha)
                eigvals = np.exp(np.clip(log_eig_scaled, -30, 30))

            # Reconstruct covariance with (potentially) rescaled eigenvalues
            self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            self.C = 0.5 * (self.C + self.C.T)

        except np.linalg.LinAlgError:
            pass

        # Standard CMA-ES update on (possibly rescaled) covariance
        self.C = ((1.0 - self.ccov) * self.C + 
                  self.ccov * (rank_one + (1.0 - 1.0 / self.mueff) * self.ccov * 2.0 * rank_mu))

        # Clip to prevent numerical explosion
        self.C = np.clip(self.C, -1e10, 1e10)
        self.C = self._ensure_positive_definite(self.C)
    
    def _adapt_covariance_variant_07(self):
        """Population-aware covariance reconstruction with empirical estimation."""
        y_mean = (self.mean - self.old_mean) / self.sigma
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        try:
            eigvals = np.linalg.eigvalsh(self.C)
            eig_min = float(np.min(eigvals))
            eig_max = float(np.max(eigvals))
            cond = eig_max / max(eig_min, 1e-10)
            eig_spread = eig_min / (eig_max + 1e-10)
        except np.linalg.LinAlgError:
            cond = 1e3
            eig_spread = 1e-4

        pop_variance = float(np.mean(np.var(self.population, axis=0)))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = float(np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0))

        if not hasattr(self, 'reconstruction_count'):
            self.reconstruction_count = 0
            self.last_reconstruction_gen = -1

        generations_since_recon = self.generation - self.last_reconstruction_gen

        needs_reconstruction = (
            cond > 1e6 or 
            eig_spread < 1e-6 or 
            diversity < 0.05 or
            self.stagnation_counter > self.max_stagnation // 2 or
            (diversity < 0.2 and generations_since_recon > max(self.dim * 2, 50))
        )

        if needs_reconstruction and generations_since_recon > max(self.dim, 30):
            pop_centered = (self.population - self.mean) / self.sigma
            C_pop = np.cov(pop_centered.T) + 1e-8 * np.eye(self.dim)

            blend_weight = float(np.clip(0.4 + 0.6 * diversity, 0.3, 0.9))
            self.C = (1.0 - blend_weight) * self.C + blend_weight * C_pop

            noise_scale = min(0.15 * float(np.log10(cond + 1)), 0.5) * (1.0 - diversity)
            if noise_scale > 0.01:
                noise = np.random.randn(self.dim, self.dim) * noise_scale
                noise = 0.5 * (noise + noise.T)
                np.fill_diagonal(noise, np.abs(np.diag(noise)))
                self.C += noise

            self.pc *= 0.3
            self.reconstruction_count += 1
            self.last_reconstruction_gen = self.generation
        else:
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
        return float(np.mean(np.std(self.population, axis=0)))
    
    def _check_stagnation(self):
        """Track stagnation counter for restart logic."""
        if float(self.f_opt) < float(self.f_opt_prev) - 1e-12:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        self.f_opt_prev = float(self.f_opt)
    
    def _restart_if_needed(self):
        """Restart population if stagnated or diversity lost."""
        diversity = self._compute_diversity()
        
        has_nan = np.any(np.isnan(self.C)) if hasattr(self, 'C') else False
        
        if (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            has_nan):
            
            # Save best solution
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]
            
            self._initialize_population()
            
            # Restore elite
            if len(self.population) > 0:
                self.population[0] = elite
                self.fitness[0] = elite_fit
            
            if hasattr(elite_fit, '__len__'):
                self.f_opt = float(np.asarray(elite_fit).flatten()[0])
            else:
                self.f_opt = float(elite_fit)
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0
