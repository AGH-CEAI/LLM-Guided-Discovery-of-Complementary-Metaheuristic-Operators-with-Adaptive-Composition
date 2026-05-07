import numpy as np


class AdaptiveSamplingCMAES:
    """
    CMA-ES optimizer with adaptive _sample_trials_batch strategy selection.
    Automatically learns which sampling strategy works best during optimization
    using Thompson Sampling with sliding window credit assignment.
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
        
        # ============================================================
        # ADAPTIVE SAMPLING OPERATOR SELECTION (replaces fixed _sample_trials_batch)
        # ============================================================
        self.num_operators = 3
        self.operator_names = ['original_cholesky', 'eigendecomposition', 'eigenvalue_weighted']
        
        # Thompson Sampling with Beta distributions
        self.alpha = np.ones(self.num_operators, dtype=np.float64)
        self.beta = np.ones(self.num_operators, dtype=np.float64)
        
        # Sliding window for credit assignment
        self.reward_window_size = 15
        self.operator_rewards = {i: [] for i in range(self.num_operators)}
        self.operator_counts = np.zeros(self.num_operators, dtype=np.float64)
        self.selection_counts = np.zeros(self.num_operators, dtype=np.float64)
        
        # Running statistics per operator
        self.operator_improvements = {i: [] for i in range(self.num_operators)}
        self.operator_diversity = {i: [] for i in range(self.num_operators)}
        
        # Runtime state
        self.generation = 0
        self.current_operator = 0
        self.trials = None
        self.trial_fitness = None
        self.population = None
        self.fitness = None
        self.mean = None
        self.old_mean = None
        self.C = None
        self.sigma = None
        self.ps = None
        self.pc = None
        self.f_opt = None
        self.x_opt = None
        self.f_opt_prev = None
        self.stagnation_counter = 0
        
        # Running best per operator for credit assignment
        self.operator_best_fitness = {i: float('inf') for i in range(self.num_operators)}
    
    def _clip_to_bounds(self, x):
        """Clip solution to bounds safely."""
        return np.clip(x, self.lb, self.ub)
    
    def _ensure_positive_definite(self, C):
        """Ensure matrix is symmetric and positive definite."""
        C = 0.5 * (C + C.T)
        min_eig = np.min(np.linalg.eigvalsh(C))
        if min_eig < 1e-10:
            C += (1e-7 - min_eig) * np.eye(self.dim)
        return C
    
    def __call__(self, func, stopping_condition):
        """Main optimization loop."""
        self.func = func
        self._initialize_population()
        
        while not stopping_condition():
            # Select and apply adaptive sampling operator
            self._select_operator_thompson()
            self._sample_trials_batch()
            
            if stopping_condition():
                break
                
            self._evaluate_batch()
            
            # Safety checks
            if self.trials is None or len(self.trials) == 0:
                break
            if self.trial_fitness is None or len(self.trial_fitness) == 0:
                break
                
            # Ensure consistent sizes
            n_valid = min(len(self.trials), len(self.trial_fitness))
            if n_valid < self.NP:
                break
                
            self._update_best()
            
            if stopping_condition():
                break
                
            self._select_survivors_batch()
            self._adapt_step_size()
            self._adapt_covariance()
            
            # Update operator rewards based on performance
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
        self.sigma = float(np.clip(self.sigma, 1e-10, 10.0))
        self.ps = np.zeros(self.dim)
        self.pc = np.zeros(self.dim)
        
        self.fitness = self.func(self.population)
        self.fitness = np.asarray(self.fitness).flatten().astype(np.float64)
        
        best_idx = np.argmin(self.fitness)
        self.f_opt = float(self.fitness[best_idx])
        self.x_opt = self.population[best_idx].copy()
        self.f_opt_prev = float(self.f_opt)
        
        self.stagnation_counter = 0
        self.generation = 0
        self.current_operator = 0
        
        # Reset operator-specific best fitness tracking
        self.operator_best_fitness = {i: float('inf') for i in range(self.num_operators)}
        self._reset_operator_state()
    
    def _reset_operator_state(self):
        """Reset all operator-specific state variables."""
        if hasattr(self, 'pc_weighted'):
            self.pc_weighted = np.zeros(self.dim)
        if hasattr(self, 'p_cross'):
            self.p_cross = np.zeros(self.dim)
        if hasattr(self, 'prev_y_mean'):
            delattr(self, 'prev_y_mean')
        if hasattr(self, 'prev_f_opt'):
            self.prev_f_opt = self.f_opt
            self.improvement_ema = 1.0
        if hasattr(self, 'exploration_temp'):
            self.exploration_temp = 1.0
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions."""
        # Ensure valid alpha/beta
        self.alpha = np.maximum(self.alpha, 1e-10)
        self.beta = np.maximum(self.beta, 1e-10)
        
        # Sample from Beta distributions
        samples = np.random.beta(self.alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        self.selection_counts[self.current_operator] += 1
        return self.current_operator
    
    def _update_operator_rewards(self):
        """
        Update operator rewards using sliding window credit assignment.
        Credit is assigned based on improvement and diversity achieved.
        """
        # Calculate improvement from this generation
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        # Calculate diversity contribution
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = float(np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0))
        
        # Calculate reward: combination of improvement and diversity
        reward = float(np.log1p(improvement * 1e10) / 10.0 + 0.1 * diversity)
        reward = float(np.clip(reward, -5.0, 5.0))
        
        op = self.current_operator
        
        # Update sliding window
        self.operator_rewards[op].append(reward)
        if len(self.operator_rewards[op]) > self.reward_window_size:
            self.operator_rewards[op].pop(0)
        
        # Track improvements per operator
        self.operator_improvements[op].append(improvement)
        if len(self.operator_improvements[op]) > self.reward_window_size:
            self.operator_improvements[op].pop(0)
        
        # Track diversity per operator
        self.operator_diversity[op].append(diversity)
        if len(self.operator_diversity[op]) > self.reward_window_size:
            self.operator_diversity[op].pop(0)
        
        # Update best fitness seen under this operator
        if self.f_opt < self.operator_best_fitness[op]:
            self.operator_best_fitness[op] = float(self.f_opt)
        
        # Update Beta distribution parameters
        n = len(self.operator_rewards[op])
        if n >= 3:
            mean_reward = float(np.mean(self.operator_rewards[op]))
            var_reward = float(np.var(self.operator_rewards[op]))
            
            # Bayesian update for Beta distribution
            denom = max(var_reward * n + 1e-10, 1e-10)
            self.alpha[op] = float(max(1.0, mean_reward * (mean_reward * (n - 1) / denom + 1)))
            self.beta[op] = float(max(1.0, (1.0 - mean_reward) * ((n - 1) * (1.0 - mean_reward) / denom + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.operator_rewards[op]))
            if sum_reward > 0:
                self.alpha[op] = 1.0 + sum_reward
            else:
                self.beta[op] = 1.0 - sum_reward
        
        # Apply exploration bonus for under-sampled operators
        min_selections = float(np.min(self.selection_counts))
        if min_selections > 0:
            for i in range(self.num_operators):
                if self.selection_counts[i] < min_selections * 2:
                    self.alpha[i] *= 1.1
                    self.beta[i] *= 0.95
    
    def _sample_trials_batch(self):
        """
        ADAPTIVE SAMPLING: Select and apply the best sampling strategy.
        Dispatches to one of three winning strategies based on Thompson Sampling.
        """
        if self.current_operator == 0:
            self._sample_trials_original()
        elif self.current_operator == 1:
            self._sample_trials_variant_01()
        else:
            self._sample_trials_variant_03()
    
    def _sample_trials_original(self):
        """Sample new trial population using Cholesky decomposition (original.py)."""
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
    
    def _sample_trials_variant_01(self):
        """
        Sample using eigendecomposition with eigenvalue bounds (variant_01_idea_0.py).
        3 wins on tasks: 19, 21, 22
        """
        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        # Eigendecomposition: C = V @ D @ V.T
        eigvals, eigvecs = np.linalg.eigh(self.C)

        # Robustness: ensure positive eigenvalues and bound condition number
        eigvals = np.maximum(eigvals, 1e-10)
        max_eig = np.max(eigvals)
        min_eig_val = np.min(eigvals)
        if max_eig > 1e6 * min_eig_val:
            eigvals = np.clip(eigvals, min_eig_val, 1e6 * min_eig_val)

        # Sample from standard normal and transform: mean + sigma * V @ sqrt(D) @ z
        z = np.random.randn(self.dim, self.NP)
        D_sqrt = np.sqrt(eigvals)
        self.trials = self.mean[:, np.newaxis] + self.sigma * (eigvecs * D_sqrt) @ z
        self.trials = self.trials.T

        # Clip to bounds
        self.trials = self._clip_to_bounds(self.trials)
    
    def _sample_trials_variant_03(self):
        """
        Sample with eigenvalue-weighted diversity and stagnation-triggered mixing (variant_03_idea_0.py).
        5 wins on tasks: 10, 12, 13, 17, 18
        """
        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        # Eigendecomposition for eigenvalue-weighted sampling
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-10)

        # Detect stagnation: high condition number means elongated covariance
        cond = np.max(eigvals) / np.min(eigvals)
        if cond > 1000:
            # Mix with identity to isotropically expand exploration
            mix = min(0.5, (cond - 1000) / 10000)
            C_sampled = (1 - mix) * self.C + mix * np.mean(eigvals) * np.eye(self.dim)
            eigvals_s, eigvecs = np.linalg.eigh(C_sampled)
            eigvals_s = np.maximum(eigvals_s, 1e-10)
        else:
            eigvals_s, eigvecs = eigvals, eigvecs

        # Eigenvalue-weighted sampling: sample more in smaller eigenvalue directions
        # Use power-law weights: smaller eigvals get higher probability
        weights = 1.0 / (eigvals_s ** 0.25 + 1e-10)
        weights /= np.sum(weights)

        # Sample eigenvalue indices for each individual
        n_sample = max(1, int(0.1 * self.NP))
        sampled_eigen_indices = np.random.choice(self.dim, size=n_sample, p=weights, replace=True)

        # Standard normal samples
        z = np.random.randn(self.NP, self.dim)

        # Apply eigenvalue scaling: scale each direction by sqrt(eigenvalue)
        sqrt_eigvals = np.sqrt(eigvals_s)
        scales = np.ones(self.dim)
        for idx in sampled_eigen_indices:
            scales[idx] *= 1.5
        scales = np.clip(scales, 0.5, 2.0)

        # Transform samples: z * sqrt(eigvals) * scales
        transformed = z * (sqrt_eigvals * scales)

        # Apply to mean
        self.trials = self.mean + self.sigma * (transformed @ eigvecs.T)

        # Clip to bounds
        self.trials = self._clip_to_bounds(self.trials)

        # Stagnation detection for restarts
        if hasattr(self, 'stagnation_counter'):
            if self.stagnation_counter > self.max_stagnation // 2:
                # Increase sigma to escape local optima
                self.sigma = min(self.sigma * 2.0, 10.0)
                # Reinitialize poorly-performing individuals
                n_bad = min(self.NP // 2, len(self.population))
                if n_bad > 0:
                    bad_indices = np.argsort(self.fitness)[-n_bad:]
                    for idx in bad_indices:
                        if idx > 0:  # Keep best individual
                            self.population[idx] = np.random.uniform(self.lb, self.ub)
                            self.fitness[idx] = float('inf')
    
    def _evaluate_batch(self):
        """Evaluate all trial candidates in single batch call."""
        self.trial_fitness = self.func(self.trials)
        self.trial_fitness = np.asarray(self.trial_fitness).flatten().astype(np.float64)
    
    def _update_best(self):
        """Update best solution if trial population contains improvement."""
        trial_best_idx = np.argmin(self.trial_fitness)
        trial_best_fit = float(self.trial_fitness[trial_best_idx])
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
        success_rate = float(np.clip(success_rate, 0.0, 1.0))

        avg_improvement = total_improvement / max(recent_improved, 1)
        improvement_magnitude = float(np.log1p(max(avg_improvement, 1e-15)))

        if not hasattr(self, 'improvement_ema'):
            self.improvement_ema = improvement_magnitude
        self.improvement_ema = 0.7 * self.improvement_ema + 0.3 * improvement_magnitude

        target_rate = 0.25
        adaptation = (success_rate - target_rate) / target_rate
        adaptation += 0.3 * np.tanh(self.improvement_ema - 1.0)
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
    
    def _adapt_covariance(self):
        """Covariance adaptation using original CMA-ES method."""
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
        return float(np.mean(np.std(self.population, axis=0)))
    
    def _check_stagnation(self):
        """Track stagnation counter for restart logic."""
        if self.f_opt < self.f_opt_prev - 1e-12:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        self.f_opt_prev = float(self.f_opt)
    
    def _restart_if_needed(self):
        """Restart population if stagnated or diversity lost."""
        diversity = self._compute_diversity()
        
        # Check for NaN in covariance matrix
        has_nan = np.any(np.isnan(self.C)) if self.C is not None else False
        
        if (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            has_nan):
            
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = float(self.fitness[best_idx])
            
            self._initialize_population()
            
            self.population[0] = elite
            self.fitness[0] = elite_fit
            self.f_opt = float(elite_fit)
            self.x_opt = elite.copy()
            self.f_opt_prev = float(self.f_opt)
            self.stagnation_counter = 0
            
            # Reset operator selection to be more exploratory after restart
            self.alpha = np.ones(self.num_operators, dtype=np.float64)
            self.beta = np.ones(self.num_operators, dtype=np.float64)
