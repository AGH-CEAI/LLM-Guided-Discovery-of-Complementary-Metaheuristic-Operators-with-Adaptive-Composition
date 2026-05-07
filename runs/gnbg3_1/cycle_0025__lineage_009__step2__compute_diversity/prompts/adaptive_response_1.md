```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best diversity computation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 6 diversity computation strategies (adaptive selection)
    - Thompson Sampling for operator selection
    - Sliding window credit assignment with predictive reward
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
        
        # Adaptive diversity operator selection parameters
        self.num_diversity_ops = 6
        self.diversity_op_names = [
            'ess_original', 'pairwise_euclidean', 'hybrid_fitness', 
            'direct_population', 'pairwise_detailed', 'percentile_condition'
        ]
        
        # Thompson Sampling for diversity operator selection
        self.alpha_div = np.ones(self.num_diversity_ops)
        self.beta_div = np.ones(self.num_diversity_ops)
        
        # Sliding window for diversity operator credit assignment
        self.reward_window_size = 15
        self.div_operator_rewards = {i: [] for i in range(self.num_diversity_ops)}
        self.div_operator_counts = np.zeros(self.num_diversity_ops)
        
        # Diversity history for reward computation
        self.diversity_history_len = 10
        self.diversity_history = []
        self.fitness_history = []
        self.div_op_history = []
        
        # Runtime state
        self.generation = 0
        self.current_diversity_op = 0
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
            self._adapt_covariance()
            
            # Track history and update diversity operator rewards
            self._track_history()
            self._update_diversity_operator_rewards()
            
            self._check_stagnation()
            self._restart_if_needed()
        
        return self.f_opt, self.x_opt
    
    def _initialize_population(self):
        """Initialize population using Sobol quasi-random sequences."""
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(self.dim, scramble=True)
            samples = sampler.random(self.NP)
            samples = qmc.scale(samples, self.lb, self.ub)
        except Exception:
            samples = np.zeros((self.NP, self.dim))
            for d in range(self.dim):
                bins = np.linspace(self.lb[d], self.ub[d], self.NP + 1)
                bin_width = bins[1] - bins[0]
                samples[:, d] = bins[:-1] + np.random.random(self.NP) * bin_width
            for d in range(self.dim):
                samples[:, d] = samples[np.random.permutation(self.NP), d]

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
        self.current_diversity_op = 0

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
        
        # Reset diversity operator history
        self.diversity_history = []
        self.fitness_history = []
        self.div_op_history = []
    
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
    
    def _adapt_covariance(self):
        """Dispatch to the selected covariance adaptation strategy."""
        if self.current_diversity_op == 0:
            self._adapt_covariance_original()
        elif self.current_diversity_op == 1:
            self._adapt_covariance_variant_01()
        elif self.current_diversity_op == 2:
            self._adapt_covariance_variant_06()
        elif self.current_diversity_op == 3:
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
    
    # =========================================================================
    # ADAPTIVE DIVERSITY COMPUTATION METHODS
    # =========================================================================
    
    def _select_diversity_operator_thompson(self):
        """Select diversity operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha_div, self.beta_div)
        self.current_diversity_op = int(np.argmax(samples))
        self.div_operator_counts[self.current_diversity_op] += 1
        return self.current_diversity_op
    
    def _compute_diversity_ess_original(self):
        """Compute diversity using effective sample size of the covariance matrix.

        Combines trace-based spread with eigenvalue-based shape penalty.
        ESS = geometric_mean(eigenvalues) / arithmetic_mean(eigenvalues):
          ~1.0 for spherical (good) → full diversity
          ~0.0 for elongated (bad) → wasted dimensions

        Diversity = normalized_trace * sqrt(ESS) penalizes both:
          - Low overall spread (collapsed population)
          - High elongation (narrow ridge, CMA-ES losing rank)
        """
        diag = np.diag(self.C)
        trace = np.sum(diag)
        expected_trace = self.dim * ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        trace_norm = trace / max(expected_trace, 1e-10)

        eigvals = np.linalg.eigvalsh(self.C)
        eigvals = np.clip(eigvals, 1e-15, None)
        geo_mean = np.exp(np.mean(np.log(eigvals)))
        arith_mean = np.mean(eigvals)
        ess = geo_mean / max(arith_mean, 1e-15)

        diversity = trace_norm * np.sqrt(ess)
        return float(np.clip(diversity, 1e-15, None))
    
    def _compute_diversity_pairwise_euclidean(self):
        """Compute diversity using average pairwise Euclidean distance in decision space.

        Directly measures spatial spread of population rather than relying on
        covariance matrix statistics. This catches population collapse that
        eigenvalue-based metrics miss on ill-conditioned problems.

        Diversity = log(1 + mean_pairwise_dist / expected_dist)
          where expected_dist is the mean distance in a uniformly distributed population

        Log scaling provides sensitivity across orders of magnitude while avoiding
        numerical instability when distances are very small.
        """
        if len(self.population) < 2:
            return 1e-15

        pop = self.population.reshape(len(self.population), -1)
        diffs = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        pairwise_dists = np.sqrt(np.sum(diffs ** 2, axis=2))

        n = len(self.population)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        mean_pairwise_dist = np.mean(pairwise_dists[mask])

        search_space_diameter = np.sqrt(self.dim) * (self.ub[0] - self.lb[0])
        expected_dist = search_space_diameter / np.sqrt(2.0)

        dist_ratio = mean_pairwise_dist / max(expected_dist, 1e-10)
        diversity = np.log1p(dist_ratio)

        return float(np.clip(diversity, 1e-15, None))
    
    def _compute_diversity_hybrid_fitness(self):
        """Compute diversity using hybrid fitness variance, condition number, and spread.

        Unlike the eigenvalue-based ESS approach, this combines:
        1. Fitness variance (relative) — detects population collapse into local optima
        2. Condition number penalty — detects elongated covariance (narrow ridges)
        3. Actual population spread — measures true geometric extent

        This directly addresses the failure mode where error > 1e+01 implies
        the population is trapped in a local optimum with low fitness diversity
        and/or highly elongated covariance.
        """
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)
        rel_var = float(np.clip(rel_var, 1e-10, 1.0))
        fitness_component = np.sqrt(rel_var)

        eigvals = np.linalg.eigvalsh(self.C)
        eigvals = np.clip(eigvals, 1e-15, None)
        eig_min = float(np.min(eigvals))
        eig_max = float(np.max(eigvals))
        cond = eig_max / max(eig_min, 1e-15)
        cond_penalty = 1.0 / np.log1p(cond)
        cond_penalty = float(np.clip(cond_penalty, 1e-10, 1.0))

        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        spread = pop_variance / max(expected_var, 1e-10)
        spread = float(np.clip(spread, 1e-10, 1.0))

        diversity = (fitness_component * cond_penalty * spread) ** (1.0 / 3.0)
        return float(np.clip(diversity, 1e-15, None))
    
    def _compute_diversity_direct_population(self):
        """Compute diversity using direct population metrics.

        Combines three complementary measures:
        1. Normalized mean pairwise distance (actual spatial spread)
        2. Relative fitness variance (selection pressure indicator)
        3. Solution-space coverage ratio (bound utilization)

        This avoids reliance on covariance matrix eigenvalues which become
        unreliable when the algorithm struggles on difficult tasks.
        """
        pop = self.population[:len(self.fitness)]
        n = len(pop)
        if n < 2:
            return 1e-15

        sample_size = min(50, n)
        indices = np.random.choice(n, sample_size, replace=False)
        pop_sample = pop[indices]

        pairwise_dists = []
        for i in range(sample_size):
            diffs = pop_sample - pop_sample[i]
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            pairwise_dists.extend(dists[i + 1:])

        mean_pairwise_dist = np.mean(pairwise_dists) if pairwise_dists else 0.0
        max_possible_dist = np.sqrt(self.dim) * (self.ub[0] - self.lb[0])
        spatial_spread = mean_pairwise_dist / max(max_possible_dist, 1e-10)
        spatial_spread = np.clip(spatial_spread, 0.0, 1.0)

        fit_arr = np.asarray(self.fitness).flatten()
        fit_var = np.var(fit_arr)
        fit_range = np.ptp(fit_arr) + 1e-10
        fit_scale = max(abs(self.f_opt), 1.0, fit_range)
        fitness_diversity = np.clip(fit_var / (fit_scale ** 2 + 1e-10), 0.0, 1.0)

        pop_min = np.min(pop, axis=0)
        pop_max = np.max(pop, axis=0)
        span = pop_max - pop_min
        search_span = self.ub - self.lb
        coverage = np.mean(span / np.maximum(search_span, 1e-10))
        coverage = np.clip(coverage, 0.0, 1.0)

        diversity = (0.5 * spatial_spread + 0.3 * fitness_diversity + 0.2 * coverage)
        return float(np.clip(diversity, 1e-15, None))
    
    def _compute_diversity_pairwise_detailed(self):
        """Compute diversity using pairwise distances in population space.

        Directly measures spatial distribution of population via mean/median
        Euclidean distances between individuals. Normalized by search space
        diameter gives scale-invariant metric that detects:
          - Population collapse (all individuals clustered)
          - Deceptive local optima trapping (clustering away from global optimum)
          - Loss of exploration (low pairwise distances)
        """
        pop = self.population[:len(self.fitness)]
        n = len(pop)

        if n < 2:
            return float(1e-15)

        sq_dists = np.sum((pop[:, np.newaxis, :] - pop[np.newaxis, :, :]) ** 2, axis=2)
        upper_tri_indices = np.triu_indices(n, k=1)
        pairwise_sq_dists = sq_dists[upper_tri_indices]

        if len(pairwise_sq_dists) == 0:
            return float(1e-15)

        mean_sq_dist = np.mean(pairwise_sq_dists)
        median_sq_dist = np.median(pairwise_sq_dists)

        mean_dist = np.sqrt(mean_sq_dist)
        median_dist = np.sqrt(median_sq_dist)

        search_diameter = np.sqrt(self.dim) * (self.ub[0] - self.lb[0])
        expected_dist = search_diameter / np.sqrt(2.0)

        mean_dist_norm = mean_dist / max(expected_dist, 1e-10)
        median_dist_norm = median_dist / max(expected_dist, 1e-10)

        dists_to_best = np.sqrt(np.sum((pop - self.x_opt) ** 2, axis=1))
        mean_dist_to_best = np.mean(dists_to_best)
        std_dist_to_best = np.std(dists_to_best)

        clustering_factor = mean_dist_to_best / max(expected_dist, 1e-10)
        clustering_penalty = np.clip(clustering_factor, 0.0, 1.0)

        spread_metric = (mean_dist_norm + median_dist_norm) / 2.0
        spread_metric = np.clip(spread_metric, 1e-15, 1.0)

        diversity = spread_metric * np.sqrt(1.0 - 0.5 * clustering_penalty)
        return float(np.clip(diversity, 1e-15, None))
    
    def _compute_diversity_percentile_condition(self):
        """Compute diversity using percentile-robust and condition-aware metrics.

        Key differences from ESS-based approach:
        - Uses MAX-normalized spread (catches catastrophic collapse better)
        - Uses fitness percentile range (detects local optima trapping)
        - Uses condition number directly (detects rank deficiency)
        - Uses population spread ratio (detects severe elongation)

        This makes the metric more sensitive to extreme states that cause
        catastrophic failure on multimodal/ill-conditioned tasks.
        """
        pop_diffs = self.population - self.mean
        max_sq_dist = np.max(np.sum(pop_diffs ** 2, axis=1))
        expected_max_sq = self.dim * ((self.ub[0] - self.lb[0]) / 3.0) ** 2
        spread_norm = np.sqrt(max_sq_dist) / np.sqrt(max(expected_max_sq, 1e-10))
        spread_norm = np.clip(spread_norm, 0.0, 2.0)

        sorted_fit = np.sort(self.fitness)
        fit_range = sorted_fit[-1] - sorted_fit[0]
        fit_median = sorted_fit[len(sorted_fit) // 2]
        fit_p10 = sorted_fit[max(0, len(sorted_fit) // 10 - 1)]
        fit_percentile_range = (fit_median - fit_p10) / (abs(fit_median) + abs(fit_p10) + 1e-10)
        fit_percentile_range = np.clip(fit_percentile_range, 0.0, 1.0)

        eigvals = np.linalg.eigvalsh(self.C)
        eigvals = np.clip(eigvals, 1e-15, None)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        cond = eig_max / max(eig_min, 1e-15)
        cond_norm = 1.0 / np.log1p(cond)
        cond_norm = np.clip(cond_norm, 0.0, 1.0)

        diag = np.diag(self.C)
        diag = np.maximum(diag, 1e-15)
        max_var = np.max(diag)
        min_var = np.min(diag)
        spread_ratio = min_var / max_var
        spread_ratio = np.clip(spread_ratio, 0.0, 1.0)

        diversity = (
            0.25 * spread_norm +
            0.35 * fit_percentile_range +
            0.25 * cond_norm +
            0.15 * spread_ratio
        )

        return float(np.clip(diversity, 1e-15, None))
    
    def _compute_all_diversities(self):
        """Compute diversity using all available strategies."""
        diversities = np.zeros(self.num_diversity_ops)
        diversities[0] = self._compute_diversity_ess_original()
        diversities[1] = self._compute_diversity_pairwise_euclidean()
        diversities[2] = self._compute_diversity_hybrid_fitness()
        diversities[3] = self._compute_diversity_direct_population()
        diversities[4] = self._compute_diversity_pairwise_detailed()
        diversities[5] = self._compute_diversity_percentile_condition()
        return diversities
    
    def _compute_diversity(self):
        """Compute diversity using Thompson Sampling-selected strategy.
        
        This is the main adaptive diversity computation method that:
        1. Selects which diversity strategy to use based on Thompson Sampling
        2. Computes diversity using the selected strategy
        3. Tracks the selection for reward computation
        """
        self._select_diversity_operator_thompson()
        
        if self.current_diversity_op == 0:
            return self._compute_diversity_ess_original()
        elif self.current_diversity_op == 1:
            return self._compute_diversity_pairwise_euclidean()
        elif self.current_diversity_op == 2:
            return self._compute_diversity_hybrid_fitness()
        elif self.current_diversity_op == 3:
            return self._compute_diversity_direct_population()
        elif self.current_diversity_op == 4:
            return self._compute_diversity_pairwise_detailed()
        else:
            return self._compute_diversity_percentile_condition()
    
    def _track_history(self):
        """Track diversity history and fitness history for reward computation."""
        current_diversity = self._compute_diversity()
        
        self.diversity_history.append(float(current_diversity))
        self.fitness_history.append(float(self.f_opt))
        self.div_op_history.append(int(self.current_diversity_op))
        
        if len(self.diversity_history) > self.diversity_history_len:
            self.diversity_history.pop(0)
            self.fitness_history.pop(0)
            self.div_op_history.pop(0)
    
    def _compute_predictive_reward(self, op_idx):
        """Compute predictive reward for a diversity operator.
        
        Reward is based on:
        1. Correlation between operator's diversity predictions and actual fitness improvements
        2. Whether diversity correctly predicted stagnation/restart needs
        3. Smoothness of the diversity signal (less noisy = better)
        """
        if len(self.diversity_history) < 3:
            return 0.5
        
        op_mask = np.array([1 if op == op_idx else 0 for op in self.div_op_history])
        
        if np.sum(op_mask) < 2:
            return 0.5
        
        div_values = np.array(self.diversity_history, dtype=np.float64)
        fit_values = np.array(self.fitness_history, dtype=np.float64)
        
        if np.any(np.isnan(div_values)) or np.any(np.isnan(fit_values)):
            return 0.5
        
        fit_improvements = np.diff(fit_values)
        
        if len(fit_improvements) < 2:
            return 0.5
        
        valid_mask = op_mask[:-1] == 1
        if np.sum(valid_mask) < 2:
            return 0.5
        
        div_valid = div_values[:-1][valid_mask]
        imp_valid = fit_improvements[valid_mask]
        
        if len(div_valid) < 2:
            return 0.5
        
        imp_valid = np.clip(imp_valid, -1e10, 1e10)
        
        try:
            if np.std(div_valid) > 1e-15 and np.std(imp_valid) > 1e-15:
                correlation = np.corrcoef(div_valid, -imp_valid)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0
        except Exception:
            correlation = 0.0
        
        correlation = float(np.clip(correlation, -1.0, 1.0))
        
        div_diff = np.diff(div_valid)
        smoothness = 1.0 / (1.0 + np.mean(np.abs(div_diff)))
        smoothness = float(np.clip(smoothness, 0.0, 1.0))
        
        reward = 0.6 * (correlation + 1.0) / 2.0 + 0.4 * smoothness
        reward = float(np.clip(reward, 0.0, 1.0))
        
        return reward
    
    def _update_diversity_operator_rewards(self):
        """Update Thompson Sampling parameters for diversity operator selection."""
        if len(self.diversity_history) < 3:
            return
        
        current_op = int(self.current_diversity_op)
        reward = self._compute_predictive_reward(current_op)
        
        self.div_operator_rewards[current_op].append(float(reward))
        
        if len(self.div_operator_rewards[current_op]) > self.reward_window_size:
            self.div_operator_rewards[current_op].pop(0)
        
        n = len(self.div_operator_rewards[current_op])
        if n >= 3:
            rewards = np.array(self.div_operator_rewards[current_op], dtype=np.float64)
            mean_reward = float(np.mean(rewards))
            var_reward = float(np.var(rewards)) + 1e-10
            
            effective_n = float(n)
            self.alpha_div[current_op] = float(max(1.0, mean_reward * (mean_reward * (effective_n - 1) / var_reward + 1)))
            self.beta_div[current_op] = float(max(1.0, (1 - mean_reward) * ((effective_n - 1) * (1 - mean_reward) / var_reward + 1)))
        elif n >= 1:
            sum_reward = float(np.sum(self.div_operator_rewards[current_op]))
            if sum_reward > 0:
                self.alpha_div[current_op] = 1.0 + sum_reward
            else:
                self.beta_div[current_op] = 1.0 - sum_reward
        
        self.alpha_div = np.clip(self.alpha_div, 1.0, 1e6)
        self.beta_div = np.clip(self.beta_div, 1