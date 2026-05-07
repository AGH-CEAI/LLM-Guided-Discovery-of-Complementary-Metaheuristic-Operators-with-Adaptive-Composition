import numpy as np


class SimplifiedCovarianceAdaptiveSearch:
    """
    A simplified CMA-ES-inspired optimizer that maintains a mean vector,
    a step size (sigma), and a simplified covariance model using a 
    diagonal matrix plus a rank-1 component. Uses evolution path concepts
    for step-size adaptation and covariance updates, but avoids full
    eigendecomposition by working with a factored representation.
    
    Additionally incorporates a secondary "exploratory" sub-population
    that uses differential perturbations to maintain diversity.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        
        # Population size: lambda
        self.lam = max(4 * dim, 20)
        # Number of parents (selected individuals)
        self.mu = self.lam // 2
        
        # Recombination weights (log-scaled)
        self._init_weights()
        
        # Step size
        self.sigma = 30.0
        
        # Mean of the distribution
        self.mean = None
        
        # Covariance model: diagonal + rank-1 approximation
        self.diag_cov = np.ones(dim)
        self.pc = np.zeros(dim)  # evolution path for covariance
        self.ps = np.zeros(dim)  # evolution path for sigma
        
        # Adaptation parameters
        self.cc = 4.0 / (dim + 4.0)
        self.cs = 3.0 / (dim + 5.0)
        self.c1 = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1.0 - self.c1, 
                       2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / 
                       ((dim + 2.0) ** 2 + self.mu_eff))
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (dim + 1.0)) - 1.0) + self.cs
        self.chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))
        
        # Explorer sub-population size
        self.n_explorers = max(dim, 10)
        
        # Restart tracking
        self.stagnation_counter = 0
        self.stagnation_limit = 10 + int(30 * dim / self.lam)
        self.best_fitness_history = []

    def _init_weights(self):
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

    def _initialize_mean(self):
        self.mean = np.random.uniform(self.lb * 0.5, self.ub * 0.5, size=self.dim)

    def _reset_covariance_state(self):
        self.diag_cov = np.ones(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.sigma = 30.0
        self.stagnation_counter = 0
        self.best_fitness_history = []

    def _sample_population_batch(self):
        """Sample lambda offspring from N(mean, sigma^2 * C) using diagonal + rank1 model."""
        sqrt_diag = np.sqrt(np.maximum(self.diag_cov, 1e-20))
        z = np.random.randn(self.lam, self.dim)
        # Apply diagonal scaling
        y = z * sqrt_diag[np.newaxis, :]
        # Add rank-1 perturbation: with probability proportional to c1,
        # inject evolution path direction
        pc_norm = np.linalg.norm(self.pc)
        if pc_norm > 1e-12:
            rank1_strength = np.sqrt(self.c1)
            alpha = np.random.randn(self.lam, 1)
            direction = self.pc / pc_norm
            y = y + rank1_strength * alpha * direction[np.newaxis, :] * pc_norm
        
        population = self.mean[np.newaxis, :] + self.sigma * y
        return population, y, z

    def _sample_explorers_batch(self, elite_archive):
        """Generate exploratory individuals via differential perturbation of elites."""
        n_archive = len(elite_archive)
        if n_archive < 3:
            return np.random.uniform(self.lb, self.ub, size=(self.n_explorers, self.dim))
        
        idx_base = np.random.randint(0, n_archive, size=self.n_explorers)
        idx_a = np.random.randint(0, n_archive, size=self.n_explorers)
        idx_b = np.random.randint(0, n_archive, size=self.n_explorers)
        
        F = 0.5 + 0.3 * np.random.randn(self.n_explorers, 1)
        explorers = elite_archive[idx_base] + F * (elite_archive[idx_a] - elite_archive[idx_b])
        
        # Add some Gaussian noise scaled by current sigma
        explorers += 0.1 * self.sigma * np.random.randn(self.n_explorers, self.dim)
        return explorers

    def _clip_to_bounds_batch(self, population):
        """Clip all individuals to the search domain."""
        return np.clip(population, self.lb, self.ub)

    def _evaluate_batch(self, func, candidates, stopping_condition):
        """Evaluate a batch of candidates, handling budget exhaustion."""
        if stopping_condition():
            return None
        candidates = self._clip_to_bounds_batch(candidates)
        fitness = func(candidates)
        if fitness is None:
            return None
        fitness = np.asarray(fitness, dtype=float)
        if len(fitness) < len(candidates):
            candidates = candidates[:len(fitness)]
        return fitness

    def _select_mu_best(self, population, fitness, y_vectors):
        """Select the mu best individuals and their corresponding y vectors."""
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return None, None, None
        
        # Replace non-finite with large values for sorting
        sort_fitness = np.where(valid_mask, fitness, np.inf)
        ranking = np.argsort(sort_fitness)
        best_indices = ranking[:self.mu]
        
        selected_pop = population[best_indices]
        selected_fit = fitness[best_indices]
        selected_y = y_vectors[best_indices] if y_vectors is not None else None
        return selected_pop, selected_fit, selected_y

    def _update_mean(self, selected_pop):
        """Update the distribution mean as weighted recombination of selected."""
        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights[:, np.newaxis] * selected_pop, axis=0)
        mean_shift = self.mean - old_mean
        return mean_shift

    def _update_evolution_path_sigma(self, mean_shift):
        """Update the conjugate evolution path for step-size control."""
        inv_sqrt_diag = 1.0 / np.sqrt(np.maximum(self.diag_cov, 1e-20))
        invsqrt_times_shift = inv_sqrt_diag * mean_shift / self.sigma
        
        c_factor = np.sqrt(self.cs * (2.0 - self.cs) * self.mu_eff)
        self.ps = (1.0 - self.cs) * self.ps + c_factor * invsqrt_times_shift

    def _update_evolution_path_cov(self, mean_shift):
        """Update the evolution path for covariance adaptation."""
        ps_norm = np.linalg.norm(self.ps)
        expected = self.chi_n
        # Heaviside function for stalling detection
        hsig = 1.0 if ps_norm / np.sqrt(1.0 - (1.0 - self.cs) ** (2 * (self.stagnation_counter + 1))) < (1.4 + 2.0 / (self.dim + 1.0)) * expected else 0.0
        
        c_factor = np.sqrt(self.cc * (2.0 - self.cc) * self.mu_eff)
        self.pc = (1.0 - self.cc) * self.pc + hsig * c_factor * mean_shift / self.sigma
        return hsig

    def _update_diagonal_covariance(self, selected_y, hsig):
        """Update the diagonal covariance using rank-mu and rank-1 updates."""
        if selected_y is None:
            return
        
        # Rank-1 update component
        rank1 = self.pc ** 2
        # Correction for hsig
        rank1_correction = (1.0 - hsig) * self.cc * (2.0 - self.cc)
        
        # Rank-mu update: weighted sum of squared steps
        weighted_y_sq = np.sum(
            self.weights[:, np.newaxis] * (selected_y ** 2), axis=0
        )
        
        # Combined update
        old_factor = 1.0 - self.c1 - self.cmu + self.c1 * rank1_correction
        self.diag_cov = (old_factor * self.diag_cov + 
                         self.c1 * rank1 + 
                         self.cmu * weighted_y_sq)
        
        # Ensure positive definiteness
        self.diag_cov = np.maximum(self.diag_cov, 1e-20)
        # Limit condition number
        max_val = np.max(self.diag_cov)
        self.diag_cov = np.maximum(self.diag_cov, max_val * 1e-12)

    def _adapt_step_size(self):
        """CSA-based sigma adaptation."""
        ps_norm = np.linalg.norm(self.ps)
        log_ratio = (ps_norm / self.chi_n - 1.0) * self.cs / self.damps
        self.sigma *= np.exp(np.clip(log_ratio, -0.5, 0.5))
        # Bound sigma
        self.sigma = np.clip(self.sigma, 1e-15, 100.0)

    def _update_elite_archive(self, archive, archive_fit, new_pop, new_fit, max_size):
        """Maintain an archive of the best individuals seen."""
        if archive is None:
            combined_pop = new_pop
            combined_fit = new_fit
        else:
            combined_pop = np.vstack([archive, new_pop])
            combined_fit = np.concatenate([archive_fit, new_fit])
        
        valid = np.isfinite(combined_fit)
        combined_pop = combined_pop[valid]
        combined_fit = combined_fit[valid]
        
        if len(combined_fit) == 0:
            return archive, archive_fit
        
        ranking = np.argsort(combined_fit)[:max_size]
        return combined_pop[ranking], combined_fit[ranking]

    def _check_stagnation(self, best_fitness):
        """Detect stagnation and trigger soft restart if needed."""
        self.best_fitness_history.append(best_fitness)
        window = min(self.stagnation_limit, len(self.best_fitness_history))
        
        if window >= self.stagnation_limit:
            recent = self.best_fitness_history[-window:]
            improvement = abs(recent[0] - recent[-1])
            scale = max(abs(recent[-1]), 1e-10)
            if improvement / scale < 1e-12:
                return True
        return False

    def _soft_restart(self, best_x):
        """Perform a soft restart: keep the best solution, reset adaptation state."""
        self.mean = best_x.copy() + 0.1 * np.random.randn(self.dim) * (self.ub - self.lb) * 0.01
        self.mean = np.clip(self.mean, self.lb, self.ub)
        self._reset_covariance_state()
        self.sigma = max(10.0, np.random.uniform(5.0, 50.0))

    def _track_best(self, population, fitness, f_opt, x_opt):
        """Update the global best solution found so far."""
        valid = np.isfinite(fitness)
        if not np.any(valid):
            return f_opt, x_opt
        
        valid_fit = np.where(valid, fitness, np.inf)
        idx_best = np.argmin(valid_fit)
        
        if valid_fit[idx_best] < f_opt:
            f_opt = valid_fit[idx_best]
            x_opt = population[idx_best].copy()
        return f_opt, x_opt

    def _merge_explorer_info(self, elite_archive, elite_fit, explorer_pop, explorer_fit):
        """Optionally shift mean toward very good explorer solutions."""
        if explorer_fit is None or elite_fit is None:
            return
        valid = np.isfinite(explorer_fit)
        if not np.any(valid):
            return
        
        best_explorer_idx = np.argmin(np.where(valid, explorer_fit, np.inf))
        best_explorer_f = explorer_fit[best_explorer_idx]
        
        # If explorer found something better than median of elite, nudge mean
        if len(elite_fit) > 0 and best_explorer_f < np.median(elite_fit):
            alpha = 0.1
            self.mean = (1.0 - alpha) * self.mean + alpha * explorer_pop[best_explorer_idx]
            self.mean = np.clip(self.mean, self.lb, self.ub)

    def __call__(self, func, stopping_condition):
        self._initialize_mean()
        self._reset_covariance_state()
        
        f_opt = np.inf
        x_opt = self.mean.copy()
        
        elite_archive = None
        elite_fit = None
        archive_max_size = 2 * self.mu
        generation = 0
        
        while not stopping_condition():
            # --- Main CMA population ---
            population, y_vectors, z_vectors = self._sample_population_batch()
            population = self._clip_to_bounds_batch(population)
            
            fitness = self._evaluate_batch(func, population, stopping_condition)
            if fitness is None:
                break
            if len(fitness) < len(population):
                population = population[:len(fitness)]
                y_vectors = y_vectors[:len(fitness)]
            
            f_opt, x_opt = self._track_best(population, fitness, f_opt, x_opt)
            
            if stopping_condition():
                break
            
            # --- Explorer sub-population (every 3rd generation) ---
            explorer_fit = None
            explorer_pop = None
            if generation % 3 == 0 and elite_archive is not None and len(elite_archive) >= 3:
                explorer_pop = self._sample_explorers_batch(elite_archive)
                explorer_pop = self._clip_to_bounds_batch(explorer_pop)
                explorer_fit = self._evaluate_batch(func, explorer_pop, stopping_condition)
                
                if explorer_fit is not None:
                    if len(explorer_fit) < len(explorer_pop):
                        explorer_pop = explorer_pop[:len(explorer_fit)]
                    f_opt, x_opt = self._track_best(explorer_pop, explorer_fit, f_opt, x_opt)
                
                if stopping_condition():
                    break
            
            # --- Selection ---
            selected_pop, selected_fit, selected_y = self._select_mu_best(
                population, fitness, y_vectors
            )
            if selected_pop is None:
                generation += 1
                continue
            
            # --- Update distribution parameters ---
            mean_shift = self._update_mean(selected_pop)
            self._update_evolution_path_sigma(mean_shift)
            hsig = self._update_evolution_path_cov(mean_shift)
            self._update_diagonal_covariance(selected_y, hsig)
            self._adapt_step_size()
            
            # --- Incorporate explorer information ---
            if explorer_pop is not None and explorer_fit is not None:
                self._merge_explorer_info(elite_archive, elite_fit, explorer_pop, explorer_fit)
            
            # --- Update elite archive ---
            elite_archive, elite_fit = self._update_elite_archive(
                elite_archive, elite_fit, population, fitness, archive_max_size
            )
            if explorer_pop is not None and explorer_fit is not None:
                elite_archive, elite_fit = self._update_elite_archive(
                    elite_archive, elite_fit, explorer_pop, explorer_fit, archive_max_size
                )
            
            # --- Stagnation check ---
            current_best = np.min(np.where(np.isfinite(fitness), fitness, np.inf))
            if self._check_stagnation(min(current_best, f_opt)):
                self._soft_restart(x_opt)
            
            self.stagnation_counter = generation
            generation += 1
        
        return f_opt, x_opt
