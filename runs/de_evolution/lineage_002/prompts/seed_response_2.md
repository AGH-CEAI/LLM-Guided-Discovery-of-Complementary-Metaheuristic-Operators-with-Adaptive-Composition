

```python
import numpy as np


class AdaptiveCMAESWithRestarts:
    """
    A simplified CMA-ES style optimizer with covariance adaptation,
    step-size control, and restart mechanism for escaping local optima.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.pop_size = max(4 * dim, 40)
        self.mu = self.pop_size // 2  # number of parents
        self.sigma = 30.0  # initial step size
        self.best_f = np.inf
        self.best_x = None
        self.stagnation_counter = 0
        self.max_stagnation = 50 + dim
        self.restart_count = 0

    def __call__(self, func, stopping_condition):
        self.best_f = np.inf
        self.best_x = np.zeros(self.dim)
        self.restart_count = 0

        while not stopping_condition():
            self._run_one_restart(func, stopping_condition)
            if stopping_condition():
                break
            self.restart_count += 1

        return self.best_f, self.best_x

    def _run_one_restart(self, func, stopping_condition):
        """Run a single CMA-ES restart cycle."""
        mean, sigma, C, p_sigma, p_c = self._initialize_state()
        eigenvalues, eigenvectors, inv_sqrt_C = self._decompose_covariance(C)
        weights = self._compute_weights()
        params = self._compute_strategy_params(weights)
        self.stagnation_counter = 0
        prev_best = np.inf
        generation = 0

        while not stopping_condition():
            # Sample population
            population, z_samples = self._sample_population_batch(
                mean, sigma, eigenvectors, eigenvalues
            )
            population = self._clip_to_bounds(population)

            # Evaluate
            fitness = func(population)
            if self._handle_truncated_eval(fitness, population):
                break
            if stopping_condition():
                break

            # Update best
            self._update_best(population, fitness)

            # Select and sort
            sorted_idx = self._rank_population(fitness)
            selected_pop = population[sorted_idx[:self.mu]]
            selected_z = z_samples[sorted_idx[:self.mu]]

            # CMA-ES updates
            old_mean = mean.copy()
            mean = self._update_mean(selected_pop, weights)
            p_sigma = self._update_evolution_path_sigma(
                p_sigma, inv_sqrt_C, mean, old_mean, sigma, params
            )
            sigma = self._adapt_step_size(sigma, p_sigma, params)
            p_c, hsig = self._update_evolution_path_c(
                p_c, mean, old_mean, sigma, p_sigma, params, generation
            )
            C = self._update_covariance(
                C, p_c, hsig, selected_z, weights, params
            )
            eigenvalues, eigenvectors, inv_sqrt_C = self._decompose_covariance(C)

            # Stagnation detection
            current_best = fitness[sorted_idx[0]]
            stagnated = self._detect_stagnation(current_best, prev_best, sigma)
            prev_best = current_best
            if stagnated:
                break

            # Sigma bounds
            sigma = self._bound_sigma(sigma)
            generation += 1

    def _initialize_state(self):
        """Initialize mean, sigma, covariance, and evolution paths for a restart."""
        if self.restart_count == 0:
            mean = np.random.uniform(self.lb, self.ub, self.dim)
        else:
            # Increase population diversity on restarts
            mean = self._restart()
        sigma = self._get_restart_sigma()
        C = np.eye(self.dim)
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        return mean, sigma, C, p_sigma, p_c

    def _restart(self):
        """Generate a new starting mean for a restart, biased toward best known."""
        if self.best_x is not None and np.random.random() < 0.3:
            # Perturb around best known solution
            perturbation = np.random.randn(self.dim) * 20.0
            mean = self.best_x + perturbation
            mean = np.clip(mean, self.lb, self.ub)
        else:
            mean = np.random.uniform(self.lb, self.ub, self.dim)
        return mean

    def _get_restart_sigma(self):
        """Get initial sigma, potentially adapted based on restart count."""
        base_sigma = 30.0
        # Vary sigma across restarts to explore different scales
        if self.restart_count > 0:
            scale_factor = 0.5 + np.random.random() * 1.5
            return base_sigma * scale_factor
        return base_sigma

    def _compute_weights(self):
        """Compute recombination weights for the mu best individuals."""
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights = raw_weights / np.sum(raw_weights)
        return weights

    def _compute_strategy_params(self, weights):
        """Compute CMA-ES strategy parameters."""
        mu_eff = 1.0 / np.sum(weights ** 2)
        dim = self.dim

        # Learning rates
        c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0, np.sqrt((mu_eff - 1.0) / (dim + 1.0)) - 1.0) + c_sigma

        c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim)
        c_1 = 2.0 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0) ** 2 + mu_eff))

        chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

        return {
            'mu_eff': mu_eff,
            'c_sigma': c_sigma,
            'd_sigma': d_sigma,
            'c_c': c_c,
            'c_1': c_1,
            'c_mu': c_mu,
            'chi_n': chi_n,
        }

    def _sample_population_batch(self, mean, sigma, eigenvectors, eigenvalues):
        """Sample a batch of candidates from the multivariate normal distribution."""
        z = np.random.randn(self.pop_size, self.dim)
        # Transform: y = eigenvectors @ diag(sqrt(eigenvalues)) @ z^T
        sqrt_eig = np.sqrt(np.maximum(eigenvalues, 1e-20))
        scaled_z = z * sqrt_eig[np.newaxis, :]
        y = scaled_z @ eigenvectors.T
        population = mean[np.newaxis, :] + sigma * y
        return population, z

    def _clip_to_bounds(self, population):
        """Clip population to search bounds."""
        return np.clip(population, self.lb, self.ub)

    def _handle_truncated_eval(self, fitness, population):
        """Check if evaluation was truncated; update best if possible. Returns True if should break."""
        valid_mask = np.isfinite(fitness)
        n_valid = np.sum(valid_mask)
        if n_valid == 0:
            return True
        if n_valid < len(fitness):
            # Still update best with valid ones
            valid_fit = fitness[valid_mask]
            valid_pop = population[valid_mask]
            best_idx = np.argmin(valid_fit)
            if valid_fit[best_idx] < self.best_f:
                self.best_f = valid_fit[best_idx]
                self.best_x = valid_pop[best_idx].copy()
            return True
        return False

    def _update_best(self, population, fitness):
        """Update the global best solution found so far."""
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return
        valid_fit = fitness[valid_mask]
        valid_pop = population[valid_mask]
        best_idx = np.argmin(valid_fit)
        if valid_fit[best_idx] < self.best_f:
            self.best_f = valid_fit[best_idx]
            self.best_x = valid_pop[best_idx].copy()

    def _rank_population(self, fitness):
        """Return indices that sort population by fitness (ascending)."""
        # Handle NaN by putting them last
        fitness_clean = np.where(np.isfinite(fitness), fitness, np.inf)
        return np.argsort(fitness_clean)

    def _update_mean(self, selected_pop, weights):
        """Compute the new mean as weighted average of selected individuals."""
        return np.sum(weights[:, np.newaxis] * selected_pop, axis=0)

    def _update_evolution_path_sigma(self, p_sigma, inv_sqrt_C, mean, old_mean, sigma, params):
        """Update the conjugate evolution path for step-size control."""
        c_sigma = params['c_sigma']
        mu_eff = params['mu_eff']
        displacement = (mean - old_mean) / sigma
        p_sigma_new = (1.0 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * (inv_sqrt_C @ displacement)
        return p_sigma_new

    def _adapt_step_size(self, sigma, p_sigma, params):
        """Adapt sigma using the cumulative step-size adaptation (CSA)."""
        c_sigma = params['c_sigma']
        d_sigma = params['d_sigma']
        chi_n = params['chi_n']
        p_norm = np.linalg.norm(p_sigma)
        sigma_new = sigma * np.exp((c_sigma / d_sigma) * (p_norm / chi_n - 1.0))
        return sigma_new

    def _update_evolution_path_c(self, p_c, mean, old_mean, sigma, p_sigma, params, generation):
        """Update the evolution path for covariance matrix adaptation."""
        c_c = params['c_c']
        mu_eff = params['mu_eff']
        chi_n = params['chi_n']

        # Heaviside function
        p_sigma_norm = np.linalg.norm(p_sigma)
        threshold = (1.4 + 2.0 / (self.dim + 1.0)) * chi_n * np.sqrt(1.0 - (1.0 - c_c) ** (2 * (generation + 1)))
        hsig = 1.0 if p_sigma_norm < threshold else 0.0

        displacement = (mean - old_mean) / sigma
        p_c_new = (1.0 - c_c) * p_c + hsig * np.sqrt(c_c * (2.0 - c_c) * mu_eff) * displacement
        return p_c_new, hsig

    def _update_covariance(self, C, p_c, hsig, selected_z, weights, params):
        """Update the covariance matrix using rank-1 and rank-mu updates."""
        c_1 = params['c_1']
        c_mu = params['c_mu']
        c_c = params['c_c']
        mu = self.mu

        # Rank-1 update
        rank1 = np.outer(p_c, p_c)
        # Correction for hsig
        delta_hsig = (1.0 - hsig) * c_c * (2.0 - c_c)

        # Rank-mu update using z vectors
        # We need to reconstruct the actual steps in the original coordinate frame
        # But for simplicity, we use the selected_z (in isotropic space)
        # This is a simplified version
        rank_mu = np.zeros_like(C)
        for i in range(mu):
            rank_mu += weights[i] * np.outer(selected_z[i], selected_z[i])

        C_new = (1.0 - c_1 - c_mu + delta_hsig * c_1) * C + c_1 * rank1 + c_mu * rank_mu

        # Ensure symmetry
        C_new = (C_new + C_new.T) / 2.0
        return C_new

    def _decompose_covariance(self, C):
        """Eigendecompose the covariance matrix and compute its inverse square root."""
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            # Ensure positive eigenvalues
            eigenvalues = np.maximum(eigenvalues, 1e-20)
            # Limit condition number
            max_eig = np.max(eigenvalues)
            eigenvalues = np.maximum(eigenvalues, max_eig * 1e-14)
        except np.linalg.LinAlgError:
            eigenvalues = np.ones(self.dim)
            eigenvectors = np.eye(self.dim)

        # Compute inverse square root
        inv_sqrt_eig = 1.0 / np.sqrt(eigenvalues)
        inv_sqrt_C = (eigenvectors * inv_sqrt_eig[np.newaxis, :]) @ eigenvectors.T
        return eigenvalues, eigenvectors, inv_sqrt_C

    def _detect_stagnation(self, current_best, prev_best, sigma):
        """Detect if the optimizer is stagnating and should restart."""
        improvement = prev_best - current_best
        # Consider relative improvement
        if prev_best != np.inf and np.isfinite(prev_best):
            rel_improvement = abs(improvement) / (abs(prev_best) + 1e-30)
            if rel_improvement < 1e-12:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = max(0, self.stagnation_counter - 1)
        else:
            self.stagnation_counter = 0

        # Also trigger restart if sigma is too small
        if sigma < 1e-15:
            self.stagnation_counter = self.max_stagnation

        # Check condition number implicitly via sigma explosion
        if sigma > 1e8:
            self.stagnation_counter = self.max_stagnation

        return self.stagnation_counter >= self.max_stagnation

    def _bound_sigma(self, sigma):
        """Keep sigma within reasonable bounds."""
        sigma = max(sigma, 1e-20)
        sigma = min(sigma, 1e6)
        return sigma
```