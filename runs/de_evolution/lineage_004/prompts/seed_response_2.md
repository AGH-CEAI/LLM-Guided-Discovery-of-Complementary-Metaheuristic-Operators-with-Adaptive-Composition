

```python
import numpy as np


class CMAESSimplified:
    """
    A simplified CMA-ES inspired optimizer with:
    - Covariance matrix adaptation using a diagonal + rank-1 update
    - Step size adaptation via cumulative path length control
    - Restart mechanism for escaping local optima
    - Weighted recombination of top candidates
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.lam = min(4 + int(3 * np.log(dim)), 300)  # population size
        self.mu = self.lam // 2  # number of parents
        self._init_weights()
        self.max_restarts = 20
        self.stagnation_window = 15
        self.stagnation_threshold = 1e-12

    def _init_weights(self):
        """Initialize recombination weights for weighted averaging."""
        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / np.sum(raw_weights)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

    def _init_strategy_parameters(self):
        """Initialize all CMA-ES strategy parameters for a fresh run."""
        dim = self.dim
        self.sigma = 40.0  # initial step size
        self.mean = np.random.uniform(self.lb, self.ub, size=dim)

        # Diagonal covariance (simplified)
        self.C_diag = np.ones(dim)
        # Low-rank component (rank-1)
        self.pc = np.zeros(dim)  # evolution path for covariance
        self.ps = np.zeros(dim)  # evolution path for sigma

        # Learning rates
        self.cc = 4.0 / (dim + 4.0)
        self.c1 = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.cs = (self.mu_eff + 2.0) / (dim + self.mu_eff + 5.0)
        self.ds = 1.0 + 2.0 * max(0.0, np.sqrt((self.mu_eff - 1.0) / (dim + 1.0)) - 1.0) + self.cs
        self.chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

        # History tracking for stagnation detection
        self.best_history = []
        self.generation = 0

    def _initialize_population(self):
        """Sample a new population from the current distribution."""
        dim = self.dim
        # Sample from N(mean, sigma^2 * C)
        sqrt_C = np.sqrt(np.maximum(self.C_diag, 1e-20))
        z = np.random.randn(self.lam, dim)
        population = self.mean[np.newaxis, :] + self.sigma * (z * sqrt_C[np.newaxis, :])
        return self._clip_to_bounds(population)

    def _clip_to_bounds(self, population):
        """Clip population to search bounds."""
        return np.clip(population, self.lb, self.ub)

    def _evaluate_batch(self, func, population, stopping_condition):
        """Evaluate a batch of solutions, handling budget exhaustion."""
        if stopping_condition():
            return None
        fitness = func(population)
        if fitness is None:
            return None
        fitness = np.asarray(fitness, dtype=float)
        if len(fitness) < len(population):
            fitness = np.concatenate([fitness, np.full(len(population) - len(fitness), np.nan)])
        return fitness

    def _select_parents(self, population, fitness):
        """Select top mu individuals based on fitness (minimization)."""
        valid_mask = ~np.isnan(fitness)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return None, None
        sorted_idx = valid_indices[np.argsort(fitness[valid_indices])]
        parent_idx = sorted_idx[:self.mu]
        return population[parent_idx], fitness[parent_idx]

    def _update_mean(self, parents):
        """Update the distribution mean via weighted recombination."""
        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights[:, np.newaxis] * parents, axis=0)
        return old_mean

    def _update_evolution_paths(self, old_mean):
        """Update cumulative evolution paths for step size and covariance."""
        dim = self.dim
        sqrt_C_inv = 1.0 / np.sqrt(np.maximum(self.C_diag, 1e-20))
        displacement = (self.mean - old_mean) / self.sigma

        # Update sigma path
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs) * self.mu_eff) * (sqrt_C_inv * displacement)

        # Heaviside function for stalling detection
        ps_norm = np.linalg.norm(self.ps)
        threshold = (1.4 + 2.0 / (dim + 1.0)) * self.chi_n * np.sqrt(1.0 - (1.0 - self.cs) ** (2.0 * (self.generation + 1)))
        h_sig = 1.0 if ps_norm < threshold else 0.0

        # Update covariance path
        self.pc = (1.0 - self.cc) * self.pc + h_sig * np.sqrt(self.cc * (2.0 - self.cc) * self.mu_eff) * displacement

    def _adapt_covariance(self, parents, old_mean):
        """Update diagonal covariance matrix with rank-1 and rank-mu updates."""
        dim = self.dim
        c_mu = min(1.0 - self.c1, 2.0 * (self.mu_eff - 2.0 + 1.0 / self.mu_eff) / ((dim + 2.0) ** 2 + self.mu_eff))

        # Rank-1 update component
        rank1 = self.pc ** 2

        # Rank-mu update component (diagonal approximation)
        diffs = (parents - old_mean[np.newaxis, :]) / self.sigma
        rank_mu = np.sum(self.weights[:, np.newaxis] * (diffs ** 2), axis=0)

        # Combined update
        self.C_diag = (1.0 - self.c1 - c_mu) * self.C_diag + self.c1 * rank1 + c_mu * rank_mu

        # Ensure positive and bounded
        self.C_diag = np.clip(self.C_diag, 1e-20, 1e10)

    def _adapt_step_size(self):
        """Adapt sigma using cumulative step-size adaptation (CSA)."""
        ps_norm = np.linalg.norm(self.ps)
        self.sigma *= np.exp((self.cs / self.ds) * (ps_norm / self.chi_n - 1.0))
        self.sigma = np.clip(self.sigma, 1e-20, 1e6)

    def _detect_stagnation(self, best_fitness):
        """Detect if the optimization has stagnated."""
        self.best_history.append(best_fitness)
        if len(self.best_history) < self.stagnation_window:
            return False
        recent = self.best_history[-self.stagnation_window:]
        improvement = abs(recent[0] - recent[-1])
        if improvement < self.stagnation_threshold:
            return True
        # Also detect if sigma has collapsed
        if self.sigma < 1e-15:
            return True
        # Detect if covariance has collapsed
        if np.max(self.C_diag) < 1e-15:
            return True
        return False

    def _restart(self, f_opt, x_opt):
        """Restart the search with a new random distribution, possibly near the best known."""
        self._init_strategy_parameters()
        # With probability 0.5, restart near the best known solution (local restart)
        if x_opt is not None and np.random.rand() < 0.3:
            self.mean = x_opt + np.random.randn(self.dim) * 20.0
            self.mean = np.clip(self.mean, self.lb, self.ub)
            self.sigma = 20.0
        # Increase population size on restarts (IPOP-like)
        self.lam = min(int(self.lam * 1.5), 500)
        self.mu = self.lam // 2
        self._init_weights()

    def _generate_parameters_batch(self):
        """Generate a batch of candidate solutions from the current distribution.
        Uses mirrored sampling for variance reduction."""
        dim = self.dim
        half = self.lam // 2
        remainder = self.lam - 2 * half

        sqrt_C = np.sqrt(np.maximum(self.C_diag, 1e-20))
        z_half = np.random.randn(half, dim)
        # Mirrored pairs
        z = np.vstack([z_half, -z_half])
        if remainder > 0:
            z = np.vstack([z, np.random.randn(remainder, dim)])

        population = self.mean[np.newaxis, :] + self.sigma * (z * sqrt_C[np.newaxis, :])
        return self._clip_to_bounds(population)

    def _select_survivors_batch(self, population, fitness, f_opt, x_opt):
        """Select survivors, update best tracking, and return parents."""
        valid_mask = ~np.isnan(fitness)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return None, None, f_opt, x_opt

        # Update best
        best_idx = valid_indices[np.argmin(fitness[valid_indices])]
        if np.isnan(f_opt) or fitness[best_idx] < f_opt:
            f_opt = fitness[best_idx]
            x_opt = population[best_idx].copy()

        # Sort and select top mu
        sorted_valid = valid_indices[np.argsort(fitness[valid_indices])]
        parent_idx = sorted_valid[:min(self.mu, len(sorted_valid))]
        parents = population[parent_idx]
        parent_fitness = fitness[parent_idx]

        return parents, parent_fitness, f_opt, x_opt

    def _inject_best_solution(self, population, fitness, x_opt, f_opt):
        """Inject the overall best solution into the population to preserve elitism."""
        if x_opt is not None and not np.isnan(f_opt):
            worst_idx = np.nanargmax(fitness)
            if np.isnan(fitness[worst_idx]) or fitness[worst_idx] > f_opt:
                population[worst_idx] = x_opt.copy()
                fitness[worst_idx] = f_opt
        return population, fitness

    def _local_search_batch(self, func, x_opt, f_opt, stopping_condition):
        """Perform a small local search around the best solution found so far."""
        if x_opt is None or stopping_condition():
            return f_opt, x_opt
        n_local = min(self.lam, 50)
        perturbations = np.random.randn(n_local, self.dim) * 0.1 * self.sigma
        candidates = x_opt[np.newaxis, :] + perturbations
        candidates = self._clip_to_bounds(candidates)
        fit = self._evaluate_batch(func, candidates, stopping_condition)
        if fit is None:
            return f_opt, x_opt
        valid = ~np.isnan(fit)
        if np.any(valid):
            best_local = np.nanargmin(fit)
            if fit[best_local] < f_opt:
                f_opt = fit[best_local]
                x_opt = candidates[best_local].copy()
        return f_opt, x_opt

    def __call__(self, func, stopping_condition):
        """Main optimization loop orchestrating CMA-ES with restarts."""
        f_opt = np.nan
        x_opt = None
        restart_count = 0

        self._init_strategy_parameters()

        while not stopping_condition() and restart_count <= self.max_restarts:
            # Run one CMA-ES epoch until stagnation or stopping
            f_opt, x_opt = self._run_epoch(func, stopping_condition, f_opt, x_opt)
            
            if stopping_condition():
                break

            # Try local search before restart
            f_opt, x_opt = self._local_search_batch(func, x_opt, f_opt, stopping_condition)
            
            if stopping_condition():
                break

            # Restart
            restart_count += 1
            self._restart(f_opt, x_opt)

        if x_opt is None:
            x_opt = np.zeros(self.dim)
        if np.isnan(f_opt):
            f_opt = float('inf')

        return f_opt, x_opt

    def _run_epoch(self, func, stopping_condition, f_opt, x_opt):
        """Run one epoch of CMA-ES until stagnation or budget exhaustion."""
        self.generation = 0

        while not stopping_condition():
            # Generate candidates
            population = self._generate_parameters_batch()

            # Evaluate
            fitness = self._evaluate_batch(func, population, stopping_condition)
            if fitness is None:
                break

            # Handle truncated evaluation
            valid_count = np.sum(~np.isnan(fitness))
            if valid_count == 0:
                break

            if stopping_condition():
                # Still update best before breaking
                valid_mask = ~np.isnan(fitness)
                if np.any(valid_mask):
                    best_idx = np.nanargmin(fitness)
                    if np.isnan(f_opt) or fitness[best_idx] < f_opt:
                        f_opt = fitness[best_idx]
                        x_opt = population[best_idx].copy()
                break

            # Select survivors and update best
            parents, parent_fitness, f_opt, x_opt = self._select_survivors_batch(
                population, fitness, f_opt, x_opt
            )

            if parents is None or len(parents) < 2:
                break

            # Ensure we have enough parents
            actual_mu = len(parents)
            if actual_mu < self.mu:
                weights_local = self.weights[:actual_mu]
                weights_local = weights_local / np.sum(weights_local)
            else:
                weights_local = self.weights

            # Update mean
            old_mean = self.mean.copy()
            self.mean = np.sum(weights_local[:actual_mu, np.newaxis] * parents[:actual_mu], axis=0)

            # Update evolution paths
            self._update_evolution_paths(old_mean)

            # Adapt covariance
            self._adapt_covariance(parents[:actual_mu], old_mean)

            # Adapt step size
            self._adapt_step_size()

            self.generation += 1

            # Check stagnation
            current_best = parent_fitness[0] if parent_fitness is not None else f_opt
            if self._detect_stagnation(current_best):
                break

        return f_opt, x_opt
```