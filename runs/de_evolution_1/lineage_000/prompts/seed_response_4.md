

```python
import numpy as np


class HybridDENelderMead:
    """
    A hybrid optimizer combining Differential Evolution with Nelder-Mead-like
    simplex refinement. The DE phase explores globally while periodic simplex
    contractions refine promising regions. Adaptive CR/F parameters per individual.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lb = -100.0
        self.ub = 100.0
        self.np_size = min(max(6 * dim, 30), 600)
        self.simplex_size = dim + 1
        self.f_opt = np.inf
        self.x_opt = None
        # Per-individual adaptive parameters
        self.F = None
        self.CR = None
        # Stagnation tracking
        self.stagnation_counter = 0
        self.stagnation_threshold = 50
        self.generation = 0

    def _initialize_population(self):
        pop = np.random.uniform(self.lb, self.ub, (self.np_size, self.dim))
        self.F = np.full(self.np_size, 0.5)
        self.CR = np.full(self.np_size, 0.9)
        self.stagnation_counter = 0
        self.generation = 0
        return pop

    def _clip_to_bounds(self, x):
        return np.clip(x, self.lb, self.ub)

    def _evaluate_batch(self, func, candidates):
        candidates = self._clip_to_bounds(candidates)
        fitness = func(candidates)
        n_valid = len(fitness)
        candidates = candidates[:n_valid]
        fitness = np.asarray(fitness, dtype=np.float64)
        return candidates, fitness, n_valid < len(candidates)

    def _update_best(self, population, fitness):
        valid_mask = np.isfinite(fitness)
        if not np.any(valid_mask):
            return
        best_idx = np.argmin(np.where(valid_mask, fitness, np.inf))
        if fitness[best_idx] < self.f_opt:
            self.f_opt = fitness[best_idx]
            self.x_opt = population[best_idx].copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

    def _adapt_parameters(self, success_mask):
        """Self-adaptive F and CR using jittering based on success."""
        # Generate new candidate parameters
        new_F = self.F + 0.1 * np.random.standard_cauchy(self.np_size)
        new_CR = self.CR + 0.1 * np.random.randn(self.np_size)
        new_F = np.clip(new_F, 0.1, 1.0)
        new_CR = np.clip(new_CR, 0.0, 1.0)
        # Only adopt new params where trial was successful
        self.F = np.where(success_mask, new_F, self.F)
        self.CR = np.where(success_mask, new_CR, self.CR)

    def _mutate_batch_current_to_pbest(self, population, fitness):
        """DE/current-to-pbest/1 mutation strategy operating on entire population."""
        NP = len(population)
        # Select p-best (top 10-20%)
        p = max(2, int(0.15 * NP))
        sorted_indices = np.argsort(fitness)
        pbest_indices = sorted_indices[:p]
        # Random p-best for each individual
        chosen_pbest = pbest_indices[np.random.randint(0, p, NP)]
        # Random indices r1 != i, r2 != i, r1 != r2
        r1 = np.random.randint(0, NP, NP)
        r2 = np.random.randint(0, NP, NP)
        for i in range(NP):
            while r1[i] == i:
                r1[i] = np.random.randint(0, NP)
            while r2[i] == i or r2[i] == r1[i]:
                r2[i] = np.random.randint(0, NP)
        F_col = self.F[:, np.newaxis]
        mutant = (population
                  + F_col * (population[chosen_pbest] - population)
                  + F_col * (population[r1] - population[r2]))
        return mutant

    def _crossover_batch_binomial(self, population, mutant):
        """Binomial crossover operating on entire population at once."""
        NP, D = population.shape
        CR_mat = self.CR[:, np.newaxis] * np.ones((NP, D))
        rand_mat = np.random.rand(NP, D)
        j_rand = np.random.randint(0, D, NP)
        # Ensure at least one dimension comes from mutant
        mask = rand_mat < CR_mat
        rows = np.arange(NP)
        mask[rows, j_rand] = True
        trial = np.where(mask, mutant, population)
        return trial

    def _select_survivors_batch(self, population, fitness, trial, trial_fitness):
        """Greedy selection: keep trial if it's at least as good."""
        n = min(len(fitness), len(trial_fitness))
        population = population[:n]
        fitness = fitness[:n]
        trial = trial[:n]
        trial_fitness = trial_fitness[:n]
        
        improved = trial_fitness <= fitness
        # Handle NaN in trial_fitness: don't select NaN
        nan_mask = ~np.isfinite(trial_fitness)
        improved = improved & (~nan_mask)
        
        new_pop = np.where(improved[:, np.newaxis], trial, population)
        new_fit = np.where(improved, trial_fitness, fitness)
        return new_pop, new_fit, improved

    def _build_simplex_from_best(self, population, fitness):
        """Select the best simplex_size individuals to form a simplex."""
        k = min(self.simplex_size, len(population))
        sorted_idx = np.argsort(fitness)[:k]
        simplex = population[sorted_idx].copy()
        simplex_fit = fitness[sorted_idx].copy()
        return simplex, simplex_fit

    def _simplex_reflect_batch(self, simplex, simplex_fit):
        """Perform Nelder-Mead-like reflection, expansion, contraction as batch ops."""
        n = len(simplex)
        if n < 2:
            return simplex, simplex_fit, False
        
        order = np.argsort(simplex_fit)
        simplex = simplex[order]
        simplex_fit = simplex_fit[order]
        
        # Centroid of all points except worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        
        # Reflected point
        reflected = centroid + alpha * (centroid - worst)
        reflected = self._clip_to_bounds(reflected)
        
        # Expanded point
        expanded = centroid + gamma * (reflected - centroid)
        expanded = self._clip_to_bounds(expanded)
        
        # Contracted (outside)
        contracted_out = centroid + rho * (centroid - worst)
        contracted_out = self._clip_to_bounds(contracted_out)
        
        # Contracted (inside)
        contracted_in = centroid - rho * (centroid - worst)
        contracted_in = self._clip_to_bounds(contracted_in)
        
        # Stack all candidates for batch evaluation
        candidates = np.stack([reflected, expanded, contracted_out, contracted_in])
        return simplex, simplex_fit, candidates

    def _simplex_update(self, simplex, simplex_fit, candidates, cand_fit):
        """Update simplex based on evaluated candidates (reflected, expanded, contracted)."""
        # candidates: [reflected, expanded, contracted_out, contracted_in]
        n_valid = min(len(cand_fit), len(candidates))
        if n_valid == 0:
            return simplex, simplex_fit
        
        candidates = candidates[:n_valid]
        cand_fit = cand_fit[:n_valid]
        
        best_cand_idx = np.argmin(cand_fit)
        best_cand = candidates[best_cand_idx]
        best_cand_f = cand_fit[best_cand_idx]
        
        worst_idx = np.argmax(simplex_fit)
        if best_cand_f < simplex_fit[worst_idx]:
            simplex[worst_idx] = best_cand
            simplex_fit[worst_idx] = best_cand_f
        
        return simplex, simplex_fit

    def _simplex_shrink_batch(self, simplex, simplex_fit):
        """Shrink all simplex points toward the best point."""
        best_idx = np.argmin(simplex_fit)
        best = simplex[best_idx]
        sigma = 0.5
        shrunk = best + sigma * (simplex - best)
        shrunk[best_idx] = best  # Don't move the best
        shrunk = self._clip_to_bounds(shrunk)
        return shrunk

    def _inject_simplex_into_population(self, population, fitness, simplex, simplex_fit):
        """Replace worst members of population with improved simplex members."""
        worst_indices = np.argsort(fitness)[-len(simplex):]
        for i, wi in enumerate(worst_indices):
            if i < len(simplex_fit) and simplex_fit[i] < fitness[wi]:
                population[wi] = simplex[i]
                fitness[wi] = simplex_fit[i]
        return population, fitness

    def _compute_diversity(self, population):
        """Compute population diversity as mean distance from centroid."""
        centroid = np.mean(population, axis=0)
        diffs = population - centroid
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        return np.mean(dists)

    def _restart_if_stagnant(self, population, fitness):
        """Restart a fraction of the population if stagnation detected."""
        if self.stagnation_counter < self.stagnation_threshold:
            return population, fitness, False
        
        n_replace = max(1, self.np_size // 2)
        worst_indices = np.argsort(fitness)[-n_replace:]
        new_individuals = np.random.uniform(self.lb, self.ub, (n_replace, self.dim))
        
        # Optionally bias some toward best known
        if self.x_opt is not None:
            n_biased = n_replace // 3
            scale = 30.0 * np.random.rand(n_biased, self.dim)
            new_individuals[:n_biased] = self.x_opt + scale * np.random.randn(n_biased, self.dim)
            new_individuals = self._clip_to_bounds(new_individuals)
        
        population[worst_indices] = new_individuals
        fitness[worst_indices] = np.inf  # Will be re-evaluated
        self.stagnation_counter = 0
        self.F = np.full(self.np_size, 0.5)
        self.CR = np.full(self.np_size, 0.9)
        return population, fitness, True

    def _opposition_based_learning(self, population):
        """Generate opposition-based candidates for diversity injection."""
        opp = self.lb + self.ub - population
        # Add some noise
        noise = np.random.randn(*population.shape) * 5.0
        opp = opp + noise
        return self._clip_to_bounds(opp)

    def __call__(self, func, stopping_condition):
        population = self._initialize_population()
        population, fitness, truncated = self._evaluate_batch(func, population)
        if truncated or stopping_condition():
            self._update_best(population, fitness)
            return self.f_opt, self.x_opt
        
        self._update_best(population, fitness)
        self.np_size = len(population)
        
        simplex_interval = 5  # Run simplex refinement every N generations

        while not stopping_condition():
            self.generation += 1
            
            # --- DE Phase ---
            mutant = self._mutate_batch_current_to_pbest(population, fitness)
            trial = self._crossover_batch_binomial(population, mutant)
            trial = self._clip_to_bounds(trial)
            
            trial, trial_fitness, truncated = self._evaluate_batch(func, trial)
            if truncated or stopping_condition():
                self._update_best(trial, trial_fitness)
                # Do partial selection with what we have
                n = min(len(fitness), len(trial_fitness))
                improved = trial_fitness[:n] <= fitness[:n]
                improved = improved & np.isfinite(trial_fitness[:n])
                population[:n] = np.where(improved[:, np.newaxis], trial[:n], population[:n])
                fitness[:n] = np.where(improved, trial_fitness[:n], fitness[:n])
                self._update_best(population, fitness)
                break
            
            population, fitness, success_mask = self._select_survivors_batch(
                population, fitness, trial, trial_fitness
            )
            self.np_size = len(population)
            self._adapt_parameters(success_mask)
            self._update_best(population, fitness)
            
            if stopping_condition():
                break
            
            # --- Simplex Refinement Phase ---
            if self.generation % simplex_interval == 0:
                simplex, simplex_fit = self._build_simplex_from_best(population, fitness)
                
                # One step of NM-like operations
                simplex_sorted, simplex_fit_sorted, candidates = self._simplex_reflect_batch(
                    simplex, simplex_fit
                )
                
                if candidates is not False and len(candidates) > 0:
                    candidates, cand_fit, truncated = self._evaluate_batch(func, candidates)
                    if truncated or stopping_condition():
                        self._update_best(candidates, cand_fit)
                        break
                    
                    simplex_sorted, simplex_fit_sorted = self._simplex_update(
                        simplex_sorted, simplex_fit_sorted, candidates, cand_fit
                    )
                    self._update_best(simplex_sorted, simplex_fit_sorted)
                    
                    # If no improvement, try shrink
                    if np.min(cand_fit) >= np.min(simplex_fit):
                        shrunk = self._simplex_shrink_batch(simplex_sorted, simplex_fit_sorted)
                        shrunk, shrunk_fit, truncated = self._evaluate_batch(func, shrunk)
                        if truncated or stopping_condition():
                            self._update_best(shrunk, shrunk_fit)
                            break
                        simplex_sorted = shrunk
                        simplex_fit_sorted = shrunk_fit
                        self._update_best(shrunk, shrunk_fit)
                    
                    population, fitness = self._inject_simplex_into_population(
                        population, fitness, simplex_sorted, simplex_fit_sorted
                    )
            
            if stopping_condition():
                break
            
            # --- Stagnation handling ---
            population, fitness, restarted = self._restart_if_stagnant(population, fitness)
            if restarted:
                # Re-evaluate individuals with inf fitness
                inf_mask = np.isinf(fitness)
                if np.any(inf_mask):
                    inf_indices = np.where(inf_mask)[0]
                    reinit_pop = population[inf_indices]
                    reinit_pop, reinit_fit, truncated = self._evaluate_batch(func, reinit_pop)
                    if truncated or stopping_condition():
                        n_valid = len(reinit_fit)
                        for j in range(n_valid):
                            fitness[inf_indices[j]] = reinit_fit[j]
                            population[inf_indices[j]] = reinit_pop[j]
                        self._update_best(population, fitness)
                        break
                    for j in range(len(reinit_fit)):
                        fitness[inf_indices[j]] = reinit_fit[j]
                        population[inf_indices[j]] = reinit_pop[j]
                    self._update_best(population, fitness)

        if self.x_opt is None:
            valid = np.isfinite(fitness)
            if np.any(valid):
                best_idx = np.argmin(np.where(valid, fitness, np.inf))
                self.f_opt = fitness[best_idx]
                self.x_opt = population[best_idx].copy()
            else:
                self.x_opt = population[0].copy()
                self.f_opt = float('inf')

        return self.f_opt, self.x_opt
```