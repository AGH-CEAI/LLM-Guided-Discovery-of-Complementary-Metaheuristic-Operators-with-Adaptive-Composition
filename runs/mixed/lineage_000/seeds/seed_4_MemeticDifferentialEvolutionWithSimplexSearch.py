import numpy as np


class MemeticDifferentialEvolutionWithSimplexSearch:
    """
    A hybrid population-based optimizer that combines DE-style mutation and
    crossover with a simplex-based local search (Nelder–Mead-like) refinement.
    The algorithm maintains a population of NP candidates, performs generational
    updates using DE operators, and periodically applies a local search to the
    most promising individuals. All major operators are implemented as batch
    operations to ensure efficient vectorised evaluation of the objective
    function.
    """

    def __init__(self, dim, *args, **kwargs):
        """
        Initialise the optimizer.

        Parameters
        ----------
        dim : int
            Problem dimension (search space is [-100, 100]^dim).
        *args, **kwargs : optional
            Additional hyperparameters:
                - F : float, initial mutation scaling factor (default 0.5).
                - CR : float, initial crossover probability (default 0.7).
                - local_search_interval : int, generations between local
                  search applications (default 5).
                - local_search_fraction : float, fraction of population to
                  undergo local search (default 0.1).
                - stagnation_threshold : int, generations without improvement
                  before a restart (default 20).
                - local_search_step : float, initial step size for the simplex
                  (default 5% of the bound range).
        """
        self.dim = dim
        # Population size: modest (4–10 times dimension) but never exceeds 1000.
        self.NP = min(1000, max(4 * dim, int(5 * dim)))
        self.lower = -100.0
        self.upper = 100.0

        # Hyperparameters (with defaults)
        self.F_init = kwargs.get('F', 0.5)
        self.CR_init = kwargs.get('CR', 0.7)
        self.local_search_interval = kwargs.get('local_search_interval', 5)
        self.local_search_fraction = kwargs.get('local_search_fraction', 0.1)
        self.stagnation_threshold = kwargs.get('stagnation_threshold', 20)
        bound_range = self.upper - self.lower
        self.local_search_step = kwargs.get('local_search_step', 0.05 * bound_range)
        self.max_step_size = 0.2 * bound_range
        self.min_step_size = 1e-4 * bound_range

        # Internal state flags
        self._budget_exhausted = False

    def __call__(self, func, stopping_condition):
        """
        Run the optimisation.

        Parameters
        ----------
        func : callable
            Objective function. Must accept a 2D array of shape (N, dim) and
            return a 1D array of fitness values of shape (N,).
        stopping_condition : callable
            A callable that returns True when the optimisation should stop.

        Returns
        -------
        f_opt : float
            Best fitness found.
        x_opt : np.ndarray
            Corresponding solution vector of shape (dim,).
        """
        # Initialise population
        pop = self._initialize_population()
        pop = self._clip_bounds(pop)

        # First batch evaluation
        fitness = self._evaluate_batch(pop)
        if self._budget_exhausted or len(fitness) == 0:
            # No feasible evaluations – return whatever we have
            best_idx = np.argmin(fitness) if len(fitness) > 0 else 0
            return fitness[best_idx], pop[best_idx]

        # Track best solution
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_x = pop[best_idx].copy()
        prev_best_fitness = best_fitness

        # Counters and parameters
        gen = 0
        no_improvement_count = 0
        F = self.F_init
        CR = self.CR_init

        # Main optimisation loop
        while not stopping_condition():
            # ---- DE operators (batch) ----
            mutants = self._mutate_batch(pop, best_idx, F)
            trials = self._crossover_batch(pop, mutants, CR)
            trials = self._clip_bounds(trials)

            # Evaluate trial vectors in one batch
            trial_fitness = self._evaluate_batch(trials)
            if self._budget_exhausted:
                break  # Budget exhausted – exit gracefully

            # ---- Survivor selection (batch) ----
            pop, fitness = self._select_survivors_batch(pop, fitness, trials, trial_fitness)

            # Update global best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_fitness = fitness[best_idx]
                best_x = pop[best_idx].copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Check stopping condition after selection
            if stopping_condition():
                break

            # ---- Periodic local search (simplex) on top individuals ----
            if gen > 0 and gen % self.local_search_interval == 0:
                pop, fitness = self._apply_local_search(pop, fitness, stopping_condition)
                # Refresh best after local search
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < best_fitness:
                    best_fitness = fitness[best_idx]
                    best_x = pop[best_idx].copy()
                if stopping_condition():
                    break

            # ---- Adapt mutation parameters (F, CR) ----
            improvement = prev_best_fitness - best_fitness
            F, CR = self._adapt_mutation_parameters(F, CR, improvement)
            prev_best_fitness = best_fitness

            # ---- Restart if stagnant ----
            if no_improvement_count >= self.stagnation_threshold:
                pop, fitness, reset_idx = self._restart_if_stagnant(pop, fitness)
                # Evaluate newly reinitialised individuals
                if len(reset_idx) > 0:
                    new_fitness = self._evaluate_batch(pop[reset_idx])
                    fitness[reset_idx] = new_fitness
                    if self._budget_exhausted:
                        break
                no_improvement_count = 0
                # Update best after restart
                best_idx = np.argmin(fitness)
                best_fitness = fitness[best_idx]
                best_x = pop[best_idx].copy()

            gen += 1

        return best_fitness, best_x

    # -----------------------------------------------------------------
    # Population initialisation
    # -----------------------------------------------------------------
    def _initialize_population(self):
        """
        Generate the initial random population within the bounds.

        Returns
        -------
        pop : np.ndarray
            Array of shape (NP, dim) containing the initial candidates.
        """
        # Uniform random sampling in the search space
        pop = np.random.uniform(self.lower, self.upper, size=(self.NP, self.dim))
        return pop

    # -----------------------------------------------------------------
    # Bound handling
    # -----------------------------------------------------------------
    def _clip_bounds(self, pop):
        """
        Clip all candidate vectors to the admissible interval [-100, 100].

        Parameters
        ----------
        pop : np.ndarray
            Array of shape (N, dim) to be clipped.

        Returns
        -------
        np.ndarray
            Clipped array with the same shape.
        """
        # Ensure all values lie within the predefined bounds
        pop = np.clip(pop, self.lower, self.upper)
        return pop

    # -----------------------------------------------------------------
    # Batch evaluation with budget handling
    # -----------------------------------------------------------------
    def _evaluate_batch(self, pop):
        """
        Evaluate the objective function on a whole population at once.

        Parameters
        ----------
        pop : np.ndarray
            Array of shape (N, dim).

        Returns
        -------
        fitness : np.ndarray
            Array of shape (N,) containing the fitness values.
        """
        # Call the user-provided function on the entire batch
        fitness = func(pop)

        # Convert to numpy array and replace NaNs with +inf
        fitness = np.asarray(fitness, dtype=np.float64)
        fitness = np.where(np.isnan(fitness), np.inf, fitness)

        # Detect budget exhaustion (function returns fewer values than asked)
        if len(fitness) < pop.shape[0]:
            self._budget_exhausted = True
            # Fill missing entries with +inf to indicate unknown fitness
            full = np.full(pop.shape[0], np.inf, dtype=np.float64)
            full[:len(fitness)] = fitness
            return full

        return fitness

    # -----------------------------------------------------------------
    # DE Mutation (batch)
    # -----------------------------------------------------------------
    def _mutate_batch(self, pop, best_idx, F):
        """
        Generate mutant vectors using the DE/current-to-best/1 mutation scheme.

        Parameters
        ----------
        pop : np.ndarray
            Current population of shape (NP, dim).
        best_idx : int
            Index of the current best individual.
        F : float
            Mutation scaling factor.

        Returns
        -------
        mutants : np.ndarray
            Mutant vectors of shape (NP, dim).
        """
        NP, dim = pop.shape
        mutants = np.empty((NP, dim), dtype=np.float64)

        # For each target vector, pick three distinct random indices
        for i in range(NP):
            # Sample three distinct indices different from i
            idx = np.random.choice(NP, 3, replace=False)
            while i in idx:
                idx = np.random.choice(NP, 3, replace=False)
            a, b, c = idx

            # DE/current-to-best/1 mutation
            mutants[i] = pop[i] + F * (pop[best_idx] - pop[a]) + F * (pop[b] - pop[c])

        return mutants

    # -----------------------------------------------------------------
    # DE Crossover (batch)
    # -----------------------------------------------------------------
    def _crossover_batch(self, pop, mutants, CR):
        """
        Perform binomial crossover between target and mutant vectors.

        Parameters
        ----------
        pop : np.ndarray
            Target vectors (NP, dim).
        mutants : np.ndarray
            Mutant vectors (NP, dim).
        CR : float
            Crossover probability.

        Returns
        -------
        trials : np.ndarray
            Trial vectors after crossover (NP, dim).
        """
        NP, dim = pop.shape
        # Random mask for each dimension
        mask = np.random.rand(NP, dim) < CR
        trials = np.where(mask, mutants, pop)

        # Guarantee at least one dimension comes from the mutant
        guarantee_idx = np.random.randint(0, dim, size=NP)
        trials[np.arange(NP), guarantee_idx] = mutants[np.arange(NP), guarantee_idx]

        return trials

    # -----------------------------------------------------------------
    # Survivor selection (batch)
    # -----------------------------------------------------------------
    def _select_survivors_batch(self, pop, fitness, trials, trial_fitness):
        """
        Select the better individual between target and trial for each index.

        Parameters
        ----------
        pop : np.ndarray
            Current population (NP, dim).
        fitness : np.ndarray
            Current fitness values (NP,).
        trials : np.ndarray
            Trial vectors (NP, dim).
        trial_fitness : np.ndarray
            Trial fitness values (NP,).

        Returns
        -------
        new_pop : np.ndarray
            Selected population for next generation.
        new_fitness : np.ndarray
            Corresponding fitness values.
        """
        # Choose the better of the two for each individual
        better = trial_fitness < fitness
        new_pop = np.where(better[:, np.newaxis], trials, pop)
        new_fitness = np.where(better, trial_fitness, fitness)
        return new_pop, new_fitness

    # -----------------------------------------------------------------
    # Local search (simplex) – batched for top individuals
    # -----------------------------------------------------------------
    def _apply_local_search(self, pop, fitness, stopping_condition):
        """
        Apply a Nelder–Mead‑like simplex search to the top fraction of the
        population. All simplex points are generated and evaluated in a single
        batch call to the objective function.

        Parameters
        ----------
        pop : np.ndarray
            Current population (NP, dim).
        fitness : np.ndarray
            Current fitness values (NP,).
        stopping_condition : callable
            Stopping condition to check after local search.

        Returns
        -------
        pop : np.ndarray
            Updated population.
        fitness : np.ndarray
            Updated fitness values.
        """
        # Determine how many individuals will receive local search
        n_ls = max(1, int(np.ceil(self.NP * self.local_search_fraction)))
        # Indices of the best individuals (lowest fitness)
        top_indices = np.argsort(fitness)[:n_ls]

        # Generate simplex points for each selected individual
        points_batch, mapping = self._generate_simplex_points(pop, top_indices, self.local_search_step)

        # Evaluate all simplex points in one batch
        points_fitness = self._evaluate_batch(points_batch)
        if self._budget_exhausted or len(points_fitness) < len(points_batch):
            # If budget is exhausted, skip the update
            return pop, fitness

        # Reshape fitness to (n_ls, dim+1) for easy per‑individual selection
        fitness_matrix = points_fitness.reshape(n_ls, self.dim + 1)

        # Find the best point for each individual
        best_local_idx = np.argmin(fitness_matrix, axis=1)
        best_local_fitness = fitness_matrix[np.arange(n_ls), best_local_idx]

        # Update those individuals that improve
        improved = best_local_fitness < fitness[top_indices]
        for offset, (i, k) in enumerate(zip(top_indices, best_local_idx)):
            if improved[offset]:
                point_pos = offset * (self.dim + 1) + k
                pop[i] = points_batch[point_pos]
                fitness[i] = points_fitness[point_pos]

        # Adapt the simplex step size based on the improvement rate
        improvement_rate = np.mean(improved)
        if improvement_rate > 0.5:
            # Many improvements – increase step to explore more
            self.local_search_step = min(self.local_search_step * 1.2, self.max_step_size)
        else:
            # Few improvements – shrink step to refine
            self.local_search_step = max(self.local_search_step * 0.8, self.min_step_size)

        return pop, fitness

    def _generate_simplex_points(self, pop, indices, step_size):
        """
        Generate simplex (Nelder‑Mead) points around each selected individual.

        For each individual a base point plus dim random direction vectors are
        created, yielding dim+1 points per individual.

        Parameters
        ----------
        pop : np.ndarray
            Current population.
        indices : np.ndarray
            Indices of individuals to process.
        step_size : float
            Length of the direction vectors.

        Returns
        -------
        points_batch : np.ndarray
            All simplex points stacked (n_ls*(dim+1), dim).
        mapping : list of tuple
            List mapping each point to (individual_index, local_point_id).
        """
        n_ls = len(indices)
        dim = self.dim
        points = []
        mapping = []

        for offset, i in enumerate(indices):
            base = pop[i]
            # Add the base point itself
            points.append(base)
            mapping.append((i, 0))

            # Generate dim random direction vectors, normalised to step_size
            directions = np.random.randn(dim, dim)
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            directions = directions / (norms + 1e-12) * step_size

            # Create perturbed points and clip to bounds
            for k in range(dim):
                perturbed = base + directions[k]
                perturbed = np.clip(perturbed, self.lower, self.upper)
                points.append(perturbed)
                mapping.append((i, k + 1))

        points_batch = np.stack(points, axis=0)
        return points_batch, mapping

    # -----------------------------------------------------------------
    # Adaptive mutation parameter tuning
    # -----------------------------------------------------------------
    def _adapt_mutation_parameters(self, F, CR, improvement):
        """
        Adapt the mutation scaling factor F and crossover probability CR based
        on whether the global best fitness improved in the last generation.

        Parameters
        ----------
        F : float
            Current scaling factor.
        CR : float
            Current crossover probability.
        improvement : float
            Difference between previous and current best fitness (positive if
            improved).

        Returns
        -------
        F : float
            Updated scaling factor.
        CR : float
            Updated crossover probability.
        """
        if improvement > 0:
            # Improvement observed – exploit: reduce F, increase CR
            F = max(F * 0.9, 0.1)
            CR = min(CR + 0.05, 1.0)
        else:
            # No improvement – explore: increase F, decrease CR
            F = min(F * 1.1, 2.0)
            CR = max(CR - 0.05, 0.0)

        return F, CR

    # -----------------------------------------------------------------
    # Stagnation handling – partial reinitialisation
    # -----------------------------------------------------------------
    def _restart_if_stagnant(self, pop, fitness):
        """
        Reinitialise a fraction of the worst individuals when the optimisation
        has stagnated.

        Parameters
        ----------
        pop : np.ndarray
            Current population.
        fitness : np.ndarray
            Current fitness values.

        Returns
        -------
        pop : np.ndarray
            Updated population (some individuals replaced).
        fitness : np.ndarray
            Updated fitness values (reset to +inf for new individuals).
        reset_idx : np.ndarray
            Indices of the reinitialised individuals.
        """
        # Reinitialise 20% of the population (the worst individuals)
        n_reset = max(1, int(self.NP * 0.2))
        worst_indices = np.argsort(fitness)[::-1][:n_reset]

        # Generate new random candidates
        new_candidates = np.random.uniform(self.lower, self.upper, size=(n_reset, self.dim))
        pop[worst_indices] = new_candidates
        # Mark fitness as unknown (+inf) – will be evaluated by the caller
        fitness[worst_indices] = np.inf

        return pop, fitness, worst_indices

    # -----------------------------------------------------------------
    # Diversity measure (optional, useful for diagnostics)
    # -----------------------------------------------------------------
    def _compute_diversity(self, pop):
        """
        Compute the average Euclidean distance between all pairs of individuals
        as a simple diversity metric.

        Parameters
        ----------
        pop : np.ndarray
            Population array of shape (NP, dim).

        Returns
        -------
        avg_dist : float
            Mean pairwise distance.
        """
        # Compute pairwise differences using broadcasting
        diff = pop[:, np.newaxis, :] - pop[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        # Exclude the diagonal (self-distances)
        mask = ~np.eye(self.NP, dtype=bool)
        avg_dist = distances[mask].mean()
        return avg_dist
