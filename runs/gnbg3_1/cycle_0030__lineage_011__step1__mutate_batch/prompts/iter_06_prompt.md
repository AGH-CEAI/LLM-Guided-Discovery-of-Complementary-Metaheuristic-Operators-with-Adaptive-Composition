This is iteration 6 of 10. Propose ONE replacement implementation for `_mutate_batch`.

Requirements:
- Keep the EXACT function signature: `def _mutate_batch(self, population, fitness, selected_operator):`
- Propose a SINGLE, FUNDAMENTALLY DIFFERENT strategy vs. the variants below
- You may read any `self` attribute but do NOT modify `__init__` or other methods
- The fenced code block must contain ONLY the single replacement function
- Ensure numerical robustness (no division by zero, handle edge dims, clip to bounds)

WHY PER-TASK COVERAGE MATTERS:
After all variants are generated, an ADAPTIVE algorithm will be built that
selects the best operator FOR EACH TASK at runtime (e.g. via Thompson Sampling
or multi-armed bandit). This means:
- We do NOT need a single variant that wins everywhere.
- We DO need at least one variant that reaches error <= 1e-08 on EACH task.
- Tasks marked UNSOLVED below are critical gaps. The bigger the remaining
  error, the higher the priority — your variant should specifically target
  the WORST unsolved tasks at the top of the priority list.

TASK COVERAGE SUMMARY: 2 SOLVED (<= 1e-08), 22 UNSOLVED (of which 0 crashed) — out of 24.
Goal: drive every task to error <= 1e-08. Focus first on the WORST unsolved tasks (top of the UNSOLVED list).

UNSOLVED TASKS (worst best-error first — these are the priority targets):
  Task 17: *** UNSOLVED *** — best=2.270e+03 by variant_01_idea_0.py      (target=1e-08, ~+11.4 decades above target)
  Task 16: *** UNSOLVED *** — best=1.857e+03 by variant_01_idea_0.py      (target=1e-08, ~+11.3 decades above target)
  Task 11: *** UNSOLVED *** — best=2.155e+02 by original.py               (target=1e-08, ~+10.3 decades above target)
  Task 19: *** UNSOLVED *** — best=6.321e+01 by original.py               (target=1e-08, ~+9.8 decades above target)
  Task 23: *** UNSOLVED *** — best=4.138e+01 by variant_04_idea_0.py      (target=1e-08, ~+9.6 decades above target)
  Task 18: *** UNSOLVED *** — best=3.017e+01 by variant_01_idea_0.py      (target=1e-08, ~+9.5 decades above target)
  Task  6: *** UNSOLVED *** — best=2.826e+01 by original.py               (target=1e-08, ~+9.5 decades above target)
  Task 20: *** UNSOLVED *** — best=2.122e+01 by original.py               (target=1e-08, ~+9.3 decades above target)
  Task  5: *** UNSOLVED *** — best=1.696e+01 by variant_01_idea_0.py      (target=1e-08, ~+9.2 decades above target)
  Task 22: *** UNSOLVED *** — best=1.220e+01 by original.py               (target=1e-08, ~+9.1 decades above target)
  Task  9: *** UNSOLVED *** — best=7.865e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.9 decades above target)
  Task 13: *** UNSOLVED *** — best=5.096e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.7 decades above target)
  Task 21: *** UNSOLVED *** — best=4.739e+00 by original.py               (target=1e-08, ~+8.7 decades above target)
  Task 14: *** UNSOLVED *** — best=3.384e+00 by variant_04_idea_0.py      (target=1e-08, ~+8.5 decades above target)
  Task 15: *** UNSOLVED *** — best=3.334e+00 by original.py               (target=1e-08, ~+8.5 decades above target)
  Task  3: *** UNSOLVED *** — best=2.453e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.4 decades above target)
  Task  7: *** UNSOLVED *** — best=2.109e+00 by variant_01_idea_0.py      (target=1e-08, ~+8.3 decades above target)
  Task  8: *** UNSOLVED *** — best=8.359e-01 by original.py               (target=1e-08, ~+7.9 decades above target)
  Task  1: *** UNSOLVED *** — best=4.718e-01 by variant_01_idea_0.py      (target=1e-08, ~+7.7 decades above target)
  Task  4: *** UNSOLVED *** — best=9.657e-02 by original.py               (target=1e-08, ~+7.0 decades above target)
  Task 12: *** UNSOLVED *** — best=5.573e-02 by original.py               (target=1e-08, ~+6.7 decades above target)
  Task 10: *** UNSOLVED *** — best=2.716e-02 by variant_01_idea_0.py      (target=1e-08, ~+6.4 decades above target)

SOLVED TASKS (already at or below target — do not regress these):
  Task  0: SOLVED  — best=1.000e-08 by original.py                   
  Task  2: SOLVED  — best=1.000e-08 by original.py                   

PER-TASK BENCHMARK ERRORS (raw numbers, lower is better):
Task original.py           variant_01_idea_0.py  variant_02_idea_0.py  variant_03_idea_0.py  variant_04_idea_0.py  variant_05_idea_0.py  
-----------------------------------------------------------------------------------------------------------------------------------------
0    1.0000e-08            6.0592e-05            6.4216e+00            -inf                  5.0910e+01            -inf                  
1    2.0433e+00            4.7178e-01            2.9668e+00            -inf                  1.3827e+02            -inf                  
2    1.0000e-08            6.8717e-04            1.6588e+01            -inf                  4.9132e+02            -inf                  
3    2.9291e+00            2.4534e+00            1.8059e+01            -inf                  2.5888e+02            -inf                  
4    9.6571e-02            1.7566e-01            1.7998e+00            -inf                  4.9274e+00            -inf                  
5    1.3428e+02            1.6962e+01            6.5538e+03            -inf                  9.8087e+08            -inf                  
6    2.8257e+01            3.7862e+01            6.1368e+02            -inf                  3.3762e+03            -inf                  
7    3.2450e+00            2.1095e+00            1.1796e+01            -inf                  3.1008e+02            -inf                  
8    8.3587e-01            1.5128e+00            6.0101e+00            -inf                  4.3161e+01            -inf                  
9    8.5918e+00            7.8652e+00            4.8846e+01            -inf                  1.5806e+02            -inf                  
10   5.2455e-02            2.7159e-02            2.3601e+02            -inf                  1.1537e+02            -inf                  
11   2.1552e+02            7.9366e+02            9.9911e+02            -inf                  4.3977e+02            -inf                  
12   5.5732e-02            1.4691e+00            1.7731e+02            -inf                  4.2430e+01            -inf                  
13   2.3793e+01            5.0958e+00            4.8062e+01            -inf                  3.2870e+01            -inf                  
14   4.0870e+00            1.6588e+01            6.4048e+00            -inf                  3.3836e+00            -inf                  
15   3.3345e+00            4.0581e+00            4.0081e+00            -inf                  3.5517e+00            -inf                  
16   4.3688e+03            1.8573e+03            1.5154e+04            -inf                  1.0661e+04            -inf                  
17   3.7425e+04            2.2705e+03            2.6347e+05            -inf                  7.5557e+04            -inf                  
18   4.0922e+01            3.0166e+01            8.6072e+01            -inf                  5.7742e+01            -inf                  
19   6.3212e+01            4.2207e+02            2.2526e+02            -inf                  9.6644e+01            -inf                  
20   2.1217e+01            2.5540e+01            4.5686e+01            -inf                  2.1942e+01            -inf                  
21   4.7394e+00            5.7968e+00            5.8765e+00            -inf                  5.7020e+00            -inf                  
22   1.2199e+01            2.0377e+01            1.7352e+01            -inf                  1.3542e+01            -inf                  
23   5.2424e+01            8.1598e+01            5.4911e+01            -inf                  4.1377e+01            -inf                  

PRIORITY TARGETS — UNSOLVED TASKS, WORST FIRST (drive these toward 1e-08):
  Task 17: best error so far = 2.270e+03  (target = 1e-08)
  Task 16: best error so far = 1.857e+03  (target = 1e-08)
  Task 11: best error so far = 2.155e+02  (target = 1e-08)
  Task 19: best error so far = 6.321e+01  (target = 1e-08)
  Task 23: best error so far = 4.138e+01  (target = 1e-08)
  Task 18: best error so far = 3.017e+01  (target = 1e-08)
  Task  6: best error so far = 2.826e+01  (target = 1e-08)
  Task 20: best error so far = 2.122e+01  (target = 1e-08)
  ... and 14 other unsolved task(s).

LABELS OF PREVIOUSLY PROPOSED VARIANTS (do NOT repeat these ideas):
Idea 0, Idea 0, Idea 0, Idea 0, Idea 0

Your new proposal MUST be designed to crush the error on the WORST unsolved
tasks above. It is acceptable — even expected — for the new variant to be
worse than existing variants on already-SOLVED tasks; the adaptive selector
will handle that. Reason explicitly about the priority targets:
1. What property of those WORST unsolved tasks (multimodality, ill-conditioning,
   separability, ruggedness, noise, deceptive local optima, narrow basins,
   non-separable rotation, etc.) is preventing existing operators from reaching
   1e-08? Use the per-task error magnitudes as evidence — errors
   stuck at ~1e+1 vs ~1e-3 vs ~1e-6 imply different failure modes.
2. What specific mechanism in your proposed operator is designed to break
   through that exact obstacle and push the error several orders of magnitude
   lower?
3. Why is this approach fundamentally different from the prior variants —
   especially from whichever variant currently holds the best (but still
   insufficient) error on the priority tasks?

Current implementation:
```python
def _mutate_batch(self, population, fitness, selected_operator):
        """Generate mutant vectors using selected DE mutation strategy."""
        np_pop = len(population)
        
        # Build ring topology indices
        indices = np.arange(np_pop)
        ring_prev = np.roll(indices, 1)
        ring_next = np.roll(indices, -1)
        
        # Get sorted indices by fitness for pbest selection
        sorted_idx = np.argsort(fitness)
        pbest_idx = sorted_idx[:max(1, int(np_pop * 0.1))]  # Top 10%
        
        # Initialize mutant population
        mutants = np.empty_like(population)
        
        # Sample indices for mutation
        r1 = np.array([np.delete(np.arange(np_pop), i) for i in range(np_pop)])
        r1 = np.array([np.random.choice(r1[i]) for i in range(np_pop)])
        r2 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i]])) 
                       for i in range(np_pop)])
        
        # Adaptive F with slight individual variation
        f_values = self.f_base + self.f_adapt * np.random.randn(np_pop)
        f_values = np.clip(f_values, 0.1, 2.0)
        
        # Apply selected mutation strategy
        if selected_operator == 0:  # rand/1
            mutants = (population[r1] + f_values[:, np.newaxis] * 
                      (population[r2] - population[np.roll(r1, -1)]))
        elif selected_operator == 1:  # best/1
            best_idx = np.argmin(fitness)
            mutants = (population[best_idx] + f_values[:, np.newaxis] * 
                      (population[r1] - population[r2]))
        elif selected_operator == 2:  # current-to-pbest/1
            pbest = population[np.random.choice(pbest_idx, np_pop)]
            mutants = (population + f_values[:, np.newaxis] * 
                      (pbest - population[r1] + population[r1] - population[r2]))
        else:  # rand-to-best/1
            best_idx = np.argmin(fitness)
            mutants = ((population[r1] + f_values[:, np.newaxis] * 
                       (population[best_idx] - population[r1] + population[r2] - population[r1])))
        
        # Clip to bounds
        mutants = np.clip(mutants, self.lower_bound, self.upper_bound)
        
        return mutants
```

Full algorithm for context:
```python
import numpy as np


class AdaptiveMemeticDE:
    """
    Adaptive Memetic Differential Evolution with Success-Based Operator Pool.
    
    Key features:
    - Pool of DE mutation strategies selected via success history (UCB-based)
    - Ring topology for information exchange
    - Periodic Nelder-Mead local refinement on elite individuals
    - Diversity monitoring with adaptive restarts
    - Step-size adaptation based on success history
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.lower_bound = -100.0
        self.upper_bound = 100.0
        
        # Population size: 5*dim (balance between exploration and efficiency)
        self.np = min(max(5 * dim, 20), 200)
        
        # Mutation strategies pool
        self.mutation_strategies = ['rand', 'best', 'current_to_pbest', 'rand_to_best']
        self.n_strategies = len(self.mutation_strategies)
        
        # Success history for operator selection (UCB)
        self.operator_rewards = np.zeros(self.n_strategies)
        self.operator_counts = np.ones(self.n_strategies)
        self.ucb_c = 2.0  # Exploration weight for UCB
        
        # DE parameters
        self.cr = 0.9
        self.f_base = 0.5
        self.f_adapt = 0.1
        
        # Local search parameters
        self.local_search_interval = 5  # Apply LS every N generations
        self.local_search_budget = 0.02  # Fraction of func budget for LS
        
        # Diversity and stagnation
        self.diversity_threshold = 1e-6
        self.stagnation_counter = 0
        self.stagnation_limit = 50
        self.best_history = []
        
        # Archive for DE (optional, bounded)
        self.archive = None
        self.archive_max_size = self.np
        
    def _initialize_population(self, func):
        """Initialize population using Latin Hypercube Sampling."""
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.np, self.dim))
        
        # LHS for better spread
        grid = np.linspace(0, 1, self.np + 1)[:-1] + np.random.uniform(0, 1/self.np, self.np)
        for d in range(self.dim):
            perm = np.random.permutation(self.np)
            pop[:, d] = self.lower_bound + (self.upper_bound - self.lower_bound) * grid[perm]
        
        # Evaluate initial population
        fitness = func(pop)
        fitness = self._handle_budget_truncation(fitness, pop)
        
        return pop, fitness
    
    def _handle_budget_truncation(self, fitness, population):
        """Handle cases where func returns fewer values due to budget exhaustion."""
        if len(fitness) < len(population):
            # Truncate population to match fitness
            actual_len = len(fitness)
            # Keep first `actual_len` individuals
            return fitness
        return fitness
    
    def _select_mutation_operator_batch(self):
        """Select mutation operator using UCB-based operator selection."""
        # UCB formula
        ucb_values = (self.operator_rewards / np.maximum(self.operator_counts, 1) + 
                      self.ucb_c * np.sqrt(np.log(sum(self.operator_counts)) / 
                      np.maximum(self.operator_counts, 1)))
        
        # Select operator for each individual (with small random perturbation)
        selected = np.argmax(ucb_values) + np.random.randint(-1, 2)
        selected = np.clip(selected, 0, self.n_strategies - 1)
        return selected
    
    def _mutate_batch(self, population, fitness, selected_operator):
        """Generate mutant vectors using selected DE mutation strategy."""
        np_pop = len(population)
        
        # Build ring topology indices
        indices = np.arange(np_pop)
        ring_prev = np.roll(indices, 1)
        ring_next = np.roll(indices, -1)
        
        # Get sorted indices by fitness for pbest selection
        sorted_idx = np.argsort(fitness)
        pbest_idx = sorted_idx[:max(1, int(np_pop * 0.1))]  # Top 10%
        
        # Initialize mutant population
        mutants = np.empty_like(population)
        
        # Sample indices for mutation
        r1 = np.array([np.delete(np.arange(np_pop), i) for i in range(np_pop)])
        r1 = np.array([np.random.choice(r1[i]) for i in range(np_pop)])
        r2 = np.array([np.random.choice(np.delete(np.arange(np_pop), [i, r1[i]])) 
                       for i in range(np_pop)])
        
        # Adaptive F with slight individual variation
        f_values = self.f_base + self.f_adapt * np.random.randn(np_pop)
        f_values = np.clip(f_values, 0.1, 2.0)
        
        # Apply selected mutation strategy
        if selected_operator == 0:  # rand/1
            mutants = (population[r1] + f_values[:, np.newaxis] * 
                      (population[r2] - population[np.roll(r1, -1)]))
        elif selected_operator == 1:  # best/1
            best_idx = np.argmin(fitness)
            mutants = (population[best_idx] + f_values[:, np.newaxis] * 
                      (population[r1] - population[r2]))
        elif selected_operator == 2:  # current-to-pbest/1
            pbest = population[np.random.choice(pbest_idx, np_pop)]
            mutants = (population + f_values[:, np.newaxis] * 
                      (pbest - population[r1] + population[r1] - population[r2]))
        else:  # rand-to-best/1
            best_idx = np.argmin(fitness)
            mutants = ((population[r1] + f_values[:, np.newaxis] * 
                       (population[best_idx] - population[r1] + population[r2] - population[r1])))
        
        # Clip to bounds
        mutants = np.clip(mutants, self.lower_bound, self.upper_bound)
        
        return mutants
    
    def _crossover_batch(self, population, mutants):
        """Binomial crossover between target and mutant vectors."""
        np_pop = len(population)
        
        # Random crossover mask
        cr_mask = np.random.rand(np_pop, self.dim) < self.cr
        
        # Ensure at least one dimension is crossed
        j_rand = np.random.randint(0, self.dim, size=np_pop)
        cr_mask[np.arange(np_pop), j_rand] = True
        
        # Create trial vectors
        trials = np.where(cr_mask, mutants, population)
        
        return trials
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
        """Selection: greedy replacement based on fitness."""
        # Better fitness wins (assuming minimization)
        better_mask = trial_fitness < fitness
        
        # Update population and fitness
        new_population = np.where(better_mask[:, np.newaxis], trials, population)
        new_fitness = np.where(better_mask, trial_fitness, fitness)
        
        return new_population, new_fitness, better_mask
    
    def _apply_local_refinement(self, population, fitness, func):
        """Apply Nelder-Mead-like local search to top individuals."""
        if len(population) < self.dim + 1:
            return population, fitness
        
        n_elite = min(3, len(population))  # Apply to top 3
        
        for i in range(n_elite):
            # Get current best position
            x_best = population[i].copy()
            f_best = fitness[i]
            
            # Build simplex around current best
            simplex = np.tile(x_best, (self.dim + 1, 1))
            simplex[0] = x_best
            
            # Add small perturbations for other vertices
            step_size = 0.5
            for j in range(1, self.dim + 1):
                simplex[j] = x_best.copy()
                simplex[j, (j-1) % self.dim] += step_size * (self.upper_bound - self.lower_bound)
                simplex[j] = np.clip(simplex[j], self.lower_bound, self.upper_bound)
            
            # Evaluate simplex
            simplex_fitness = func(simplex)
            simplex_fitness = self._handle_budget_truncation(simplex_fitness, simplex)
            
            if len(simplex_fitness) < len(simplex):
                break
            
            # Simple Nelder-Mead iteration
            alpha, gamma, rho = 1.0, 2.0, 0.5
            
            # Sort simplex
            sort_idx = np.argsort(simplex_fitness)
            simplex = simplex[sort_idx]
            simplex_fitness = simplex_fitness[sort_idx]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, self.lower_bound, self.upper_bound)
            f_r = func(x_r.reshape(1, -1))[0]
            
            if f_r < simplex_fitness[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, self.lower_bound, self.upper_bound)
                f_e = func(x_e.reshape(1, -1))[0]
                
                if f_e < f_r:
                    new_point, new_f = x_e, f_e
                else:
                    new_point, new_f = x_r, f_r
            elif f_r < simplex_fitness[-2]:
                new_point, new_f = x_r, f_r
            else:
                # Shrink
                new_point = simplex[0] + rho * (simplex[-1] - simplex[0])
                new_point = np.clip(new_point, self.lower_bound, self.upper_bound)
                new_f = func(new_point.reshape(1, -1))[0]
            
            # Update if improvement found
            if new_f < f_best:
                population[i] = new_point
                fitness[i] = new_f
        
        return population, fitness
    
    def _update_operator_rewards(self, selected_operator, success_mask):
        """Update operator rewards based on success rate."""
        success_rate = np.mean(success_mask)
        
        # Reward successful operators
        self.operator_counts[selected_operator] += 1
        if success_rate > 0:
            self.operator_rewards[selected_operator] += success_rate
        
        # Decay rewards of non-selected operators slightly
        for i in range(self.n_strategies):
            if i != selected_operator:
                self.operator_rewards[i] *= 0.99
    
    def _adapt_step_size(self, success_rate):
        """Adapt F and CR based on recent success."""
        if success_rate > 0.2:
            # Good success: maintain or slightly increase exploration
            self.f_adapt = min(0.5, self.f_adapt * 1.1)
        elif success_rate < 0.05:
            # Poor success: reduce step size for exploitation
            self.f_adapt = max(0.01, self.f_adapt * 0.9)
        
        # Adapt CR: increase if stuck, decrease if unstable
        if success_rate > 0.3:
            self.cr = min(0.95, self.cr * 1.01)
        elif success_rate < 0.1:
            self.cr = max(0.5, self.cr * 0.99)
    
    def _compute_diversity(self, population):
        """Compute population diversity using average Euclidean distance."""
        if len(population) < 2:
            return 0.0
        
        # Sample for efficiency
        sample_size = min(50, len(population))
        indices = np.random.choice(len(population), sample_size, replace=False)
        sample = population[indices]
        
        # Pairwise distances
        diff = sample[:, np.newaxis, :] - sample[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Average of upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(distances, dtype=bool), k=1)
        avg_distance = np.mean(distances[mask])
        
        return avg_distance
    
    def _check_stagnation(self, fitness):
        """Check if optimization has stagnated."""
        current_best = np.min(fitness)
        self.best_history.append(current_best)
        
        if len(self.best_history) > self.stagnation_limit:
            self.best_history.pop(0)
        
        if len(self.best_history) >= self.stagnation_limit:
            improvement = self.best_history[-1] - self.best_history[0]
            if abs(improvement) < 1e-8:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
            return self.stagnation_counter >= 2
        
        return False
    
    def _restart_if_needed(self, population, fitness, func, force=False):
        """Restart population if stagnation or diversity loss detected."""
        diversity = self._compute_diversity(population)
        
        if force or diversity < self.diversity_threshold or self.stagnation_counter >= 2:
            # Reinitialize portion of population
            n_reinit = int(0.7 * self.np)
            reinit_idx = np.argsort(fitness)[-n_reinit:]  # Replace worst
            
            # Generate new individuals using LHS
            for d in range(self.dim):
                grid = np.linspace(0, 1, n_reinit + 1)[:-1] + np.random.uniform(0, 1/n_reinit, n_reinit)
                perm = np.random.permutation(n_reinit)
                population[reinit_idx, d] = (self.lower_bound + 
                    (self.upper_bound - self.lower_bound) * grid[perm])
            
            # Evaluate reinitialized portion
            new_fitness = func(population[reinit_idx])
            if len(new_fitness) < len(reinit_idx):
                new_fitness = np.pad(new_fitness, (0, len(reinit_idx) - len(new_fitness)), 
                                     constant_values=np.inf)
            fitness[reinit_idx] = new_fitness
            
            # Reset stagnation counter
            self.stagnation_counter = 0
            self.best_history = []
            
        return population, fitness
    
    def _clip_to_bounds(self, population):
        """Ensure all individuals are within bounds."""
        return np.clip(population, self.lower_bound, self.upper_bound)
    
    def __call__(self, func, stopping_condition):
        """
        Run the optimizer.
        
        Args:
            func: Fitness function accepting (N, dim) array, returning (N,) fitness
            stopping_condition: Callable returning True when to stop
            
        Returns:
            (f_opt, x_opt): Best fitness and corresponding solution
        """
        # Initialize
        population, fitness = self._initialize_population(func)
        
        # Check budget after initialization
        if stopping_condition():
            best_idx = np.argmin(fitness)
            return fitness[best_idx], population[best_idx]
        
        generation = 0
        local_search_counter = 0
        
        # Main loop
        while not stopping_condition():
            # Select mutation operator
            selected_operator = self._select_mutation_operator_batch()
            
            # Generate mutants
            mutants = self._mutate_batch(population, fitness, selected_operator)
            
            # Crossover
            trials = self._crossover_batch(population, mutants)
            
            # Clip trials to bounds
            trials = self._clip_to_bounds(trials)
            
            # Check stopping condition before evaluation
            if stopping_condition():
                break
            
            # Evaluate trials
            trial_fitness = func(trials)
            
            # Handle budget truncation
            if len(trial_fitness) < len(trials):
                valid_trials = len(trial_fitness)
                trials = trials[:valid_trials]
                trial_fitness = trial_fitness[:valid_trials]
                population = population[:valid_trials]
                fitness = fitness[:valid_trials]
                self.np = valid_trials
                if self.np < 10:
                    break
            
            # Check stopping after partial evaluation
            if stopping_condition():
                break
            
            # Selection
            population, fitness, success_mask = self._select_survivors_batch(
                population, fitness, trials, trial_fitness
            )
            
            # Update operator rewards
            self._update_operator_rewards(selected_operator, success_mask)
            
            # Adapt step size
            success_rate = np.mean(success_mask)
            self._adapt_step_size(success_rate)
            
            # Periodic local search
            local_search_counter += 1
            if local_search_counter >= self.local_search_interval:
                population, fitness = self._apply_local_refinement(population, fitness, func)
                local_search_counter = 0
                
                if stopping_condition():
                    break
            
            # Check stagnation
            if self._check_stagnation(fitness):
                population, fitness = self._restart_if_needed(population, fitness, func)
            
            # Periodic restart check based on diversity
            if generation % 10 == 0:
                population, fitness = self._restart_if_needed(population, fitness, func)
            
            generation += 1
        
        # Return best solution
        best_idx = np.argmin(fitness)
        return fitness[best_idx], population[best_idx]

```

Respond with exactly ONE idea using the format:
**Idea: Short Name**
One-line description.
```python
def _mutate_batch(self, ...):
    ...
```