Given benchmark results on 24 optimization test functions, create an ADAPTIVE version of the algorithm that automatically selects the best `_select_survivors_batch` strategy during optimization.

BENCHMARK RESULTS — per-task errors (lower = better, -inf = crashed):
Task   original.py             variant_01_idea_0.py    variant_02_idea_0.py    variant_03_idea_0.py    variant_04_idea_0.py    variant_05_idea_0.py    variant_06_idea_0.py    variant_07_idea_0.py    variant_08_idea_0.py    variant_09_idea_0.py    variant_10_idea_0.py    
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0      5.581447e-02            9.431159e-01            1.429044e+01            7.107882e-02            7.798456e-01            1.458152e+02            5.605277e-02            -inf                    5.945102e+00            3.948667e+01            2.403244e+01            
1      4.225025e-02            9.951561e-01            2.620190e+01            3.114219e-01            6.522157e-01            2.884579e+02            3.944812e-02            -inf                    3.876921e+00            7.676326e+01            3.934245e+01            
2      1.550907e-01            7.317754e+00            1.393210e+02            6.064704e+00            3.839803e+00            1.132511e+03            1.447677e-01            -inf                    1.456204e+01            3.476995e+02            2.490324e+02            
3      1.731658e-01            7.299587e+00            8.082337e+01            5.517719e+00            6.458938e+00            4.282329e+02            1.348500e-01            -inf                    8.960210e+00            1.735627e+02            1.217776e+02            
4      5.133402e-01            1.602752e+00            3.778741e+00            8.794281e-01            1.245305e+00            6.577591e+00            5.116451e-01            -inf                    1.552157e+00            4.570400e+00            4.397651e+00            
5      1.328620e-04            1.150671e+02            9.096319e+06            1.096424e+02            1.007622e+02            1.797801e+10            9.913827e-05            -inf                    2.530511e+02            2.268514e+08            1.398628e+07            
6      1.577348e-02            2.349061e+02            9.868771e+02            1.869211e-01            3.338955e+01            1.679298e+04            1.562840e-02            -inf                    3.152722e+02            2.969435e+03            9.853636e+02            
7      8.061778e-02            5.075038e+00            7.007701e+01            3.579734e+00            3.624329e+00            6.238349e+02            8.550361e-02            -inf                    1.113270e+01            1.790562e+02            1.254877e+02            
8      2.568841e-01            3.525069e+00            1.844011e+01            2.226276e+00            2.971315e+00            6.489830e+01            2.949587e-01            -inf                    3.201104e+00            3.242788e+01            3.029020e+01            
9      9.359551e-02            1.314317e+01            8.962781e+01            2.224634e+00            1.505538e+00            5.653659e+02            7.237658e-02            -inf                    1.250991e+01            1.932095e+02            6.157522e+01            
10     1.004256e+01            1.122455e+02            7.028471e+01            1.673007e+01            1.251356e+02            5.670481e+02            5.379250e+01            -inf                    2.214016e+01            3.349672e+02            1.621356e+02            
11     1.611530e+01            4.138419e+02            2.581300e+02            4.060257e+01            2.835795e+02            2.212466e+03            1.112419e+02            -inf                    6.318537e+01            1.243870e+03            4.582428e+02            
12     1.431520e+01            1.252856e+02            6.912613e+01            2.063501e+01            1.031410e+02            3.480543e+02            4.955929e+01            -inf                    2.128125e+01            2.254414e+02            1.511083e+02            
13     3.185759e+00            2.365486e+01            2.281779e+01            5.954663e+00            2.081289e+01            6.019802e+01            8.429661e+00            -inf                    1.110711e+01            4.309866e+01            3.336864e+01            
14     2.765349e+00            9.459018e+00            9.739954e+00            2.785644e+00            3.487899e+00            2.260057e+01            2.794531e+00            -inf                    4.325956e+00            1.801391e+01            8.858764e+00            
15     2.708644e+00            3.542430e+00            3.391276e+00            2.923188e+00            3.646815e+00            4.507390e+00            3.268229e+00            -inf                    2.760636e+00            4.312907e+00            3.826012e+00            
16     7.913191e+01            1.107466e+03            2.176049e+03            1.464651e+02            2.837547e+01            5.299838e+04            1.164794e+02            -inf                    4.332460e+02            1.389057e+04            1.313701e+03            
17     4.045684e+01            4.014830e+04            3.047897e+04            2.125268e+03            5.207409e+04            4.759524e+05            9.117668e+03            -inf                    1.604785e+03            2.006874e+05            9.176499e+04            
18     1.283604e+01            3.855690e+01            4.248719e+01            1.823641e+01            4.924371e+01            1.103699e+02            2.740403e+01            -inf                    2.012064e+01            8.064135e+01            5.994715e+01            
19     1.339200e+01            7.099093e+01            1.415007e+02            1.737406e+01            9.564903e+01            6.618199e+02            1.880135e+01            -inf                    4.220874e+01            4.364005e+02            1.319661e+02            
20     1.828716e+01            2.820777e+01            3.174129e+01            1.858934e+01            3.232065e+01            8.233431e+01            2.335172e+01            -inf                    2.448520e+01            6.123205e+01            3.690373e+01            
21     4.361961e+00            5.537830e+00            5.515852e+00            4.648923e+00            5.009904e+00            6.114549e+00            4.638819e+00            -inf                    4.676062e+00            5.919717e+00            5.648366e+00            
22     8.522054e+00            1.513495e+01            1.289889e+01            8.097668e+00            1.411253e+01            2.523612e+01            1.224348e+01            -inf                    1.041756e+01            2.142152e+01            1.599444e+01            
23     1.942306e+01            4.437308e+01            5.622802e+01            1.875027e+01            2.240480e+01            1.783652e+02            2.188021e+01            -inf                    3.995852e+01            1.205107e+02            4.976410e+01            

BEST VARIANT PER FUNCTION (non-trivial tasks only):
Task  0: original.py  (error=5.581447e-02)
Task  1: variant_06_idea_0.py  (error=3.944812e-02)
Task  2: variant_06_idea_0.py  (error=1.447677e-01)
Task  3: variant_06_idea_0.py  (error=1.348500e-01)
Task  4: variant_06_idea_0.py  (error=5.116451e-01)
Task  5: variant_06_idea_0.py  (error=9.913827e-05)
Task  6: variant_06_idea_0.py  (error=1.562840e-02)
Task  7: original.py  (error=8.061778e-02)
Task  8: original.py  (error=2.568841e-01)
Task  9: variant_06_idea_0.py  (error=7.237658e-02)
Task 10: original.py  (error=1.004256e+01)
Task 11: original.py  (error=1.611530e+01)
Task 12: original.py  (error=1.431520e+01)
Task 13: original.py  (error=3.185759e+00)
Task 14: original.py  (error=2.765349e+00)
Task 15: original.py  (error=2.708644e+00)
Task 16: variant_04_idea_0.py  (error=2.837547e+01)
Task 17: original.py  (error=4.045684e+01)
Task 18: original.py  (error=1.283604e+01)
Task 19: original.py  (error=1.339200e+01)
Task 20: original.py  (error=1.828716e+01)
Task 21: original.py  (error=4.361961e+00)
Task 22: variant_03_idea_0.py  (error=8.097668e+00)
Task 23: variant_03_idea_0.py  (error=1.875027e+01)

WIN COUNTS:
  original.py: 14 wins
  variant_06_idea_0.py: 7 wins
  variant_03_idea_0.py: 2 wins
  variant_04_idea_0.py: 1 wins

WINNING VARIANT IMPLEMENTATIONS:
# --- From variant_03_idea_0.py (2 wins) ---
```python
def _select_survivors_batch(self):
        """Select survivors via deterministic crowding for diversity maintenance."""
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])

        sorted_indices = np.argsort(combined_fit)

        # Adaptive diversity threshold based on convergence state
        fit_variance = np.var(combined_fit)
        fit_range = np.max(combined_fit) - np.min(combined_fit) + 1e-10
        normalized_variance = fit_variance / fit_range

        # Base threshold scales with problem dimension
        base_threshold = (self.ub[0] - self.lb[0]) * 0.005 * np.sqrt(self.dim / 100.0)

        # Increase threshold when population converges to escape local optima
        if normalized_variance < 0.01:
            diversity_threshold = base_threshold * 20.0
        elif normalized_variance < 0.05:
            diversity_threshold = base_threshold * 10.0
        elif normalized_variance < 0.1:
            diversity_threshold = base_threshold * 5.0
        else:
            diversity_threshold = base_threshold * 2.0

        survivors_pop = []
        survivors_fit = []

        # Always preserve the best individual (elitism)
        best_idx = sorted_indices[0]
        survivors_pop.append(combined_pop[best_idx].copy())
        survivors_fit.append(combined_fit[best_idx])

        # Deterministic crowding: select diverse, high-quality individuals
        for idx in sorted_indices[1:]:
            candidate = combined_pop[idx]
            candidate_fit = combined_fit[idx]

            # Find most similar survivor
            min_dist = float('inf')
            most_similar_idx = 0
            for s_idx, surv in enumerate(survivors_pop):
                dist = np.linalg.norm(candidate - surv)
                if dist < min_dist:
                    min_dist = dist
                    most_similar_idx = s_idx

            # If candidate is diverse enough, add it
            if min_dist >= diversity_threshold:
                survivors_pop.append(candidate.copy())
                survivors_fit.append(candidate_fit)
            # If candidate is better than its most similar survivor, replace it
            elif candidate_fit < survivors_fit[most_similar_idx]:
                survivors_pop[most_similar_idx] = candidate.copy()
                survivors_fit[most_similar_idx] = candidate_fit

            if len(survivors_pop) >= self.NP:
                break

        # If we haven't filled NP slots, fill with best remaining (no diversity check)
        if len(survivors_pop) < self.NP:
            for idx in sorted_indices:
                if combined_pop[idx].tolist() not in [p.tolist() for p in survivors_pop]:
                    survivors_pop.append(combined_pop[idx].copy())
                    survivors_fit.append(combined_fit[idx])
                if len(survivors_pop) >= self.NP:
                    break

        self.population = np.array(survivors_pop[:self.NP])
        self.fitness = np.array(survivors_fit[:self.NP])

        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
```

# --- From variant_04_idea_0.py (1 wins) ---
```python
def _select_survivors_batch(self):
        """Select survivors via deterministic crowding: match trials to nearest parents by distance, keep the better of each pair."""
        pop_size = self.NP
        num_trials = min(len(self.trials), pop_size)

        if num_trials == 0 or len(self.population) == 0:
            combined_pop = np.vstack([self.population, self.trials])
            combined_fit = np.concatenate([self.fitness, self.trial_fitness])
            sorted_indices = np.argsort(combined_fit)
            self.population = combined_pop[sorted_indices[:self.NP]]
            self.fitness = combined_fit[sorted_indices[:self.NP]]
        else:
            # Compute pairwise distances between trials and population
            # trials: (num_trials, dim), population: (pop_size, dim)
            # dists[i, j] = distance from trial i to population member j
            dists = np.zeros((num_trials, pop_size))
            for i in range(num_trials):
                diffs = self.trials[i] - self.population  # (pop_size, dim)
                dists[i] = np.sum(diffs * diffs, axis=1)  # squared distances

            # For each trial, find the nearest parent
            nearest_parent = np.argmin(dists, axis=1)

            # Build candidate set: compare each trial to its nearest parent
            # Keep the better one for each pair
            candidates_pop = []
            candidates_fit = []
            used_parents = set()

            # Sort trials by fitness (best first) to prioritize improvements
            trial_order = np.argsort(self.trial_fitness[:num_trials])

            for ti in trial_order:
                parent_idx = nearest_parent[ti]
                trial_fit = self.trial_fitness[ti]
                parent_fit = self.fitness[parent_idx]

                if trial_fit <= parent_fit:
                    # Trial is better or equal - use trial
                    candidates_pop.append(self.trials[ti])
                    candidates_fit.append(trial_fit)
                    used_parents.add(parent_idx)
                else:
                    # Parent is better - use parent (if not already selected)
                    if parent_idx not in used_parents:
                        candidates_pop.append(self.population[parent_idx])
                        candidates_fit.append(parent_fit)
                        used_parents.add(parent_idx)

            # Add remaining unused parents (fill up to NP)
            for i in range(pop_size):
                if i not in used_parents:
                    candidates_pop.append(self.population[i])
                    candidates_fit.append(self.fitness[i])
                    if len(candidates_pop) >= pop_size:
                        break

            # Trim or pad to exactly NP
            candidates_pop = np.array(candidates_pop)
            candidates_fit = np.array(candidates_fit)

            if len(candidates_pop) > pop_size:
                # If more candidates than slots, keep the best NP
                keep_idx = np.argsort(candidates_fit)[:pop_size]
                self.population = candidates_pop[keep_idx]
                self.fitness = candidates_fit[keep_idx]
            elif len(candidates_pop) < pop_size:
                # If fewer candidates, add random individuals to fill
                self.population = list(candidates_pop)
                self.fitness = list(candidates_fit)
                remaining = pop_size - len(candidates_pop)
                rand_indices = np.random.choice(pop_size, remaining, replace=True)
                for idx in rand_indices:
                    self.population.append(self.population[idx])
                    self.fitness.append(self.fitness[idx])
                self.population = np.array(self.population)
                self.fitness = np.array(self.fitness)
            else:
                self.population = candidates_pop
                self.fitness = candidates_fit

        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
```

# --- From variant_06_idea_0.py (7 wins) ---
```python
def _select_survivors_batch(self):
        """Select survivors via deterministic crowding with niching to preserve diversity."""
        combined_pop = np.vstack([self.population, self.trials])
        combined_fit = np.concatenate([self.fitness, self.trial_fitness])

        # Determine number of offspring
        n_trials = len(self.trials)
        n_parents = len(self.population)

        # Compute pairwise distances between trials and parents
        # Use efficient broadcasting: (n_trials, 1, dim) - (1, n_parents, dim)
        trials_exp = self.trials[:, np.newaxis, :]  # (n_trials, 1, dim)
        parents_exp = self.population[np.newaxis, :, :]  # (1, n_parents, dim)
        dists = np.linalg.norm(trials_exp - parents_exp, axis=2)  # (n_trials, n_parents)

        # For each trial, find its most similar parent
        closest_parent = np.argmin(dists, axis=1)  # (n_trials,)

        # Build new population through deterministic crowding
        new_pop = []
        new_fit = []

        # Add all parents as candidates
        for i in range(n_parents):
            new_pop.append(self.population[i])
            new_fit.append(self.fitness[i])

        # For each trial, compare with its closest parent
        for i in range(n_trials):
            parent_idx = closest_parent[i]
            trial_fit = combined_fit[n_parents + i]
            parent_fit = self.fitness[parent_idx]

            # Keep whichever has better fitness (minimization)
            if trial_fit < parent_fit:
                new_pop.append(self.trials[i])
                new_fit.append(trial_fit)
            else:
                # Still add parent (already in list, will deduplicate later)
                pass

        # If population exceeds NP, use fitness-based pruning with diversity bonus
        if len(new_pop) > self.NP:
            new_pop = np.array(new_pop)
            new_fit = np.array(new_fit)

            # Compute diversity contribution of each individual
            pop_stack = np.vstack(new_pop)

            # Pairwise distances for diversity
            pdists = np.linalg.norm(pop_stack[:, np.newaxis, :] - pop_stack[np.newaxis, :, :], axis=2)

            # Average distance to others (diversity score)
            avg_dist = np.mean(pdists, axis=1)

            # Combined score: prioritize fitness but add small diversity bonus
            # Normalize fitness to [0, 1] range for combination
            fit_min, fit_max = np.min(new_fit), np.max(new_fit)
            fit_range = max(fit_max - fit_min, 1e-10)
            fit_norm = (new_fit - fit_min) / fit_range

            # Diversity-normalized selection score
            diversity_weight = 0.05
            selection_score = fit_norm - diversity_weight * (avg_dist / (np.max(avg_dist) + 1e-10))

            # Select best NP individuals
            best_indices = np.argsort(selection_score)[:self.NP]

            new_pop = pop_stack[best_indices]
            new_fit = new_fit[best_indices]
        else:
            new_pop = np.array(new_pop[:self.NP]) if len(new_pop) >= self.NP else np.array(new_pop)
            new_fit = np.array(new_fit[:self.NP]) if len(new_fit) >= self.NP else np.array(new_fit)

        # Ensure we have exactly NP individuals
        if len(new_pop) < self.NP:
            # Pad with random individuals from combined population
            deficit = self.NP - len(new_pop)
            combined_sorted = np.argsort(combined_fit)
            padding_indices = combined_sorted[len(new_pop):len(new_pop) + deficit]
            padding_pop = combined_pop[padding_indices]
            padding_fit = combined_fit[padding_indices]
            new_pop = np.vstack([new_pop, padding_pop])
            new_fit = np.concatenate([new_fit, padding_fit])

        self.population = new_pop[:self.NP]
        self.fitness = new_fit[:self.NP]

        self.old_mean = self.mean.copy()
        self.mean = np.mean(self.population, axis=0)
        self.generation += 1
```

Requirements:
- Respond with the COMPLETE class code inside a single ```python``` block
- Include `import numpy as np` at the top
- Use an adaptive mechanism (Thompson Sampling, Multi-Armed Bandit, or sliding window credit assignment) to learn which operator works best during a run
- You may add new attributes in __init__ and new private helper methods
- Do NOT change signatures of existing public methods (__call__, crossover, mutation, selection, etc.)
- The class MUST be callable as: `obj(func, stopping_condition)` → (f_opt, x_opt)
- Handle ALL edge cases: no -inf, no crashes, no NaN, clip to bounds
- Use integer-index-based operator tracking (not id()-based)
- Cast all reward/fitness values to float scalars before storing

Original algorithm:
```python
import numpy as np


class AdaptiveCovarianceEvolutionStrategy:
    """
    Adaptive optimizer that automatically selects the best covariance adaptation
    strategy during optimization using Thompson Sampling with sliding window credit assignment.
    
    Features:
    - 5 covariance adaptation strategies (original + 4 variants)
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
        
        # Adaptive operator selection parameters
        self.num_operators = 5
        self.operator_names = ['original', 'variant_01', 'variant_06', 'variant_08', 'variant_09']
        
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
        
        best_idx = np.argmin(self.fitness)
        self.f_opt = float(np.asarray(self.fitness[best_idx]).flatten()[0])
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
    
    def _sample_trials_batch(self):
        """Sample new trial population using Cholesky decomposition."""
        # Ensure positive definiteness before Cholesky
        min_eig = np.min(np.linalg.eigvalsh(self.C))
        if min_eig < 1e-8:
            self.C += (1e-7 - min_eig) * np.eye(self.dim)

        # Compute Cholesky factor L where C = L @ L.T
        # More efficient than eigendecomposition for sampling
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
    
    def _select_operator_thompson(self):
        """Select operator using Thompson Sampling from Beta distributions."""
        samples = np.random.beta(self.alpha, self.beta)
        self.current_operator = int(np.argmax(samples))
        self.operator_counts[self.current_operator] += 1
        return self.current_operator
    
    def _update_operator_rewards(self):
        """Update operator rewards using sliding window credit assignment."""
        improvement = max(0.0, self.f_opt_prev - self.f_opt)
        
        pop_variance = np.mean(np.var(self.population, axis=0))
        expected_var = ((self.ub[0] - self.lb[0]) / 6.0) ** 2
        diversity = np.clip(pop_variance / (expected_var + 1e-10), 0.0, 1.0)
        
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
    
    def _adapt_covariance(self):
        """Dispatch to the selected covariance adaptation strategy."""
        if self.current_operator == 0:
            self._adapt_covariance_original()
        elif self.current_operator == 1:
            self._adapt_covariance_variant_01()
        elif self.current_operator == 2:
            self._adapt_covariance_variant_06()
        elif self.current_operator == 3:
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

        # Initialize archive for tracking distinct best solutions
        if not hasattr(self, 'archive_positions'):
            self.archive_positions = []
            self.archive_fitness = []
            self.archive_generations = []

        # Add current best to archive if distinct enough
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
            # Replace worst entry
            worst_idx = np.argmax(self.archive_fitness)
            self.archive_positions[worst_idx] = self.x_opt.copy()
            self.archive_fitness[worst_idx] = self.f_opt
            self.archive_generations[worst_idx] = self.generation

        # Compute diversity and condition metrics
        fit_var = np.var(self.fitness)
        fit_scale = max(abs(self.f_opt), 1.0)
        rel_var = fit_var / (fit_scale ** 2 + 1e-10)

        eigvals = np.linalg.eigvalsh(self.C)
        eig_min = np.min(eigvals)
        eig_max = np.max(eigvals)
        eig_spread = eig_min / (eig_max + 1e-10)
        cond = eig_max / (max(eig_min, 1e-10))

        # Detect trapping: low variance OR ill-conditioned OR collapsed spectrum
        is_trapped = (rel_var < 1e-3) or (cond > 1e5) or (eig_spread < 1e-5)

        # Compute archive-based escape direction
        escape_perturb = np.zeros((self.dim, self.dim))
        if is_trapped and len(self.archive_positions) >= 3:
            # Direction from mean toward archive centroid, weighted by diversity
            centroid = np.mean(self.archive_positions, axis=0)
            toward_centroid = centroid - self.mean
            norm_toward = np.linalg.norm(toward_centroid)
            if norm_toward > 1e-10:
                toward_centroid /= norm_toward
                # Also consider directions to individual archive members
                for arch_pos in self.archive_positions:
                    dir_to_arch = arch_pos - self.mean
                    norm_dir = np.linalg.norm(dir_to_arch)
                    if norm_dir > 1e-10:
                        dir_to_arch /= norm_dir
                        # Perturb along diverse directions
                        escape_perturb += 0.3 * np.outer(dir_to_arch, dir_to_arch)
            # Add global exploration along principal axes
            escape_perturb += 0.1 * np.eye(self.dim)

        # Adaptive learning rates with exploration boost when trapped
        if is_trapped:
            ccov_scale = 0.8
            cc_scale = 0.8
        else:
            ccov_scale = min(1.0, 0.2 + 2.0 * rel_var)
            cc_scale = 1.0

        ccov_1 = ccov_scale * 1.0 / (self.dim + 2.0)
        ccov_mu = ccov_scale * self.mueff / (self.dim + 2.0) / (self.dim + 4.0)
        cc_adapt = cc_scale * (4.0 + self.mueff / self.dim) / (self.dim + 4.0 + 2.0 * self.mueff / self.dim)

        # Evolution path update
        self.pc = (1.0 - cc_adapt) * self.pc + np.sqrt(cc_adapt * (2.0 - cc_adapt)) * y_mean

        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i in range(self.mu):
            diff = (self.population[i] - self.old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(diff, diff)

        # Combine updates with archive-based perturbation
        self.C = ((1.0 - ccov_1 - ccov_mu) * self.C +
                  ccov_1 * rank_one +
                  ccov_mu * rank_mu)

        # Inject escape perturbation when trapped
        if is_trapped and len(self.archive_positions) >= 3:
            self.C = self.C + escape_perturb

        self.C = self._ensure_positive_definite(self.C)

        # Full restart on extreme ill-conditioning
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

        # Evolution path update (unchanged)
        self.pc = (1.0 - self.cc) * self.pc + np.sqrt(self.cc * (2.0 - self.cc)) * y_mean

        # Rank-one update
        rank_one = np.outer(self.pc, self.pc)

        # Rank-mu update with positive/negative decomposition
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

        # Normalize by sum of positive and negative weights separately
        pos_sum = max(np.sum(self.weights[:self.mu][self.weights[:self.mu] > 0]), 1e-10)
        neg_sum = max(np.sum(np.abs(self.weights[:self.mu][self.weights[:self.mu] < 0])), 1e-10)

        rank_mu_pos /= pos_sum
        rank_mu_neg /= neg_sum

        # Combine positive and negative updates (active CMA-ES core)
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
    
    def _restart_if_needed(self):
        """Restart population if stagnated or diversity lost."""
        diversity = self._compute_diversity()
        
        if (self.stagnation_counter > self.max_stagnation or 
            diversity < self.min_diversity or 
            np.any(np.isnan(self.C))):
            
            best_idx = np.argmin(self.fitness)
            elite = self.population[best_idx].copy()
            elite_fit = self.fitness[best_idx]
            
            self._initialize_population()
            
            self.population[0] = elite
            self.fitness[0] = elite_fit
            self.f_opt = float(np.asarray(elite_fit).flatten()[0])
            self.x_opt = elite.copy()
            self.f_opt_prev = self.f_opt
            self.stagnation_counter = 0

```