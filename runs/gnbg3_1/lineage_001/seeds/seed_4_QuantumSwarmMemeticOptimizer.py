import numpy as np


class QuantumSwarmMemeticOptimizer:
    """
    Hybrid optimizer combining:
    - Quantum-inspired state collapse mutation (tunneling between solution states)
    - Swarm-informed gradient approximation (population structure as gradient proxy)
    - Ring-lattice small-world topology for information flow
    - Adaptive metropolis-coupled acceptance (simulated annealing flavor)
    - Periodic Nelder-Mead local search on diverse elite subset
    - Archive of historically good solutions to enrich mutation pool
    """
    
    def __init__(self, dim, NP=None, F_base=0.6, CR=0.85, local_search_freq=8):
        self.dim = dim
        self.NP = NP if NP else min(max(4 * dim, 40), 200)
        self.F_base = F_base
        self.CR = CR
        self.local_search_freq = local_search_freq
        self.bounds = np.array([[-100.0, 100.0]] * dim)
        self.archive = []
        self.archive_max = self.NP * 2
        
    def __call__(self, func, stopping_condition):
        pop = self._initialize_population()
        fitness = func(pop)
        if len(fitness) < self.NP:
            fitness = np.pad(fitness, (0, self.NP - len(fitness)), 'edge')
        f_opt, x_opt = self._update_best(pop, fitness, np.inf, None)
        generation = 0
        
        while not stopping_condition():
            grad = self._approximate_gradient_from_topology(pop, fitness)
            F_adapt = self._adapt_scaling_factor()
            trials = self._quantum_collapse_mutation_batch(pop, grad, F_adapt)
            trials = self._binomial_recombination_batch(pop, trials)
            trials = self._clip_to_bounds_batch(trials)
            
            if stopping_condition():
                break
            trial_fit = func(trials)
            
            if len(trial_fit) < len(trials):
                valid = len(trial_fit)
                trials, trial_fit = trials[:valid], trial_fit[:valid]
                pop, fitness = pop[:valid], fitness[:valid]
            
            if stopping_condition():
                break
            pop, fitness = self._metropolis_select_batch(pop, fitness, trials, trial_fit)
            self._update_archive_batch(pop, fitness)
            f_opt, x_opt = self._update_best(pop, fitness, f_opt, x_opt)
            
            if generation % self.local_search_freq == 0 and generation > 0:
                pop, fitness, f_opt, x_opt = self._apply_conditional_local_search(
                    pop, fitness, func, f_opt, x_opt)
            
            if self._check_diversity_stagnation(fitness):
                pop, fitness = self._reinitialize_diverse_subset(pop, fitness)
            
            generation += 1
        
        return f_opt, x_opt
    
    def _initialize_population(self):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        pop = np.random.uniform(low, high, (self.NP, self.dim))
        return pop
    
    def _get_ring_lattice_neighbors(self, i):
        left = (i - 1) % self.NP
        right = (i + 1) % self.NP
        return np.array([left, right])
    
    def _rewire_shortcut(self, i, rng):
        k = rng.integers(0, self.NP)
        if k != i and k != (i - 1) % self.NP and k != (i + 1) % self.NP:
            return k
        return rng.integers(0, self.NP)
    
    def _approximate_gradient_from_topology(self, pop, fitness):
        grad = np.zeros_like(pop)
        rng = np.random.default_rng()
        for i in range(self.NP):
            neighbors = self._get_ring_lattice_neighbors(i)
            shortcuts = [self._rewire_shortcut(i, rng) for _ in range(2)]
            all_neighbors = np.concatenate([neighbors, shortcuts])
            all_neighbors = np.unique(all_neighbors)
            best_nb = all_neighbors[np.argmin(fitness[all_neighbors])]
            grad[i] = (pop[best_nb] - pop[i]) / (self.NP + 1)
        return grad
    
    def _quantum_collapse_mutation_batch(self, pop, grad, F):
        rng = np.random.default_rng()
        n = len(pop)
        indices = rng.choice(n, n, replace=False)
        r1, r2 = indices[:2]
        archive_sample = pop[r1:r1+1] if len(self.archive) > 0 else pop[:1]
        if len(self.archive) > 0:
            idx = rng.integers(0, min(len(self.archive), n))
            archive_sample = np.atleast_2d(self.archive[idx % len(self.archive)])
        tunnel_vector = pop[r2:r2+1] - archive_sample
        quantum_collapse = pop[r1:r1+1] + F * tunnel_vector
        quantum_collapse = np.clip(quantum_collapse, self.bounds[:, 0], self.bounds[:, 1])
        grad_component = grad * rng.uniform(0.0, 0.3)
        mutant = pop + F * (quantum_collapse - pop) + grad_component
        return mutant
    
    def _binomial_recombination_batch(self, target, mutant):
        rng = np.random.default_rng()
        mask = rng.random((self.NP, self.dim)) < self.CR
        trials = np.where(mask, mutant, target)
        j_rand = rng.integers(0, self.dim, size=self.NP)
        trials[np.arange(self.NP), j_rand] = mutant[np.arange(self.NP), j_rand]
        return trials
    
    def _clip_to_bounds_batch(self, pop):
        return np.clip(pop, self.bounds[:, 0], self.bounds[:, 1])
    
    def _metropolis_select_batch(self, pop, fitness, trials, trial_fit):
        temp = max(1e-6, np.std(fitness) * 0.1)
        delta = trial_fit - fitness
        accept_prob = np.exp(-delta / temp)
        accepted = np.random.random(len(fitness)) < accept_prob
        pop[accepted] = trials[accepted]
        fitness[accepted] = trial_fit[accepted]
        return pop, fitness
    
    def _update_archive_batch(self, pop, fitness):
        sorted_indices = np.argsort(fitness)
        for idx in sorted_indices[:self.NP // 4]:
            if len(self.archive) >= self.archive_max:
                remove_idx = np.random.randint(0, len(self.archive))
                self.archive.pop(remove_idx)
            self.archive.append(pop[idx].copy())
    
    def _adapt_scaling_factor(self):
        jitter = np.random.uniform(-0.1, 0.1)
        return np.clip(self.F_base + jitter, 0.3, 1.2)
    
    def _update_best(self, pop, fitness, f_opt_prev, x_opt_prev):
        best_idx = np.argmin(fitness)
        f_opt = fitness[best_idx]
        x_opt = pop[best_idx].copy() if f_opt < f_opt_prev else x_opt_prev
        return f_opt, x_opt
    
    def _check_diversity_stagnation(self, fitness):
        if len(fitness) < 2:
            return True
        return np.std(fitness) < 1e-8
    
    def _reinitialize_diverse_subset(self, pop, fitness):
        n_replace = self.NP // 3
        sorted_idx = np.argsort(fitness)
        keep = sorted_idx[n_replace:]
        replaced = np.random.choice(keep, n_replace, replace=False)
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        pop[replaced] = np.random.uniform(low, high, (n_replace, self.dim))
        fitness[replaced] = np.inf
        return pop, fitness
    
    def _apply_conditional_local_search(self, pop, fitness, func, f_opt, x_opt):
        elite_count = max(2, self.NP // 10)
        sorted_idx = np.argsort(fitness)
        improved = False
        for k in range(elite_count):
            idx = sorted_idx[k]
            simplex_idx = sorted_idx[:max(2, self.dim + 1)]
            if len(simplex_idx) < self.dim + 1:
                simplex_idx = np.concatenate([simplex_idx, np.delete(np.arange(self.NP), simplex_idx)[:self.dim + 1 - len(simplex_idx)]])
            simplex = pop[simplex_idx]
            simplex_fit = fitness[simplex_idx]
            new_point, new_fit = self._nelder_mead_step(simplex, simplex_fit, func)
            if new_fit < fitness[idx]:
                pop[idx] = new_point
                fitness[idx] = new_fit
                if new_fit < f_opt:
                    f_opt = new_fit
                    x_opt = new_point.copy()
                    improved = True
        if improved:
            self.archive = []
        return pop, fitness, f_opt, x_opt
    
    def _nelder_mead_step(self, simplex, simplex_fit, func):
        n = len(simplex)
        if n < 2:
            return simplex[0], simplex_fit[0]
        sorted_idx = np.argsort(simplex_fit)
        simplex, simplex_fit = simplex[sorted_idx], simplex_fit[sorted_idx]
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        reflected = centroid + (centroid - worst)
        reflected = np.clip(reflected, self.bounds[:, 0], self.bounds[:, 1])
        reflected_fit = func(reflected.reshape(1, -1))[0]
        if reflected_fit < simplex_fit[0]:
            expanded = centroid + 2.0 * (reflected - centroid)
            expanded = np.clip(expanded, self.bounds[:, 0], self.bounds[:, 1])
            expanded_fit = func(expanded.reshape(1, -1))[0]
            if expanded_fit < reflected_fit:
                return expanded, expanded_fit
            return reflected, reflected_fit
        elif reflected_fit < simplex_fit[-2]:
            return reflected, reflected_fit
        else:
            contracted = centroid + 0.5 * (worst - centroid)
            contracted = np.clip(contracted, self.bounds[:, 0], self.bounds[:, 1])
            contracted_fit = func(contracted.reshape(1, -1))[0]
            if contracted_fit < simplex_fit[-1]:
                return contracted, contracted_fit
            else:
                shrink = simplex[0] + 0.5 * (simplex - simplex[0])
                shrink = np.clip(shrink, self.bounds[:, 0], self.bounds[:, 1])
                shrink_fit = func(shrink.reshape(1, -1))[0]
                return shrink, shrink_fit
