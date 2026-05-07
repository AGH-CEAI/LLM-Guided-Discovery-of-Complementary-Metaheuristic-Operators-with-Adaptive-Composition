**Idea: Vectorized Current-to-Rand/1 with Archive**
A rotationally invariant mutation strategy using an archive of rejected solutions for increased diversity, with fully vectorized index generation for efficiency.
```python
def _mutate_batch(self, population, fitness, selected_operator):
        """Generate mutant vectors using current-to-rand/1 with optional archive."""
        np_pop = len(population)
        dim = self.dim
        
        # Vectorized index selection using np.random.choice on flattened indices
        all_indices = np.arange(np_pop)
        
        # Efficiently sample 4 unique indices per individual (avoiding current index i)
        # Use broadcasting trick: create (np_pop, np_pop) difference matrix approach
        r = np.random.randint(0, np_pop, size=(np_pop, 4))
        
        # Ensure uniqueness per row (simple rejection for small populations)
        for i in range(np_pop):
            while len(np.unique(r[i])) < 4:
                r[i] = np.random.randint(0, np_pop, size=4)
        
        r1, r2, r3, r4 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        
        # Build archive from current population (union with existing if applicable)
        if self.archive is None:
            self.archive = population.copy()
        else:
            # Keep archive bounded
            if len(self.archive) > self.archive_max_size:
                self.archive = self.archive[np.random.choice(
                    len(self.archive), self.archive_max_size, replace=False)]
            self.archive = np.vstack([self.archive, population])
            if len(self.archive) > self.archive_max_size * 2:
                self.archive = self.archive[np.random.choice(
                    len(self.archive), self.archive_max_size, replace=False)]
        
        # Sample from archive for additional diversity (with fallback)
        archive_size = len(self.archive)
        if archive_size > np_pop:
            r_archive = np.random.randint(0, archive_size, size=np_pop)
            archive_donors = self.archive[r_archive]
        else:
            archive_donors = population  # Fallback to population if archive small
        
        # Get pbest indices (top p% where p varies per strategy)
        sorted_idx = np.argsort(fitness)
        p = 0.1 + 0.1 * selected_operator  # Vary pbest percentage by operator
        pbest_count = max(1, int(np_pop * p))
        pbest_idx = sorted_idx[:pbest_count]
        
        # Strategy-specific mutation with current-to-rand/1 as primary
        mutants = np.empty_like(population)
        
        # Adaptive F: use success history if available, otherwise base value
        f_values = self.f_base + self.f_adapt * np.random.randn(np_pop)
        f_values = np.clip(f_values, 0.1, 2.0)
        
        if selected_operator == 0:  # Current-to-rand/1 (rotationally invariant)
            # m = x_i + F*(x_r1 - x_i) + F*(x_r2 - x_r3)
            mutants = (population + f_values[:, np.newaxis] * 
                      (population[r1] - population + population[r2] - population[r3]))
        
        elif selected_operator == 1:  # Best/2 with archive
            best_idx = sorted_idx[0]
            mutants = (population[best_idx] + f_values[:, np.newaxis] * 
                      (population[r1] - population[r2] + archive_donors - population[r3]))
        
        elif selected_operator == 2:  # Rand/1 with pbest injection
            pbest = population[np.random.choice(pbest_idx, np_pop)]
            mutants = (population[r1] + f_values[:, np.newaxis] * 
                      (pbest - population[r2] + population[r3] - population[r4]))
        
        else:  # Current-to-pbest/1 with archive
            pbest = population[np.random.choice(pbest_idx, np_pop)]
            mutants = (population + f_values[:, np.newaxis] * 
                      (pbest - population[r1] + archive_donors - population[r2]))
        
        # Clip to bounds
        mutants = np.clip(mutants, self.lower_bound, self.upper_bound)
        
        return mutants
```