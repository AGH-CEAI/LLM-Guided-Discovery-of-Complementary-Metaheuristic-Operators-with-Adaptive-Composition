**Idea: JADE-style Mutation with Success-Archive**
JADE is a well-established DE variant using current-to-pbest/1 mutation with an archive of rejected solutions, providing fundamentally different exploration dynamics compared to the cultural memory blending approach.

```python
def _mutate_current_to_pbest_with_culture(self, population, fitness):
    # Initialize archive if needed
    if not hasattr(self, 'archive') or self.archive is None:
        self.archive = np.zeros((0, self.dim))
    
    # Select p-best individuals
    sorted_idx = np.argsort(fitness)
    n_best = max(1, int(self.NP * self.p_best_rate))
    p_best_indices = sorted_idx[:n_best]
    p_best_idx = p_best_indices[np.random.randint(0, n_best, size=self.NP)]
    p_best = population[p_best_idx]
    
    # Generate random indices r1, r2 (different from current index i)
    r1 = np.random.randint(0, self.NP, size=self.NP)
    r2 = np.zeros(self.NP, dtype=int)
    
    for i in range(self.NP):
        while True:
            r2[i] = np.random.randint(0, self.NP)
            if r2[i] != i and r2[i] != r1[i]:
                break
    
    # JADE mutation: current-to-pbest/1 with archive
    # Use archive members if available, otherwise only current population
    archive_size = len(self.archive)
    total_size = self.NP + archive_size
    
    if archive_size > 0:
        # Sample from combined population + archive
        all_indices = np.concatenate([np.arange(self.NP), self.archive])
        r3_indices = np.random.randint(0, total_size, size=self.NP)
        r3 = np.where(r3_indices < self.NP, r3_indices, -(r3_indices - self.NP + 1))
        
        # Handle cases where r3 == current index
        for i in range(self.NP):
            if r3[i] == i:
                if r3[i] < self.NP - 1:
                    r3[i] += 1
                else:
                    r3[i] = 0
        r3_pop = np.where(r3 >= 0, population[r3], self.archive[-(r3 + 1)])
    else:
        # No archive: use population only with careful index handling
        r3 = np.random.randint(0, self.NP, size=self.NP)
        for i in range(self.NP):
            while r3[i] == i or r3[i] == r1[i]:
                r3[i] = (r3[i] + 1) % self.NP
        r3_pop = population[r3]
    
    # Apply JADE mutation: x_i + F_i * (pbest - x_i) + F_i * (r3 - r1)
    # Use per-individual F with bounded random component
    F_i = np.clip(self.F + np.random.randn(self.NP) * 0.1, 0.0, 2.0)
    
    donors = population + F_i[:, np.newaxis] * (p_best - population) + F_i[:, np.newaxis] * (r3_pop - population[r1])
    
    # Clip to bounds for numerical stability
    donors = np.clip(donors, self.lower, self.upper)
    
    # Update archive with rejected solutions (called from select after we know rejections)
    # Store for later use - actual archive update happens in survivor selection
    self._pending_archive_additions = None
    
    return donors
```