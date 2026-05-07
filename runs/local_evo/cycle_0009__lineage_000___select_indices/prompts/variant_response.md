**Idea 1: Roulette Wheel Fitness‑Proportional Selection**  
Select individuals with probability proportional to the inverse of their fitness (lower fitness → higher chance).  
```python
def _select_indices(self):
    fitness = self.current_fitness
    n = len(fitness)
    if n == 0:
        return np.array([], dtype=int)
    num_selected = max(1, n // 4)
    # Small offset to avoid division by zero
    offset = 1e-12
    min_fit = np.nanmin(fitness)
    weights = 1.0 / (fitness - min_fit + offset)
    probs = weights / weights.sum()
    cumsum = np.cumsum(probs)
    selected = []
    rng = self._rng
    while len(selected) < num_selected:
        r = rng.random()
        idx = int(np.searchsorted(cumsum, r))
        if idx >= n:
            idx = n - 1
        if idx not in selected:
            selected.append(idx)
    return np.array(selected, dtype=int)
```  

**Idea 2: Binary Tournament Selection**  
Randomly pair individuals in tournaments of size √n and pick the best (lowest fitness) from each tournament.  
```python
def _select_indices(self):
    fitness = self.current_fitness
    n = len(fitness)
    if n == 0:
        return np.array([], dtype=int)
    num_selected = max(1, n // 4)
    tournament_size = max(2, int(np.sqrt(n)))
    selected = []
    rng = self._rng
    for _ in range(num_selected):
        cand = rng.choice(n, size=tournament_size, replace=True)
        best = cand[np.argmin(fitness[cand])]
        selected.append(best)
    return np.array(selected, dtype=int)
```  

**Idea 3: Linear Ranking Selection**  
Assign ranks by sorting fitness; probability grows linearly with rank (best rank gets highest weight).  
```python
def _select_indices(self):
    fitness = self.current_fitness
    n = len(fitness)
    if n == 0:
        return np.array([], dtype=int)
    num_selected = max(1, n // 4)
    sorted_idx = np.argsort(fitness)
    # Linear ranking weight: higher for better (lower) rank
    ranks = np.arange(n)
    weights = (n - ranks).astype(float)
    probs = weights / weights.sum()
    cumsum = np.cumsum(probs)
    selected = []
    rng = self._rng
    while len(selected) < num_selected:
        r = rng.random()
        pos = int(np.searchsorted(cumsum, r))
        if pos >= n:
            pos = n - 1
        idx = sorted_idx[pos]
        if idx not in selected:
            selected.append(idx)
    return np.array(selected, dtype=int)
```  

**Idea 4: Exponential Crossover Style Selection**  
Use an exponential decay weighting of ranks to emulate exponential crossover bias toward top individuals.  
```python
def _select_indices(self):
    fitness = self.current_fitness
    n = len(fitness)
    if n == 0:
        return np.array([], dtype=int)
    num_selected = max(1, n // 4)
    sorted_idx = np.argsort(fitness)
    lam = max(1, n // 10)  # decay length
    ranks = np.arange(n, dtype=float)
    weights = np.exp(-ranks / lam)
    probs = weights / weights.sum()
    cumsum = np.cumsum(probs)
    selected = []
    rng = self._rng
    while len(selected) < num_selected:
        r = rng.random()
        pos = int(np.searchsorted(cumsum, r))
        if pos >= n:
            pos = n - 1
        idx = sorted_idx[pos]
        if idx not in selected:
            selected.append(idx)
    return np.array(selected, dtype=int)
```  

**Idea 5: Arithmetic Blend (BLX‑α) Selection**  
Blend the top k individuals (weighted by inverse fitness) into a target point; select the individuals nearest that point.  
```python
def _select_indices(self):
    fitness = self.current_fitness
    pop = self.current_population
    n = len(fitness)
    if n == 0:
        return np.array([], dtype=int)
    num_selected = max(1, n // 4)
    k = max(2, num_selected)
    top_k_idx = np.argsort(fitness)[:k]
    top_fitness = fitness[top_k_idx]
    offset = 1e-12
    inv_w = 1.0 / (top_fitness - top_fitness.min() + offset)
    inv_w /= inv_w.sum()
    target = np.average(pop[top_k_idx], axis=0, weights=inv_w)
    distances = np.linalg.norm(pop - target, axis=1)
    selected = np.argsort(distances)[:num_selected]
    return selected.astype(int)
```  

**Idea 6: Segment‑Based Selection**  
Pick a random contiguous segment of the sorted fitness list of length ⌊pop/4⌋.  
```python
def _select_indices(self):
    fitness = self.current_fitness
    n = len(fitness)
    if n == 0:
        return np.array([], dtype=int)
    num_selected = max(1, n // 4)
    sorted_idx = np.argsort(fitness)
    if n <= num_selected:
        start = 0
    else:
        start = self._rng.integers(0, n - num_selected + 1)
    segment = sorted_idx[start:start + num_selected]
    return segment.astype(int)
```  

**Idea 7: Eigenvector‑Rotated (Mahalanobis) Selection**  
Rotate the population by the covariance eigenvectors, compute Mahalanobis distances, and select based on a combined rank of fitness and distance.  
```python
def _select_indices(self):
    fitness = self.current_fitness
    pop = self.current_population
    n = len(fitness)
    if n == 0:
        return np.array([], dtype=int)
    num_selected = max(1, n // 4)
    mean = self.mean
    cov = self.covariance
    diff = pop - mean
    try:
        cov_inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.eye(self.dim)
    mahal_sq = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)
    mahal_sq = np.clip(mahal_sq, 0.0, None)
    rank_fitness = np.argsort(np.argsort(fitness))
    rank_mahal = np.argsort(np.argsort(mahal_sq))
    combined = rank_fitness + rank_mahal
    selected = np.argsort(combined)[:num_selected]
    return selected.astype(int)
```  

**Idea 8: Diversity‑Aware Greedy (Maximin) Selection**  
Start with the best individual and greedily add the remaining candidate that maximizes the minimal distance to the already‑selected set while also preferring low fitness.  
```python
def _select_indices(self):
    fitness = self.current_fitness
    pop = self.current_population
    n = len(fitness)
    if n == 0:
        return np.array([], dtype=int)
    num_selected = max(1, n // 4)
    best_idx = int(np.argmin(fitness))
    selected = [best_idx]
    remaining = set(range(n)) - {best_idx}
    if not remaining:
        return np.array(selected, dtype=int)
    # pairwise Euclidean distances
    dists = np.linalg.norm(pop[:, np.newaxis, :] - pop[np.newaxis, :, :], axis=2)
    while len(selected) < num_selected:
        max_dist_to_selected = np.array([np.max(dists[i, selected]) for i in remaining])
        fitness_ranks = np.argsort(np.argsort(fitness))
        scores = -fitness_ranks[list(remaining)] + max_dist_to_selected
        next_idx = list(remaining)[np.argmax(scores)]
        selected.append(next_idx)
        remaining.remove(next_idx)
    return np.array(selected, dtype=int)
```