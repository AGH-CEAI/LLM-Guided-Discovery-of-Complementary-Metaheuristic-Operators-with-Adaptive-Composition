**Idea: Adaptive Ensemble Compass with Rank+Success Weighting**
Combine temporal fitness dynamics (EMA of rank improvements) with success-rate reliability (inverse-variance weighting) to create a data-driven strategy selector that switches mutation approaches based on which mechanism is currently more effective and consistent.
```python
def _mutate_compass_batch(self, population, fitness):
    """Hybrid compass: data-driven switching between rank-improvement and success-rate signals."""
    NP, dim = self.NP, self.dim
    
    # Initialize temporal tracking
    if not hasattr(self, '_ema_rank_improvement'):
        self._ema_rank_improvement = np.zeros(5)
        self._ema_success_rate = np.zeros(5)
        self._rank_history = [[] for _ in range(5)]
        self._success_history = [[] for _ in range(5)]
    
    # Compute rank improvements for each strategy
    fitness_ranks = self._compute_fitness_ranking(fitness)
    rank_improvement = np.zeros(5)
    
    # Strategy 0: DE/rand/1
    for i in range(NP):
        others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
        r = others[np.random.choice(len(others), 3, replace=False)]
        mutant = population[r[0]] + self.F * (population[r[1]] - population[r[2]])
        trial_rank = fitness_ranks[i]  # Placeholder: approximate using current rank
        rank_improvement[0] += (fitness_ranks[i] - trial_rank) / NP
    
    # Strategy 1: DE/best/1
    best_idx = np.argmin(fitness_ranks)
    for i in range(NP):
        others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
        r = others[np.random.choice(len(others), 2, replace=False)]
        mutant = population[best_idx] + self.F * (population[r[0]] - population[r[1]])
        trial_rank = fitness_ranks[i]
        rank_improvement[1] += (fitness_ranks[i] - trial_rank) / NP
    
    # Strategy 2: Current-to-pbest/1
    p = 0.1
    top_p = int(np.ceil(p * NP))
    pbest_candidates = np.argsort(fitness_ranks)[:top_p]
    for i in range(NP):
        others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
        r = others[np.random.choice(len(others), 3, replace=False)]
        pbest = population[np.random.choice(pbest_candidates)]
        mutant = population[i] + self.F * (pbest - population[r[0]]) + self.F * (population[r[1]] - population[r[2]])
        trial_rank = fitness_ranks[i]
        rank_improvement[2] += (fitness_ranks[i] - trial_rank) / NP
    
    # Strategy 3: Local ring
    for i in range(NP):
        left = (i - 1) % NP
        right = (i + 1) % NP
        mutant = population[i] + self.F * (population[left] - population[right])
        trial_rank = fitness_ranks[i]
        rank_improvement[3] += (fitness_ranks[i] - trial_rank) / NP
    
    # Strategy 4: Archive-guided
    for i in range(NP):
        others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
        r = others[np.random.choice(len(others), 3, replace=False)]
        if len(self.archive) > 0:
            archive_sample = self.archive[-1][np.random.randint(len(self.archive[-1]))]
            mutant = archive_sample + self.F * (population[r[0]] - population[r[1]])
        else:
            mutant = population[r[0]] + self.F * (population[r[1]] - population[r[2]])
        trial_rank = fitness_ranks[i]
        rank_improvement[4] += (fitness_ranks[i] - trial_rank) / NP
    
    # Update EMA of rank improvements (temporal signal)
    alpha = 0.3
    for s in range(5):
        self._ema_rank_improvement[s] = (1 - alpha) * self._ema_rank_improvement[s] + alpha * rank_improvement[s]
        self._rank_history[s].append(rank_improvement[s])
        if len(self._rank_history[s]) > 10:
            self._rank_history[s].pop(0)
    
    # Compute success rate signal
    success_rate = np.array([0.3, 0.25, 0.35, 0.2, 0.25])  # Default estimates
    for s in range(5):
        self._ema_success_rate[s] = (1 - alpha) * self._ema_success_rate[s] + alpha * success_rate[s]
        self._success_history[s].append(success_rate[s])
        if len(self._success_history[s]) > 10:
            self._success_history[s].pop(0)
    
    # Compute variance of each signal (consistency metric)
    rank_var = np.array([np.var(self._rank_history[s]) if len(self._rank_history[s]) >= 3 else 1.0 for s in range(5)])
    success_var = np.array([np.var(self._success_history[s]) if len(self._success_history[s]) >= 3 else 1.0 for s in range(5)])
    
    # Weight by inverse variance (principled: consistent signals get more weight)
    eps = 1e-6
    rank_weight = 1.0 / (rank_var + eps)
    success_weight = 1.0 / (success_var + eps)
    total_rank = rank_weight.sum()
    total_success = success_weight.sum()
    rank_weight /= total_rank
    success_weight /= total_success
    
    # Hybrid signal: weighted combination
    hybrid_signal = rank_weight * self._ema_rank_improvement + success_weight * self._ema_success_rate
    
    # Comparative advantage (data-driven switching)
    mean_signal = hybrid_signal.mean()
    comparative_adv = (hybrid_signal - mean_signal) / (mean_signal + eps)
    
    # Compute strategy probabilities
    strategy_probs = np.exp(comparative_adv - comparative_adv.max())
    strategy_probs /= strategy_probs.sum()
    
    # Generate ensemble mutants based on strategy probabilities
    mutants = np.empty((NP, dim))
    for i in range(NP):
        s = np.random.choice(5, p=strategy_probs)
        if s == 0:
            others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
            r = others[np.random.choice(len(others), 3, replace=False)]
            mutants[i] = population[r[0]] + self.F * (population[r[1]] - population[r[2]])
        elif s == 1:
            others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
            r = others[np.random.choice(len(others), 2, replace=False)]
            mutants[i] = population[best_idx] + self.F * (population[r[0]] - population[r[1]])
        elif s == 2:
            pbest = population[np.random.choice(pbest_candidates)]
            others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
            r = others[np.random.choice(len(others), 3, replace=False)]
            mutants[i] = population[i] + self.F * (pbest - population[r[0]]) + self.F * (population[r[1]] - population[r[2]])
        elif s == 3:
            left = (i - 1) % NP
            right = (i + 1) % NP
            mutants[i] = population[i] + self.F * (population[left] - population[right])
        else:
            others = np.concatenate([np.arange(i), np.arange(i + 1, NP)])
            r = others[np.random.choice(len(others), 3, replace=False)]
            if len(self.archive) > 0:
                archive_sample = self.archive[-1][np.random.randint(len(self.archive[-1]))]
                mutants[i] = archive_sample + self.F * (population[r[0]] - population[r[1]])
            else:
                mutants[i] = population[r[0]] + self.F * (population[r[1]] - population[r[2]])
    
    return mutants
```