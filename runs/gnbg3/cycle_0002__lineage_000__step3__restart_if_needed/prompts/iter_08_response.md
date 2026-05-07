**Idea: Fitness-Distribution + Rank-Entropy Hybrid Restart**

Category H: Combines fitness variance analysis (convergence detection) with rank-entropy measurement (diversity collapse detection) using data-driven signal weighting to choose between mild preservation restart and aggressive full reinitialization.

```python
def _restart_if_needed(self, population, fitness):
    """Hybrid restart combining two distinct mechanisms with data-driven switching.
    
    Mechanism 1 - Fitness Variance Analysis (Category D): Detects convergence by
    measuring how similar fitness values are. Low variance = population has settled.
    
    Mechanism 2 - Rank Distribution Entropy (Category D): Measures diversity collapse
    by checking if ranks are uniformly distributed. Skewed ranks = few solutions dominate.
    
    The weighting between mild vs aggressive restart is determined by comparing
    the relative magnitudes of these two signals (principled, not arbitrary).
    """
    valid_mask = ~np.isnan(fitness)
    if valid_mask.sum() < 2:
        return population, np.nanmin(fitness), population[np.nanargmin(fitness)]
    
    valid_fitness = fitness[valid_mask]
    
    # Mechanism 1: Fitness variance ratio (normalized by range)
    f_min, f_max = valid_fitness.min(), valid_fitness.max()
    f_range = f_max - f_min + 1e-10
    variance_ratio = np.var(valid_fitness) / f_range
    
    # Mechanism 2: Rank distribution entropy (uniform = high entropy = diverse)
    ranks = np.argsort(np.argsort(valid_fitness))
    rank_probs = (ranks + 1) / (len(ranks) + 1)
    rank_probs = np.clip(rank_probs, 1e-10, 1 - 1e-10)
    rank_entropy = -np.mean(rank_probs * np.log(rank_probs))
    max_entropy = np.log(len(ranks) + 1)
    entropy_ratio = rank_entropy / (max_entropy + 1e-10)
    
    # Data-driven switch: compare signal magnitudes
    # Low variance_ratio + low entropy_ratio = both convergence AND collapse
    # In this case, favor aggressive restart
    convergence_signal = variance_ratio * entropy_ratio
    collapse_signal = (1 - entropy_ratio) * (1 - variance_ratio)
    
    # Determine restart aggressiveness
    threshold = 1e-4
    is_converged = convergence_signal < threshold
    is_collapsed = collapse_signal > (1 - threshold)
    
    # Track restart history for adaptive thresholding
    if not hasattr(self, '_restart_history'):
        self._restart_history = []
    self._restart_history.append(float(is_converged or is_collapsed))
    if len(self._restart_history) > 10:
        self._restart_history.pop(0)
    
    # Adaptive threshold based on recent restart frequency
    recent_restart_rate = np.mean(self._restart_history)
    adaptive_threshold = threshold * (1 + recent_restart_rate)
    should_restart = convergence_signal < adaptive_threshold or entropy_ratio < 0.1
    
    if not should_restart:
        return population, np.nanmin(fitness), population[np.nanargmin(fitness)]
    
    # Decide between mild and aggressive based on signal dominance
    use_mild = (variance_ratio > 0.1) or (entropy_ratio > 0.3)
    
    # Prepare archive for guided restart
    n_archive = min(len(self.archive) * self.NP // 4, self.NP * 2) if self.archive else 0
    if n_archive > 0:
        archive_concat = np.vstack([a for a in self.archive if len(a) > 0])
        if len(archive_concat) > n_archive:
            idx = np.random.choice(len(archive_concat), n_archive, replace=False)
            archive_samples = archive_concat[idx]
        else:
            archive_samples = archive_concat
    else:
        archive_samples = None
    
    # Update archive with current best solutions
    n_best = max(1, self.NP // 10)
    best_idx = np.argsort(valid_fitness)[:n_best]
    if not self.archive:
        self.archive = []
    self.archive.append(population[best_idx].copy())
    if sum(len(a) for a in self.archive) > self.archive_max:
        self.archive.pop(0)
    
    if use_mild:
        # Mild restart: preserve top performers, reinitialize rest
        new_pop = population.copy()
        worst_idx = np.argsort(valid_fitness)[-max(2, self.NP // 3):]
        
        for i, idx in enumerate(worst_idx):
            if archive_samples is not None and len(archive_samples) > 0:
                base = archive_samples[i % len(archive_samples)]
                noise_scale = 15.0 * (i / len(worst_idx))
                new_pop[idx] = np.clip(base + np.random.randn(self.dim) * noise_scale, -100.0, 100.0)
            else:
                new_pop[idx] = np.random.uniform(-100.0, 100.0, self.dim)
        
        population = new_pop
    else:
        # Aggressive restart: full reinitialization with archive guidance
        new_pop = np.empty((self.NP, self.dim))
        
        if archive_samples is not None and len(archive_samples) > 0:
            # Fill first portion with archive-guided samples
            n_archive_fill = min(len(archive_samples), self.NP // 2)
            for i in range(n_archive_fill):
                base = archive_samples[i % len(archive_samples)]
                new_pop[i] = np.clip(base + np.random.randn(self.dim) * 20.0, -100.0, 100.0)
            
            # Fill remaining with random initialization
            for i in range(n_archive_fill, self.NP):
                new_pop[i] = np.random.uniform(-100.0, 100.0, self.dim)
        else:
            for i in range(self.NP):
                new_pop[i] = np.random.uniform(-100.0, 100.0, self.dim)
        
        # Perturb top performers from old population
        for i in range(min(3, len(best_idx))):
            old_best = population[best_idx[i]]
            new_pop[i] = np.clip(old_best + np.random.randn(self.dim) * 25.0, -100.0, 100.0)
        
        population = new_pop
    
    best_fitness = np.nanmin(fitness)
    best_solution = population[np.nanargmin(fitness)]
    
    return population, best_fitness, best_solution
```