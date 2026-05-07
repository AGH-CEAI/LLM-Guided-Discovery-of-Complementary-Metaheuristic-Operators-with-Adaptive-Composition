Looking at the priority targets (errors of 1e+01 to 1e+04), I need a hybrid approach combining two distinct mechanisms with principled switching. The current implementation uses spectral condition number for diversity modulation. I'll add temporal stagnation detection as a second mechanism with schedule-free, data-driven regime switching.

**Idea: Temporal-Regime Hybrid with Stagnation-Triggered Diversity**

Combines: (1) Temporal regime detection via EMA divergence tracking generational fitness trends, with (2) Conditional switching between greedy acceptance and rank-based acceptance. The switching is data-driven: when short-term success rate falls below long-term EMA, we switch to rank-based to preserve diversity.

```python
def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
    """Hybrid: temporal regime detection + conditional acceptance switching."""
    improved_mask = trial_fitness < fitness
    improvement = fitness - trial_fitness

    # --- Mechanism 1: Temporal regime detection ---
    if not hasattr(self, 'ema_short'):
        self.ema_short = 0.0
        self.ema_long = 0.0
        self.fitness_history = []

    current_best = np.min(fitness)
    self.fitness_history.append(current_best)
    if len(self.fitness_history) > 30:
        self.fitness_history.pop(0)

    # Compute fitness delta over recent window
    if len(self.fitness_history) >= 5:
        delta_short = self.fitness_history[-1] - self.fitness_history[-5]
    else:
        delta_short = 0.0

    if len(self.fitness_history) >= 15:
        delta_long = self.fitness_history[-1] - self.fitness_history[-15]
    else:
        delta_long = delta_short if len(self.fitness_history) > 1 else 0.0

    # EMA for regime tracking
    alpha_short, alpha_long = 0.3, 0.1
    self.ema_short = alpha_short * delta_short + (1 - alpha_short) * self.ema_short
    self.ema_long = alpha_long * delta_long + (1 - alpha_long) * self.ema_long

    # Regime: stagnation when short-term improvement stalls vs long-term trend
    stagnation_threshold = 0.1
    in_stagnation = (self.ema_short < stagnation_threshold) and (self.ema_short < self.ema_long)

    # --- Mechanism 2: Conditional acceptance switching ---
    if in_stagnation:
        # Rank-based acceptance to preserve diversity
        combined_fitness = np.where(improved_mask, trial_fitness, fitness)
        ranks = np.argsort(np.argsort(combined_fitness))
        n = len(fitness)
        rank_threshold = int(n * 0.4)
        accept_mask = ranks <= rank_threshold
    else:
        # Greedy acceptance for exploitation
        accept_mask = improved_mask

    new_population = population.copy()
    new_fitness = fitness.copy()

    new_population[accept_mask] = trials[accept_mask]
    new_fitness[accept_mask] = trial_fitness[accept_mask]

    return new_population, new_fitness, accept_mask
```