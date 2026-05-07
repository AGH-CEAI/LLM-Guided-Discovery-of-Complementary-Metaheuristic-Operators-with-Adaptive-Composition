Propose a NEW population-based metaheuristic optimizer as a complete Python class.
 
HARD REQUIREMENTS (your code will be rejected if any of these are violated)
- Start with `import numpy as np`.
- One ```python``` code block only, nothing else.
- The class is instantiated as `obj = YourClass(dim=30)`. Signature MUST be
  `def __init__(self, dim, *args, **kwargs)` with NO other required arguments.
- Callable as: `obj(func, stopping_condition) -> (f_opt, x_opt)`.
- Problem dimension is `dim` (passed to __init__); search bounds are
  [-100, 100]^dim.
 
CRITICAL PERFORMANCE CONTRACT (violating these causes infinite-looking runs)
- `func` accepts a 2D array of shape (N, dim) and returns a 1D array of
  shape (N,) of fitness values. N can be any positive integer.
- You MUST call `func` on the WHOLE POPULATION (or whole batch of trials)
  at once per generation, e.g.:
      fitness = func(population)            # population shape (NP, dim)
      trial_fitness = func(trial_batch)     # trial_batch shape (NP, dim)
  NEVER evaluate candidates one-by-one in a Python for-loop. Per-individual
  calls like `func(x.reshape(1, -1))[0]` inside a loop over the population
  are PROHIBITED — they are ~100x slower and will exhaust wall-clock time.
- You MUST check `stopping_condition()` INSIDE any loop that evaluates
  fitness, not just once per generation. Typical correct pattern:
      while not stopping_condition():
          trials = build_trials_batched(population)      # vectorised, (NP, dim)
          trial_fit = func(trials)                       # ONE batched call
          if stopping_condition(): break
          population, fitness = select(population, fitness, trials, trial_fit)
- Near budget exhaustion `func` may return fewer values than you passed (it
  returns NaN past the budget, or truncates). Handle that: after every call
  check `len(trial_fit) < len(trials)` and truncate, and break cleanly.
- Keep NP modest for dim=30: roughly 4*dim to 10*dim is reasonable.
  Do NOT use NP > 1000.
- Clip every candidate to the bounds [-100, 100] BEFORE evaluation.
- Robust to edge cases: no NaN propagation into the best-tracking logic,
  no division by zero, no unbounded population growth.
 
STRUCTURAL REQUIREMENTS (very important)
- Decompose the algorithm into MANY METHODS, each with a single
  responsibility. Good examples (note the _batch suffix — these operate
  on the whole population at once, not one individual at a time):
      _initialize_population, _mutate_batch, _crossover_batch,
      _select_survivors_batch, _adapt_step_size, _restart_if_stagnant,
      _compute_diversity, ...
  Each method should be short (ideally between 5 - 25 lines) so it can later be
  swapped out in isolation. But each method must operate on BATCHES
  (2D arrays of shape (NP, dim)), not single individuals.
- Do NOT inline all logic inside __call__. __call__ should be a thin
  driver that orchestrates calls to the batched helper methods.
- Use clear, snake_case method names describing the operator.
- Avoid making methods that are too small (below < 4 lines)
 
CREATIVE DIRECTION
Use a hybrid: DE-style mutation + local search (e.g. Nelder–Mead-like) refinement.
 
Previously proposed seed classes in this run — propose something STRATEGICALLY
DIFFERENT from these (different operator family, different adaptation scheme,
different population topology, etc.):
- AdaptiveMultiStrategyDE
- CovarianceAdaptiveSwarmOptimizer
- AdaptiveTopologyCulturalPSO
 
FEEDBACK FROM PREVIOUS RUNS (use this to design a better algorithm):
Best algorithm so far: geomean = 1.417657e+01
  from lineage 4, cycle 13

Per-task errors of the best algorithm (lower = better):
  Tasks with high error are where you should focus improvement.
  Task  0: error=1.470e-01
  Task  1: error=2.982e+00  *** NEEDS IMPROVEMENT ***
  Task  2: error=1.909e-01
  Task  3: error=5.779e+00  *** NEEDS IMPROVEMENT ***
  Task  4: error=1.257e+00  *** NEEDS IMPROVEMENT ***
  Task  5: error=1.040e+03  *** NEEDS IMPROVEMENT ***
  Task  6: error=5.914e+02  *** NEEDS IMPROVEMENT ***
  Task  7: error=6.035e+00  *** NEEDS IMPROVEMENT ***
  Task  8: error=2.636e+00  *** NEEDS IMPROVEMENT ***
  Task  9: error=5.224e+01  *** NEEDS IMPROVEMENT ***
  Task 10: error=2.567e-01
  Task 11: error=6.491e+01  *** NEEDS IMPROVEMENT ***
  Task 12: error=4.297e-01
  Task 13: error=1.507e+01  *** NEEDS IMPROVEMENT ***
  Task 14: error=8.189e+00  *** NEEDS IMPROVEMENT ***
  Task 15: error=3.230e+00  *** NEEDS IMPROVEMENT ***
  Task 16: error=1.235e+03  *** NEEDS IMPROVEMENT ***
  Task 17: error=1.586e+04  *** NEEDS IMPROVEMENT ***
  Task 18: error=3.625e+01  *** NEEDS IMPROVEMENT ***
  Task 19: error=9.296e+01  *** NEEDS IMPROVEMENT ***
  Task 20: error=2.088e+01  *** NEEDS IMPROVEMENT ***
  Task 21: error=5.408e+00  *** NEEDS IMPROVEMENT ***
  Task 22: error=1.129e+01  *** NEEDS IMPROVEMENT ***
  Task 23: error=5.818e+01  *** NEEDS IMPROVEMENT ***

History stats: 14 cycles, 9 improvements, 0 crashes
Methods that responded well to adaptation:
  - _position_update_batch (2 successful adaptation(s))
  - _adapt_inertia_weight (2 successful adaptation(s))
  - _position_update_temporal_drift (2 successful adaptation(s))
  - _velocity_update_batch (1 successful adaptation(s))
  - _position_update_fitness_rank (1 successful adaptation(s))
  - _adapt_neighborhood_size (1 successful adaptation(s))

Design your algorithm so it has methods similar to the responsive ones above — they are more likely to benefit from later operator-level tuning.
