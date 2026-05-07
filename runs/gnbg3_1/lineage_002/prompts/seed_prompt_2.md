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
Use a CMA-ES style covariance-adaptation scheme but with a simplified update.
 
Previously proposed seed classes in this run — propose something STRATEGICALLY
DIFFERENT from these (different operator family, different adaptation scheme,
different population topology, etc.):
- AdaptiveNeighborhoodDE
 
FEEDBACK FROM PREVIOUS RUNS (use this to design a better algorithm):
Best algorithm so far: geomean = 2.588790e+00
  from lineage 1, cycle 2

Per-task errors of the best algorithm (lower = better):
  Tasks with high error are where you should focus improvement.
  Task  0: error=5.807e-02
  Task  1: error=4.800e-02
  Task  2: error=3.872e-01
  Task  3: error=7.284e-01
  Task  4: error=5.374e-01
  Task  5: error=3.807e-04
  Task  6: error=1.302e+00  *** NEEDS IMPROVEMENT ***
  Task  7: error=1.164e-01
  Task  8: error=8.042e-01
  Task  9: error=7.005e+00  *** NEEDS IMPROVEMENT ***
  Task 10: error=1.497e+01  *** NEEDS IMPROVEMENT ***
  Task 11: error=1.438e+01  *** NEEDS IMPROVEMENT ***
  Task 12: error=2.598e+01  *** NEEDS IMPROVEMENT ***
  Task 13: error=2.821e+00  *** NEEDS IMPROVEMENT ***
  Task 14: error=2.823e+00  *** NEEDS IMPROVEMENT ***
  Task 15: error=2.764e+00  *** NEEDS IMPROVEMENT ***
  Task 16: error=1.373e+02  *** NEEDS IMPROVEMENT ***
  Task 17: error=5.272e+02  *** NEEDS IMPROVEMENT ***
  Task 18: error=2.138e+01  *** NEEDS IMPROVEMENT ***
  Task 19: error=1.449e+01  *** NEEDS IMPROVEMENT ***
  Task 20: error=2.341e+01  *** NEEDS IMPROVEMENT ***
  Task 21: error=4.485e+00  *** NEEDS IMPROVEMENT ***
  Task 22: error=9.351e+00  *** NEEDS IMPROVEMENT ***
  Task 23: error=2.207e+01  *** NEEDS IMPROVEMENT ***

History stats: 3 cycles, 3 improvements, 0 crashes
Methods that responded well to adaptation:
  - _adapt_covariance (1 successful adaptation(s))
  - _adapt_step_size (1 successful adaptation(s))
  - _sample_trials_batch (1 successful adaptation(s))

Design your algorithm so it has methods similar to the responsive ones above — they are more likely to benefit from later operator-level tuning.
