Propose a NEW population-based metaheuristic optimizer as a complete Python class.

HARD REQUIREMENTS
- Start with `import numpy as np`.
- One ```python``` code block only, nothing else.
- The class must be instantiable with no required args and callable as:
      obj(func, stopping_condition)  ->  (f_opt, x_opt)
  where `func` is a callable taking a 2D array (pop x dim) of candidates and
  returning a 1D array of fitness values, and `stopping_condition` is a
  callable taking no argument and returning True when the algorithm should stop.
- Problem dimension is 30; search bounds are [-100, 100]^30.
- Clip every candidate to the bounds before evaluation.
- Robust to edge cases: no NaN, no division by zero, no runaway population size.

STRUCTURAL REQUIREMENTS (very important)
- Decompose the algorithm into MANY SMALL METHODS, each with a single
  responsibility. Good examples:
      _initialize_population, _mutate, _crossover, _select_survivors,
      _adapt_step_size, _restart_if_stagnant, _compute_diversity, ...
  Each such method should be short (ideally < 25 lines) so it can later be
  swapped out in isolation.
- Do NOT inline all logic inside __call__. __call__ should be a thin driver
  that orchestrates calls to the helper methods.
- Use clear, snake_case method names describing the operator.

CREATIVE DIRECTION
Use a hybrid: DE-style mutation + local search (e.g. Nelder–Mead-like) refinement.

Previously proposed seed classes in this run — propose something STRATEGICALLY
DIFFERENT from these (different operator family, different adaptation scheme,
different population topology, etc.):
- AdaptiveArchiveDEA
- CovarianceGuidedEvolutionStrategy
- TopologyAdaptiveParticleSwarmOptimizer
