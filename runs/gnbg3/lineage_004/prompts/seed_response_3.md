```python
import numpy as np


class TopologyAdaptiveParticleSwarm:
    """
    Particle Swarm Optimizer with adaptive topology and multi-strategy selection.
    Uses ring topology with dynamically adapting neighborhood size, multiple
    velocity update strategies that are selected based on historical performance.
    """

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.pop_size = min(max(kwargs.get('pop_size', 6 * dim), 60), 300)
        self.bounds = kwargs.get('bounds', (-100.0, 100.0))
        
        # Population state
        self.population = None
        self.velocity = None
        self.local_best_pos = None
        self.local_best_fit = None
        self.best_pos = None
        self.best_fit = None
        
        # Strategy system
        self.strategies = ['standard', 'explorative', 'exploitative', 'hybrid']
        self.strategy_scores = np.ones(len(self.strategies))
        self.strategy_idx = 0
        
        # Adaptive PSO parameters
        self.inertia = 0.729
        self.cognitive = 1.493
        self.social = 1.493
        self.min_inertia = 0.3
        self.max_inertia = 0.9
        
        # Topology
        self.neighbor_size = max(3, dim // 10)
        self.ring_indices = None
        
        # Diversity and stagnation
        self.diversity_history = []
        self.stagnation_count = 0
        self.stagnation_limit = 30
        self.improvement_history = []
        
        # Tracking
        self.func = None
        self.generation = 0
        
    def __call__(self, func, stopping_condition):
        self.func = func
        self._initialize()
        
        while not stopping_condition():
            self.generation += 1
            
            # Build neighborhood structure
            r1 = np.random.random((self.pop_size, self.dim))
            r2 = np.random.random((self.pop_size, self.dim))
            self._build_ring_topology()
            
            # Select and apply strategy
            self.strategy_idx = self._select_strategy()
            self._adapt_strategy_parameters()
            
            # Generate trials
            trials = self._generate_trials_batch(r1, r2)
            trials = self._clip_to_bounds(trials)
            
            # Evaluate batch
            trial_fit = func(trials)
            
            # Handle budget exhaustion
            if len(trial_fit) < len(trials):
                valid = len(trial_fit)
                trials = trials[:valid]
                trial_fit = trial_fit[:valid]
            
            if stopping_condition():
                return self.best_fit, self.best_pos.copy()
            
            # Selection and tracking
            self._select_survivors_batch(trials, trial_fit)
            self._update_strategy_scores()
            self._update_adaptive_parameters()
            self._record_diversity()
            self._check_stagnation()
            
            if self.stagnation_count >= self.stagnation_limit:
                self._restart_if_needed()
        
        return self.best_fit, self.best_pos.copy()

    def _initialize(self):
        """Initialize population, velocities, and tracking structures."""
        lo, hi = self.bounds
        width = hi - lo
        
        self.population = np.random.uniform(lo, hi, (self.pop_size, self.dim))
        self.velocity = np.random.uniform(-0.1 * width, 0.1 * width, (self.pop_size, self.dim))
        
        self.local_best_pos = self.population.copy()
        
        # Evaluate initial population
        fit = self.func(self.population)
        self.local_best_fit = fit.copy()
        
        best_idx = np.nanargmin(fit)
        self.best_pos = self.population[best_idx].copy()
        self.best_fit = fit[best_idx]
        
        # Initialize ring topology
        self._build_ring_topology()
        
        self.generation = 0
        self.diversity_history = []
        self.improvement_history = []
        self.stagnation_count = 0

    def _build_ring_topology(self):
        """Build ring topology indices for neighborhood communication."""
        indices = np.arange(self.pop_size)
        self.ring_indices = np.zeros((self.pop_size, self.neighbor_size), dtype=int)
        
        half = self.neighbor_size // 2
        for i in range(self.pop_size):
            neighbors = np.concatenate([
                indices[(i - half) % self.pop_size:i],
                indices[i + 1:(i + half + 1) % self.pop_size]
            ])
            if len(neighbors) < half:
                neighbors = np.concatenate([neighbors, indices[:half - len(neighbors)]])
            self.ring_indices[i] = neighbors[:half]

    def _select_strategy(self):
        """Select strategy using roulette wheel on normalized scores."""
        scores = self.strategy_scores.copy()
        scores = np.maximum(scores, 0.01)
        probs = scores / scores.sum()
        idx = np.random.choice(len(self.strategies), p=probs)
        return idx

    def _adapt_strategy_parameters(self):
        """Modify PSO parameters based on selected strategy."""
        if self.strategies[self.strategy_idx] == 'explorative':
            self.inertia = max(self.min_inertia, self.inertia * 0.95)
            self.cognitive = 2.0
            self.social = 1.0
        elif self.strategies[self.strategy_idx] == 'exploitative':
            self.inertia = min(self.max_inertia, self.inertia * 1.05)
            self.cognitive = 1.0
            self.social = 2.0
        elif self.strategies[self.strategy_idx] == 'hybrid':
            self.inertia = 0.5 * (self.min_inertia + self.max_inertia)
            self.cognitive = 1.5
            self.social = 1.5
        else:
            self.inertia = 0.729
            self.cognitive = 1.493
            self.social = 1.493

    def _generate_trials_batch(self, r1, r2):
        """Generate trial positions using selected velocity strategy."""
        lbest = self._compute_local_best_batch()
        pbest = self.local_best_pos
        
        # Base velocity update
        cognitive_component = self.cognitive * r1 * (pbest - self.population)
        social_component = self.social * r2 * (lbest - self.population)
        velocity = self.inertia * self.velocity + cognitive_component + social_component
        
        # Strategy-specific velocity modifications
        strategy = self.strategies[self.strategy_idx]
        if strategy == 'explorative':
            velocity += 0.3 * np.random.randn(self.pop_size, self.dim)
        elif strategy == 'exploitative':
            velocity *= 0.8
        elif strategy == 'hybrid':
            # Add differential mutation component
            idx_a, idx_b, idx_c = np.random.randint(0, self.pop_size, (3, self.pop_size))
            diff = self.population[idx_b] - self.population[idx_c]
            velocity += 0.2 * diff
        
        # Apply velocity and update position
        trials = self.population + velocity
        return trials

    def _compute_local_best_batch(self):
        """Compute best position in each particle's neighborhood."""
        lbest = np.zeros((self.pop_size, self.dim))
        for i in range(self.pop_size):
            neighbor_fits = self.local_best_fit[self.ring_indices[i]]
            best_neighbor = self.ring_indices[i][np.nanargmin(neighbor_fits)]
            lbest[i] = self.local_best_pos[best_neighbor]
        return lbest

    def _clip_to_bounds(self, candidates):
        """Clip all candidates to search bounds."""
        lo, hi = self.bounds
        return np.clip(candidates, lo, hi)

    def _select_survivors_batch(self, trials, trial_fit):
        """Select survivors between current population and trials."""
        # Update local bests
        improved_local = trial_fit < self.local_best_fit
        for i in range(len(trials)):
            if improved_local[i]:
                self.local_best_fit[i] = trial_fit[i]
                self.local_best_pos[i] = trials[i].copy()
        
        # Update global best
        valid_mask = ~np.isnan(trial_fit)
        if np.any(valid_mask):
            best_trial_idx = np.nanargmin(trial_fit)
            if trial_fit[best_trial_idx] < self.best_fit:
                self.best_fit = trial_fit[best_trial_idx]
                self.best_pos = trials[best_trial_idx].copy()
                self.stagnation_count = 0
        
        # Greedy selection: keep better of current vs trial
        for i in range(len(trials)):
            if trial_fit[i] < self.local_best_fit[i]:
                self.population[i] = trials[i]
                self.velocity[i] = trials[i] - self.population[i]
            else:
                # Update velocity as difference for non-improving particles
                self.velocity[i] *= 0.9

    def _update_strategy_scores(self):
        """Update strategy scores based on recent performance."""
        decay = 0.95
        self.strategy_scores *= decay
        
        # Reward current strategy if global best improved
        if len(self.improvement_history) > 0 and self.improvement_history[-1]:
            self.strategy_scores[self.strategy_idx] += 1.0
        
        # Bonus for maintaining diversity
        if len(self.diversity_history) > 0:
            current_div = self.diversity_history[-1]
            if current_div > np.mean(self.diversity_history[-10:]) if len(self.diversity_history) >= 10 else True:
                self.strategy_scores[self.strategy_idx] += 0.5

    def _update_adaptive_parameters(self):
        """Adapt inertia based on diversity and progress."""
        if len(self.diversity_history) >= 10:
            recent_div = np.mean(self.diversity_history[-10:])
            if recent_div < 0.1 * (self.bounds[1] - self.bounds[0]):
                self.inertia = min(self.max_inertia, self.inertia * 1.1)
            else:
                self.inertia = max(self.min_inertia, self.inertia * 0.99)
        
        # Adapt neighborhood size based on stagnation
        if self.stagnation_count > 10:
            self.neighbor_size = min(self.neighbor_size + 1, self.pop_size // 2)
        elif self.stagnation_count == 0 and self.neighbor_size > 3:
            self.neighbor_size = max(3, self.neighbor_size - 1)

    def _record_diversity(self):
        """Record population diversity measure."""
        centroid = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - centroid, axis=1)
        diversity = np.mean(distances)
        self.diversity_history.append(diversity)
        
        # Track improvement
        current_best = self.best_fit
        if len(self.improvement_history) > 0:
            improved = current_best < self.improvement_history[-1]
        else:
            improved = False
        self.improvement_history.append(current_best)
        
        if not improved or np.isnan(current_best):
            self.stagnation_count += 1
        else:
            self.stagnation_count = max(0, self.stagnation_count - 1)

    def _check_stagnation(self):
        """Check if optimization has stagnated."""
        return self.stagnation_count >= self.stagnation_limit

    def _restart_if_needed(self):
        """Reinitialize portion of population when stagnated."""
        lo, hi = self.bounds
        width = hi - lo
        
        # Reinitialize worst half
        sorted_indices = np.argsort(self.local_best_fit)[::-1]
        restart_count = self.pop_size // 2
        
        for i in range(restart_count):
            idx = sorted_indices[i]
            self.population[idx] = np.random.uniform(lo, hi, self.dim)
            self.velocity[idx] = np.random.uniform(-0.1 * width, 0.1 * width, self.dim)
            self.local_best_pos[idx] = self.population[idx].copy()
        
        # Re-evaluate restarted particles
        restart_mask = np.zeros(self.pop_size, dtype=bool)
        restart_mask[sorted_indices[:restart_count]] = True
        
        if np.any(restart_mask):
            fit = self.func(self.population[restart_mask])
            self.local_best_fit[restart_mask] = fit
        
        # Reset counters
        self.stagnation_count = 0
        self.diversity_history = []
        
        # Perturb global best slightly
        if self.best_pos is not None:
            perturbation = np.random.randn(self.dim) * 0.1 * width
            self.best_pos = np.clip(self.best_pos + perturbation, lo, hi)
```