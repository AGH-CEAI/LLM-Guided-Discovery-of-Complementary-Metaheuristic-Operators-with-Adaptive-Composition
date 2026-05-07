import numpy as np


class AdaptiveDirectionalDE:
    """
    Adaptive Directional Differential Evolution (ADDE)
    
    A DE variant with:
    - Ring topology for mutation parent selection
    - Directional memory that biases mutation toward successful directions
    - Adaptive F and Cr based on historical success rates
    - Composite mutation: directional + rand/1 blended adaptively
    - Diversity-triggered restart mechanism
    """
    
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.np = min(max(4 * dim, 60), 300)  # Population size
        self.lower = -100.0
        self.upper = 100.0
        
        # Control parameters
        self.F_base = 0.5
        self.Cr_base = 0.3
        self.F = self.F_base
        self.Cr = self.Cr_base
        
        # Directional memory
        self.direction_memory_size = 5
        self.successful_directions = []
        self.direction_decay = 0.95
        
        # Adaptation tracking
        self.adaptation_window = 10
        self.F_success_history = []
        self.Cr_success_history = []
        
        # Restart control
        self.stagnation_threshold = 50
        self.min_diversity_threshold = 1e-6
        self.generation = 0
        
    def __call__(self, func, stopping_condition):
        return self._optimize(func, stopping_condition)
    
    def _initialize_population(self, func):
        """Initialize using Latin Hypercube Sampling for better spread."""
        sampler = np.linspace(self.lower, self.upper, self.np)
        population = np.zeros((self.np, self.dim))
        
        for d in range(self.dim):
            indices = np.random.permutation(self.np)
            offsets = np.random.uniform(0, 1, self.np)
            population[:, d] = sampler[indices] + offsets * (sampler[1] - sampler[0])
        
        population = self._clip_to_bounds(population)
        fitness = func(population)
        
        if len(fitness) < self.np:
            fitness = np.resize(fitness, self.np)
            fitness[np.isnan(fitness)] = np.inf
        
        best_idx = np.argmin(fitness)
        return population, fitness, fitness[best_idx], population[best_idx].copy()
    
    def _clip_to_bounds(self, population):
        """Hard clip all candidates to search bounds."""
        return np.clip(population, self.lower, self.upper)
    
    def _build_ring_topology(self):
        """Create ring neighbor indices for each population member."""
        indices = np.arange(self.np)
        ring_prev = np.roll(indices, 1)
        ring_next = np.roll(indices, -1)
        return ring_prev, ring_next
    
    def _compute_population_centroid(self, population, fitness):
        """Compute fitness-weighted centroid for directional bias."""
        weights = 1.0 / (fitness - np.min(fitness) + 1e-10)
        weights = weights / np.sum(weights)
        centroid = np.sum(population * weights[:, np.newaxis], axis=0)
        return centroid
    
    def _update_direction_memory(self, successful_mutations, successful_mask):
        """Track successful mutation directions for adaptive biasing."""
        if not np.any(successful_mask):
            self.successful_directions = [
                d * self.direction_decay for d in self.successful_directions
            ]
            return
        
        for mutation_vec in successful_mutations[successful_mask]:
            self.successful_directions.append(mutation_vec.copy())
        
        if len(self.successful_directions) > self.direction_memory_size:
            self.successful_directions = self.successful_directions[-self.direction_memory_size:]
    
    def _compute_directional_bias(self, target_indices):
        """Compute directional bias from historical successful mutations."""
        if len(self.successful_directions) == 0:
            return np.zeros((len(target_indices), self.dim))
        
        direction_stack = np.array(self.successful_directions)
        weights = np.exp(-0.5 * np.arange(len(direction_stack))[::-1])
        weights = weights / np.sum(weights)
        
        weighted_direction = np.sum(direction_stack * weights[:, np.newaxis], axis=0)
        directional_bias = np.tile(weighted_direction, (len(target_indices), 1))
        
        return 0.3 * directional_bias
    
    def _mutate_batch(self, population, fitness, ring_prev, ring_next):
        """
        Generate mutation vectors using composite strategy:
        - Ring-based rand/1 component
        - Directional component from historical memory
        """
        np_pop = len(population)
        trials = np.zeros((np_pop, self.dim))
        
        centroid = self._compute_population_centroid(population, fitness)
        directional_bias = self._compute_directional_bias(np.arange(np_pop))
        
        # Select ring-based parents
        r1 = ring_prev
        r2 = ring_next
        
        # Standard DE rand/1 mutation
        rand1_vectors = population[r1]
        rand2_vectors = population[r2]
        
        # Adaptive F with perturbation based on diversity
        current_F = self.F * (1.0 + 0.1 * np.random.randn())
        current_F = np.clip(current_F, 0.1, 2.0)
        
        # Base mutation: ring-based differential
        base_mutation = rand1_vectors + current_F * (rand2_vectors - population)
        
        # Directional component: bias toward successful regions
        directional_component = directional_bias
        
        # Blend based on adaptation success
        blend_weight = 0.2 + 0.3 * (1.0 - self.Cr)
        trials = (1 - blend_weight) * base_mutation + blend_weight * (centroid + directional_component)
        
        return trials, current_F
    
    def _crossover_batch(self, population, mutant_batch):
        """Binomial crossover with adaptive Cr."""
        np_pop, dim = population.shape
        trials = np.zeros((np_pop, dim))
        
        # Per-individual Cr with bounded distribution
        Cr_individual = np.clip(
            self.Cr + 0.1 * np.random.randn(np_pop),
            0.1, 0.9
        )
        
        # Ensure at least one dimension is crossed
        j_rand = np.random.randint(0, dim, size=np_pop)
        
        for i in range(np_pop):
            mask = np.random.rand(dim) < Cr_individual[i]
            mask[j_rand[i]] = True
            trials[i] = np.where(mask, mutant_batch[i], population[i])
        
        return trials, Cr_individual
    
    def _evaluate_batch(self, batch, func):
        """Evaluate batch, handle budget exhaustion gracefully."""
        batch = self._clip_to_bounds(batch)
        fitness = func(batch)
        
        if len(fitness) < len(batch):
            valid_len = len(fitness)
            fitness = np.resize(fitness, len(batch))
            fitness[valid_len:] = np.inf
        
        fitness = np.where(np.isnan(fitness), np.inf, fitness)
        return fitness
    
    def _select_survivors_batch(self, population, fitness, trials, trial_fitness):
            """Greedy selection with spectral diversity modulation."""
            improved_mask = trial_fitness < fitness
            improvement = fitness - trial_fitness

            # Compute population covariance eigenvalue spread
            centered = population - np.mean(population, axis=0)
            cov = np.cov(centered.T)

            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1]

            # Eigenvalue spread: ratio of largest to smallest non-zero eigenvalue
            # High spread → anisotropic population → reduce selection pressure
            if eigvals[0] > 1e-12 and eigvals[-1] > 1e-12:
                condition = eigvals[0] / eigvals[-1]
            else:
                condition = 1.0

            # Modulation: in ill-conditioned populations, accept smaller improvements
            # log(condition) maps: 1→0, 10→2.3, 100→4.6, 1000→6.9
            diversity_bonus = np.log1p(condition)

            # Scaled improvement relative to population fitness range
            fit_range = np.ptp(fitness)
            fit_range = max(fit_range, 1e-10)
            scaled_improvement = improvement / fit_range

            # Accept if improved OR if scaled improvement exceeds diversity threshold
            accept_threshold = 0.01 / (1.0 + 0.5 * diversity_bonus)
            accept_mask = improved_mask | (scaled_improvement > accept_threshold)

            new_population = population.copy()
            new_fitness = fitness.copy()

            new_population[accept_mask] = trials[accept_mask]
            new_fitness[accept_mask] = trial_fitness[accept_mask]

            return new_population, new_fitness, accept_mask
    
    def _compute_diversity(self, population):
        """Compute population diversity via average pairwise distance."""
        centroid = np.mean(population, axis=0)
        distances = np.linalg.norm(population - centroid, axis=1)
        return np.mean(distances)
    
    def _adapt_parameters(self, improved_mask, F_used, Cr_used):
        """Adapt F and Cr using temporal autocorrelation, rate-of-change, and drift."""
        success_rate = np.mean(improved_mask)

        # Initialize temporal state on first call
        if not hasattr(self, 'success_history'):
            self.success_history = []
            self.F_ewma = self.F
            self.Cr_ewma = self.Cr
            self.stagnation_window = []
            self.stagnation_counter = 0
            self.last_best = np.inf

        # Append to rolling time series
        self.success_history.append(success_rate)
        if len(self.success_history) > 20:
            self.success_history.pop(0)

        # Track improvement for stagnation window
        current_best = np.min(self.population_fitness) if hasattr(self, 'population_fitness') else np.inf
        improved_this_gen = current_best < self.last_best
        self.last_best = current_best

        self.stagnation_window.append(1 if improved_this_gen else 0)
        if len(self.stagnation_window) > 12:
            self.stagnation_window.pop(0)

        # Compute rate-of-change: derivative-like signal from EMA difference
        alpha_fast = 0.4
        alpha_slow = 0.15
        if not hasattr(self, 'ewma_fast'):
            self.ewma_fast = success_rate
            self.ewma_slow = success_rate
        self.ewma_fast = alpha_fast * success_rate + (1 - alpha_fast) * self.ewma_fast
        self.ewma_slow = alpha_slow * success_rate + (1 - alpha_slow) * self.ewma_slow
        rate_of_change = self.ewma_fast - self.ewma_slow  # Positive = improving, negative = declining

        # Compute autocorrelation at lag-1 using time series
        n_hist = len(self.success_history)
        autocorr = 0.0
        if n_hist >= 4:
            series = np.array(self.success_history)
            mean_series = np.mean(series)
            var_series = np.var(series)
            if var_series > 1e-12:
                autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
                autocorr = np.clip(autocorr, -1.0, 1.0)

        # Compute temporal drift: EMA-weighted parameter history
        self.F_ewma = 0.7 * self.F_ewma + 0.3 * F_used
        self.Cr_ewma = 0.7 * self.Cr_ewma + 0.3 * Cr_used
        F_drift_ratio = self.F / (self.F_ewma + 1e-12)
        Cr_drift_ratio = self.Cr / (self.Cr_ewma + 1e-12)

        # Stagnation severity from time-window
        stagnation_ratio = np.mean(self.stagnation_window)  # Fraction of generations with improvement

        # Regime detection via temporal signals
        is_oscillating = autocorr > 0.4 and abs(rate_of_change) < 0.05
        is_declining = rate_of_change < -0.02
        is_stagnant = stagnation_ratio < 0.2
        is_converging = rate_of_change > 0.02 and stagnation_ratio > 0.5

        # Temporal adjustment logic
        if is_oscillating:
            # Cyclic behavior detected: inject diversity via F boost
            F_adjustment = 1.25 * np.clip(F_drift_ratio, 0.8, 1.3)
            Cr_adjustment = 0.85
            self.stagnation_counter += 1
        elif is_declining and is_stagnant:
            # Declining with stagnation: strong exploration push
            F_adjustment = 1.15 * (1.1 - stagnation_ratio) / np.clip(F_drift_ratio, 0.8, 1.4)
            Cr_adjustment = 0.88
            self.stagnation_counter += 1
        elif is_stagnant and stagnation_ratio < 0.1:
            # Deep stagnation: large perturbation
            F_adjustment = 0.7
            Cr_adjustment = 1.2
            self.stagnation_counter += 2
        elif is_converging:
            # Steady improvement: fine-tune toward exploitation
            F_adjustment = 0.95 * np.clip(F_drift_ratio, 0.9, 1.1)
            Cr_adjustment = 1.05
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
        else:
            # Neutral: gentle correction based on rate
            F_adjustment = 1.0 + 0.1 * rate_of_change
            Cr_adjustment = 1.0 - 0.05 * rate_of_change

        # Apply adjustments
        self.F = np.clip(self.F * F_adjustment, 0.1, 2.0)
        self.Cr = np.clip(self.Cr * Cr_adjustment, 0.05, 0.95)

        # Decay stagnation counter slowly
        self.stagnation_counter = max(0, self.stagnation_counter - 0.1)
    
    def _restart_if_stagnant(self, population, fitness, best_fitness, best_solution):
        """Reinitialize portion of population if stagnation detected."""
        current_best = np.min(fitness)
        
        if current_best >= best_fitness:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            best_fitness = current_best
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx].copy()
        
        diversity = self._compute_diversity(population)
        
        if (self.stagnation_counter > self.stagnation_threshold or 
            diversity < self.min_diversity_threshold):
            
            n_replace = max(self.np // 5, 2)
            replace_indices = np.random.choice(self.np, n_replace, replace=False)
            
            for idx in replace_indices:
                population[idx] = np.random.uniform(
                    self.lower, self.upper, self.dim
                )
            
            fitness[replace_indices] = np.inf
            self.stagnation_counter = 0
            self.successful_directions = []
        
        return population, fitness, best_fitness, best_solution
    
    def _optimize(self, func, stopping_condition):
        """Main optimization loop orchestrating all batched operations."""
        population, fitness, best_fitness, best_solution = self._initialize_population(func)
        
        self.stagnation_counter = 0
        self.generation = 0
        
        ring_prev, ring_next = self._build_ring_topology()
        
        while not stopping_condition():
            self.generation += 1
            
            # Build mutation batch
            mutant_batch, F_used = self._mutate_batch(
                population, fitness, ring_prev, ring_next
            )
            
            # Build crossover batch
            trial_batch, Cr_used = self._crossover_batch(population, mutant_batch)
            
            # Check budget before evaluation
            if stopping_condition():
                break
            
            # Evaluate trial batch
            trial_fitness = self._evaluate_batch(trial_batch, func)
            
            if stopping_condition():
                break
            
            # Select survivors
            population, fitness, improved_mask = self._select_survivors_batch(
                population, fitness, trial_batch, trial_fitness
            )
            
            # Update best solution
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
            
            # Compute successful mutations for direction memory
            successful_mutations = trial_batch - population
            self._update_direction_memory(successful_mutations, improved_mask)
            
            # Adapt control parameters
            self._adapt_parameters(improved_mask, F_used, Cr_used)
            
            # Check for stagnation and restart if needed
            population, fitness, best_fitness, best_solution = self._restart_if_stagnant(
                population, fitness, best_fitness, best_solution
            )
            
            # Rebuild topology after potential restart
            ring_prev, ring_next = self._build_ring_topology()
        
        return best_fitness, best_solution
